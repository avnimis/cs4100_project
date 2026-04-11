import json
import math
import re
import sys
import torch
import torch.nn as nn
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split

DATA_DIR      = "labeled"        # folder containing one .json file per class
MODEL_PATH    = "best_model.pth" # where to save the best checkpoint
BATCH_SIZE    = 32               # how many emails per gradient descent step
LEARNING_RATE = 1e-3             # alpha (α): w = w - α∇L
NUM_EPOCHS    = 30               # full passes over training data
VAL_SPLIT     = 0.2              # 20% held out for validation
HIDDEN_SIZE   = 128              # width of hidden layers
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_LABELS = [
    "duplicate",      # Near-duplicate of another email
    "expired_offer",  # Contains expired promotions/offers
    "keep",           # Email worth keeping
    "old_alert",      # Alert/notification that is no longer relevant
    "past_event",     # Contains dates for events that have passed
    "spam",           # General spam
]
NUM_CLASSES   = len(ALL_LABELS)                        # 6
LABEL_TO_IDX  = {label: i for i, label in enumerate(ALL_LABELS)}
IDX_TO_LABEL  = {i: label for label, i in LABEL_TO_IDX.items()}

SPAM_WORDS = {
    "unsubscribe", "click here", "free", "winner", "selected", "prize",
    "offer", "deal", "sale", "discount", "limited time", "act now",
    "congratulations", "claim", "gift card", "reward", "opt-out",
    "no-reply", "noreply", "promotion", "coupon", "expires", "promo",
}

EVENT_WORDS = {
    "webinar", "conference", "event", "meetup", "workshop", "seminar",
    "reminder", "starts at", "join us", "register", "rsvp", "session",
    "talk", "lecture", "summit", "hackathon",
}

ALERT_WORDS = {
    "shipped", "delivered", "tracking", "order", "receipt", "invoice",
    "verification", "code", "otp", "alert", "notification", "login",
    "sign-in", "password", "reset", "confirm", "two-factor", "2fa",
}


# LOAD DATA
def load_json_file(path: str) -> list[dict]:
    """Load emails from a JSON array or JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def load_dataset(data_dir: str) -> list[dict]:
    """
    Scan data_dir for *.json files. Stamp each email with its label
    (the filename without extension) — exactly as ImageFolder stamps
    each image with its parent subfolder name.

    Prints the class distribution so you can spot imbalances early.
    """
    data_dir = Path(data_dir)
    all_emails: list[dict] = []

    json_files = sorted(data_dir.glob("*.jsonl"))
    if not json_files:
        raise FileNotFoundError(f"No .jsonl files found in '{data_dir}'. "
                                f"Expected one file per class, e.g. keep.jsonl, spam.jsonl")

    print(f"Classes found in '{data_dir}':")
    for path in json_files:
        label = path.stem 
        if label not in LABEL_TO_IDX:
            print(f"  WARNING: '{label}.jsonl' is not in ALL_LABELS — skipping")
            continue
        emails = load_json_file(str(path))
        for e in emails:
            e["label"] = label
        print(f"  {label:<16} → {len(emails):>5} emails  (class index {LABEL_TO_IDX[label]})")
        all_emails.extend(emails)

    print(f"\nTotal: {len(all_emails)} emails across {len(json_files)} classes\n")
    return all_emails

FEATURE_DIM = 30

def extract_features(email: dict, now: datetime | None = None) -> list[float]:
    """
    Convert a raw email dict into a fixed-length numeric feature vector.

    Features (30 total):
      [0]  has_attachment (0/1)
      [1]  is_noreply sender (0/1)
      [2]  subject length (log-scaled)
      [3]  body length (log-scaled)
      [4]  num links (log-scaled)
      [5]  link density: links / max(body_words, 1)
      [6]  spam word hit count (log-scaled)
      [7]  event word hit count (log-scaled)
      [8]  alert word hit count (log-scaled)
      [9]  has list-unsubscribe header (0/1)
      [10] has spam headers (0/1)
      [11] is in Trash gmail label (0/1)
      [12] is in Spam gmail label (0/1)
      [13] is in Updates category (0/1)
      [14] is in Promotions category (0/1)
      [15] email age in days (log-scaled, 0 if unknown)
      [16] sent on a weekend (0/1)
      [17] sent in business hours 9–17 UTC (0/1)
      [18] all-caps word ratio in subject
      [19] exclamation count in subject (log-scaled)
      [20] question mark count in body (log-scaled)
      [21] ratio of uppercase chars in body
      [22] digit ratio in body (numbers-heavy → receipts/codes)
      [23] url ratio: unique domains / max(links, 1)
      [24] from domain is free mail provider (gmail/yahoo/hotmail)
      [25] subject contains date pattern (0/1)
      [26] body contains date pattern (0/1)
      [27] body contains price/currency pattern (0/1)
      [28] is HTML email (0/1)
      [29] duplicate subject flag — filled in by mark_duplicates()
    """
    if now is None:
        now = datetime.now(timezone.utc)

    subject    = str(email.get("subject", "") or "")
    body       = str(email.get("body_text", "") or "")
    from_email = str(email.get("from_email", "") or "").lower()
    labels     = [str(l).lower() for l in (email.get("gmail_labels") or [])]
    links      = email.get("links") or []
    content_t  = str(email.get("content_type", "") or "").lower()

    subject_l  = subject.lower()
    body_l     = body.lower()
    body_words = body.split()

    email_age_days = 0.0
    sent_weekend   = 0.0
    sent_biz_hours = 0.0
    date_str = email.get("date_iso") or email.get("date_raw") or ""
    if date_str:
        try:
            sent = datetime.fromisoformat(date_str)
            if sent.tzinfo is None:
                sent = sent.replace(tzinfo=timezone.utc)
            email_age_days = max((now - sent).days, 0)
            sent_weekend   = float(sent.weekday() >= 5)
            sent_biz_hours = float(9 <= sent.hour <= 17)
        except (ValueError, TypeError):
            pass

    combined   = subject_l + " " + body_l
    spam_hits  = sum(1 for w in SPAM_WORDS  if w in combined)
    event_hits = sum(1 for w in EVENT_WORDS if w in combined)
    alert_hits = sum(1 for w in ALERT_WORDS if w in combined)

    subject_words = subject.split()
    allcaps_ratio = (
        sum(1 for w in subject_words if w.isupper() and len(w) > 1)
        / max(len(subject_words), 1)
    )
    exclaim_count  = subject.count("!")
    question_count = body.count("?")

    upper_ratio = sum(1 for c in body if c.isupper()) / max(len(body), 1)
    digit_ratio = sum(1 for c in body if c.isdigit()) / max(len(body), 1)

    domains = set()
    for link in links:
        m = re.search(r"https?://([^/]+)", str(link))
        if m:
            domains.add(m.group(1).lower())
    url_ratio = len(domains) / max(len(links), 1)

    free_providers = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com"}
    sender_domain  = from_email.split("@")[-1] if "@" in from_email else ""
    is_freemail    = float(sender_domain in free_providers)
    is_noreply     = float("noreply" in from_email or "no-reply" in from_email)

    date_pattern   = r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2},? \d{4})\b"
    price_pattern  = r"(\$|€|£|usd|eur|gbp)\s*\d"
    subj_has_date  = float(bool(re.search(date_pattern, subject, re.I)))
    body_has_date  = float(bool(re.search(date_pattern, body,    re.I)))
    body_has_price = float(bool(re.search(price_pattern, body_l)))

    def logp1(x):
        return math.log1p(float(x))

    features = [
        float(bool(email.get("has_attachment"))),           # [0]
        is_noreply,                                         # [1]
        logp1(len(subject)),                                # [2]
        logp1(len(body)),                                   # [3]
        logp1(len(links)),                                  # [4]
        len(links) / max(len(body_words), 1),               # [5]
        logp1(spam_hits),                                   # [6]
        logp1(event_hits),                                  # [7]
        logp1(alert_hits),                                  # [8]
        float(bool(email.get("list_unsubscribe"))),         # [9]
        float(bool(email.get("spam_headers"))),             # [10]
        float("trash" in labels),                           # [11]
        float("spam" in labels),                            # [12]
        float(any("updates" in l for l in labels)),         # [13]
        float(any("promotions" in l for l in labels)),      # [14]
        logp1(email_age_days),                              # [15]
        sent_weekend,                                       # [16]
        sent_biz_hours,                                     # [17]
        allcaps_ratio,                                      # [18]
        logp1(exclaim_count),                               # [19]
        logp1(question_count),                              # [20]
        upper_ratio,                                        # [21]
        digit_ratio,                                        # [22]
        url_ratio,                                          # [23]
        is_freemail,                                        # [24]
        subj_has_date,                                      # [25]
        body_has_date,                                      # [26]
        body_has_price,                                     # [27]
        float("html" in content_t),                        # [28]
        0.0,
    ]

    assert len(features) == FEATURE_DIM, f"Expected {FEATURE_DIM} features, got {len(features)}"
    return features


def mark_duplicates(emails: list[dict]) -> None:
    """
    Set '_duplicate_flag' = 1.0 on any email whose normalized subject appears
    more than once in the batch. Mutates emails in-place.

    Requires seeing the full batch first — analogous to how the photo classifier
    needs all images loaded before it can detect near-duplicates across folders.
    """
    subject_counts = Counter(
        re.sub(r"\s+", " ", str(e.get("subject", "")).lower().strip())
        for e in emails
    )
    for email in emails:
        key = re.sub(r"\s+", " ", str(email.get("subject", "")).lower().strip())
        email["_duplicate_flag"] = float(subject_counts[key] > 1)


def build_feature_tensor(email: dict, now: datetime | None = None) -> torch.Tensor:
    """Extract features and return as a float32 tensor of shape (FEATURE_DIM,)."""
    feats = extract_features(email, now)
    feats[29] = email.get("_duplicate_flag", 0.0)
    return torch.tensor(feats, dtype=torch.float32)


# DATASET
class EmailDataset(Dataset):
    def __init__(self, emails: list[dict], now: datetime | None = None):
        mark_duplicates(emails)
        self.now    = now or datetime.now(timezone.utc)
        self.emails = emails

        dist = Counter(e.get("label", "unknown") for e in emails)
        print(f"Dataset: {len(emails)} emails")
        print(f"Class distribution: {dict(sorted(dist.items()))}")

    def __len__(self):
        return len(self.emails)

    def __getitem__(self, idx):
        email    = self.emails[idx]
        features = build_feature_tensor(email, self.now)
        class_idx = LABEL_TO_IDX[email["label"]]
        return features, torch.tensor(class_idx, dtype=torch.long)


# NEURAL NETWORK
class EmailNet(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_size: int = HIDDEN_SIZE):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(hidden_size // 4, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass — PyTorch records this computation graph.
        loss.backward() traverses it in reverse (chain rule → ∂L/∂w per weight).
        optimizer.step() then performs: w = w - α * ∂L/∂w
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for features, labels in loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)

        logits = model(features)
        
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss += loss.item() * features.size(0)
        predicted   = logits.argmax(dim=1)
        correct    += (predicted == labels).sum().item()
        total      += features.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            logits = model(features)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * features.size(0)
            predicted   = logits.argmax(dim=1)
            correct    += (predicted == labels).sum().item()
            total      += features.size(0)

    return total_loss / total, correct / total

def main():
    print(f"Using device: {DEVICE}\n")

    all_emails = load_dataset(DATA_DIR)

    dataset = EmailDataset(all_emails)
    n       = len(dataset)
    n_val   = int(n * VAL_SPLIT)
    n_train = n - n_val

    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {n_train} | Val: {n_val}\n")

    model = EmailNet().to(DEVICE)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {total_params:,}\n")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>10} | {'Val Loss':>10} | {'Val Acc':>10}")
    print("-" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            flag = " ← best"
        else:
            flag = ""

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>10.2%} | "
              f"{val_loss:>10.4f} | {val_acc:>10.2%}{flag}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {MODEL_PATH}")


def predict(emails: list[dict], model_path: str = MODEL_PATH) -> list[dict]:
    """
    load saved model and classify each email into one of NUM_CLASSES categories.

    return list of result dicts — one per email:
      {
        "id":              "00001",
        "subject":         "Flash Sale ...",
        "label":           "expired_offer",
        "probabilities": {
          "keep":           0.0312,
          "duplicate":      0.0091,
          "expired_offer":  0.8743,
          "old_alert":      0.0421,
          "past_event":     0.0280,
          "spam":           0.0153
        }
      }
    """
    model = EmailNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    mark_duplicates(emails)
    now = datetime.now(timezone.utc)

    results = []
    with torch.no_grad():
        for email in emails:
            tensor = build_feature_tensor(email, now).unsqueeze(0).to(DEVICE)
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze(0)

            pred_idx   = probs.argmax().item()
            pred_label = IDX_TO_LABEL[pred_idx]

            results.append({
                "id":      email.get("id"),
                "subject": email.get("subject", ""),
                "label":   pred_label,
                "probabilities": {
                    IDX_TO_LABEL[i]: round(probs[i].item(), 4)
                    for i in range(NUM_CLASSES)
                },
            })

    return results


def predict_file(input_path: str, output_path: str) -> None:
    """Convenience wrapper: load emails from file, predict, write output JSON."""
    emails  = load_json_file(input_path)
    results = predict(emails)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    label_counts = Counter(r["label"] for r in results)
    keep_n   = label_counts.pop("keep", 0)
    delete_n = len(results) - keep_n
    print(f"\nResults: {len(results)} total — {keep_n} keep, {delete_n} delete")
    print(f"Output written to {output_path}")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        predict_file(sys.argv[1], sys.argv[2])
    else:
        main()
