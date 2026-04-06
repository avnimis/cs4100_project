"""
Gmail Storage Manager - Email Classifier
=========================================
Data: labeled JSONL files, one per category:
- duplicate.jsonl
- expired_offer.jsonl
- old_alert.jsonl
- past_event.jsonl
- spam.jsonl

Architecture:
  - Feature extraction  
  - Vectorization 
  - Neural Network 
  - CrossEntropyLoss 
  - Gradient descent 
"""

import re
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from datetime import datetime


# LABELS
# defines the 5 categories we want to classify emails into
# each category maps to an integer index that the model predicts
LABEL_MAP = {
    "duplicate":      0,
    "expired_offer":  1,
    "old_alert":      2,
    "past_event":     3,
    "spam":           4,
}
# reverses integer index to category
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES   = len(LABEL_MAP)

# tells the app what to do with each predicted category
ACTIONS = {
    "duplicate":     "DELETE — duplicate email",
    "expired_offer": "FLAG  — expired deal found",
    "old_alert":     "DELETE — old notification",
    "past_event":    "DELETE — event already passed",
    "spam":          "DELETE — spam",
}


# DATA LOADING (training)
# reads each JSONL files from disk + every line in a JSONL file in one email as a JSON object
# pairs each email with its integer class label from LABEL_MAP
# returns two parallel lists - all emails + all their labels
def load_jsonl_folder(folder_path: str):
    emails, labels = [], []

    for filename, label_idx in [
        ("duplicate.jsonl",     LABEL_MAP["duplicate"]),
        ("expired_offer.jsonl", LABEL_MAP["expired_offer"]),
        ("old_alert.jsonl",     LABEL_MAP["old_alert"]),
        ("past_event.jsonl",    LABEL_MAP["past_event"]),
        ("spam.jsonl",          LABEL_MAP["spam"]),
    ]:
        filepath = os.path.join(folder_path, filename)
        if not os.path.exists(filepath):
            print(f"  [WARNING] {filename} not found, skipping.")
            continue

        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    email = json.loads(line)
                    emails.append(email)
                    labels.append(label_idx)
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"  [WARNING] Skipping malformed line in {filename}: {e}")

        print(f"  Loaded {count:>5} emails from {filename}")

    print(f"\n  Total: {len(emails)} emails across {NUM_CLASSES} classes\n")
    return emails, labels


# FEATURE EXTRACTION (training)
# converts one raw email dict into a 25-number feature vector
# grouped into 4 categories of signals:
#
# KEYWORD SCORES (0–4)  — fraction of category-specific words found
#      0  urgency_score    "expires", "act now", "deadline"
#      1  spam_score       "winner", "free", "guarantee"
#      2  alert_score      "shipped", "otp", "verification code"
#      3  event_score      "rsvp", "register", "you're invited"
#      4  offer_score      "% off", "coupon", "promo"
#
# SENDER SIGNALS (5–11) — who sent it and from where
#      5  is_noreply       sender is a no-reply address
#      6  is_social        sender is LinkedIn, Facebook, Twitter, Instagram
#      7  is_shopping      sender is Amazon, eBay, Etsy, Walmart, etc.
#      8  is_personal      sender is Gmail, Yahoo, Hotmail, Outlook
#      9  is_newsletter    sender domain contains "mail", "news", "notify", "info"
#     10  domain_length    length of sender domain (longer = more corporate)
#     11  is_known_alert   sender is a known transactional sender (bank, UPS, etc.)
#
# SUBJECT LINE SIGNALS (12–17) — patterns in the subject text
#     12  is_reply_fwd     subject starts with "Re:" or "Fwd:"
#     13  has_brackets     subject contains [Newsletter], [Alert], etc.
#     14  caps_ratio       fraction of uppercase letters (HIGH CAPS = spammy)
#     15  exclamation_ct   number of exclamation marks (normalised)
#     16  has_date_in_subj subject contains a month name or year
#     17  subject_length   length of subject (normalised)
#
# BODY SIGNALS (18–24) — patterns inside the email body
#     18  has_unsubscribe  "unsubscribe" in body (strong newsletter signal)
#     19  link_count       number of http/https links (normalised)
#     20  caps_ratio_body  fraction of uppercase letters in body
#     21  body_length_norm log-scaled body length
#     22  has_order_num    body contains order/tracking number patterns
#     23  has_coupon_code  body contains coupon/promo code patterns
#     24  days_old_norm    how old the email is (0–1 over 2 years)
URGENCY_WORDS = ["urgent", "immediately", "action required", "expires",
                 "deadline", "limited time", "act now", "last chance"]
SPAM_WORDS    = ["winner", "congratulations", "free", "click here",
                 "no cost", "risk free", "guarantee", "million dollars"]
ALERT_WORDS   = ["your order", "shipped", "delivered", "security alert",
                 "sign-in attempt", "verification code", "otp", "receipt"]
EVENT_WORDS   = ["invitation", "you're invited", "rsvp", "join us",
                 "conference", "webinar", "register", "calendar"]
OFFER_WORDS   = ["% off", "discount", "coupon", "promo", "deal",
                 "sale", "offer", "save", "exclusive", "expires"]

SOCIAL_DOMAINS   = ["linkedin", "facebook", "twitter", "instagram",
                    "tiktok", "pinterest", "snapchat", "reddit"]
SHOPPING_DOMAINS = ["amazon", "ebay", "etsy", "walmart", "target",
                    "shopify", "bestbuy", "wayfair", "chewy"]
PERSONAL_DOMAINS = ["gmail", "yahoo", "hotmail", "outlook", "icloud", "aol"]
ALERT_DOMAINS    = ["chase", "bankofamerica", "wellsfargo", "paypal",
                    "ups", "fedex", "usps", "dhl", "apple", "google",
                    "microsoft", "netflix", "spotify"]
MONTH_NAMES      = ["january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november", "december"]

def _word_score(text: str, word_list: list) -> float:
    t = text.lower()
    return sum(1 for w in word_list if w in t) / len(word_list)

def _get_domain(sender: str) -> str:
    # extracts domain from "Name <email@domain.com>" or "email@domain.com"
    match = re.search(r'@([\w.\-]+)', sender)
    return match.group(1).lower() if match else ""

def extract_features(email: dict) -> np.ndarray:
    subject  = email.get("subject", "")
    body     = email.get("body", "")
    sender   = email.get("sender", "")
    date_str = email.get("date", "")
    text     = subject + " " + body
    subj_low = subject.lower()
    body_low = body.lower()
    domain   = _get_domain(sender)

    try:
        sent = datetime.strptime(date_str, "%Y-%m-%d")
        days = (datetime.now() - sent).days
    except Exception:
        days = 0

    # keyword scores
    urgency_score = _word_score(text, URGENCY_WORDS)
    spam_score    = _word_score(text, SPAM_WORDS)
    alert_score   = _word_score(text, ALERT_WORDS)
    event_score   = _word_score(text, EVENT_WORDS)
    offer_score   = _word_score(text, OFFER_WORDS)

    # sender signals
    is_noreply    = float("noreply" in sender.lower() or "no-reply" in sender.lower())
    is_social     = float(any(s in domain for s in SOCIAL_DOMAINS))
    is_shopping   = float(any(s in domain for s in SHOPPING_DOMAINS))
    is_personal   = float(any(s in domain for s in PERSONAL_DOMAINS))
    is_newsletter = float(any(s in domain for s in ["mail", "news", "notify", "info", "updates"]))
    domain_length = min(len(domain), 50) / 50.0
    is_known_alert = float(any(s in domain for s in ALERT_DOMAINS))

    # subject line signals
    is_reply_fwd   = float(subj_low.startswith("re:") or subj_low.startswith("fwd:") or subj_low.startswith("fw:"))
    has_brackets   = float(bool(re.search(r'\[.+?\]', subject)))
    caps_ratio     = sum(1 for c in subject if c.isupper()) / max(len(subject), 1)
    exclamation_ct = min(subject.count("!"), 5) / 5.0
    has_date_subj  = float(any(m in subj_low for m in MONTH_NAMES) or bool(re.search(r'\b20\d{2}\b', subject)))
    subject_length = min(len(subject), 200) / 200.0

    # body signals
    has_unsubscribe = float("unsubscribe" in body_low)
    link_count      = min(len(re.findall(r'https?://', body)), 20) / 20.0
    caps_ratio_body = sum(1 for c in body if c.isupper()) / max(len(body), 1)
    body_length     = np.log1p(len(body)) / 10.0
    has_order_num   = float(bool(re.search(r'(order|tracking|shipment)\s*#?\s*\d+', body_low)))
    has_coupon_code = float(bool(re.search(r'(code|coupon|promo)[:\s]+[A-Z0-9]{4,}', body)))
    days_old_norm   = min(days, 730) / 730.0

    return np.array([
        urgency_score, spam_score, alert_score, event_score, offer_score,
        is_noreply, is_social, is_shopping, is_personal, is_newsletter,
        domain_length, is_known_alert,
        is_reply_fwd, has_brackets, caps_ratio, exclamation_ct,
        has_date_subj, subject_length,
        has_unsubscribe, link_count, caps_ratio_body, body_length,
        has_order_num, has_coupon_code, days_old_norm,
    ], dtype=np.float32)

HANDCRAFTED_DIM = 25

# PYTORCH DATASET (training)
# prepares all emails into tensors the model can consume
# does two things then combines them:
#   1. TF-IDF vectorization — converts email text into a 500-number vector
#      representing how important each word is in that email vs. the whole
#      dataset. Fitted once on training data, then reused for val/test
#   2. Hand-crafted features — the 10-number vector from Section 3
#   3. Concatenation — [tfidf(500) | hand_crafted(10)] → one 510-number
#      vector per email. This is the final input to the neural network
class EmailDataset(Dataset):
    def __init__(self, emails, labels=None, vectorizer=None, tfidf_dim=500):
        self.labels = labels

        texts = [e.get("subject","") + " " + e.get("body","") for e in emails]

        if vectorizer is None:          # training: fit the vectorizer on training text
            self.vectorizer = TfidfVectorizer(
                max_features=tfidf_dim,
                stop_words="english",
                sublinear_tf=True,      # log-scale term frequencies
            )
            tfidf = self.vectorizer.fit_transform(texts).toarray()
        else:                           # val/test/inference: reuse the already-fitted one
            self.vectorizer = vectorizer
            tfidf = vectorizer.transform(texts).toarray()

        hc = np.stack([extract_features(e) for e in emails])

        # Final input: TF-IDF text features + hand-crafted features side by side
        self.X = np.hstack([tfidf, hc]).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        if self.labels is not None:
            return x, torch.tensor(self.labels[idx], dtype=torch.long)
        return x


# NEURAL NETWORK ARCHITECTURE (training)
# defines the shape of the model (Linear -> ReLU -> Linear -> ReLU -> Linear)

# input layer 1:  510 numbers (500 TF-IDF + 10 hand-crafted)
# hidden layer 1: 256 neurons with ReLU activation + dropout
# hidden layer 2: 128 neurons with ReLU activation + dropout
# output layer : 5 neurons — one score (logit) per class

# ropout randomly zeros some neurons during training to prevent the model
# from memorizing the training data (overfitting)
class EmailClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 525 → 128  (smaller first layer)
            nn.ReLU(),
            nn.Dropout(dropout),                # 0.5 dropout = drops half the neurons

            nn.Linear(hidden_dim, num_classes), # 128 → 5  (removed second hidden layer)
        )

    def forward(self, x):
        return self.network(x)


# TRAIN LOOP (training)
# runs one full pass over the training data (one epoch)
# for each batch of emails:
#   1. Forward pass  — feed inputs through the network to get predictions
#   2. Loss          — measure how wrong the predictions are (CrossEntropyLoss)
#   3. Backward pass — compute gradients via backpropagation (loss.backward())
#   4. Update        — adjust weights in the direction that reduces loss
#   5. Zero grads    — clear gradients so they don't accumulate next batch
# returns average loss and accuracy for this epoch
def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in dataloader:
        X, y   = X.to(device), y.to(device)
        pred   = model(X)                   # 1. forward pass
        loss   = loss_fn(pred, y)           # 2. compute loss

        loss.backward()                     # 3. backpropagation
        optimizer.step()                    # 4. update weights
        optimizer.zero_grad()               # 5. clear gradients

        total_loss += loss.item() * len(y)
        correct    += (pred.argmax(1) == y).sum().item()
        n          += len(y)

    return total_loss / n, correct / n

# VALIDATION LOOP (evaluative)
# runs on the held-out validation split after each training epoch
# identical to train_loop EXCEPT:
#   - model.eval() disables Dropout so all neurons are active
#   - torch.no_grad() skips gradient computation (faster, uses less memory)
#   - No optimizer.step() — weights are NOT updated here
def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():                   # no gradient tracking needed
        for X, y in dataloader:
            X, y   = X.to(device), y.to(device)
            logits = model(X)
            total_loss += loss_fn(logits, y).item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            n          += len(y)

    return total_loss / n, correct / n


# FULL TRAINING PIPELINE
# orchestrates the entire training process end-to-end:
#   1. Detects best available device (GPU → Apple MPS → CPU)
#   2. Loads emails from your JSONL folder (Section 2)
#   3. Splits data into 85% training / 15% validation
#      (stratified = same class proportions in both splits)
#   4. Wraps data in Dataset + DataLoader (batching, shuffling)
#   5. Instantiates the model, loss function, and optimizer
#   6. Runs train_loop + test_loop for each epoch, printing a progress table
# returns the trained model, fitted vectorizer, and device
def train_model(folder_path, tfidf_dim=500, hidden_dim=128,
                epochs=50, batch_size=64, lr=1e-3, patience=5):

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}\n")

    print("Loading data...")
    emails, labels = load_jsonl_folder(folder_path)

    tr_idx, val_idx = train_test_split(
        range(len(emails)), test_size=0.15, stratify=labels, random_state=42
    )
    tr_emails,  tr_labels  = [emails[i] for i in tr_idx],  [labels[i] for i in tr_idx]
    val_emails, val_labels = [emails[i] for i in val_idx], [labels[i] for i in val_idx]

    tr_ds  = EmailDataset(tr_emails,  tr_labels,  vectorizer=None,             tfidf_dim=tfidf_dim)
    val_ds = EmailDataset(val_emails, val_labels, vectorizer=tr_ds.vectorizer, tfidf_dim=tfidf_dim)

    class_counts = Counter(tr_labels)
    sample_weights = [1.0 / class_counts[l] for l in tr_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(tr_labels), replacement=True)
    tr_dl  = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    total = len(tr_labels)
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * class_counts[i]) for i in range(NUM_CLASSES)],
        dtype=torch.float
    ).to(device)

    input_dim = tfidf_dim + HANDCRAFTED_DIM
    model     = EmailClassifier(input_dim, hidden_dim).to(device)
    loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss   = float("inf")
    best_state      = None
    epochs_no_improve = 0

    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc':<14}{'Val Loss':<14}{'Val Acc'}")
    print("─" * 60)
    for epoch in range(1, epochs + 1):
        tr_loss,  tr_acc  = train_loop(tr_dl,  model, loss_fn, optimizer, device)
        val_loss, val_acc = test_loop(val_dl, model, loss_fn, device)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = " ✓"
        else:
            epochs_no_improve += 1
            marker = f" (no improve {epochs_no_improve}/{patience})"

        print(f"{epoch:<8}{tr_loss:<14.4f}{tr_acc:<14.3f}{val_loss:<14.4f}{val_acc:.3f}{marker}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    print(f"Best val loss: {best_val_loss:.4f}")
    return model, tr_ds.vectorizer, device


# INFERENCE (evaluative)
# uses the trained model to classify new, unseen emails
# steps:
#   1. Wraps emails in a Dataset using the already-fitted vectorizer
#      (important: same vectorizer from training so features match)
#   2. Runs a forward pass with no gradient tracking
#   3. Applies Softmax to convert raw logits → probabilities per class
#   4. Picks the highest-probability class as the prediction
#   5. Returns human-readable results: label, confidence %, and action

# safe_delete = True means the model is confident enough (≥ threshold)
# to act on the prediction automatically
def classify_emails(emails, model, vectorizer, device="cpu", confidence_threshold=0.6):
    ds = EmailDataset(emails, labels=None, vectorizer=vectorizer)
    dl = DataLoader(ds, batch_size=64)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for X in dl:
            logits = model(X.to(device))
            probs  = nn.Softmax(dim=1)(logits)  # convert logits → probabilities
            all_probs.append(probs.cpu())
    all_probs = torch.cat(all_probs)

    results = []
    for i, email in enumerate(emails):
        pred_idx   = all_probs[i].argmax().item()
        confidence = all_probs[i][pred_idx].item()
        label      = IDX_TO_LABEL[pred_idx]
        results.append({
            "subject":     email.get("subject", "(no subject)"),
            "sender":      email.get("sender",  "(unknown)"),
            "label":       label,
            "confidence":  round(confidence, 3),
            "action":      ACTIONS[label],
            "safe_delete": confidence >= confidence_threshold,
        })
    return results


# SAVE / LOAD
# save_model: persists the trained weights + fitted vectorizer to a .pth file
#   - model.state_dict() saves all learned weight tensors (same as lab)
#   - vectorizer must be saved alongside so inference uses identical features
#
# load_model: reconstructs the model from disk for later use
#   - rebuilds the architecture then loads weights via load_state_dict()
#   - call model.eval() before inference (disables Dropout)
def save_model(model, vectorizer, path="email_model.pth"):
    cpu_state  = {k: v.cpu() for k, v in model.state_dict().items()}
    input_dim  = cpu_state["network.0.weight"].shape[1]  # cols of first layer
    hidden_dim = cpu_state["network.0.weight"].shape[0]  # rows of first layer
    torch.save({
        "model_state": cpu_state,
        "input_dim":   input_dim,
        "hidden_dim":  hidden_dim,
        "vectorizer":  vectorizer,
    }, path)
    print(f"\nModel saved → {path}")

def load_model(path="email_model.pth", device="cpu"):
    ckpt  = torch.load(path, weights_only=False, map_location="cpu")
    model = EmailClassifier(ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt["vectorizer"]


# MAIN 
# entry point - two phases:
#   TRAINING   — load data → train model → save to disk
#   EVALUATION — load saved model → classify sample emails → print results
if __name__ == "__main__":
    DATA_FOLDER  = "../emails/labeled"
    MODEL_PATH   = "./email_model.pth"

    # TRAINING
    model, vectorizer, device = train_model(
        folder_path = DATA_FOLDER,
        tfidf_dim   = 500,
        hidden_dim  = 256,
        epochs      = 20,
        batch_size  = 64,
        lr          = 1e-3,
    )
    save_model(model, vectorizer, path=MODEL_PATH)

    # EVALUATION
    model, vectorizer = load_model(path=MODEL_PATH, device=device)

    with open(os.path.join(DATA_FOLDER, "spam.jsonl")) as f:
        sample = [json.loads(l) for l in f if l.strip()][:5]

    results = classify_emails(sample, model, vectorizer, device)
    print("\n── Sample Predictions ──")
    for r in results:
        flag = "✓" if r["safe_delete"] else "?"
        print(f"  [{flag}] {r['action']:<35} ({r['confidence']:.0%})  \"{r['subject']}\"")