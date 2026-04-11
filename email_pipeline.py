import os
import sys

import torch
import pandas as pd
from torch.utils.data import DataLoader

from email_dataset_class import EmailDataset
from emails_model        import EmailClassifier
from email_preprocess    import EMAIL_LABELS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "client"))
from auth import authenticate
from gmail_client import scrape_emails, apply_deletion_label


# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 16
THRESHOLD    = 0.5   # sigmoid probability above which an email is flagged
MAX_EMAILS   = 500
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "email_classifier.pt")
# ──────────────────────────────────────────────────────────────────────────────


def build_dataframe(emails: list[dict]) -> pd.DataFrame:
    """Convert raw email dicts from the Gmail client into a model-ready DataFrame."""
    rows = []
    for email in emails:
        subject   = email.get("subject", "") or ""
        body_text = email.get("body_text", "") or ""
        combined  = f"{subject}\n\n{body_text}".strip()

        row = {
            "id":               email["id"],
            "text":             combined,
            "subject":          subject,
            "from_email":       email.get("from_email", ""),
            "date_iso":         email.get("date_iso", ""),
            "gmail_labels":     email.get("gmail_labels", []),
            "list_unsubscribe": email.get("list_unsubscribe", ""),
            "spam_headers":     email.get("spam_headers", {}),
            "link_count":       len(email.get("links", [])),
        }
        for label in EMAIL_LABELS:
            row[label] = 0  # no ground-truth labels for live inference

        rows.append(row)

    return pd.DataFrame(rows)


def run_inference():
    # ── 1. Fetch emails ───────────────────────────────────────────────────────
    print("Step 1: Authenticating and fetching emails...")
    creds = authenticate()
    emails = scrape_emails(creds, max_results=MAX_EMAILS)
    print(f"  → {len(emails)} emails fetched")

    # ── 2. Build dataset ──────────────────────────────────────────────────────
    print("Step 2: Building dataset...")
    df = build_dataframe(emails)
    dataset = EmailDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── 3. Load model ─────────────────────────────────────────────────────────
    print("Step 3: Loading model...")
    model = EmailClassifier(num_labels=len(EMAIL_LABELS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # ── 4. Run inference ──────────────────────────────────────────────────────
    print("Step 4: Running inference...")
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["rule_features"].to(DEVICE),
            )
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)

    all_probs = torch.cat(all_probs, dim=0)  # [N, num_labels]

    # ── 5. Flag emails for deletion ───────────────────────────────────────────
    id_to_email = {e["id"]: e for e in emails}
    flagged_count = 0

    for i, row in df.iterrows():
        probs = all_probs[i]
        predicted_labels = [
            EMAIL_LABELS[j] for j, p in enumerate(probs) if p.item() >= THRESHOLD
        ]
        if predicted_labels:
            email = id_to_email[row["id"]]
            email["flagged_for_deletion"] = True
            flagged_count += 1

    print(f"  → {flagged_count}/{len(emails)} emails flagged for deletion")

    # ── 6. Apply Gmail label ──────────────────────────────────────────────────
    print("Step 5: Applying Gmail label...")
    apply_deletion_label(creds, emails)
    print("Done.")


if __name__ == "__main__":
    run_inference()
