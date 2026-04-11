import os
import sys

import torch
from torch.utils.data import DataLoader

from email_data   import load_emails_from_api, EMAIL_LABELS, EmailDataset
from emails_model import EmailClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "client"))
from auth import authenticate
from gmail_client import apply_deletion_label


# Config
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 16
THRESHOLD    = 0.5   # sigmoid probability above which an email is flagged
MAX_EMAILS   = 500
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "email_classifier.pt")


def run_inference():
    # 1. Fetch emails and build DataFrame 
    print("Step 1: Authenticating and fetching emails...")
    creds = authenticate()
    df = load_emails_from_api(max_results=MAX_EMAILS)
    print(f"  → {len(df)} emails fetched")

    # 2. Build dataset
    print("Step 2: Building dataset...")
    dataset = EmailDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Load model 
    print("Step 3: Loading model...")
    model = EmailClassifier(num_labels=len(EMAIL_LABELS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 4. Run inference 
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

    # 5. Flag emails for deletion
    flagged_ids = []
    for i, row in df.iterrows():
        probs = all_probs[i]
        if any(p.item() >= THRESHOLD for p in probs):
            flagged_ids.append(row["id"])

    print(f"  → {len(flagged_ids)}/{len(df)} emails flagged for deletion")

    # 6. Apply Gmail label
    print("Step 5: Applying Gmail label...")
    flagged_emails = [{"id": eid, "flagged_for_deletion": True} for eid in flagged_ids]
    apply_deletion_label(creds, flagged_emails)
    print("Done.")


if __name__ == "__main__":
    run_inference()
