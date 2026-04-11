import os
import sys
import re
import json
import glob as glob_module
import torch
import pandas as pd
from datetime import datetime, timezone
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "client"))
from auth import authenticate
from gmail_client import scrape_emails

EMAIL_LABELS = ["duplicate", "expired_offer", "old_alert", "past_event", "spam"]

TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), "emails_training_data")


def _build_row(email: dict, default_labels: bool = False) -> dict:
    subject   = email.get("subject", "") or ""
    body_text = email.get("body_text", "") or ""
    combined  = f"{subject}\n\n{body_text}".strip()

    row = {
        "id":               email.get("id", ""),
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
        row[label] = 0 if default_labels else email.get(label, 0)

    return row


def load_all_emails() -> pd.DataFrame:
    """
    Loads training emails from all JSONL files inside emails_training_data/.
    Each line in a JSONL file should be a JSON email record that may include
    ground-truth label fields (duplicate, expired_offer, old_alert, past_event, spam).
    """
    rows = []
    pattern = os.path.join(TRAINING_DATA_DIR, "**", "*.jsonl")
    for path in glob_module.glob(pattern, recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(_build_row(json.loads(line)))

    return pd.DataFrame(rows)


def load_emails_from_api(max_results: int = 500) -> pd.DataFrame:
    """
    Fetches live emails from Gmail via the API client and returns a DataFrame
    with the same schema expected by the model pipeline.
    Labels default to 0 (unclassified) for live inference.
    """
    creds = authenticate()
    raw_emails = scrape_emails(creds, max_results=max_results)
    rows = [_build_row(email, default_labels=True) for email in raw_emails]
    return pd.DataFrame(rows)


def extract_features(row: pd.Series) -> dict:
    """
    Rule-based signals extracted from the rich JSON fields.
    These 6 floats are fused into the classifier head alongside BERT.
    """
    features = {}
    now = datetime.now(timezone.utc)

    # ── 1. Past date from ISO timestamp ──────────────────────────────────────
    try:
        from dateutil import parser as dp
        email_date = dp.parse(row["date_iso"])
        if email_date.tzinfo is None:
            email_date = email_date.replace(tzinfo=timezone.utc)
        features["email_is_old"] = (now - email_date).days > 30
    except Exception:
        features["email_is_old"] = False

    # ── 2. Expiry language in subject + body ─────────────────────────────────
    text_lower = row["text"].lower()
    expiry_kw  = ["expires", "expiring", "valid until", "limited time",
                  "ends on", "offer ends", "last chance", "today only"]
    features["has_expiry_language"] = any(k in text_lower for k in expiry_kw)

    # ── 3. Spam signals — use the gmail_labels field directly ─────────────────
    gmail_labels = row["gmail_labels"] if isinstance(row["gmail_labels"], list) else []
    features["gmail_marked_spam"] = "Spam" in gmail_labels

    # ── 4. Unsubscribe link present (strong promo/spam signal) ────────────────
    features["has_unsubscribe"] = bool(row["list_unsubscribe"])

    # ── 5. Tracking URL pattern in body ──────────────────────────────────────
    tracking_pattern = r'https?://tracking\.|/t/\d+/|utm_source='
    features["has_tracking_links"] = bool(re.search(tracking_pattern, row["text"]))

    # ── 6. Spam header count ──────────────────────────────────────────────────
    spam_hdr_count = len(row["spam_headers"]) if isinstance(row["spam_headers"], dict) else 0
    features["spam_header_count"] = min(spam_hdr_count, 5) / 5.0  # normalize to [0,1]

    return features


class EmailDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_length: int = 256):
        self.df         = df.reset_index(drop=True)
        self.max_length = max_length
        self.tokenizer  = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row  = self.df.iloc[idx]
        text = str(row["text"])

        # 1. Tokenize for BERT
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 2. Rule-based features
        feats = extract_features(row)
        rule_features = torch.tensor([
            float(feats["email_is_old"]),
            float(feats["has_expiry_language"]),
            float(feats["gmail_marked_spam"]),
            float(feats["has_unsubscribe"]),
            float(feats["has_tracking_links"]),
            float(feats["spam_header_count"]),
        ], dtype=torch.float32)

        # 3. Label vector (one float per class)
        labels = torch.tensor(
            row[EMAIL_LABELS].values.astype(float),
            dtype=torch.float32
        )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "rule_features":  rule_features,
            "labels":         labels,
        }
