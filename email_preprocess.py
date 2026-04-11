import os
import sys
import re
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "client"))
from auth import authenticate
from gmail_client import scrape_emails

EMAIL_LABELS = ["duplicate", "expired_offer", "old_alert", "past_event", "spam"]


def load_all_emails(max_results: int = 500) -> pd.DataFrame:
    """
    Fetches emails from Gmail via the API client and returns a DataFrame
    with the same schema expected by the model pipeline.
    Labels default to 0 (unclassified) for live inference.
    """
    creds = authenticate()
    raw_emails = scrape_emails(creds, max_results=max_results)

    rows = []
    for email in raw_emails:
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
            row[label] = 0

        rows.append(row)

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