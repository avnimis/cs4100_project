import json
import os
import pandas as pd

LABEL_FILES = {
    "duplicate":      "emails/categorized/duplicate.json",
    "expired_offer":  "emails/categorized/expired_offers.json",
    "old_alert":      "emails/categorized/old_alert.json",
    "past_event":     "emails/categorized/past_event.json",
    "spam":           "emails/categorized/spam.json",
}

EMAIL_LABELS = list(LABEL_FILES.keys())  # ["duplicate", "expired_offer", ...]


def load_all_emails() -> pd.DataFrame:
    """
    Reads all category JSON files and merges them into a single DataFrame.
    Each email gets a binary column per label (1 = belongs to that category).
    An email can belong to multiple categories (multi-label).
    """
    # email_id -> { "text": ..., "duplicate": 0, "spam": 1, ... }
    master: dict[str, dict] = {}

    for label, filepath in LABEL_FILES.items():
        with open(filepath, "r", encoding="utf-8") as f:
            emails = json.load(f)

        for email in emails:
            eid  = email["id"]
            text = email["text"]

            if eid not in master:
                # First time seeing this email — initialize all labels to 0
                master[eid] = {"id": eid, "text": text}
                for l in EMAIL_LABELS:
                    master[eid][l] = 0

            # Mark this specific label as positive
            master[eid][label] = 1

    df = pd.DataFrame(list(master.values()))
    return df  # columns: id, text, duplicate, expired_offer, old_alert, past_event, spam


def extract_features(email_text: str) -> dict:
    """Rule-based signals fused into the model alongside BERT embeddings."""
    import re
    from dateutil import parser as dateparser
    from datetime import datetime

    features = {}

    date_pattern = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b'
    found_dates  = re.findall(date_pattern, email_text, re.IGNORECASE)
    past_dates   = []
    for d in found_dates:
        try:
            parsed = dateparser.parse(d)
            if parsed and parsed < datetime.now():
                past_dates.append(parsed)
        except Exception:
            pass
    features["has_past_date"] = len(past_dates) > 0

    expiry_kw = ["expires", "expiring", "valid until", "limited time", "ends on", "offer ends"]
    features["has_expiry_language"] = any(k in email_text.lower() for k in expiry_kw)

    spam_kw = ["unsubscribe", "click here", "act now", "free offer", "you've been selected"]
    features["spam_keyword_count"] = sum(1 for k in spam_kw if k in email_text.lower())

    return features