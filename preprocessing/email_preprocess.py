import json
import re
import pandas as pd
from datetime import datetime, timezone

LABEL_FILES = {
    "duplicate":      "emails/labeled/rc_duplicate.jsonl",
    "expired_offer":  "emails/labeled/rc_expired_offer.jsonl",
    "old_alert":      "emails/labeled/rc_old_alert.jsonl",
    "past_event":     "emails/labeled/rc_past_event.jsonl",
    "spam":           "emails/labeled/rc_spam.jsonl",
}

EMAIL_LABELS = list(LABEL_FILES.keys())


def load_all_emails() -> pd.DataFrame:
    """
    Reads all category JSONs and merges into one DataFrame.
    Preserves all rich fields from your actual JSON structure.
    """
    master: dict[str, dict] = {}

    for label, filepath in LABEL_FILES.items():
        with open(filepath, "r", encoding="utf-8") as f:
            emails = [json.loads(line) for line in f if line.strip()]

        for email in emails:
            eid = email["id"]

            if eid not in master:
                # Combine subject + body as the text the model reads
                subject   = email.get("subject", "") or ""
                body_text = email.get("body_text", "") or ""
                combined  = f"{subject}\n\n{body_text}".strip()

                master[eid] = {
                    "id":               eid,
                    "text":             combined,          # model input
                    "subject":          subject,
                    "from_email":       email.get("from_email", ""),
                    "date_iso":         email.get("date_iso", ""),
                    "gmail_labels":     email.get("gmail_labels", []),
                    "list_unsubscribe": email.get("list_unsubscribe", ""),
                    "spam_headers":     email.get("spam_headers", {}),
                    "link_count":       len(email.get("links", [])),
                }
                for l in EMAIL_LABELS:
                    master[eid][l] = 0

            master[eid][label] = 1

    return pd.DataFrame(list(master.values()))


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