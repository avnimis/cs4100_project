import json
import re
import email
import email.header
from pathlib import Path

# ── decoders ──────────────────────────────────────────────────────────────────

def decode_mime_words(text):
    """Decode encoded subject lines like =?UTF-8?B?...?= or =?UTF-8?Q?...?="""
    if not text:
        return ""
    decoded = email.header.decode_header(text)
    parts = []
    for part, enc in decoded:
        if isinstance(part, bytes):
            parts.append(part.decode(enc or "utf-8", errors="ignore"))
        else:
            parts.append(str(part))
    return " ".join(parts)

# ── cleaners ──────────────────────────────────────────────────────────────────

def remove_urls(text):
    return re.sub(r'https?://\S+', ' ', text)

def remove_html_entities(text):
    # &zwnj; &amp; &nbsp; etc.
    return re.sub(r'&[a-zA-Z]+;', ' ', text)

def remove_zero_width_chars(text):
    # zero-width space, non-joiner, joiner, BOM, etc.
    return re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff\u00a0]', ' ', text)

def remove_inline_image_refs(text):
    # AE-style [http://...] image blocks
    return re.sub(r'\[https?://\S+\]', ' ', text)

def remove_tracking_noise(text):
    # Long base64-looking strings (tracking tokens)
    return re.sub(r'[A-Za-z0-9+/=_\-]{60,}', ' ', text)

def normalize_whitespace(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_coupon_codes(text):
    """Find promo codes — must appear directly after coupon trigger keywords."""
    # only extract codes that follow an explicit trigger phrase
    pattern = r'(?:use\s+code|promo\s+code|coupon\s+code|discount\s+code|code)[:\s]+([A-Z0-9]{4,12})\b'
    candidates = re.findall(pattern, text, re.IGNORECASE)
    # must contain at least one letter (not just a number) and be all caps/digits
    codes = [c.upper() for c in candidates if re.search(r'[A-Z]', c) and re.fullmatch(r'[A-Z0-9]+', c.upper())]
    return list(set(codes))

def extract_sender_domain(from_field):
    """Pull domain from 'Name <email@domain.com>' or 'email@domain.com'."""
    match = re.search(r'@([\w.\-]+)', from_field or "")
    return match.group(1).lower() if match else ""

# ── main cleaner ──────────────────────────────────────────────────────────────

def clean_text(text):
    if not text:
        return ""
    text = remove_inline_image_refs(text)
    text = remove_urls(text)
    text = remove_html_entities(text)
    text = remove_zero_width_chars(text)
    text = remove_tracking_noise(text)
    text = normalize_whitespace(text)
    return text

def preprocess_email(email_obj):
    """Return a cleaned, enriched version of one email dict."""
    raw_subject = email_obj.get("subject", "")
    raw_body    = email_obj.get("body", "")
    raw_from    = email_obj.get("from", "")

    clean_subject = clean_text(decode_mime_words(raw_subject))
    clean_body    = clean_text(raw_body)

    # combined text for NLP — subject weighted 2x since it's more signal-dense
    combined_text = f"{clean_subject} {clean_subject} {clean_body}".strip()

    return {
        **email_obj,
        "clean_subject":  clean_subject,
        "clean_body":     clean_body,
        "combined_text":  combined_text,
        "sender_domain":  extract_sender_domain(raw_from),
        "coupon_codes":   extract_coupon_codes(combined_text),
        "body_length":    len(clean_body.split()),
        "is_empty_body":  len(clean_body.strip()) == 0,
    }

# ── run ───────────────────────────────────────────────────────────────────────

def preprocess_file(input_path, output_path):
    with open(input_path) as f:
        emails = json.load(f)

    processed = [preprocess_email(e) for e in emails]

    with open(output_path, "w") as f:
        json.dump(processed, f, indent=2)

    # quick stats
    total       = len(processed)
    empty_body  = sum(1 for e in processed if e["is_empty_body"])
    with_coupons = sum(1 for e in processed if e["coupon_codes"])
    domains     = set(e["sender_domain"] for e in processed if e["sender_domain"])

    print(f"Preprocessed {total} emails → {output_path}")
    print(f"  Empty bodies:     {empty_body}/{total}")
    print(f"  Coupon codes found in: {with_coupons} emails")
    print(f"  Unique sender domains: {len(domains)}")
    print(f"\nSample coupons found:")
    for e in processed:
        if e["coupon_codes"]:
            label = e.get("label", "unlabeled")
            print(f"  [{label}] {e['clean_subject'][:50]!r} → {e['coupon_codes']}")


if __name__ == "__main__":
    preprocess_file(
        input_path="data/emails/all_emails.json",
        output_path="data/emails/all_emails_clean.json"
    )