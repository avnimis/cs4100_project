"""
mbox Preprocessor for Email Deletion Classification
=====================================================
Parses a .mbox file and outputs structured JSON records, one per email,
with all fields needed to classify emails for deletion.

Output fields per record:
  id              – zero-padded index (stable within a run)
  subject         – decoded subject line
  from_name       – sender display name
  from_email      – sender email address
  to              – recipient(s)
  date_raw        – original Date header string
  date_iso        – ISO-8601 UTC datetime (or null if unparseable)
  gmail_labels    – list of Gmail labels (Category Promotions, Trash, etc.)
  content_type    – top-level MIME type
  has_attachment  – bool
  body_text       – plain-text body (preferred) or HTML→text fallback, stripped
  body_length     – character count of body_text
  links           – list of unique URLs extracted from body
  list_unsubscribe – value of List-Unsubscribe header if present (signals mailing list)
  spam_headers    – dict of spam-related headers (X-Spam-*, X-Mailer, etc.)
  message_id      – Message-ID header

Usage:
  python preprocess.py emails/Trash.mbox emails/output.jsonl 

  If output path is omitted, writes to <input_stem>_preprocessed.jsonl
  alongside the input file.
"""

import mailbox
import json
import re
import sys
from pathlib import Path
from email.header import decode_header as _decode_header
from email.utils import parsedate_to_datetime, getaddresses
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_header(value: str | None) -> str:
    """Decode RFC 2047-encoded header value to plain unicode string."""
    if not value:
        return ""
    parts = _decode_header(value)
    out = []
    for chunk, enc in parts:
        if isinstance(chunk, bytes):
            out.append(chunk.decode(enc or "utf-8", errors="replace"))
        else:
            out.append(chunk)
    return " ".join(out).strip()


def parse_address(header_value: str | None):
    """Return (display_name, email_address) from a single address header."""
    if not header_value:
        return "", ""
    pairs = getaddresses([header_value])
    if not pairs:
        return "", ""
    name, addr = pairs[0]
    return name.strip(), addr.strip().lower()


def parse_date(date_str: str | None) -> str | None:
    """Return ISO-8601 UTC string or None."""
    if not date_str:
        return None
    try:
        dt = parsedate_to_datetime(date_str)
        # Normalise to UTC
        from datetime import timezone
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


_URL_RE = re.compile(
    r"https?://[^\s\"'<>)\]]+",
    re.IGNORECASE,
)

# Invisible/zero-width Unicode characters that email marketers use to pad
# preview text – they add noise to NLP features and should be stripped.
_INVISIBLE_RE = re.compile(
    r"[\u034f\u00ad\u200b\u200c\u200d\u2007\ufeff\u180e]",
)

def clean_body(text: str) -> str:
    """Remove invisible Unicode filler, collapse whitespace."""
    text = _INVISIBLE_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_urls(text: str) -> list[str]:
    """Return deduplicated list of URLs found in text."""
    found = _URL_RE.findall(text)
    # strip trailing punctuation artefacts
    cleaned = [u.rstrip(".,;:)>") for u in found]
    return list(dict.fromkeys(cleaned))  # preserve order, deduplicate


_SPAM_HEADER_PREFIXES = (
    "x-spam",
    "x-mailer",
    "x-originating-ip",
    "x-bulk",
    "x-marketing",
    "x-campaign",
    "x-mailgun",
    "x-feedback-id",
    "x-ses",
    "x-sendgrid",
    "precedence",
)

def extract_spam_headers(msg) -> dict:
    headers = {}
    for key in msg.keys():
        if any(key.lower().startswith(p) for p in _SPAM_HEADER_PREFIXES):
            headers[key.lower()] = msg.get(key, "")
    return headers


def html_to_text(html: str) -> str:
    """Convert HTML to readable plain text."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove invisible / noisy elements
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # Collapse whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_body(msg) -> str:
    """
    Extract the most useful body text from a (possibly multipart) message.
    Priority: text/plain > HTML→text
    """
    plain_parts = []
    html_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype in ("text/plain", "text/html"):
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
                if ctype == "text/plain":
                    plain_parts.append(text)
                else:
                    html_parts.append(text)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                html_parts.append(text)
            else:
                plain_parts.append(text)

    if plain_parts:
        body = "\n".join(plain_parts).strip()
    elif html_parts:
        body = html_to_text("\n".join(html_parts))
    else:
        body = ""

    return body


def has_attachment(msg) -> bool:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                return True
    return False


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_mbox(mbox_path: str, output_path: str | None = None) -> str:
    mbox_path = Path(mbox_path)
    if output_path is None:
        output_path = mbox_path.parent / (mbox_path.stem + "_preprocessed.jsonl")
    else:
        output_path = Path(output_path)

    mb = mailbox.mbox(str(mbox_path))
    messages = list(mb)
    total = len(messages)
    print(f"Found {total} messages in {mbox_path.name}", file=sys.stderr)

    records_written = 0
    errors = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, msg in enumerate(messages):
            try:
                subject = decode_header(msg.get("Subject"))
                from_name, from_email = parse_address(msg.get("From"))
                to_raw = decode_header(msg.get("To"))
                date_raw = msg.get("Date", "")
                date_iso = parse_date(date_raw)

                labels_raw = msg.get("X-Gmail-Labels", "")
                gmail_labels = [l.strip() for l in labels_raw.split(",") if l.strip()]

                body = clean_body(extract_body(msg))
                links = extract_urls(body)

                record = {
                    "id": f"{idx:05d}",
                    "subject": subject,
                    "from_name": from_name,
                    "from_email": from_email,
                    "to": to_raw,
                    "date_raw": date_raw,
                    "date_iso": date_iso,
                    "gmail_labels": gmail_labels,
                    "content_type": msg.get_content_type(),
                    "has_attachment": has_attachment(msg),
                    "body_text": body,
                    "body_length": len(body),
                    "links": links,
                    "list_unsubscribe": msg.get("List-Unsubscribe", ""),
                    "spam_headers": extract_spam_headers(msg),
                    "message_id": msg.get("Message-ID", "").strip(),
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1

            except Exception as e:
                errors += 1
                print(f"  ERROR on message {idx}: {e}", file=sys.stderr)

    print(f"Done — {records_written} records written to {output_path}", file=sys.stderr)
    if errors:
        print(f"  ({errors} messages skipped due to errors)", file=sys.stderr)
    return str(output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess_mbox.py <input.mbox> [output.jsonl]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    process_mbox(inp, out)