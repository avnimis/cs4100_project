import re
import base64
import pickle
from email.utils import parsedate_to_datetime
from auth import authenticate
from googleapiclient.discovery import build
from google.auth.transport.requests import Request


SYSTEM_LABEL_NAMES = {
    "INBOX": "Inbox", "SENT": "Sent", "TRASH": "Trash", "SPAM": "Spam",
    "DRAFT": "Draft", "STARRED": "Starred", "IMPORTANT": "Important",
    "UNREAD": "Unread", "CATEGORY_UPDATES": "Category Updates",
    "CATEGORY_PROMOTIONS": "Category Promotions", "CATEGORY_SOCIAL": "Category Social",
    "CATEGORY_FORUMS": "Category Forums", "CATEGORY_PERSONAL": "Category Personal",
}

SPAM_HEADER_KEYS = {
    "x-mailgun-sid", "x-feedback-id", "x-mailgun-sending-ip", "x-mailgun-tag",
    "x-mailgun-track-clicks", "x-mailgun-variables", "x-ses-outgoing",
    "x-ses-receipt", "x-google-dkim-signature", "x-forwarded-to",
}


def get_service(creds):
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build('gmail', 'v1', credentials=creds)


# Scrape emails

def scrape_emails(creds, max_results=500):
    """
    Fetches emails and returns a list of email dicts matching the Takeout MBOX-parsed schema.
    """
    service = get_service(creds)
    label_map = _build_label_map(service)
    emails = []
    next_page_token = None

    print("Scraping emails...")

    while True:
        results = service.users().messages().list(
            userId="me",
            maxResults=min(max_results - len(emails), 100),
            pageToken=next_page_token,
        ).execute()

        for msg in results.get("messages", []):
            raw = service.users().messages().get(
                userId="me",
                id=msg["id"],
                format="full"
            ).execute()

            headers = {h["name"]: h["value"] for h in raw["payload"]["headers"]}
            body_text = _extract_body(raw["payload"])
            from_name, from_email = _parse_from(headers.get("From", ""))
            date_raw = headers.get("Date", "")

            emails.append({
                "id": raw["id"],
                "subject": headers.get("Subject", "(no subject)"),
                "from_name": from_name,
                "from_email": from_email,
                "to": headers.get("To", ""),
                "date_raw": date_raw,
                "date_iso": _to_iso(date_raw),
                "gmail_labels": _resolve_labels(raw.get("labelIds", []), label_map),
                "content_type": raw["payload"].get("mimeType", ""),
                "has_attachment": _has_attachment(raw["payload"]),
                "body_text": body_text,
                "body_length": len(body_text),
                "links": _extract_links(body_text),
                "list_unsubscribe": headers.get("List-Unsubscribe", ""),
                "spam_headers": _extract_spam_headers(headers),
                "message_id": headers.get("Message-ID", ""),
                "flagged_for_deletion": False,  # your classifier sets this
            })

        print(f"  Fetched {len(emails)} emails so far...")
        next_page_token = results.get("nextPageToken")
        if not next_page_token or len(emails) >= max_results:
            break

    print(f"Done. Scraped {len(emails)} emails.")
    return emails


# Apply deletion label

def apply_deletion_label(creds, emails, label_name="Selected For Deletion"):
    """
    Creates a Gmail label if it doesn't exist, then applies it
    to all emails flagged for deletion.
    """
    service = get_service(creds)
    label_id = _get_or_create_label(service, label_name)
    print(f"Using label '{label_name}' (id: {label_id})")

    flagged = [e for e in emails if e["flagged_for_deletion"]]
    print(f"Applying label to {len(flagged)} emails...")

    chunk_size = 1000
    for i in range(0, len(flagged), chunk_size):
        chunk = flagged[i:i + chunk_size]
        service.users().messages().batchModify(
            userId="me",
            body={
                "ids": [e["id"] for e in chunk],
                "addLabelIds": [label_id],
            }
        ).execute()
        print(f"  Labeled {min(i + chunk_size, len(flagged))}/{len(flagged)}")

    print("Done.")
    return label_id


# Helpers

def _build_label_map(service):
    resp = service.users().labels().list(userId="me").execute()
    m = {l["id"]: l["name"] for l in resp.get("labels", [])}
    m.update(SYSTEM_LABEL_NAMES)
    return m


def _resolve_labels(label_ids, label_map):
    """
    Converts label IDs to human-readable names matching the X-Gmail-Labels
    format found in Takeout MBOX exports.
    'Opened' in MBOX = absence of UNREAD in the API.
    """
    resolved = []
    is_read = "UNREAD" not in label_ids

    for lid in label_ids:
        if lid == "UNREAD":
            continue
        name = label_map.get(lid) or SYSTEM_LABEL_NAMES.get(lid)
        if not name:
            name = lid.replace("_", " ").title()
        resolved.append(name)

    if is_read:
        resolved.append("Opened")

    return resolved


def _parse_from(from_header):
    """Splits 'Name <email>' into (name, email). Falls back gracefully."""
    pattern = r'^(.*?)\s*<(.+?)>\s*$'
    match = re.match(pattern, from_header)
    if match:
        return match.group(1).strip().strip('"'), match.group(2).strip()
    return "", from_header.strip()


def _to_iso(date_raw):
    try:
        return parsedate_to_datetime(date_raw).isoformat()
    except Exception:
        return ""


def _extract_body(payload):
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
    for part in payload.get("parts", []):
        result = _extract_body(part)
        if result:
            return result
    return ""


def _has_attachment(payload):
    if payload.get("filename"):
        return True
    return any(_has_attachment(p) for p in payload.get("parts", []))


def _extract_links(text):
    return re.findall(r'https?://\S+', text)


def _extract_spam_headers(headers):
    return {
        k.lower(): v for k, v in headers.items()
        if k.lower() in SPAM_HEADER_KEYS
    }


def _get_or_create_label(service, label_name):
    existing = service.users().labels().list(userId="me").execute()
    for label in existing.get("labels", []):
        if label["name"] == label_name:
            return label["id"]
    new_label = service.users().labels().create(
        userId="me",
        body={
            "name": label_name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show",
        }
    ).execute()
    print(f"Created new label: '{label_name}'")
    return new_label["id"]


# client demo 

if __name__ == "__main__":
    creds = authenticate()
    emails = scrape_emails(creds, max_results=100)

    # Placeholder: flag emails with no subject as a test
    for e in emails:
        if e["subject"] == "":
            e["flagged_for_deletion"] = True

    apply_deletion_label(creds, emails)