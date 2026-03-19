import json
import re
from datetime import datetime, timezone

# data parsing
DATE_PATTERNS = [
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
    r'\b\d{1,2}/\d{1,2}/\d{4}\b',
    r'\b\d{4}-\d{2}-\d{2}\b',
]

def extract_dates(text):
    """Return list of datetime objects found in text."""
    dates = []
    for pattern in DATE_PATTERNS:
        for match in re.findall(pattern, text):
            for fmt in ('%B %d, %Y', '%m/%d/%Y', '%Y-%m-%d'):
                try:
                    dates.append(datetime.strptime(match.strip(), fmt))
                    break
                except ValueError:
                    continue
    return dates

def is_expired(email, now):
    """Return True if any date in subject+body is clearly in the past."""
    text = (email.get('subject', '') + ' ' + email.get('body', '')).lower()
    expiry_keywords = ['expires', 'expir', 'ends', 'valid through',
                       'valid until', 'last day', 'deadline', 'by']
    has_expiry_language = any(kw in text for kw in expiry_keywords)
    if not has_expiry_language:
        return False
    dates = extract_dates(email.get('subject', '') + ' ' + email.get('body', ''))
    return any(d < now for d in dates)

# auto-labeling heuristics
SPAM_SIGNALS = [
    'make money', 'earn $', 'work from home', 'no experience needed',
    'click here immediately', 'act fast', 'limited time', 'you have been selected',
    'claim your reward', 'winner', 'suspended', 'verify your account',
    'unusual activity', 'fast-cash', 'promo-rewards', 'wealthcoaching.biz',
]
PROMO_SIGNALS = [
    '% off', 'sale ends', 'use code', 'coupon', 'discount', 'shop now',
    'promo code', 'deal', 'offer expires', 'save $', 'clearance',
]
ALERT_SIGNALS = [
    'security alert', 'password reset', 'storage is', 'storage full',
    'vulnerability', 'suspicious', 'unauthorized', 'verify your identity',
    'dependabot', 'two-factor',
]
TRANSACTIONAL_SIGNALS = [
    'order #', 'order confirmed', 'has shipped', 'receipt', 'payment of',
    'subscription has been charged', 'confirmation code', 'booking confirmation',
    'transaction id', 'invoice',
]
IMPORTANT_SIGNALS = [
    'interview', 'appointment', 'reminder:', 'check-in', 'deadline',
    'internship', 'offer', 'recruiter', 'rsvp', 'registration', 'invited',
    'hackathon', 'flight', 'travel',
]
NEWSLETTER_SIGNALS = [
    'weekly digest', 'newsletter', 'this week', 'unsubscribe',
    'top stories', 'new arrivals', 'wrapped', 'streak', 'digest',
]

def auto_label(email):
    text = (email.get('subject', '') + ' ' + email.get('body', '')).lower()
    sender = email.get('from', '').lower()

    scores = {
        'spam':          sum(1 for s in SPAM_SIGNALS if s in text or s in sender),
        'promo':         sum(1 for s in PROMO_SIGNALS if s in text),
        'alert':         sum(1 for s in ALERT_SIGNALS if s in text),
        'transactional': sum(1 for s in TRANSACTIONAL_SIGNALS if s in text),
        'important':     sum(1 for s in IMPORTANT_SIGNALS if s in text),
        'newsletter':    sum(1 for s in NEWSLETTER_SIGNALS if s in text),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'unknown'


def label_emails(input_path, output_path):
    with open(input_path) as f:
        emails = json.load(f)

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    labeled, skipped = 0, 0

    for email in emails:
        # only auto-label if no label exists yet
        if not email.get('label'):
            email['label'] = auto_label(email)
            labeled += 1
        else:
            skipped += 1

        # always (re)compute date fields
        body_text = email.get('subject', '') + ' ' + email.get('body', '')
        dates = extract_dates(body_text)
        email['has_date'] = len(dates) > 0
        email['expired']  = is_expired(email, now)
        email['dates_found'] = [d.strftime('%Y-%m-%d') for d in dates]

    with open(output_path, 'w') as f:
        json.dump(emails, f, indent=2)

    print(f"Done! Auto-labeled {labeled} emails, kept existing labels for {skipped}.")
    print(f"Saved to {output_path}")

    # summary
    from collections import Counter
    label_counts = Counter(e['label'] for e in emails)
    expired_count = sum(1 for e in emails if e.get('expired'))
    print(f"\nLabel breakdown: {dict(label_counts)}")
    print(f"Expired emails: {expired_count}/{len(emails)}")


if __name__ == '__main__':
    label_emails(
        input_path='data/emails/all_emails.json',
        output_path='data/emails/all_emails_labeled.json'
    )