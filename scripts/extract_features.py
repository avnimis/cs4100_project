import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix

# ── load data ─────────────────────────────────────────────────────────────────

def load_emails(path):
    with open(path) as f:
        emails = json.load(f)
    # drop emails with no usable label
    emails = [e for e in emails if e.get("label") not in (None, "unknown")]
    print(f"Loaded {len(emails)} labeled emails")
    return emails

# ── structured features ───────────────────────────────────────────────────────

# domains strongly associated with a category — used as binary flags
SPAM_DOMAINS    = {"promo-rewards-claim.net", "fast-cash-now.biz", "wealthcoaching.biz",
                   "secure-account-verify.com", "midcareercoaching.info"}
PROMO_DOMAINS   = {"shein.com", "nike.com", "ae.com", "oldnavy.com", "grubhub.com",
                   "starbucks.com", "uber.com", "amenify.com"}
TRANS_DOMAINS   = {"amazon.com", "netflix.com", "venmo.com", "united.com"}
ALERT_DOMAINS   = {"github.com", "google.com"}
NEWS_DOMAINS    = {"quizlet.com", "mobbin.com", "technologyreview.com", "linkedin.com",
                   "duolingo.com", "spotify.com", "jobright.ai"}

def structured_features(email_obj):
    """Return a 1D list of hand-crafted numeric features."""
    domain       = email_obj.get("sender_domain", "")
    has_coupon   = int(len(email_obj.get("coupon_codes", [])) > 0)
    has_date     = int(email_obj.get("has_date", False))
    expired      = int(email_obj.get("expired", False))
    empty_body   = int(email_obj.get("is_empty_body", False))
    body_len     = min(email_obj.get("body_length", 0), 1000) / 1000  # normalize 0-1

    # domain category flags
    is_spam_dom  = int(domain in SPAM_DOMAINS)
    is_promo_dom = int(domain in PROMO_DOMAINS)
    is_trans_dom = int(domain in TRANS_DOMAINS)
    is_alert_dom = int(domain in ALERT_DOMAINS)
    is_news_dom  = int(domain in NEWS_DOMAINS)

    # subject signal flags
    subj = email_obj.get("clean_subject", "").lower()
    has_pct_off  = int("% off" in subj or "off" in subj)
    has_urgent   = int(any(w in subj for w in ["urgent", "suspended", "verify", "act now"]))
    has_reminder = int("reminder" in subj or "appointment" in subj)
    has_order    = int(any(w in subj for w in ["order", "shipped", "receipt", "payment", "invoice"]))
    has_alert    = int(any(w in subj for w in ["alert", "security", "reset", "storage"]))

    return [
        has_coupon, has_date, expired, empty_body, body_len,
        is_spam_dom, is_promo_dom, is_trans_dom, is_alert_dom, is_news_dom,
        has_pct_off, has_urgent, has_reminder, has_order, has_alert,
    ]

STRUCTURED_FEATURE_NAMES = [
    "has_coupon", "has_date", "expired", "empty_body", "body_len_norm",
    "is_spam_domain", "is_promo_domain", "is_trans_domain", "is_alert_domain", "is_news_domain",
    "subj_pct_off", "subj_urgent", "subj_reminder", "subj_order", "subj_alert",
]

# ── build feature matrix ──────────────────────────────────────────────────────

def build_features(emails, vectorizer=None, fit=True):
    """
    Returns:
        X       — combined sparse matrix (TF-IDF + structured)
        y       — label array
        le      — fitted LabelEncoder
        vectorizer — fitted TfidfVectorizer
    """
    texts     = [e.get("combined_text", "") for e in emails]
    labels    = [e["label"] for e in emails]
    struct    = np.array([structured_features(e) for e in emails], dtype=float)

    # TF-IDF on combined text
    if fit or vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=500,       # keep vocab small given dataset size
            ngram_range=(1, 2),     # unigrams + bigrams
            sublinear_tf=True,      # log-scale TF
            stop_words="english",
            min_df=1,
        )
        tfidf = vectorizer.fit_transform(texts)
    else:
        tfidf = vectorizer.transform(texts)

    # combine TF-IDF sparse matrix with structured features
    X = hstack([tfidf, csr_matrix(struct)])

    # encode labels
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    return X, y, le, vectorizer

# ── save artifacts ────────────────────────────────────────────────────────────

def save_artifacts(X, y, le, vectorizer, out_dir="data/features"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{out_dir}/y_labels.npy", y)
    with open(f"{out_dir}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(f"{out_dir}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    # save sparse X
    from scipy.sparse import save_npz
    save_npz(f"{out_dir}/X_features.npz", X)
    print(f"Saved feature matrix {X.shape} → {out_dir}/")
    print(f"Labels: {dict(zip(le.classes_, np.bincount(y)))}")

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    emails = load_emails("data/emails/all_emails_clean.json")
    X, y, le, vectorizer = build_features(emails, fit=True)
    save_artifacts(X, y, le, vectorizer)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"  TF-IDF features:     500")
    print(f"  Structured features: {len(STRUCTURED_FEATURE_NAMES)}")
    print(f"  Total features:      {X.shape[1]}")
    print(f"\nLabel distribution:")
    for label, count in zip(le.classes_, np.bincount(y)):
        print(f"  {label:<15} {count}")