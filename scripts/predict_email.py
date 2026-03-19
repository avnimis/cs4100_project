"""
predict_email.py — full inference pipeline for a single email.

Usage:
    python scripts/predict_email.py path/to/email.eml
    python scripts/predict_email.py --json path/to/email.json
"""

import sys
import os
import json
import pickle
import argparse
import email
import email.header
from datetime import datetime, timezone
from scipy.sparse import load_npz, hstack, csr_matrix
import numpy as np

# allow imports from scripts/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_emails import eml_to_json
from preprocess_emails import clean_text, decode_mime_words, extract_coupon_codes, extract_sender_domain
from extract_features import structured_features, STRUCTURED_FEATURE_NAMES

# ── load saved artifacts ──────────────────────────────────────────────────────

def load_model(feat_dir="data/features"):
    with open(f"{feat_dir}/final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{feat_dir}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(f"{feat_dir}/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, le, vectorizer

# ── preprocess a single email dict ───────────────────────────────────────────

def preprocess_one(raw_email):
    """Clean a raw email dict and add all derived fields."""
    from label_emails import extract_dates, is_expired

    raw_subject = raw_email.get("subject", "")
    raw_body    = raw_email.get("body", "")
    raw_from    = raw_email.get("from", "")

    clean_subject = clean_text(decode_mime_words(raw_subject))
    clean_body    = clean_text(raw_body)
    combined_text = f"{clean_subject} {clean_subject} {clean_body}".strip()

    now   = datetime.now(timezone.utc).replace(tzinfo=None)
    dates = extract_dates(combined_text)

    return {
        **raw_email,
        "clean_subject":  clean_subject,
        "clean_body":     clean_body,
        "combined_text":  combined_text,
        "sender_domain":  extract_sender_domain(raw_from),
        "coupon_codes":   extract_coupon_codes(combined_text),
        "body_length":    len(clean_body.split()),
        "is_empty_body":  len(clean_body.strip()) == 0,
        "has_date":       len(dates) > 0,
        "expired":        is_expired(raw_email, now),
        "dates_found":    [d.strftime("%Y-%m-%d") for d in dates],
    }

# ── featurize a single email ──────────────────────────────────────────────────

def featurize_one(email_obj, vectorizer):
    text   = email_obj.get("combined_text", "")
    tfidf  = vectorizer.transform([text])
    struct = csr_matrix(np.array([structured_features(email_obj)], dtype=float))
    return hstack([tfidf, struct])

# ── predict ───────────────────────────────────────────────────────────────────

def predict(email_obj, model, le, vectorizer):
    X          = featurize_one(email_obj, vectorizer)
    pred_idx   = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]
    label      = le.inverse_transform([pred_idx])[0]
    confidence = proba[pred_idx]

    # top 3 predictions with confidence
    top3_idx   = np.argsort(proba)[::-1][:3]
    top3       = [(le.inverse_transform([i])[0], proba[i]) for i in top3_idx]

    return label, confidence, top3

# ── format output ─────────────────────────────────────────────────────────────

def format_result(email_obj, label, confidence, top3):
    lines = []
    lines.append("\n" + "="*55)
    lines.append("  EMAIL ANALYSIS RESULT")
    lines.append("="*55)
    lines.append(f"  Subject : {email_obj['clean_subject'][:60]}")
    lines.append(f"  From    : {email_obj.get('from', '')[:60]}")
    lines.append(f"  Date    : {email_obj.get('date', '')[:40]}")
    lines.append("-"*55)
    lines.append(f"  Label      : {label.upper()}")
    lines.append(f"  Confidence : {confidence:.0%}")
    lines.append(f"\n  Top predictions:")
    for lbl, prob in top3:
        bar = "█" * int(prob * 20)
        lines.append(f"    {lbl:<15} {prob:.0%}  {bar}")

    # actionable flags
    lines.append("\n  Flags:")
    if email_obj.get("coupon_codes"):
        lines.append(f"    🎟  Coupon codes found: {', '.join(email_obj['coupon_codes'])}")
        if email_obj.get("expired"):
            lines.append(f"    ⚠️  Offer may be EXPIRED")
        else:
            lines.append(f"    ✅  Offer appears still valid")
    if email_obj.get("dates_found"):
        lines.append(f"    📅  Dates mentioned: {', '.join(email_obj['dates_found'])}")
    if email_obj.get("is_empty_body"):
        lines.append(f"    ⚠️  Email body is empty (HTML-only email)")
    if label == "spam":
        lines.append(f"    🚫  Recommended action: DELETE")
    elif label == "promo" and email_obj.get("expired"):
        lines.append(f"    🗑  Recommended action: DELETE (expired offer)")
    elif label in ("newsletter", "promo") and not email_obj.get("coupon_codes"):
        lines.append(f"    🗑  Recommended action: consider deleting")
    elif label == "important":
        lines.append(f"    ⭐  Recommended action: KEEP")

    lines.append("="*55)
    return "\n".join(lines)

# ── main ──────────────────────────────────────────────────────────────────────

def run(path, from_json=False):
    # load raw email
    if from_json:
        with open(path) as f:
            raw = json.load(f)
        # handle both single email and list
        if isinstance(raw, list):
            raw = raw[0]
    else:
        raw = eml_to_json(path)

    # run pipeline
    model, le, vectorizer = load_model()
    processed = preprocess_one(raw)
    label, confidence, top3 = predict(processed, model, le, vectorizer)
    print(format_result(processed, label, confidence, top3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict email label")
    parser.add_argument("path", help="Path to .eml file or .json email file")
    parser.add_argument("--json", action="store_true", help="Input is JSON instead of .eml")
    args = parser.parse_args()
    run(args.path, from_json=args.json)