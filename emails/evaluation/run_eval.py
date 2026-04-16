"""
Evaluate the trained EmailClassifier against the held-out labelled samples
in evaluation/data/*.jsonl and produce per-class confusion matrices.

Usage:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --threshold 0.4
    python evaluation/run_eval.py --no-plot
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, multilabel_confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizerFast

_EVAL_DIR   = os.path.dirname(os.path.abspath(__file__))
_EMAILS_DIR = os.path.dirname(_EVAL_DIR)
sys.path.insert(0, _EMAILS_DIR)

from emails_model import EmailClassifier

# ── Constants (mirror email_data.py) ─────────────────────────────────────────
EMAIL_LABELS = ["duplicate", "expired_offer", "old_alert", "past_event", "spam"]
DATA_DIR     = os.path.join(_EVAL_DIR, "data")
MODEL_PATH   = os.path.join(_EMAILS_DIR, "email_classifier.pt")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 16
OUTPUT_DIR   = os.path.join(_EVAL_DIR, "results")


# ── Data helpers (self-contained, no Gmail imports) ───────────────────────────

def _build_row(email: dict) -> dict:
    subject   = email.get("subject", "") or ""
    body_text = email.get("body_text", "") or ""
    row = {
        "id":               email.get("id", ""),
        "text":             f"{subject}\n\n{body_text}".strip(),
        "subject":          subject,
        "from_email":       email.get("from_email", ""),
        "date_iso":         email.get("date_iso", ""),
        "gmail_labels":     email.get("gmail_labels", []),
        "list_unsubscribe": email.get("list_unsubscribe", ""),
        "spam_headers":     email.get("spam_headers", {}),
        "link_count":       len(email.get("links", [])),
    }
    for label in EMAIL_LABELS:
        row[label] = email.get(label, 0)
    return row


def _extract_features(row: pd.Series) -> list[float]:
    now = datetime.now(timezone.utc)
    try:
        from dateutil import parser as dp
        email_date = dp.parse(row["date_iso"])
        if email_date.tzinfo is None:
            email_date = email_date.replace(tzinfo=timezone.utc)
        email_is_old = (now - email_date).days > 30
    except Exception:
        email_is_old = False

    text_lower       = row["text"].lower()
    expiry_kw        = ["expires", "expiring", "valid until", "limited time",
                        "ends on", "offer ends", "last chance", "today only"]
    has_expiry       = any(k in text_lower for k in expiry_kw)
    gmail_labels     = row["gmail_labels"] if isinstance(row["gmail_labels"], list) else []
    gmail_spam       = "Spam" in gmail_labels
    has_unsubscribe  = bool(row["list_unsubscribe"])
    has_tracking     = bool(re.search(r'https?://tracking\.|/t/\d+/|utm_source=', row["text"]))
    spam_hdr_count   = len(row["spam_headers"]) if isinstance(row["spam_headers"], dict) else 0
    spam_hdr_norm    = min(spam_hdr_count, 5) / 5.0

    return [float(email_is_old), float(has_expiry), float(gmail_spam),
            float(has_unsubscribe), float(has_tracking), spam_hdr_norm]


class EmailDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_length: int = 256):
        self.df         = df.reset_index(drop=True)
        self.max_length = max_length
        self.tokenizer  = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "rule_features":  torch.tensor(_extract_features(row), dtype=torch.float32),
            "labels":         torch.tensor(row[EMAIL_LABELS].values.astype(float), dtype=torch.float32),
        }


def load_eval_emails() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # Labels are nested under a "labels" key in the eval data
                nested = record.get("labels", {})
                for label in EMAIL_LABELS:
                    record[label] = nested.get(label, 0)
                rows.append(_build_row(record))
    return pd.DataFrame(rows)


# ── Main evaluation (mirrors email_evaluation.py) ────────────────────────────

def evaluate(threshold: float = 0.5, plot: bool = True):
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"eval_held_out_{timestamp}.txt")

    def log(msg=""):
        print(msg)
        f.write(msg + "\n")

    with open(output_path, "w") as f:
        log("EMAIL MODEL EVALUATION — held-out eval set")
        log(f"Run timestamp : {timestamp}")
        log(f"Model         : {MODEL_PATH}")
        log(f"Threshold     : {threshold}")
        log(f"Device        : {DEVICE}")
        log()

        log("Loading eval data...")
        df = load_eval_emails()
        if df.empty:
            log(f"No JSONL files found in {DATA_DIR}")
            return
        log(f"Evaluating on {len(df)} labelled emails")
        log()

        val_loader = DataLoader(EmailDataset(df), batch_size=BATCH_SIZE, shuffle=False)

        log("Loading model...")
        model = EmailClassifier(num_labels=len(EMAIL_LABELS)).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        all_probs, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                logits  = model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                    batch["rule_features"].to(DEVICE),
                )
                probs   = torch.sigmoid(logits).cpu().numpy()
                preds   = (probs >= threshold).astype(int)
                targets = batch["labels"].numpy().astype(int)
                all_probs.append(probs)
                all_preds.append(preds)
                all_targets.append(targets)

        y_prob = np.vstack(all_probs)
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)

        log("=" * 60)
        log("PER-CLASS METRICS")
        log("=" * 60)
        log(classification_report(y_true, y_pred, target_names=EMAIL_LABELS, zero_division=0))

        log("=" * 60)
        log("PER-CLASS CONFUSION MATRICES")
        log("  (TN  FP)")
        log("  (FN  TP)")
        log("=" * 60)
        cms = multilabel_confusion_matrix(y_true, y_pred)
        for label, cm in zip(EMAIL_LABELS, cms):
            tn, fp, fn, tp = cm.ravel()
            log(f"\n{label}:")
            log(f"  TN={tn}  FP={fp}")
            log(f"  FN={fn}  TP={tp}")

        # Aggregate "not_delete" confusion matrix
        # Positive class (1) = email should be kept (no delete label set)
        not_del_true = (y_true.sum(axis=1) == 0).astype(int)
        not_del_pred = (y_pred.sum(axis=1) == 0).astype(int)
        cm_nd = confusion_matrix(not_del_true, not_del_pred, labels=[0, 1])
        tn, fp, fn, tp = cm_nd.ravel()
        log(f"\nnot_delete (keep email):")
        log(f"  TN={tn}  FP={fp}")
        log(f"  FN={fn}  TP={tp}")

        log()
        log("=" * 60)
        log("AVERAGE PREDICTED PROBABILITY PER CLASS")
        log("=" * 60)
        for i, label in enumerate(EMAIL_LABELS):
            log(f"  {label:<20} {y_prob[:, i].mean():.4f}")

        log()
        log("=" * 60)
        log("THRESHOLD SENSITIVITY  (overall F1 at different cutoffs)")
        log("=" * 60)
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            f1     = f1_score(y_true, (y_prob >= t).astype(int), average="macro", zero_division=0)
            marker = " ← current" if t == threshold else ""
            log(f"  threshold={t:.1f}  macro-F1={f1:.4f}{marker}")

        log()
        log(f"Results saved to: {output_path}")

    if plot:
        n = len(EMAIL_LABELS)
        plot_cms   = list(cms) + [cm_nd]
        plot_labels = EMAIL_LABELS + ["not_delete"]
        fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
        fig.suptitle(
            f"Confusion Matrices — held-out eval set (n={len(df)}, threshold={threshold})",
            fontsize=13, fontweight="bold",
        )
        for ax, label, cm in zip(axes, plot_labels, plot_cms):
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticks([0, 1], ["0", "1"])
            ax.set_yticks([0, 1], ["0", "1"])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            thresh = cm.max() / 2.0
            for i in range(2):
                for j in range(2):
                    cell_label = ["TN", "FP", "FN", "TP"][i * 2 + j]
                    ax.text(j, i, f"{cell_label}\n{cm[i, j]}", ha="center", va="center",
                            fontsize=12, color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"confusion_matrices_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    evaluate(threshold=args.threshold, plot=not args.no_plot)
