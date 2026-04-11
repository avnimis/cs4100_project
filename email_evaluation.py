import torch
import numpy as np
import os
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from email_data          import load_all_emails, EMAIL_LABELS, EmailDataset
from email_deduplication import apply_deduplication
from emails_model        import EmailClassifier


# Config
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH     = "email_classifier.pt"
THRESHOLD      = 0.5
BATCH_SIZE     = 16
VAL_SPLIT      = 0.15
OUTPUT_DIR     = "results"


def evaluate():
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"eval_{timestamp}.txt")

    def log(msg=""):
        """Print to console and write to file simultaneously."""
        print(msg)
        f.write(msg + "\n")

    with open(output_path, "w") as f:
        log(f"EMAIL MODEL EVALUATION")
        log(f"Run timestamp : {timestamp}")
        log(f"Model         : {MODEL_PATH}")
        log(f"Threshold     : {THRESHOLD}")
        log(f"Device        : {DEVICE}")
        log()

        log("Loading data...")
        df = load_all_emails()
        df = apply_deduplication(df)

        full_dataset = EmailDataset(df)
        val_size     = int(len(full_dataset) * VAL_SPLIT)
        train_size   = len(full_dataset) - val_size

        _, val_ds = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        log(f"Evaluating on {val_size} validation emails")
        log()

        log("Loading model...")
        model = EmailClassifier(num_labels=len(EMAIL_LABELS)).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        all_probs   = []
        all_preds   = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                    batch["rule_features"].to(DEVICE),
                )
                probs   = torch.sigmoid(logits).cpu().numpy()
                preds   = (probs >= THRESHOLD).astype(int)
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

        log()
        log("=" * 60)
        log("AVERAGE PREDICTED PROBABILITY PER CLASS")
        log("=" * 60)
        for i, label in enumerate(EMAIL_LABELS):
            avg_prob = y_prob[:, i].mean()
            log(f"  {label:<20} {avg_prob:.4f}")

        log()
        log("=" * 60)
        log("THRESHOLD SENSITIVITY  (overall F1 at different cutoffs)")
        log("=" * 60)
        from sklearn.metrics import f1_score
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            preds_t = (y_prob >= t).astype(int)
            f1      = f1_score(y_true, preds_t, average="macro", zero_division=0)
            marker  = " ← current" if t == THRESHOLD else ""
            log(f"  threshold={t:.1f}  macro-F1={f1:.4f}{marker}")

        log()
        log(f"Results saved to: {output_path}")


if __name__ == "__main__":
    evaluate()