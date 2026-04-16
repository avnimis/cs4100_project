import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from email_data          import load_all_emails, EmailDataset
from email_deduplication import apply_deduplication
from emails_model        import EmailClassifier


# Config 
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 15
BATCH_SIZE = 16
LR         = 2e-5
VAL_SPLIT  = 0.15  # 15% of data used for validation


def main():
    print("Step 1: Loading JSON files...")
    df = load_all_emails()
    print(f"  → {len(df)} unique emails loaded across all categories")

    print("Step 2: Skipping deduplication — training data is pre-labeled in JSONL files.")
    dupes = df["duplicate"].sum()
    print(f"  → {dupes} duplicate emails already labeled in training data")

    print("Step 3: Building Dataset...")
    full_dataset = EmailDataset(df)

    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    print(f"  → Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print("Step 4: Initializing model...")
    model     = EmailClassifier(num_labels=5).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Inverse-frequency weighting: down-weights majority classes (e.g. duplicate)
    # and up-weights rare classes (e.g. old_alert, past_event).
    # pos_weight[i] = (N - n_pos_i) / n_pos_i
    EMAIL_LABELS_LOCAL = ["duplicate", "expired_offer", "old_alert", "past_event", "spam"]
    n_total = len(df)
    pos_weight = torch.tensor(
        [(n_total - df[lbl].sum()) / max(df[lbl].sum(), 1) for lbl in EMAIL_LABELS_LOCAL],
        dtype=torch.float32,
    ).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=EPOCHS * len(train_loader),
    )

    timestamp    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_logs")
    os.makedirs(logs_dir, exist_ok=True)
    step_csv     = os.path.join(logs_dir, f"step_loss_{timestamp}.csv")
    epoch_csv    = os.path.join(logs_dir, f"epoch_loss_{timestamp}.csv")

    step_records  = []   # {global_step, epoch, step, train_loss}
    epoch_records = []   # {epoch, avg_train_loss, avg_val_loss}
    global_step   = 0

    print("Step 5: Training...\n")
    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_loader, 1):
            global_step += 1
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["rule_features"].to(DEVICE),
            )
            loss = loss_fn(logits, batch["labels"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            step_records.append({"global_step": global_step, "epoch": epoch + 1,
                                  "step": step, "train_loss": loss.item()})
            print(f"  Epoch {epoch+1}/{EPOCHS}  step {step}/{len(train_loader)}  loss={loss.item():.4f}", end="\r")

        # ── Validate ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                    batch["rule_features"].to(DEVICE),
                )
                val_loss += loss_fn(logits, batch["labels"].to(DEVICE)).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        epoch_records.append({"epoch": epoch + 1, "avg_train_loss": avg_train, "avg_val_loss": avg_val})
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # ── Save CSVs ──
    with open(step_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["global_step", "epoch", "step", "train_loss"])
        writer.writeheader()
        writer.writerows(step_records)

    with open(epoch_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "avg_train_loss", "avg_val_loss"])
        writer.writeheader()
        writer.writerows(epoch_records)

    print(f"\nLoss logs saved to: {logs_dir}")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training run {timestamp}", fontsize=12, fontweight="bold")

    ax1.plot([r["global_step"] for r in step_records],
             [r["train_loss"]  for r in step_records], linewidth=0.8)
    ax1.set_title("Per-step train loss")
    ax1.set_xlabel("Global step")
    ax1.set_ylabel("Loss")

    epochs_x  = [r["epoch"]          for r in epoch_records]
    ax2.plot(epochs_x, [r["avg_train_loss"] for r in epoch_records], marker="o", label="train")
    ax2.plot(epochs_x, [r["avg_val_loss"]   for r in epoch_records], marker="o", label="val")
    ax2.set_title("Per-epoch avg loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(logs_dir, f"loss_curves_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Loss plot saved to: {plot_path}")
    plt.show()

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "email_classifier.pt")
    print("\nSaving model...")
    torch.save(model.state_dict(), model_path)
    print(f"Done! Model saved to {model_path}")


if __name__ == "__main__":
    main()