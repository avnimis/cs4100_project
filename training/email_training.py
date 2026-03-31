import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from preprocessing.email_preprocess   import load_all_emails
from deduplication.email_deduplication import apply_deduplication
from dataset_class.email_dataset_class import EmailDataset
from models.emails_model                import EmailClassifier


# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 5
BATCH_SIZE = 16
LR         = 2e-5
VAL_SPLIT  = 0.15  # 15% of data used for validation
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print("Step 1: Loading JSON files...")
    df = load_all_emails()
    print(f"  → {len(df)} unique emails loaded across all categories")

    print("Step 2: Running deduplication...")
    df = apply_deduplication(df)
    dupes = df["duplicate"].sum()
    print(f"  → {dupes} near-duplicate emails flagged")

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
    loss_fn   = nn.BCEWithLogitsLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=EPOCHS * len(train_loader),
    )

    print("Step 5: Training...\n")
    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0
        for batch in train_loader:
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

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}")

    print("\nSaving model...")
    torch.save(model.state_dict(), "models/email_classifier.pt")
    print("Done! Model saved to models/email_classifier.pt")


if __name__ == "__main__":
    main()