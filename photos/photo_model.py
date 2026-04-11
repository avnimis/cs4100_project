"""
Photo Deletion Classifier
=========================
Classifies photos as KEEP (important_photos) or DELETE (everything else).

Folder structure expected:
    photos/
        important_photos/   ← KEEP (label 0)
        blurry/             ← DELETE (label 1)
        clothing_tag/
        ... (all other folders = delete)

Fix applied: class imbalance
  - The dataset has ~1 keep folder vs 12 delete folders (~94% delete).
  - A naive model learns to always predict "delete" and gets 94% accuracy
    while learning nothing. This is called the "majority class collapse."
  - Solution: weighted BCE loss + weighted sampler to balance training.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from collections import Counter

# ---------------------------------------------------------------------------
# 0. Hyperparameters
# ---------------------------------------------------------------------------
DATA_DIR      = "photos"
IMG_SIZE      = 128
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS    = 20
VAL_SPLIT     = 0.2
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEEP_CLASS    = "important_photos"

# ---------------------------------------------------------------------------
# 1. Transforms
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# 2. Dataset — Binary Labels
# ---------------------------------------------------------------------------
class BinaryPhotoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root=root, transform=transform)
        self.keep_idx = self.dataset.class_to_idx.get(KEEP_CLASS, -1)
        if self.keep_idx == -1:
            raise ValueError(f"'{KEEP_CLASS}' folder not found in {root}")
        print(f"Classes found: {self.dataset.classes}")
        print(f"'{KEEP_CLASS}' → KEEP (0) | all others → DELETE (1)\n")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        binary = 0 if label == self.keep_idx else 1
        return img, torch.tensor(binary, dtype=torch.float32)

# ---------------------------------------------------------------------------
# 3. Weighted Sampler
#
#    WHY: If 94% of images are "delete", every batch will be ~94% delete.
#    The model sees almost no "keep" examples and learns to ignore that class.
#
#    WeightedRandomSampler assigns each sample a draw probability inversely
#    proportional to its class frequency, so each batch ends up roughly 50/50
#    keep/delete regardless of the true class ratio.
#
#    This is the sampling equivalent of what Laplace smoothing does in Naive
#    Bayes from your notes — preventing any class from dominating.
# ---------------------------------------------------------------------------
def make_weighted_sampler(dataset):
    # Count how many keep (0) vs delete (1) in the training split
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    counts = Counter(labels)                          # {0: n_keep, 1: n_delete}
    n_total = len(labels)

    # Weight per class = 1 / frequency  (rare class gets higher weight)
    class_weights = {cls: n_total / count for cls, count in counts.items()}
    print(f"Class counts  — keep: {counts[0]}, delete: {counts[1]}")
    print(f"Sample weights — keep: {class_weights[0]:.2f}x, "
          f"delete: {class_weights[1]:.2f}x\n")

    # Assign each sample its class weight
    sample_weights = [class_weights[lbl] for lbl in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True      # sample with replacement so minority class repeats
    )

# ---------------------------------------------------------------------------
# 4. Weighted Loss
#
#    WHY: Even with a balanced sampler, we add a loss weight as a second
#    safeguard. BCELoss(pos_weight=w) multiplies the loss for positive
#    (delete) class errors by w, but more importantly we set it so that
#    missing a KEEP photo is penalized more — those are the photos the user
#    actually cares about not losing.
#
#    From your notes: the SVM loss function has a hyperparameter C that
#    controls the tradeoff between margin and misclassification penalty.
#    pos_weight here plays the same role — it controls how much we care
#    about one type of error vs. the other.
# ---------------------------------------------------------------------------
def make_loss(train_dataset):
    labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    counts = Counter(labels)
    # If delete is 12x more common, upweight keep errors by factor of 12
    pos_weight = torch.tensor([counts[1] / max(counts[0], 1)], dtype=torch.float32)
    print(f"BCE pos_weight (penalizes missing KEEP): {pos_weight.item():.2f}x\n")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

# ---------------------------------------------------------------------------
# 5. Model — CNN built from scratch
#
#    NOTE: Because we switched to BCEWithLogitsLoss (which is numerically
#    more stable and accepts raw logits), we REMOVE the Sigmoid from the
#    model. The loss function applies it internally. At inference time we
#    apply sigmoid manually to get a probability.
# ---------------------------------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1: learns low-level features (edges, colors)
        # Input: (batch, 3, 128, 128)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # → (batch, 32, 64, 64)
        )

        # Block 2: combines edges into textures/shapes
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # → (batch, 64, 32, 32)
        )

        # Block 3: combines shapes into higher-level concepts
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # → (batch, 128, 16, 16)
        )

        # Classifier head — vectorize spatial features → single logit
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # → (batch, 32768)
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            # NO Sigmoid here — BCEWithLogitsLoss handles it internally
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x).squeeze(1)  # raw logit per image

# ---------------------------------------------------------------------------
# 6. Per-class accuracy helper
#
#    Overall accuracy is misleading with imbalanced classes (a model that
#    always says "delete" scores 94%). We instead report:
#      - Keep accuracy:   how often we correctly identify important photos
#      - Delete accuracy: how often we correctly identify deletable photos
#    Both should be high for the model to be useful.
# ---------------------------------------------------------------------------
def per_class_acc(preds_prob, labels):
    preds = (preds_prob >= 0.5).float()
    keep_mask   = (labels == 0)
    delete_mask = (labels == 1)
    keep_acc   = (preds[keep_mask]   == 0).float().mean().item() if keep_mask.any()   else 0.0
    delete_acc = (preds[delete_mask] == 1).float().mean().item() if delete_mask.any() else 0.0
    return keep_acc, delete_acc

# ---------------------------------------------------------------------------
# 7. Training / Evaluation loops
# ---------------------------------------------------------------------------
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)                      # raw scores (no sigmoid yet)
        loss = criterion(logits, labels)          # BCEWithLogitsLoss

        optimizer.zero_grad()
        loss.backward()                           # backpropagation
        optimizer.step()                          # w = w - α * ∂L/∂w

        total_loss += loss.item() * imgs.size(0)
        all_probs.append(torch.sigmoid(logits).detach().cpu())
        all_labels.append(labels.cpu())

    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    keep_acc, delete_acc = per_class_acc(probs, labels)
    return total_loss / len(loader.dataset), keep_acc, delete_acc


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    keep_acc, delete_acc = per_class_acc(probs, labels)
    return total_loss / len(loader.dataset), keep_acc, delete_acc

# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
def main():
    print(f"Using device: {DEVICE}\n")

    full_dataset = BinaryPhotoDataset(root=DATA_DIR, transform=transform)
    n       = len(full_dataset)
    n_val   = int(n * VAL_SPLIT)
    n_train = n - n_val

    train_set, val_set = random_split(full_dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    print(f"Total: {n} | Train: {n_train} | Val: {n_val}\n")

    # Weighted sampler so each training batch is ~50/50 keep/delete
    sampler      = make_weighted_sampler(train_set)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    model     = ConvNet().to(DEVICE)
    criterion = make_loss(train_set)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss  = float('inf')
    best_model_path = "best_photo_classifier.pth"

    print(f"{'Ep':>4} | {'TrLoss':>8} | {'Tr Keep%':>9} | {'Tr Del%':>8} "
          f"| {'VaLoss':>8} | {'Va Keep%':>9} | {'Va Del%':>8}")
    print("-" * 72)

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_keep, tr_del = train(model, train_loader, criterion, optimizer)
        va_loss, va_keep, va_del = evaluate(model, val_loader, criterion)
        scheduler.step(va_loss)

        flag = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), best_model_path)
            flag = " ←"

        print(f"{epoch:>4} | {tr_loss:>8.4f} | {tr_keep:>9.2%} | {tr_del:>8.2%} "
              f"| {va_loss:>8.4f} | {va_keep:>9.2%} | {va_del:>8.2%}{flag}")

    print(f"\nDone. Best val loss: {best_val_loss:.4f} → {best_model_path}")


# ---------------------------------------------------------------------------
# 9. Inference
# ---------------------------------------------------------------------------
def predict(image_path: str, model_path: str = "best_photo_classifier.pth") -> dict:
    from PIL import Image
    model = ConvNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit        = model(tensor).item()
        prob_delete  = torch.sigmoid(torch.tensor(logit)).item()

    return {
        "image":       image_path,
        "prob_delete": round(prob_delete, 4),
        "prob_keep":   round(1 - prob_delete, 4),
        "decision":    "DELETE" if prob_delete >= 0.5 else "KEEP",
    }


if __name__ == "__main__":
    main()