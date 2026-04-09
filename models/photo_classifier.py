import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR    = "photos"          # root folder containing one subfolder per class
BATCH_SIZE  = 32
NUM_EPOCHS  = 30
LR          = 1e-4
IMG_SIZE    = 224
NUM_WORKERS = 0  # macOS multiprocessing fix
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {DEVICE}")

if __name__ == '__main__':

# ──────────────────────────────────────────────
# TRANSFORMS
# Heavy augmentation for small datasets
# ──────────────────────────────────────────────
  train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean/std
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),              # randomly mask patches
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ──────────────────────────────────────────────
# DATASET — expects photos/<class_name>/*.jpg
# ──────────────────────────────────────────────
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # allow truncated images

class SafeImageFolder(ImageFolder):
    """ImageFolder that skips corrupt images instead of crashing."""
    def __getitem__(self, idx):
        while True:
            try:
                return super().__getitem__(idx)
            except Exception:
                # Remove bad index and try the next one
                self.samples.pop(idx)
                self.targets.pop(idx)
                if idx >= len(self.samples):
                    idx = 0

full_dataset = SafeImageFolder(DATA_DIR, transform=train_transforms)
class_names  = full_dataset.classes
num_classes  = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# 80/20 train-val split
n_total = len(full_dataset)
n_val   = int(0.2 * n_total)
n_train = n_total - n_val
train_set, val_set = torch.utils.data.random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)
# val set uses val transforms
val_set.dataset = copy.deepcopy(full_dataset)
val_set.dataset.transform = val_transforms

# ──────────────────────────────────────────────
# WEIGHTED SAMPLER — handles class imbalance
# ──────────────────────────────────────────────
train_labels  = [full_dataset.targets[i] for i in train_set.indices]
class_counts  = np.bincount(train_labels, minlength=num_classes)
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = [class_weights[lbl] for lbl in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=NUM_WORKERS)

# ──────────────────────────────────────────────
# MODEL — ResNet-50 with custom head
# ──────────────────────────────────────────────
def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer3, layer4, and fc for fine-tuning
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final FC with a custom classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

model = build_model(num_classes).to(DEVICE)

# ──────────────────────────────────────────────
# LOSS, OPTIMIZER, SCHEDULER
# ──────────────────────────────────────────────
# Class-weighted cross entropy for extra imbalance handling
tensor_class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=tensor_class_weights)

# Only optimize unfrozen params
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)

# Cosine annealing: smoothly reduces LR each epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc  = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, preds, labels_list = evaluate(model, val_loader, criterion)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, "best_photo_classifier.pth")
        print(f"  ✓ New best model saved (val_acc={best_val_acc:.3f})")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")

# ──────────────────────────────────────────────
# EVALUATION — classification report + confusion matrix
# ──────────────────────────────────────────────
model.load_state_dict(best_model_wts)
_, _, preds, true_labels = evaluate(model, val_loader, criterion)

print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=class_names, labels=list(range(num_classes)), zero_division=0))

cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
            yticklabels=class_names, cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Loss / accuracy curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="Train")
ax1.plot(history["val_loss"],   label="Val")
ax1.set_title("Loss"); ax1.legend()
ax2.plot(history["train_acc"], label="Train")
ax2.plot(history["val_acc"],   label="Val")
ax2.set_title("Accuracy"); ax2.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()