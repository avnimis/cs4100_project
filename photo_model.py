"""
Photo Deletion Classifier
=========================
Classifies photos as KEEP (important_photos) or DELETE (everything else).

Folder structure expected:
    data/
        important_photos/
        blurry/
        clothing_tag/
        lecture_notes/
        photo_of_paper/
        presentation_photo/
        products/
        qr_code/
        random_object/
        receipt/
        screenshot_chat/
        screenshot_wallpaper/
        similar_duplicate/

Lecture concepts implemented (Ch. 7 notes):
  - Feature extraction / vectorization     → CNN layers extract pixel features into vectors
  - Perceptrons & activation functions     → each Conv2d + ReLU is a layer of perceptrons
  - Neural network (forward pass)          → ConvNet.forward()
  - Backpropagation / computation graph   → loss.backward() in PyTorch (automatic diff)
  - Gradient descent                       → optimizer.step() with learning rate alpha
  - Logistic regression output             → sigmoid on final linear layer → probability
  - Binary Cross Entropy loss              → nn.BCELoss() (same BCE from logistic regression)
  - Train/validation split                 → explicit split before training loop
  - Hyperparameter tuning                  → learning_rate, num_epochs, hidden_size

TA note (transfer learning vs. from scratch):
  This model is built ENTIRELY from scratch — no pretrained weights, no external model
  libraries like torchvision's ResNet. Every conv filter starts randomly initialized and
  learns purely from your training data via gradient descent + backpropagation.

  If you later want to compare against a pretrained approach:
    - Base model: ResNet-18 pretrained on ImageNet (1.2M images, 1000 classes)
    - Transfer: replace final fc layer with Linear(512, 1), freeze early layers
    - Fine-tune: unfreeze all layers and train at a low lr (1e-4)
  The scratch model here is what your notes teach. The pretrained version would be
  what the TA referenced — you'd need to explain ResNet's residual blocks and how
  ImageNet features transfer to your domain.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Hyperparameters
#    (These are what your notes call "hyperparameters" — values we choose
#     before training that control the learning process itself.)
# ---------------------------------------------------------------------------
DATA_DIR      = "photos"         # root folder containing class subfolders
IMG_SIZE      = 128              # resize all images to 128x128 pixels
BATCH_SIZE    = 32               # how many images per gradient descent step
LEARNING_RATE = 1e-3             # alpha (α) from gradient descent update: w = w - α∇L
NUM_EPOCHS    = 20               # how many full passes over the training data
VAL_SPLIT     = 0.2              # 20% of data held out for validation
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The one class we want to KEEP — everything else is a deletion candidate
KEEP_CLASS = "important_photos"


# ---------------------------------------------------------------------------
# 1. Feature Extraction / Vectorization
#    Your notes explain this for emails (converting "has attachment" → 0 or 1).
#    For images, the raw pixels ARE the features, but they're high-dimensional
#    and spatially structured. We normalize them into a consistent numeric range.
#
#    transforms.Normalize(mean, std) standardizes each color channel so that
#    pixel values have mean ≈ 0 and std ≈ 1 — same idea as normalizing typo
#    counts by email length in the notes.
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),   # all images → same spatial size
    transforms.ToTensor(),                      # pixel values → float tensor in [0, 1]
    transforms.Normalize(                       # standardize: x = (x - mean) / std
        mean=[0.485, 0.456, 0.406],            # per-channel means (standard values)
        std=[0.229, 0.224, 0.225]              # per-channel stds
    ),
])


# ---------------------------------------------------------------------------
# 2. Dataset — Binary Labels
#    ImageFolder loads each subfolder as a class. We then remap all labels:
#      important_photos → 0  (KEEP)
#      everything else  → 1  (DELETE)
#    This gives us a binary classification problem, exactly like spam (−1/+1)
#    in your notes, but using 0/1 for BCE loss compatibility.
# ---------------------------------------------------------------------------
class BinaryPhotoDataset(torch.utils.data.Dataset):
    """
    Wraps ImageFolder and converts multi-class labels to binary:
      0 = keep (important_photos)
      1 = delete (all other categories)
    """
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root=root, transform=transform)
        # index of the KEEP class in ImageFolder's sorted class list
        self.keep_idx = self.dataset.class_to_idx.get(KEEP_CLASS, -1)
        if self.keep_idx == -1:
            raise ValueError(f"'{KEEP_CLASS}' folder not found in {root}")
        print(f"Classes found: {self.dataset.classes}")
        print(f"'{KEEP_CLASS}' mapped to keep (label 0); all others → delete (label 1)")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Remap: keep=0, delete=1
        binary_label = 0 if label == self.keep_idx else 1
        return img, torch.tensor(binary_label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 3. The Neural Network — Built From Scratch
#
#    Architecture: CNN (Convolutional Neural Network)
#
#    WHY CNNs?
#    Your notes explain that a perceptron computes activation(w·x + b).
#    A fully-connected (dense) perceptron on a 128×128 RGB image would need
#    128×128×3 = 49,152 weights per neuron — too many to learn efficiently.
#    A convolutional layer instead applies a small learned *filter* (e.g. 3×3)
#    that slides across the image. This means:
#      - Far fewer parameters (3×3×3 = 27 weights per filter)
#      - Spatial structure is preserved (nearby pixels processed together)
#      - The same filter detects the same pattern anywhere in the image
#
#    LAYERS EXPLAINED:
#
#    Conv2d(in_channels, out_channels, kernel_size):
#      → A layer of perceptrons where each neuron looks at a 3×3 patch.
#        'out_channels' is how many different filters (feature detectors) we learn.
#        Early filters learn edges/colors; later filters learn shapes/objects.
#
#    BatchNorm2d:
#      → Normalizes activations within a batch. Keeps gradient magnitudes stable
#        so gradient descent doesn't overshoot (analogous to why we normalize
#        features before feeding them to the model).
#
#    ReLU (Rectified Linear Unit): f(x) = max(0, x)
#      → The activation function from your notes. Introduces non-linearity so
#        the network can learn non-linear decision boundaries (like XOR).
#        Without it, stacking linear layers is still just one linear layer.
#
#    MaxPool2d(2, 2):
#      → Downsamples spatial dimensions by factor of 2 (takes the max value
#        in each 2×2 region). Reduces computation and makes features
#        position-invariant (a "blurry" pattern is still blurry if shifted 1px).
#
#    Dropout(p):
#      → Randomly zeros out p% of neurons during training. Prevents overfitting
#        by forcing the network not to rely on any single neuron.
#
#    Linear(in, out):
#      → Fully-connected perceptron layer: output = w·x + b
#        This is the same as your notes' weight vector dot product + bias.
#
#    Sigmoid (output):
#      → From your logistic regression section: maps w·x + b → probability ∈ [0,1]
#        Output > 0.5 → predict DELETE; ≤ 0.5 → predict KEEP
# ---------------------------------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Convolutional Block 1 ---
        # Input: (batch, 3, 128, 128)  — 3 color channels, 128×128 pixels
        # Each Conv2d filter learns to detect a different low-level feature
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32 filters, 3×3 each
            nn.BatchNorm2d(32),
            nn.ReLU(),                                    # non-linear activation
            nn.MaxPool2d(2, 2),                           # 128×128 → 64×64
        )

        # --- Convolutional Block 2 ---
        # Deeper filters combine the edges/textures from block1 into shapes
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 64×64 → 32×32
        )

        # --- Convolutional Block 3 ---
        # Even deeper: combines shapes into higher-level visual concepts
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 32×32 → 16×16
        )

        # --- Classifier Head ---
        # Flatten spatial features into a 1-D vector (vectorization step),
        # then apply two fully-connected layers → logistic output.
        # 128 filters × 16 × 16 spatial = 32,768 features after flattening
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # (batch,128,16,16) → (batch,32768)
            nn.Linear(128 * 16 * 16, 256),      # perceptron layer: 32768 → 256
            nn.ReLU(),
            nn.Dropout(0.5),                     # regularization (prevents overfitting)
            nn.Linear(256, 1),                   # final perceptron: 256 → single score
            nn.Sigmoid(),                        # logistic function → probability in [0,1]
        )

    def forward(self, x):
        """
        Forward pass — this is the computation graph PyTorch records.
        When we call loss.backward(), PyTorch traverses this graph in reverse
        (backpropagation) applying the chain rule at each node to compute
        ∂L/∂w for every weight. Then optimizer.step() updates: w = w - α * ∂L/∂w
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x.squeeze(1)   # shape: (batch,) — one probability per image


# ---------------------------------------------------------------------------
# 4. Loss Function — Binary Cross Entropy
#
#    From your logistic regression notes, BCE is:
#      L = -[y·log(p) + (1-y)·log(1-p)]
#    where y is the true label (0=keep, 1=delete) and p is the model's
#    predicted probability of delete.
#
#    This is minimized when p ≈ y:
#      - True label = 1 (delete) and model predicts p ≈ 1 → loss ≈ 0
#      - True label = 0 (keep)   and model predicts p ≈ 0 → loss ≈ 0
#      - Any misprediction → loss increases (penalizes confidence in wrong direction)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5. Training Loop
#
#    Implements the gradient descent algorithm from your notes:
#      while not converged:
#          gradient = ∂L/∂w   (computed by backpropagation)
#          w = w - α * gradient
# ---------------------------------------------------------------------------
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # --- Forward pass (builds computation graph) ---
        preds = model(imgs)                     # predicted probabilities

        # --- Compute loss ---
        loss = criterion(preds, labels)         # BCE: how wrong are we?

        # --- Backward pass (traverse graph, compute all ∂L/∂w via chain rule) ---
        optimizer.zero_grad()                   # clear previous gradients
        loss.backward()                         # backpropagation

        # --- Gradient descent step: w = w - α * ∂L/∂w ---
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * imgs.size(0)
        predicted = (preds >= 0.5).float()      # threshold sigmoid output
        correct += (predicted == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():   # don't build computation graph during validation
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, labels)
            total_loss += loss.item() * imgs.size(0)
            predicted = (preds >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# 6. Main — Data loading, train/val split, training loop
# ---------------------------------------------------------------------------
def main():
    print(f"Using device: {DEVICE}\n")

    # Load dataset with binary labels
    dataset = BinaryPhotoDataset(root=DATA_DIR, transform=transform)
    n = len(dataset)
    n_val = int(n * VAL_SPLIT)
    n_train = n - n_val

    # Train/validation split
    # The model never sees val data during training — used only to tune hyperparameters
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset: {n} images | Train: {n_train} | Val: {n_val}\n")

    # Build model from scratch (no pretrained weights)
    model = ConvNet().to(DEVICE)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {total_params:,}\n")

    # Loss: Binary Cross Entropy (from logistic regression section of notes)
    criterion = nn.BCELoss()

    # Optimizer: Adam (an adaptive variant of gradient descent)
    # Adam adjusts the learning rate per-parameter based on gradient history,
    # which tends to converge faster than vanilla gradient descent.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler: halve lr if val loss doesn't improve for 3 epochs
    # This is analogous to taking smaller steps as you approach the minimum.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss = float('inf')
    best_model_path = "best_model.pth"

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>10} | {'Val Loss':>10} | {'Val Acc':>10}")
    print("-" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        # Step scheduler based on validation loss
        scheduler.step(val_loss)

        # Save best model (evaluated on validation set — never test set)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            flag = " ← best"
        else:
            flag = ""

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>10.2%} | "
              f"{val_loss:>10.4f} | {val_acc:>10.2%}{flag}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")


# ---------------------------------------------------------------------------
# 7. Inference — Predict on a single image
# ---------------------------------------------------------------------------
def predict(image_path: str, model_path: str = "best_model.pth") -> dict:
    """
    Load saved model and predict whether an image should be deleted.
    Returns a dict with the probability and decision.
    """
    from PIL import Image

    model = ConvNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)  # add batch dimension

    with torch.no_grad():
        prob_delete = model(tensor).item()   # sigmoid output: P(delete)

    return {
        "image": image_path,
        "prob_delete": round(prob_delete, 4),
        "prob_keep":   round(1 - prob_delete, 4),
        "decision":    "DELETE" if prob_delete >= 0.5 else "KEEP",
    }


if __name__ == "__main__":
    main()