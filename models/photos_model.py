import torch.nn as nn
import torchvision.models as models

class ImageClassifier(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        base = models.efficientnet_b3(weights="IMAGENET1K_V1")

        # Freeze early layers, fine-tune later ones
        for name, param in base.named_parameters():
            if "features.6" not in name and "features.7" not in name and "features.8" not in name:
                param.requires_grad = False

        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()  # Remove original head
        self.backbone = base

        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_labels)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)