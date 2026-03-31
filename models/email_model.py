import torch
import torch.nn as nn
from transformers import DistilBertModel

class EmailClassifier(nn.Module):
    def __init__(self, num_labels: int, rule_feature_dim: int = 3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden = self.bert.config.hidden_size  # 768

        # Fuse BERT output with hand-crafted rule features
        self.classifier = nn.Sequential(
            nn.Linear(hidden + rule_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, rule_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        combined = torch.cat([cls_output, rule_features], dim=1)
        return self.classifier(combined)  # Raw logits — use BCEWithLogitsLoss