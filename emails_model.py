import torch
import torch.nn as nn
from transformers import DistilBertModel


class EmailClassifier(nn.Module):
    def __init__(self, num_labels: int = 5, rule_feature_dim: int = 6): 
        super().__init__()
        self.bert   = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.bert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + rule_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask, rule_features):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]       # [batch, 768]
        combined   = torch.cat([cls_output, rule_features], dim=1)  # [batch, 771]
        return self.classifier(combined)                      # [batch, 5] — raw logits