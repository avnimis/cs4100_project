from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast
import pandas as pd, torch

class EmailDataset(Dataset):
    def __init__(self, csv_path, max_length=256):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.max_length = max_length
        self.label_cols = ["past_event", "expired_offer", "spam", "duplicate", "old_alert"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        enc = self.tokenizer(
            text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        feats = extract_features(text)
        rule_features = torch.tensor([
            float(feats["has_past_date"]),
            float(feats["has_expiry_language"]),
            float(feats["spam_keyword_count"]) / 5.0,  # normalize
        ])
        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "rule_features": rule_features,
            "labels": labels,
        }