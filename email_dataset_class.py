import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

from email_preprocess import extract_features, EMAIL_LABELS


class EmailDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_length: int = 256):
        """
        Args:
            df: The merged DataFrame from load_all_emails() + apply_deduplication()
            max_length: Max token length for DistilBERT (256 is a good default)
        """
        self.df         = df.reset_index(drop=True)
        self.max_length = max_length
        self.tokenizer  = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row  = self.df.iloc[idx]
        text = str(row["text"])

        # 1. Tokenize for BERT
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 2. Rule-based features — now uses the full row, not just text
        feats = extract_features(row)
        rule_features = torch.tensor([
            float(feats["email_is_old"]),
            float(feats["has_expiry_language"]),
            float(feats["gmail_marked_spam"]),
            float(feats["has_unsubscribe"]),
            float(feats["has_tracking_links"]),
            float(feats["spam_header_count"]),
        ], dtype=torch.float32)

        # 3. Label vector (one float per class)
        labels = torch.tensor(
            row[EMAIL_LABELS].values.astype(float),
            dtype=torch.float32
        )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # [max_length]
            "attention_mask": enc["attention_mask"].squeeze(0),  # [max_length]
            "rule_features":  rule_features,                     # [3]
            "labels":         labels,                            # [5]
        }