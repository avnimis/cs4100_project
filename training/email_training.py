from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

EPOCHS = 5
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = EmailClassifier(num_labels=5).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
loss_fn = nn.BCEWithLogitsLoss()

train_loader = DataLoader(EmailDataset("train.csv"), batch_size=BATCH_SIZE, shuffle=True)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=100,
    num_training_steps=EPOCHS * len(train_loader)
)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
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
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")