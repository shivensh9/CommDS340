import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
import json
import random

# === CONFIG ===
MAX_LEN = 40
EMBED_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 16
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# === DATASET ===
class CommentaryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[idx]["input_text"]
        tgt = self.data[idx]["target_text"]
        src_ids = tokenizer.encode(src, truncation=True, padding="max_length", max_length=MAX_LEN)
        tgt_ids = tokenizer.encode(tgt, truncation=True, padding="max_length", max_length=MAX_LEN)
        return torch.tensor(src_ids), torch.tensor(tgt_ids)
# === MODEL ===
class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM)
        self.encoder = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.decoder = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)
    def forward(self, src, tgt):
        src_emb = self.embed(src)
        _, (h, c) = self.encoder(src_emb)

        tgt_input = tgt[:, :-1]  # exclude last token
        tgt_emb = self.embed(tgt_input)
        out, _ = self.decoder(tgt_emb, (h, c))
        logits = self.fc(out)
        return logits
# === TRAIN FUNCTION ===
def train_model(train_loader):
    model = Seq2SeqModel(len(tokenizer)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            logits = model(src, tgt)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt[:,1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), "custom_seq2seq_model.pt")
    print("✅ Model saved.")
    return model
# === EVALUATION FUNCTION ===
def evaluate_model(model, test_loader):
    model.eval()
    total_tokens = 0
    correct_tokens = 0
    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            logits = model(src, tgt)
            preds = logits.argmax(dim=-1)
            # Ignore padding when comparing
            mask = tgt[:, 1:] != tokenizer.pad_token_id
            correct = (preds == tgt[:, 1:]) & mask
            total_tokens += mask.sum().item()
            correct_tokens += correct.sum().item()
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    print(f"Test Accuracy (Token-Level): {accuracy * 100:.2f}%")
    return accuracy
# === MAIN SCRIPT ===
    # Load data
with open("hf_dataset.json") as f:
    full_data = json.load(f)

random.shuffle(full_data)
split_index = int(len(full_data) * 0.85)
train_data = full_data[:split_index]
test_data = full_data[split_index:]

    # Datasets
train_ds = CommentaryDataset(train_data)
test_ds = CommentaryDataset(test_data)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Train and Evaluate
model = train_model(train_loader)
evaluate_model(model, test_loader)
