import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import json
import random
from nltk.translate.bleu_score import corpus_bleu
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

# === MODEL with ATTENTION ===
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch, seq_len, hidden_dim)
        energy = energy.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # (batch, 1, hidden_dim)
        attn_weights = torch.bmm(v, energy).squeeze(1)  # (batch, seq_len)
        return torch.softmax(attn_weights, dim=1)

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM)
        self.encoder = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True, bidirectional=False)
        self.decoder = nn.LSTMCell(EMBED_DIM + HIDDEN_DIM, HIDDEN_DIM)
        self.attention = Attention(HIDDEN_DIM)
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embed(src)
        encoder_outputs, (hidden, cell) = self.encoder(src_emb)

        batch_size, tgt_len = tgt.size()
        outputs = torch.zeros(batch_size, tgt_len - 1, self.fc.out_features).to(DEVICE)

        dec_input = self.embed(tgt[:, 0])  # start with CLS token
        h, c = hidden.squeeze(0), cell.squeeze(0)

        for t in range(1, tgt_len):
            attn_weights = self.attention(h, encoder_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            decoder_input = torch.cat((dec_input, context), dim=1)
            h, c = self.decoder(decoder_input, (h, c))
            output = self.fc(h)
            outputs[:, t-1, :] = output
            dec_input = self.embed(tgt[:, t])  # Teacher forcing
        return outputs

# === TRAIN FUNCTION ===
def train_model(train_loader):
    model = Seq2SeqWithAttention(len(tokenizer)).to(DEVICE)
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

    torch.save(model.state_dict(), "custom_seq2seq_attention_model.pt")
    print("✅ Model with attention saved.")
    return model

# === GENERATION FUNCTION ===
def generate(model, src_ids):
    model.eval()
    with torch.no_grad():
        src = src_ids.unsqueeze(0).to(DEVICE)
        src_emb = model.embed(src)
        encoder_outputs, (hidden, cell) = model.encoder(src_emb)
        h, c = hidden.squeeze(0), cell.squeeze(0)

        outputs = []
        input_id = tokenizer.cls_token_id
        for _ in range(MAX_LEN):
            dec_input = model.embed(torch.tensor([input_id]).to(DEVICE))
            attn_weights = model.attention(h, encoder_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            decoder_input = torch.cat((dec_input, context), dim=1)
            h, c = model.decoder(decoder_input, (h, c))
            output = model.fc(h)
            input_id = output.argmax(-1).item()
            if input_id == tokenizer.sep_token_id:
                break
            outputs.append(input_id)
        return tokenizer.decode(outputs, skip_special_tokens=True)

# === EVALUATION FUNCTION ===
def evaluate_model(model, test_loader):
    references = []
    hypotheses = []
    model.eval()

    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(DEVICE)
            for i in range(src.size(0)):
                src_input = src[i]
                pred_sentence = generate(model, src_input)
                target_sentence = tokenizer.decode(tgt[i], skip_special_tokens=True)
                references.append([target_sentence.split()])
                hypotheses.append(pred_sentence.split())
    bleu_score = corpus_bleu(references, hypotheses)
    print(f"🎯 BLEU Score on Test Set: {bleu_score * 100:.2f}%")
# === MAIN SCRIPT ===
if __name__ == "__main__":
    with open("hf_dataset.json") as f:
        full_data = json.load(f)

    random.shuffle(full_data)
    split_index = int(len(full_data) * 0.85)
    train_data = full_data[:split_index]
    test_data = full_data[split_index:]

    train_ds = CommentaryDataset(train_data)
    test_ds = CommentaryDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1)  # Batch size 1 for BLEU

    model = train_model(train_loader)
    evaluate_model(model, test_loader)
