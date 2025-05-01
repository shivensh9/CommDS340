import torch
import json
from nltk.translate.bleu_score import corpus_bleu
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertTokenizer
from train_and_evaluate_custom_lstm import Seq2SeqModel
from train_and_evaluate_custom_lstm_attention import Seq2SeqWithAttention

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 40
BATCH_SIZE = 1

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
t5_tokenizer = T5Tokenizer.from_pretrained("t5_basketball_model")

# === LOAD DATA ===
with open("hf_dataset.json") as f:
    data = json.load(f)

split_index = int(len(data) * 0.85)
test_data = data[split_index:]

# === LOAD MODELS ===
t5_model = T5ForConditionalGeneration.from_pretrained("t5_basketball_model").to(DEVICE)
lstm_model = Seq2SeqModel(len(bert_tokenizer)).to(DEVICE)
lstm_model.load_state_dict(torch.load("custom_seq2seq_model.pt", map_location=DEVICE))
lstm_attn_model = Seq2SeqWithAttention(len(bert_tokenizer)).to(DEVICE)
lstm_attn_model.load_state_dict(torch.load("custom_seq2seq_attention_model.pt", map_location=DEVICE))

# === INFERENCE FUNCTIONS ===

def generate_t5(input_text):
    input_ids = t5_tokenizer("Describe play: " + input_text, return_tensors="pt", truncation=True).input_ids.to(DEVICE)
    outputs = t5_model.generate(input_ids, max_length=64, temperature=0.7)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_lstm(input_text, model):
    model.eval()
    with torch.no_grad():
        input_ids = bert_tokenizer.encode(input_text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        src_emb = model.embed(input_ids)
        encoder_outputs, (hidden, cell) = model.encoder(src_emb)
        h, c = hidden.squeeze(0), cell.squeeze(0)

        outputs = []
        input_id = bert_tokenizer.cls_token_id
        for _ in range(MAX_LEN):
            dec_input = model.embed(torch.tensor([input_id]).to(DEVICE))
            if hasattr(model, "attention"):
                attn_weights = model.attention(h, encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
                decoder_input = torch.cat((dec_input, context), dim=1)
            else:
                decoder_input = dec_input

            h, c = model.decoder(decoder_input, (h, c))
            output = model.fc(h)
            input_id = output.argmax(-1).item()
            if input_id == bert_tokenizer.sep_token_id:
                break
            outputs.append(input_id)
        return bert_tokenizer.decode(outputs, skip_special_tokens=True)

# === BLEU Score Calculation ===

def evaluate_model_bleu(generate_func, model):
    references = []
    hypotheses = []

    for item in test_data:
        src = item["input_text"]
        tgt = item["target_text"]

        pred = generate_func(src, model) if model else generate_func(src)
        references.append([tgt.split()])
        hypotheses.append(pred.split())

    bleu = corpus_bleu(references, hypotheses)
    return bleu * 100

# === MAIN COMPARISON ===

if __name__ == "__main__":
    print("Evaluating BLEU for T5 Fine-Tuned...")
    t5_bleu = evaluate_model_bleu(generate_t5, None)
    print(f"✅ T5 BLEU: {t5_bleu:.2f}%\n")

    print("Evaluating BLEU for Basic LSTM...")
    lstm_bleu = evaluate_model_bleu(generate_lstm, lstm_model)
    print(f"✅ LSTM BLEU: {lstm_bleu:.2f}%\n")

    print("Evaluating BLEU for LSTM + Attention...")
    lstm_attn_bleu = evaluate_model_bleu(generate_lstm, lstm_attn_model)
    print(f"✅ LSTM+Attention BLEU: {lstm_attn_bleu:.2f}%\n")
