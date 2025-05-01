import torch
import json
import random
from nltk.translate.bleu_score import corpus_bleu
from transformers import T5ForConditionalGeneration, T5Tokenizer
# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 40
BATCH_SIZE = 1
MODEL_DIR = "t5_basketball_model"
# === LOAD T5 Fine-Tuned Model ===
t5_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
t5_tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
# === LOAD DATA ===
with open("/Users/shivii2526/DS340Proj/CommDS340/raw_data/data/processed/tuning_dataset.jsonl") as f:
    full_data = json.load(f)
# === GENERATE FUNCTION ===
def generate_t5(input_text, temperature=0.7):
    input_ids = t5_tokenizer("Describe play: " + input_text, return_tensors="pt", truncation=True).input_ids.to(DEVICE)
    outputs = t5_model.generate(input_ids, max_length=64, temperature=temperature, do_sample=True)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
# === BLEU Evaluation ===
def evaluate_bleu(data_subset, temperature=0.7):
    references = []
    hypotheses = []
    for item in data_subset:
        src = item["input_text"]
        tgt = item["target_text"]

        pred = generate_t5(src, temperature=temperature)
        references.append([tgt.split()])
        hypotheses.append(pred.split())

    bleu = corpus_bleu(references, hypotheses)
    return bleu * 100
# === RUN EXPERIMENTS ===
def run_experiments():
    random.shuffle(full_data)
    # Different data amounts
    data_fractions = [0.2, 0.5, 1.0]  # 20%, 50%, 100% of data
    temperatures = [0.3, 0.7, 1.0]    # Different creativity levels
    results = []
    for fraction in data_fractions:
        split_size = int(len(full_data) * fraction)
        data_subset = full_data[:split_size]

        for temp in temperatures:
            bleu = evaluate_bleu(data_subset, temperature=temp)
            results.append({
                "data_fraction": fraction,
                "temperature": temp,
                "bleu_score": bleu
            })
            print(f"✅ Data Fraction: {fraction*100:.0f}%, Temperature: {temp}, BLEU: {bleu:.2f}%")
    return results
# === MAIN ===
if __name__ == "__main__":
    final_results = run_experiments()
    print("\n=== Final Experiment Results ===")
    for r in final_results:
        print(r)
