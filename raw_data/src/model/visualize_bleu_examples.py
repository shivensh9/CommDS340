import random
import torch
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "t5_basketball_model"
MAX_LEN = 64

# === LOAD MODEL ===
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)

# === LOAD DATA ===
with open("raw_data/data/processed/tuning_dataset.jsonl") as f:
    full_data = json.load(f)

# === GENERATE FUNCTION ===
def generate_t5(input_text, temperature=0.7):
    input_ids = tokenizer("Describe play: " + input_text, return_tensors="pt", truncation=True).input_ids.to(DEVICE)
    outputs = model.generate(
        input_ids,
        max_length=MAX_LEN,
        temperature=temperature,
        do_sample=True,
        num_beams=1  # Important: No beam search for this run
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === PLOT EXAMPLES ===
def plot_examples(data_subset, num_examples=5):
    examples = random.sample(data_subset, num_examples)

    fig, axs = plt.subplots(num_examples, 1, figsize=(12, num_examples * 3))

    if num_examples == 1:
        axs = [axs]  # ensure it's iterable

    for i, item in enumerate(examples):
        input_text = item["input_text"]
        reference = item["target_text"]
        prediction = generate_t5(input_text)

        text = f"Event: {input_text}\n\nReference: {reference}\n\nPrediction: {prediction}"

        axs[i].text(0, 0.5, text, fontsize=12, va="center", ha="left", wrap=True)
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()

# === MAIN ===
if __name__ == "__main__":
    plot_examples(full_data, num_examples=5)
