import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset

# Configuration
MODEL_NAME = "google/flan-t5-small"  # or "t5-small"
MAX_INPUT_LENGTH = 64
MAX_TARGET_LENGTH = 64
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
OUTPUT_DIR = "t5_basketball_model"
dataset = load_dataset("json", data_files={"train": "hf_dataset.json"}, split="train")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
def preprocess(example):
    input_str = "Describe play: " + example["input_text"]
    model_input = tokenizer(input_str, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(example["target_text"], max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
    model_input["labels"] = labels["input_ids"]
    return model_input
tokenized_dataset = dataset.map(preprocess, remove_columns=["input_text", "target_text"])
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="no",
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    save_total_limit=1,
    logging_steps=10,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none"
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Fine-tuning complete! Model saved to: {OUTPUT_DIR}")
