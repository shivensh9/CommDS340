import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset

# === CONFIG ===
MODEL_NAME = "google/flan-t5-small"   # Pretrained base
OUTPUT_DIR = "t5_basketball_model2"    # Where fine-tuned model will be saved
MAX_INPUT_LENGTH = 64
MAX_TARGET_LENGTH = 64
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100

# === LOAD TOKENIZER + BASE MODEL ===
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# === LOAD YOUR DATA ===
raw_dataset = load_dataset("json", data_files={"train": "hf_dataset.json"}, split="train")

# Split into 90% train, 10% validation
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

# === PREPROCESS FUNCTION ===
def preprocess_function(examples):
    inputs = ["Describe play: " + ex for ex in examples["input_text"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess separately for train and validation
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])
val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])

# === TRAINING ARGUMENTS ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",  # <-- evaluate after every epoch
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=WARMUP_STEPS,
    save_total_limit=1,
    save_strategy="epoch",
    logging_steps=50,
    report_to="none",  # No wandb or tensorboard unless you want
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
)

# === COLLATOR (Handles padding dynamically) ===
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,   # <-- adding validation here
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === START TRAINING ===
trainer.train()

# === SAVE MODEL AND TOKENIZER ===
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Fine-tuning completed and model saved to:", OUTPUT_DIR)
