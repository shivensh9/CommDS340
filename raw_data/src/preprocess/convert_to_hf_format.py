import json
input_path = "raw_data/data/processed/all_enhanced.jsonl"          
output_path = "raw_data/data/processed/tuning_dataset.jsonl"             
data = []
with open(input_path, "r") as infile:
    for line in infile:
        entry = json.loads(line)
        prompt = entry.get("prompt", "").strip()
        completion = entry.get("completion", "").strip()
        if prompt and completion:
            data.append({
                "input_text": prompt,
                "target_text": completion
            })
with open(output_path, "w") as out:
    json.dump(data, out, indent=2)
print(f"✅ Converted {len(data)} examples to Hugging Face fine-tuning format.")
