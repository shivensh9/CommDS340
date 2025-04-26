import json

with open("enhanced_data.jsonl", "r") as f, open("openai_finetune.jsonl", "w") as out:
    for line in f:
        data = json.loads(line)
        formatted = {
            "messages": [
                {"role": "user", "content": data["prompt"]},
                {"role": "assistant", "content": data["completion"]}
            ]
        }
        out.write(json.dumps(formatted) + "\n")