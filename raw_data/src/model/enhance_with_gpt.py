import os
import json
import openai

# Use the environment variable (recommended)
openai.api_key = os.environ["OPENAI_API_KEY"]

def enhance_commentary(prompt_text, temperature=0.7):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature
    )
    enhanced_text = response.choices[0].message["content"].strip()
    return enhanced_text

def main():
    input_file = "data/processed/draft_commentary.jsonl"
    output_file = "data/processed/enhanced_commentary.jsonl"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            data = json.loads(line)
            prompt_text = data["prompt"]
            enhanced_text = enhance_commentary(prompt_text, temperature=0.7)
            data["completion"] = enhanced_text
            fout.write(json.dumps(data) + "\n")

    print(f"✅ Done! Enhanced commentary saved to {output_file}")

if __name__ == "__main__":
    main()
