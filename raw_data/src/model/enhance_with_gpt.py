import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def enhance_commentary(prompt_text, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature
        )
        enhanced_text = response.choices[0].message.content.strip()
        return enhanced_text
    except Exception as e:
        print(f"Error enhancing prompt: {prompt_text}\nError: {e}")
        return ""

def main():
    input_file = "data/processed/last_81k_draft.jsonl"
    output_file = "data/processed/last_81k_enhance.jsonl"

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
