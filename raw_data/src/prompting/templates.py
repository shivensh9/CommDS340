import json
import random

def basic_template(event_text):
    """
    Basic prompt template: simple and instructive
    """
    return f"Describe this basketball play: {event_text.strip()}"

def conversational_template(event_text):
    """
    More engaging, broadcast-style prompt
    """
    return f"Generate a live basketball commentary for the following play: \"{event_text.strip()}\""

def structured_template(event_text, game_id=None):
    """
    Include structured metadata (optional)
    """
    prompt = f"[GAME_ID: {game_id}] PLAY-BY-PLAY: {event_text.strip()}\nCOMMENTARY:"
    return prompt

def create_prompt_completion_pairs(aligned_data, template_fn=basic_template):
    """
    Converts aligned data to prompt-completion format for LLM training or inference.
    """
    pairs = []
    for example in aligned_data:
        prompt = template_fn(example["event_text"])
        completion = example["commentary"].strip()
        pairs.append({
            "prompt": prompt,
            "completion": completion
        })
    return pairs

def save_prompt_completion_data(pairs, output_path):
    with open(output_path, "w") as f:
        for item in pairs:
            f.write(json.dumps(item) + "\n")  # format for OpenAI or HuggingFace fine-tuning

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Format aligned data into prompt-completion pairs")
    parser.add_argument("--input", type=str, default="data/processed/aligned_data.json")
    parser.add_argument("--output", type=str, default="data/processed/tokenized_dataset.jsonl")
    parser.add_argument("--template", type=str, choices=["basic", "conversational", "structured"], default="basic")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        aligned_data = json.load(f)

    template_map = {
        "basic": basic_template,
        "conversational": conversational_template,
        "structured": structured_template
    }
    template_fn = template_map[args.template]

    pairs = create_prompt_completion_pairs(aligned_data, template_fn=template_fn)
    save_prompt_completion_data(pairs, args.output)

    print(f"Saved {len(pairs)} prompt-completion pairs using '{args.template}' template.")
