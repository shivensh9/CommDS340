import json
import pandas as pd
import os
import re
from difflib import SequenceMatcher

def load_play_by_play(csv_path):
    """Load cleaned play-by-play data"""
    return pd.read_csv(csv_path)

def load_transcripts(folder_path):
    """
    Load commentary transcripts from text files in a folder.
    Assumes each file corresponds to one game.
    """
    commentary = {}
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            game_id = os.path.splitext(fname)[0]
            with open(os.path.join(folder_path, fname), 'r') as f:
                lines = f.readlines()
                commentary[game_id] = [line.strip() for line in lines if line.strip()]
    return commentary

def match_commentary(event_text, commentary_lines):
    """
    Find the most similar commentary line to a given event description.
    Uses a simple similarity metric (SequenceMatcher).
    """
    best_match = ""
    best_score = 0.0
    for line in commentary_lines:
        score = SequenceMatcher(None, event_text.lower(), line.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = line
    return best_match, best_score

def align_events(df, commentary_dict, similarity_threshold=0.4):
    aligned = []
    for idx, row in df.iterrows():
        game_id = str(row['GAME_ID'])
        event_text = row['EVENT_TEXT']
        commentary_lines = commentary_dict.get(game_id, [])
        
        if not commentary_lines:
            continue  # skip if no commentary available for this game
        
        match, score = match_commentary(event_text, commentary_lines)
        if score >= similarity_threshold:
            aligned.append({
                "game_id": game_id,
                "event_text": event_text,
                "commentary": match,
                "similarity_score": score
            })

    return aligned

def save_alignment(aligned_data, output_path):
    with open(output_path, "w") as f:
        json.dump(aligned_data, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Align play-by-play events with commentary")
    parser.add_argument("--playbyplay", type=str, default="data/processed/cleaned_play_by_play.csv")
    parser.add_argument("--transcripts", type=str, default="data/commentary/")
    parser.add_argument("--output", type=str, default="data/processed/aligned_data.json")
    args = parser.parse_args()

    print("Loading data...")
    df = load_play_by_play(args.playbyplay)
    commentary_dict = load_transcripts(args.transcripts)

    print("Aligning events with commentary...")
    aligned_data = align_events(df, commentary_dict)

    print(f"Aligned {len(aligned_data)} event-commentary pairs.")
    save_alignment(aligned_data, args.output)
