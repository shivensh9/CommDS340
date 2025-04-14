import sys
import os
import pandas as pd
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.prompt_engineering.template_generator import get_template

def safe_get(row, key, default=""):
    val = row.get(key, default)
    if pd.isna(val):
        return default
    return str(val)

def clean_player_name(name):
    # Remove everything after a hyphen (if present), including the hyphen.
    if "-" in name:
        return name.split("-")[0].strip()
    return name

def determine_event_type(row):
    if pd.notna(row["ShotOutcome"]):
        if row["ShotOutcome"].lower() == "make":
            if row["ShotType"] == "3PT":
                return "3pt_made"
            else:
                return "2pt_made"
        elif row["ShotOutcome"].lower() == "miss":
            return "missed_shot"
    elif pd.notna(row["FreeThrowOutcome"]):
        return "free_throw"
    elif pd.notna(row["ReboundType"]):
        if row["ReboundType"].lower() == "off":
            return "rebound_offensive"
        else:
            return "rebound_defensive"
    elif pd.notna(row["TurnoverPlayer"]):
        return "turnover"
    elif pd.notna(row["FoulType"]):
        return "foul"
    elif pd.notna(row["EnterGame"]) and pd.notna(row["LeaveGame"]):
        return "substitution"
    elif pd.notna(row["TimeoutTeam"]):
        return "timeout"
    elif pd.notna(row["JumpballAwayPlayer"]) and pd.notna(row["JumpballHomePlayer"]):
        return "jump_ball"
    elif pd.notna(row["ViolationType"]):
        return "violation"
    return None

def extract_details(row):
    player1 = (
        safe_get(row, "Shooter", "") or
        safe_get(row, "FreeThrowShooter", "") or
        safe_get(row, "Rebounder", "") or
        safe_get(row, "Fouler", "") or
        safe_get(row, "TurnoverPlayer", "Player") or
        safe_get(row, "ViolationPlayer", "Player") or
        safe_get(row, "EnterGame", "Player") or "Player"
    )
    return {
        "PLAYER1": clean_player_name(player1),
        "PLAYER2": (
            safe_get(row, "LeaveGame", "") or
            safe_get(row, "Assister", "") or
            safe_get(row, "Blocker", "")
        ),
        "TEAM": safe_get(row, "TimeoutTeam", "Team"),
        "LOCATION": "beyond the arc" if safe_get(row, "ShotType", "") == "3PT" else "inside the paint"
    }

def main():
    input_csv = os.path.join(os.path.dirname(__file__), "NBA_PBP_2020-21.csv")
    output_jsonl = "data/processed/draft_commentary.jsonl"

    df = pd.read_csv(input_csv)
    results = []

    for _, row in df.iterrows():
        event_type = determine_event_type(row)
        if not event_type:
            continue

        details = extract_details(row)
        commentary = get_template(event_type, details)

        results.append({
            "prompt": f"Rephrase this basketball commentary for a broadcast: '{commentary}'",
            "completion": ""
        })

    os.makedirs("data/processed", exist_ok=True)
    with open(output_jsonl, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Saved {len(results)} prompt drafts to {output_jsonl}")

if __name__ == "__main__":
    main()
