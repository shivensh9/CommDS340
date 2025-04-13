import sys
import os
# Add the project root (raw_data) to sys.path to allow importing from "src"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import json
from src.prompt_engineering.template_generator import get_template

def determine_event_type(row):
    """Infer event type from available columns."""
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
    return {
        "PLAYER1": row.get("Shooter") or row.get("FreeThrowShooter") or row.get("Rebounder") or
                   row.get("Fouler") or row.get("TurnoverPlayer") or
                   row.get("ViolationPlayer") or row.get("EnterGame") or "Player",
        "PLAYER2": row.get("LeaveGame") or row.get("Assister") or row.get("Blocker") or "",
        "TEAM": row.get("TimeoutTeam") or "Team",
        "LOCATION": "beyond the arc" if row.get("ShotType") == "3PT" else "inside the paint"
    }

def main():
    # Since your CSV is in the same directory as this script (src/preprocess),
    # use this path to reference the CSV.
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
