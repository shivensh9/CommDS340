import pandas as pd
import os
import argparse

def load_play_by_play(csv_path):
    """Loads raw NBA play-by-play data"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    return df

def clean_play_by_play(df):
    """Simplify and clean the play-by-play dataset for alignment"""
    # Keep relevant columns only
    cols_to_keep = [
        'GAME_ID', 'EVENTNUM', 'EVENTMSGTYPE', 'HOMEDESCRIPTION',
        'VISITORDESCRIPTION', 'PLAYER1_NAME', 'PLAYER2_NAME',
        'PLAYER3_NAME', 'PERIOD', 'PCTIMESTRING'
    ]
    df = df[cols_to_keep]

    # Drop rows where there's no description (some events are empty or duplicates)
    df = df.dropna(subset=['HOMEDESCRIPTION', 'VISITORDESCRIPTION'], how='all')

    # Fill empty descriptions for cleaner merging
    df['HOMEDESCRIPTION'] = df['HOMEDESCRIPTION'].fillna("")
    df['VISITORDESCRIPTION'] = df['VISITORDESCRIPTION'].fillna("")

    # Create a unified 'event_text' column
    df['EVENT_TEXT'] = df['HOMEDESCRIPTION'] + " " + df['VISITORDESCRIPTION']
    df['EVENT_TEXT'] = df['EVENT_TEXT'].str.strip()

    # Drop rows with empty combined event text
    df = df[df['EVENT_TEXT'].str.len() > 0]

    return df

def save_cleaned_data(df, output_path):
    """Save cleaned DataFrame to disk"""
    print(f"Saving cleaned data to {output_path}...")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean NBA play-by-play data")
    parser.add_argument("--input", type=str, default="data/raw/play_by_play_2015_2021.csv")
    parser.add_argument("--output", type=str, default="data/processed/cleaned_play_by_play.csv")
    args = parser.parse_args()

    raw_df = load_play_by_play(args.input)
    clean_df = clean_play_by_play(raw_df)
    save_cleaned_data(clean_df, args.output)

    print(f"Done! Cleaned {len(clean_df)} events.")
