import random

# =============================================================================
# Template Dictionary: Maps common event types to a list of commentary templates.
# You can add as many variations as you like. The placeholders will be filled 
# using details extracted from your play-by-play data.
# =============================================================================

TEMPLATES = {
    # === Shot Made (general) ===
    "made_shot": [
        "{PLAYER1} drains a beautiful shot from {LOCATION}!",
        "What a shot by {PLAYER1} from {LOCATION} — simply spectacular!",
        "{PLAYER1} scores with style from {LOCATION}!",
    ],
    # === 3-Point Shot Made ===
    "3pt_made": [
        "{PLAYER1} pulls up from deep and nails a three!",
        "Nothing but net! {PLAYER1} drains a three from downtown!",
        "Bang! {PLAYER1} hits a long-range three!",
    ],
    # === 2-Point Shot Made ===
    "2pt_made": [
        "{PLAYER1} finishes the play with a smooth layup.",
        "A fine move by {PLAYER1} to score at the rim.",
        "{PLAYER1} scores inside with a powerful dunk!",
    ],
    # === Shot Missed ===
    "missed_shot": [
        "Oh no—{PLAYER1} misses from {LOCATION}.",
        "{PLAYER1} swings and misses the shot from {LOCATION}.",
        "Tough break for {PLAYER1}; the shot from {LOCATION} doesn't connect.",
    ],
    # === Free Throw Attempt ===
    "free_throw": [
        "{PLAYER1} steps to the line for a free throw.",
        "It's free throw time—{PLAYER1} has a chance to capitalize.",
        "{PLAYER1} at the charity stripe; all eyes on the free throw.",
    ],
    # === Offensive Rebound ===
    "rebound_offensive": [
        "What a hustle! {PLAYER1} secures the offensive rebound.",
        "{PLAYER1} grabs the board and gives the team another chance.",
        "{PLAYER1} snatches the offensive rebound—second chance!",
    ],
    # === Defensive Rebound ===
    "rebound_defensive": [
        "Defensive rebound for {PLAYER1}!",
        "{PLAYER1} cleans up on the glass with a strong defensive rebound.",
        "{PLAYER1} secures the defensive rebound, halting the opposing drive.",
    ],
    # === Turnover ===
    "turnover": [
        "Turnover! {PLAYER1} loses possession under pressure.",
        "A costly mistake as {PLAYER1} turns the ball over.",
        "Bad decision by {PLAYER1} results in a turnover.",
    ],
    # === Steal ===
    "steal": [
        "And it's a steal by {PLAYER1}! Fast break imminent!",
        "{PLAYER1} strips the ball away with great anticipation!",
        "Great defensive play by {PLAYER1}—steal secured!",
    ],
    # === Block ===
    "block": [
        "What a block by {PLAYER1}! The shot is denied!",
        "{PLAYER1} swats away the attempt with authority!",
        "No chance on that shot as {PLAYER1} makes a huge block!",
    ],
    # === Foul ===
    "foul": [
        "That's a foul on {PLAYER1}. The referees don't miss a beat.",
        "Flag on {PLAYER1} for a rough play on defense.",
        "{PLAYER1} is called for a foul—penalty time for the opposing team.",
    ],
    # === Substitution ===
    "substitution": [
        "Substitution: {PLAYER1} exits, and {PLAYER2} comes in.",
        "Time to switch it up—{PLAYER1} is replaced by {PLAYER2}.",
        "{PLAYER1} leaves the floor as {PLAYER2} steps up.",
    ],
    # === Timeout ===
    "timeout": [
        "Timeout is called by the {TEAM}. The players head to the bench.",
        "The clock stops as {TEAM} calls a timeout.",
        "Break in the action—timeout for the {TEAM}.",
    ],
    # === Jump Ball ===
    "jump_ball": [
        "It's a jump ball—both teams battle for control!",
        "The ball is in the air for the jump ball; who will win it?",
        "The jump ball is contested fiercely between the two teams.",
    ],
    # === Violation ===
    "violation": [
        "That's a violation on the play—possession will change.",
        "The referees call a violation, and the ball is turned over.",
        "Violation detected, and the opposing team gets the ball.",
    ],
}

# =============================================================================
# Template Generator Function
# =============================================================================
def get_template(event_type, details):
    """
    Returns a formatted commentary string for a given event type.
    
    Parameters:
    - event_type (str): One of the keys defined in TEMPLATES (e.g., '3pt_made', 'turnover', etc.).
    - details (dict): A dictionary with required placeholders. Expected keys may include:
        - 'PLAYER1': Main player involved.
        - 'PLAYER2': Secondary player (for substitutions, assists, etc.)
        - 'LOCATION': A description of the shot location or area of the court.
        - 'TEAM': The team name (for timeouts, etc.)
    
    If no template exists for the event_type, the function returns a fallback message.
    """
    templates = TEMPLATES.get(event_type, [])
    if not templates:
        return "No commentary available for this event."
    
    # Randomly select one template among the available variations.
    template = random.choice(templates)
    
    # Format the selected template using the details dictionary.
    # Any missing placeholder in 'details' can cause a KeyError, so ensure your dictionary is complete.
    try:
        commentary = template.format(**details)
    except KeyError as e:
        commentary = f"Incomplete details for commentary generation: missing {e.args[0]}"
        
    return commentary

# =============================================================================
# Example Usage for Testing (Can be removed or modified in production)
# =============================================================================
if __name__ == "__main__":
    # Sample details for various events (fill in as many values as your data provides)
    sample_details = {
        "PLAYER1": "Stephen Curry",
        "PLAYER2": "Klay Thompson",
        "LOCATION": "beyond the arc",
        "TEAM": "Golden State Warriors"
    }
    
    # List of sample event types to test all templates
    event_types = [
        "made_shot",
        "3pt_made",
        "2pt_made",
        "missed_shot",
        "free_throw",
        "rebound_offensive",
        "rebound_defensive",
        "turnover",
        "steal",
        "block",
        "foul",
        "substitution",
        "timeout",
        "jump_ball",
        "violation"
    ]
    
    print("=== Sample Draft Commentaries ===")
    for et in event_types:
        commentary = get_template(et, sample_details)
        print(f"[{et}] => {commentary}")
