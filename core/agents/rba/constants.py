"""
Constants for the Rule-Based Agent's Bayesian inference engine.
"""
from enum import Enum

class RBAEventType(Enum):
    """
    Enum for game event types that influence suspicion.
    These events are interpreted from the game's state history.
    """
    # Action-based events
    PLAYER_VOTED = "player_voted"          # A player voted for someone.
    PLAYER_WAS_VOTED_FOR = "player_was_voted_for" # A player received a vote.
    PLAYER_ACCUSED = "player_accused"        # A player accused another of being Mafia.
    PLAYER_DEFENDED = "player_defended"      # A player defended another (claimed they are Citizen).
    
    # Role-specific actions (observed by others)
    HEAL_ATTEMPT = "heal_attempt"            # Doctor healing someone.
    INVESTIGATE_ATTEMPT = "investigate_attempt" # Police investigating someone.
    
    # Outcome-based events
    ACCUSATION_WAS_WRONG = "accusation_was_wrong" # A player's accusation of "Mafia" was proven false.
    ACCUSATION_WAS_RIGHT = "accusation_was_right" # A player's accusation of "Mafia" was proven true.

# --- Decision Making Constants ---
# The entropy threshold for abstaining. If the uncertainty (entropy) of the
# probability distribution is higher than this, the agent will abstain from voting.
ABSTAIN_ENTROPY_THRESHOLD = 0.8

# Softmax temperature for voting. Higher values lead to more random (exploratory) votes,
# lower values lead to more deterministic (exploitative) votes.
INITIAL_VOTE_TEMPERATURE = 1.2
MIN_VOTE_TEMPERATURE = 0.4
VOTE_TEMP_DECAY_RATE = 2.0 # Higher value means faster decay to min temp