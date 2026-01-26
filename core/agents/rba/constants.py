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

# Likelihood Matrix: P(E|M) - Probability of an event given the player is Mafia.
# These values are weights that multiply the current suspicion.
# Values > 1.0 increase suspicion.
# Values < 1.0 decrease suspicion.
LIKELIHOOD_MATRIX = {
    # A Mafia voting is slightly suspicious as they coordinate.
    RBAEventType.PLAYER_VOTED: 1.1,
    # Being voted for is neutral; it's the outcome that matters.
    RBAEventType.PLAYER_WAS_VOTED_FOR: 1.0,
    # Accusing is a common Mafia tactic to sow discord.
    RBAEventType.PLAYER_ACCUSED: 1.5,
    # Defending is more of a Town action.
    RBAEventType.PLAYER_DEFENDED: 0.7,
    # These are strong "Town" signals. If a Mafia fakes this, it's to build trust.
    # So we slightly decrease suspicion, but not by much, to allow for deception.
    RBAEventType.HEAL_ATTEMPT: 0.5,
    RBAEventType.INVESTIGATE_ATTEMPT: 0.5,
    # If someone makes a wrong accusation, they are either a bad Townie or a lying Mafia.
    RBAEventType.ACCUSATION_WAS_WRONG: 2.0,
    # Correctly identifying a Mafia is a strong "Town" signal.
    RBAEventType.ACCUSATION_WAS_RIGHT: 0.1,
}

# --- Decision Making Constants ---
# The entropy threshold for abstaining. If the uncertainty (entropy) of the
# probability distribution is higher than this, the agent will abstain from voting.
ABSTAIN_ENTROPY_THRESHOLD = 0.8

# Softmax temperature for voting. Higher values lead to more random (exploratory) votes,
# lower values lead to more deterministic (exploitative) votes.
VOTE_TEMPERATURE = 1.0