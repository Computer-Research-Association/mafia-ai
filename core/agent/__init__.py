"""
Character module
"""

import config
from core.agent.baseAgent import BaseAgent
from core.agent.llmAgent import LLMAgent


def create_player(char_id: int, player_id: int) -> BaseAgent:
    """
    Create a rational player (personality-based agents removed).
    char_id parameter is kept for backward compatibility but ignored.
    """
    # All players now use RationalCharacter
    return LLMAgent(player_id)
