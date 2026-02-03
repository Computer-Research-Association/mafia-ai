"""
HumanAgent 클래스 - 사람 플레이어를 위한 에이전트
"""
from core.agents.base_agent import BaseAgent
from core.engine.state import GameStatus, GameAction
from config import Role


class HumanAgent(BaseAgent):
    """
    사람이 직접 플레이하는 에이전트
    
    실제 행동 결정은 UI를 통해 주입되므로,
    get_action 메서드는 더미 값을 반환합니다.
    """
    
    def __init__(self, player_id: int, role: Role = Role.CITIZEN):
        super().__init__(player_id, role)
        self.char_name = "Human"
    
    def get_action(self, status: GameStatus) -> GameAction:
        """
        UI에서 행동을 주입받으므로 더미 GameAction 반환
        실제로는 step_phase_handler에서 UI를 통해 받은 행동을 사용
        """
        return GameAction(target_id=-1, claim_role=None)
