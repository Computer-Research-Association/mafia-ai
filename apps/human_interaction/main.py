"""
마피아 게임 Human-AI Interaction을 위한 NiceGUI 웹 애플리케이션의 메인 파일.
"""
import sys
from pathlib import Path
import asyncio

# 프로젝트 루트 디렉토리를 Python 경로에 추가하여 core 모듈을 임포트할 수 있도록 함
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from nicegui import ui, app

from core.engine.game import MafiaGame
from core.agents.rule_base_agent import RuleBaseAgent
from core.engine.state import GameStatus, Role, GameAction

# --- Application State ---
# 앱의 상태를 관리하는 클래스. 게임 엔진 인스턴스를 저장합니다.
class AppState:
    def __init__(self):
        # 8명의 규칙 기반 에이전트로 게임 엔진을 초기화합니다.
        # 추후 LLMAgent나 HumanPlayerAgent 등으로 교체할 수 있습니다.
        agents = [RuleBaseAgent(i, Role.CITIZEN) for i in range(8)]
        self.game_engine = MafiaGame(agents=agents)

# 앱 전체에서 공유될 상태 객체 생성
app_state = AppState()

# --- UI Components ---
def create_header():
    """페이지 상단에 표시될 헤더를 생성합니다."""
    with ui.header().classes('bg-primary text-white p-4'):
        ui.label('Mafia AI: Human Interaction').classes('text-2xl font-bold')

def create_player_cards_grid():
    """
    게임 상태를 기반으로 플레이어 정보를 표시하는 카드 그리드를 생성합니다.
    app_state에서 직접 전체 플레이어 목록을 가져와 역할을 표시합니다.
    """
    game_engine = app_state.game_engine
    
    # game_engine.players가 비어있거나, 아직 역할이 할당되지 않은 경우(reset 전)
    if not game_engine.players or not hasattr(game_engine.players[0], 'role'):
        ui.label("게임을 시작하려면 'Start New Game' 버튼을 누르세요.")
        return

    # 플레이어 정보를 표시할 4열 그리드
    with ui.grid(columns=4).classes('w-full gap-4'):
        # 에이전트 객체를 ID 순서로 정렬하여 순회
        for player in sorted(game_engine.players, key=lambda p: p.id):
            with ui.card().classes('w-full items-center p-4'):
                ui.label(f'Player {player.id}').classes('font-bold text-lg')
                # 에이전트 객체에서 직접 역할(Role) 정보를 가져와 표시
                ui.label(player.role.name).classes('text-sm bg-gray-200 text-gray-700 px-2 py-1 rounded-full')

# --- Page Layout ---
@ui.page('/')
def main_page():
    """메인 웹 페이지를 구성합니다."""
    # --- Local Page State & Event Handlers ---
    
    async def handle_start_game():
        """'Start New Game' 버튼 클릭 시 호출될 비동기 핸들러."""
        ui.notify('새로운 게임을 시작합니다...', type='info')

        # 게임 엔진 리셋 실행
        if asyncio.iscoroutinefunction(app_state.game_engine.reset):
            game_status = await app_state.game_engine.reset()
        else:
            game_status = await asyncio.to_thread(app_state.game_engine.reset)

        # UI 업데이트
        player_grid_container.clear()
        with player_grid_container:
            create_player_cards_grid()
        
        ui.notify('게임이 준비되었습니다. 플레이어 역할이 배정되었습니다.', type='positive')

    # --- UI Definition ---

    # 페이지 상단 헤더
    create_header()

    # 메인 컨텐츠 영역
    with ui.column().classes('w-full max-w-4xl mx-auto p-8 items-center'):
        
        # 컨트롤 버튼 영역
        with ui.row().classes('mb-8 gap-4'):
            ui.button('Start New Game', on_click=handle_start_game)
            ui.button('Run Next Phase', on_click=lambda: ui.notify('아직 구현되지 않았습니다!')).props('outline')

        # 플레이어 정보가 표시될 컨테이너 (app.storage 대신 지역 변수 사용)
        player_grid_container = ui.column().classes('w-full items-center')
        with player_grid_container:
            # 초기 상태 표시
            create_player_cards_grid()

# --- App Entrypoint ---
def run_app():
    """애플리케이션을 실행합니다."""
    ui.run(title='Mafia AI', storage_secret='a_very_secret_key')

if __name__ in {"__main__", "__mp_main__"}:
    run_app()
