"""
마피아 게임 Human-AI Interaction을 위한 NiceGUI 웹 애플리케이션의 메인 파일.
(수정: 백그라운드 태스크 UI 업데이트 오류 최종 해결)
"""
import sys
from pathlib import Path
import asyncio
import json
from typing import List, Dict

# 프로젝트 루트 디렉토리를 Python 경로에 추가하여 core 모듈을 임포트할 수 있도록 함
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from nicegui import ui, app, Client
from pydantic import BaseModel

from core.engine.game import MafiaGame
from core.agents.rule_base_agent import RuleBaseAgent
from core.engine.state import GameStatus, Role

# --- Application State ---
class AppState:
    def __init__(self):
        agents = [RuleBaseAgent(i, Role.CITIZEN) for i in range(8)]
        self.game_engine = MafiaGame(agents=agents)

app.state = AppState()

# --- Browser Console Logging ---
def log_to_console(client: Client, data: BaseModel):
    """Pydantic 모델을 JSON으로 변환하여 특정 클라이언트의 브라우저 콘솔에 로깅합니다."""
    try:
        json_data = json.dumps(data.model_dump(mode='json'))
        # [수정] ui.run_javascript 대신 client.run_javascript 사용
        client.run_javascript(f'console.log(JSON.parse(String.raw`{json_data}`))')
    except Exception as e:
        print(f"콘솔 로깅 실패: {e}")
        # 에러 로깅도 client 객체 사용
        client.run_javascript(f'console.error("Failed to log data: {e}");')

# --- UI Components ---
def create_header():
    """페이지 상단에 표시될 헤더를 생성합니다."""
    with ui.header().classes('bg-primary text-white p-4'):
        ui.label('Mafia AI: Human Interaction').classes('text-2xl font-bold')

def update_player_cards(card_elements: List[Dict]):
    """미리 생성된 플레이어 카드 UI 요소의 내용을 업데이트합니다."""
    game_engine = app.state.game_engine
    for i, player in enumerate(sorted(game_engine.players, key=lambda p: p.id)):
        card_ui = card_elements[i]
        status = "alive" if player.alive else "dead"
        card_ui['status_label'].set_text(f'({status})')
        
        alive_classes = 'bg-green-200 text-green-800'
        dead_classes = 'bg-red-200 text-red-800'
        if player.alive:
            card_ui['status_label'].classes(remove=dead_classes, add=alive_classes)
        else:
            card_ui['status_label'].classes(remove=alive_classes, add=dead_classes)

# --- Game Loop ---
async def run_game_loop(client: Client, card_elements: List[Dict]):
    """게임의 메인 루프를 실행하고, 상태와 이벤트를 콘솔에 로깅합니다."""
    game_engine: MafiaGame = app.state.game_engine

    print("새로운 게임을 시작합니다...")
    await asyncio.to_thread(game_engine.reset)
    update_player_cards(card_elements)

    is_over = False
    while not is_over:
        await asyncio.sleep(2)

        living_players = [p for p in game_engine.players if p.alive]
        history_len_before = len(game_engine.history)

        async def get_single_action(player):
            player_view = game_engine.get_game_status(viewer_id=player.id)
            # [수정] log_to_console에 client 전달
            log_to_console(client, player_view)
            action = await asyncio.to_thread(player.get_action, player_view)
            return player.id, action
        
        action_tasks = [get_single_action(p) for p in living_players]
        action_results = await asyncio.gather(*action_tasks)
        actions = dict(action_results)
        
        _, is_over, is_win = await asyncio.to_thread(game_engine.step_phase, actions)
        
        new_events = game_engine.history[history_len_before:]
        for event in new_events:
            # [수정] log_to_console에 client 전달
            log_to_console(client, event)

        update_player_cards(card_elements)
    
    winner = "시민" if is_win else "마피아"
    print(f"게임 종료! {winner} 팀의 승리입니다!")
    # [수정] log_to_console에 client 전달
    log_to_console(client, game_engine.get_game_status())

@ui.page('/')
async def main_page(client: Client):
    """메인 웹 페이지를 구성합니다."""
    create_header()

    card_elements = []
    with ui.column().classes('w-full max-w-4xl mx-auto p-8 items-center'):
        with ui.grid(columns=4).classes('w-full gap-4'):
            for i in range(8):
                with ui.card().classes('w-full items-center p-4') as card:
                    id_label = ui.label(f'Player {i}').classes('font-bold text-lg')
                    status_label = ui.label('(waiting)').classes('text-sm px-2 py-1 rounded-full')
                    card_elements.append({'card': card, 'id_label': id_label, 'status_label': status_label})

    await client.connected()
    # [수정] run_game_loop에 client 전달
    asyncio.create_task(run_game_loop(client, card_elements))

# --- App Entrypoint ---
def run_app():
    """애플리케이션을 실행합니다."""
    ui.run(title='Mafia AI', storage_secret='a_very_secret_key_for_demo', reload=False)

if __name__ in {"__main__", "__mp_main__"}:
    run_app()

