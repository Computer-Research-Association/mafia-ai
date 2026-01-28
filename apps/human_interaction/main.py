"""
마피아 게임 Human-AI Interaction을 위한 NiceGUI 웹 애플리케이션의 메인 파일.
(수정: 테마 기반 낮/밤 전환)
"""
import sys
from pathlib import Path
import asyncio
import json
from typing import Optional

# 프로젝트 루트 디렉토리를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / 'static'
sys.path.append(str(PROJECT_ROOT))

from nicegui import ui, app, Client
from pydantic import BaseModel

from core.engine.game import MafiaGame
from core.agents.rule_base_agent import RuleBaseAgent
from core.engine.state import Role, Phase, EventType
from core.managers.logger import LogManager

# 정적 파일 경로 설정
app.add_static_files('/static', STATIC_DIR)

# --- Application State ---
class AppState(BaseModel):
    game_engine: MafiaGame
    log_manager: LogManager
    day_phase_text: str = "Day 0 | WAITING"
    game_over: bool = False
    previous_day: int = 0
    next_button_text: str = "NEXT PHASE"

    class Config:
        arbitrary_types_allowed = True

# 앱 상태 초기화
agents = [RuleBaseAgent(i, Role.CITIZEN) for i in range(8)]
state = AppState(
    game_engine=MafiaGame(agents=agents),
    log_manager=LogManager(experiment_name="narrative_generator", write_mode=False)
)

# --- Browser & JS Interaction ---
def log_to_console(client: Client, data: BaseModel):
    try:
        json_data = json.dumps(data.model_dump(mode='json'))
        client.run_javascript(f'console.log(JSON.parse(String.raw`{json_data}`))')
    except Exception as e:
        print(f"콘솔 로깅 실패: {e}")
        client.run_javascript(f'console.error("Failed to log data: {e}");')

def show_announcement(client: Client, text: str):
    """화면 중앙에 텍스트 애니메이션과 함께 백드롭을 표시합니다."""
    client.run_javascript("document.getElementById('announcement-backdrop').classList.add('visible');")
    client.run_javascript(f"document.getElementById('day-announcement-text').innerText = '{text}';")
    client.run_javascript("const el = document.getElementById('day-announcement'); el.classList.remove('animate'); void el.offsetWidth; el.classList.add('animate');")
    
    def hide():
        client.run_javascript("document.getElementById('announcement-backdrop').classList.remove('visible');")
    ui.timer(2.0, hide, once=True)

# --- UI Components ---
def create_card_html(player_id: int) -> str:
    return f"""
    <div class="card-container">
        <div class="speech-bubble" id="player-bubble-{player_id}"><p id="player-bubble-text-{player_id}"></p></div>
        <div class="card" id="player-card-{player_id}">
            <div class="card-content">
                <p id="player-id-{player_id}" style="font-size: 1.5em; font-weight: bold;">Player {player_id}</p>
                <p id="player-role-{player_id}" style="font-size: 1em;">(Unknown)</p>
            </div>
        </div>
    </div>
    """

def update_ui_for_game_state(client: Client):
    """UI의 모든 동적 요소를 현재 게임 상태에 맞게 업데이트합니다."""
    for player in state.game_engine.players:
        client.run_javascript(f"document.getElementById('player-role-{player.id}').innerText = '{player.role.name}';")
        client.run_javascript(f"document.getElementById('player-card-{player.id}').classList.{'add' if not player.alive else 'remove'}('dead');")

    phase_name = state.game_engine.phase.name.replace('_', ' ').title()
    state.day_phase_text = f"Day {state.game_engine.day} | {phase_name}"

    # 테마 전환
    theme = 'night' if state.game_engine.phase == Phase.NIGHT else 'day'
    client.run_javascript(f"set_theme('{theme}')")
    
    if state.game_engine.day > state.previous_day:
        show_announcement(client, f"Day {state.game_engine.day}")
        state.previous_day = state.game_engine.day

# --- Game Control ---
async def step_phase_handler(client: Client):
    """'NEXT PHASE'/'PLAY AGAIN' 버튼 클릭 시 호출되는 핸들러."""
    if state.game_over:
        await init_game(client)
        return

    history_len_before = len(state.game_engine.history)
    living_players = [p for p in state.game_engine.players if p.alive]
    
    async def get_single_action(player):
        player_view = state.game_engine.get_game_status(viewer_id=player.id)
        action = await asyncio.to_thread(player.get_action, player_view)
        return player.id, action
    
    actions = dict(await asyncio.gather(*[get_single_action(p) for p in living_players]))
    
    _, is_over, is_win = await asyncio.to_thread(state.game_engine.step_phase, actions)
    
    new_events = state.game_engine.history[history_len_before:]
    for event in new_events:
        log_to_console(client, event)
        if event.event_type == EventType.CLAIM:
            text = state.log_manager.interpret_event(event).replace('"', '\\"').replace("'", "\\'")
            if event.actor_id is not None:
                client.run_javascript(f"type_text('player-bubble-text-{event.actor_id}', '{text}');")
        elif event.event_type == EventType.VOTE and event.target_id is not None:
            client.run_javascript(f"shake_card({event.target_id});")
    
    update_ui_for_game_state(client)

    if is_over:
        state.game_over = True
        winner = "CITIZEN" if is_win else "MAFIA"
        show_announcement(client, f"{winner} TEAM WINS!")
        state.next_button_text = "PLAY AGAIN"

async def init_game(client: Client):
    """새 게임을 시작하고 UI를 초기화합니다."""
    print("새로운 게임을 시작합니다...")
    await asyncio.to_thread(state.game_engine.reset)
    
    state.game_over = False
    state.previous_day = 0
    state.next_button_text = "NEXT PHASE"
    
    client.run_javascript("set_theme('day')")
    client.run_javascript("document.getElementById('announcement-backdrop').classList.remove('visible');")
    client.run_javascript("document.getElementById('day-announcement-text').innerText = '';")
        
    update_ui_for_game_state(client)
    print("게임 초기화 완료.")

@ui.page('/')
async def main_page(client: Client):
    # 배경 div를 페이지 최상단에 추가
    ui.html('<div id="background-div" class="background-div"></div>', sanitize=False)
    
    ui.add_head_html('<link rel="stylesheet" href="/static/styles.css">')
    ui.add_head_html('<script src="/static/scripts.js"></script>')

    # --- 상단 제어 및 상태 바 ---
    with ui.row().classes('w-full h-16 bg-gray-900 text-white items-center justify-between p-4 shadow-md z-30'):
        ui.label().bind_text_from(state, 'day_phase_text').classes('text-xl font-mono')
        ui.button(on_click=lambda: step_phase_handler(client)).props('color=primary push').bind_text_from(state, 'next_button_text')

    # --- 플레이어 카드 그리드 ---
    with ui.column().classes('w-full max-w-5xl mx-auto p-8 items-center'):
        with ui.grid(columns=4).classes('w-full gap-8'):
            for i in range(8):
                ui.html(content=create_card_html(i), sanitize=False)
    
    # --- 중앙 알림 및 백드롭 ---
    ui.html('<div id="announcement-backdrop" class="announcement-backdrop"></div>', sanitize=False)
    with ui.element('div').classes('day-announcement').props('id="day-announcement"'):
        ui.label().props('id="day-announcement-text"')

    # --- 초기화 ---
    await client.connected()
    ui.run_javascript('initCardHoverEffects();')
    await init_game(client)

# --- App Entrypoint ---
def run_app():
    ui.run(title='Mafia AI', storage_secret='a_very_secret_key_for_demo', reload=False)

if __name__ in {"__main__", "__mp_main__"}:
    run_app()

