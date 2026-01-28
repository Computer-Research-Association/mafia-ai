"""
마피아 게임 Human-AI Interaction을 위한 NiceGUI 웹 애플리케이션의 메인 파일.
(수정: 내러티브 큐 시스템 및 랜덤 템플릿 적용)
"""
import sys
from pathlib import Path
import asyncio
import json
import random
from typing import Optional, Deque
from collections import deque

from pydantic import Field
# 프로젝트 루트 디렉토리를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / 'static'
sys.path.append(str(PROJECT_ROOT))

from nicegui import ui, app, Client
from pydantic import BaseModel

from core.engine.game import MafiaGame
from core.agents.rule_base_agent import RuleBaseAgent
from core.engine.state import Role, Phase, EventType, GameEvent
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
    narrative_queue: Deque = Field(default_factory=deque)
    is_processing_narrative: bool = False

    class Config:
        arbitrary_types_allowed = True

# 앱 상태 초기화
state = AppState(
    game_engine=MafiaGame(agents=[RuleBaseAgent(i, Role.CITIZEN) for i in range(8)]),
    log_manager=LogManager(experiment_name="narrative_generator", write_mode=False)
)

# --- 내러티브 래퍼 ---
narrative_variations = {
    "CLAIM_SELF_POLICE": [
        "저야말로 시민들을 지키는 경찰입니다.",
        "제가 경찰입니다. 제 말을 믿어주세요.",
        "진실을 밝히는 경찰, 바로 접니다."
    ],
    "CLAIM_OTHER_MAFIA": [
        "아무래도 {target_id}번 플레이어는 마피아 같습니다.",
        "{target_id}번, 정체를 밝히시죠. 당신 마피아잖아!",
        "제 감이 말해주고 있습니다. {target_id}번이 마피아입니다."
    ],
    "SILENCE": [
        "...",
        "(침묵)",
        "할 말이 없군요."
    ]
}

def get_random_narrative(event: GameEvent) -> str:
    """이벤트를 기반으로 다양한 내러티브를 반환합니다."""
    # 특정 조건에 맞는 키 생성
    key = None
    if event.event_type == EventType.CLAIM:
        if event.value == Role.POLICE and (event.target_id is None or event.target_id == event.actor_id):
            key = "CLAIM_SELF_POLICE"
        elif event.value == Role.MAFIA and event.target_id is not None and event.target_id != event.actor_id:
            key = "CLAIM_OTHER_MAFIA"
        elif event.value is None:
            key = "SILENCE"

    # 다양한 버전이 있는 경우, 랜덤 선택
    if key and key in narrative_variations:
        template = random.choice(narrative_variations[key])
        return template.format(target_id=event.target_id)
    
    # 기본 내러티브 반환
    return state.log_manager.interpret_event(event)

# --- Browser & JS Interaction ---
def log_to_console(client: Client, data: BaseModel):
    try:
        json_data = json.dumps(data.model_dump(mode='json'))
        client.run_javascript(f'console.log(JSON.parse(String.raw`{json_data}`))')
    except Exception as e:
        print(f"콘솔 로깅 실패: {e}")
        client.run_javascript(f'console.error("Failed to log data: {e}");')

def show_announcement(client: Client, text: str):
    client.run_javascript("document.getElementById('announcement-backdrop').classList.add('visible');")
    client.run_javascript(f"document.getElementById('day-announcement-text').innerText = '{text}';")
    client.run_javascript("const el = document.getElementById('day-announcement'); el.classList.remove('animate'); void el.offsetWidth; el.classList.add('animate');")
    
    def hide():
        client.run_javascript("document.getElementById('announcement-backdrop').classList.remove('visible');")
    ui.timer(2.0, hide, once=True)

# --- UI Components ---
def create_card_html(player_id: int) -> str:
    # ... (내용 변경 없음)
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
    # ... (내용 변경 없음)
    for player in state.game_engine.players:
        client.run_javascript(f"document.getElementById('player-role-{player.id}').innerText = '{player.role.name}';")
        client.run_javascript(f"document.getElementById('player-card-{player.id}').classList.{'add' if not player.alive else 'remove'}('dead');")

    phase_name = state.game_engine.phase.name.replace('_', ' ').title()
    state.day_phase_text = f"Day {state.game_engine.day} | {phase_name}"

    theme = 'night' if state.game_engine.phase == Phase.NIGHT else 'day'
    client.run_javascript(f"set_theme('{theme}')")
    
    if state.game_engine.day > state.previous_day:
        show_announcement(client, f"Day {state.game_engine.day}")
        state.previous_day = state.game_engine.day

# --- 내러티브 큐 처리 ---
async def process_narrative_queue(client: Client):
    if state.is_processing_narrative:
        return
    
    state.is_processing_narrative = True
    while state.narrative_queue:
        actor_id, text = state.narrative_queue.popleft()
        
        hold_duration_ms = 3000 # JS에 고정된 5초 대신 3초 유지
        
        # JS 함수 호출
        js_call = f"type_text('player-bubble-text-{actor_id}', '{text}', {hold_duration_ms})"
        client.run_javascript(js_call)
        
        # JS 애니메이션 시간과 동기화하여 대기
        typing_duration_ms = len(text) * 30 
        total_wait_sec = (typing_duration_ms + hold_duration_ms) / 1000.0
        
        await asyncio.sleep(total_wait_sec)
    
    state.is_processing_narrative = False

# --- Game Control ---
async def step_phase_handler(client: Client):
    if state.game_over:
        await init_game(client)
        return
    if state.is_processing_narrative: # 내러티브 처리 중에는 진행 방지
        return

    history_len_before = len(state.game_engine.history)
    living_players = [p for p in state.game_engine.players if p.alive]
    
    async def get_single_action(player):
        player_view = state.game_engine.get_game_status(viewer_id=player.id)
        action = await asyncio.to_thread(player.get_action, player_view)
        return player.id, action
    
    action_tasks = [get_single_action(p) for p in living_players]
    action_results = await asyncio.gather(*action_tasks)
    actions = dict(action_results)

    _, is_over, is_win = await asyncio.to_thread(state.game_engine.step_phase, actions)
    
    # 새 이벤트를 내러티브 큐에 추가
    new_events = state.game_engine.history[history_len_before:]
    for event in new_events:
        log_to_console(client, event)
        if event.event_type == EventType.CLAIM and event.actor_id is not None:
            text = get_random_narrative(event).replace('"', '\\"').replace("'", "\\'")
            state.narrative_queue.append((event.actor_id, text))
        elif event.event_type == EventType.VOTE and event.target_id is not None:
            client.run_javascript(f"shake_card({event.target_id});")
    
    update_ui_for_game_state(client)

    # 내러티브 큐 처리 시작
    if state.narrative_queue:
        asyncio.create_task(process_narrative_queue(client))

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
    state.narrative_queue.clear()
    state.is_processing_narrative = False

    client.run_javascript("set_theme('day')")
    client.run_javascript("document.getElementById('announcement-backdrop').classList.remove('visible');")
    client.run_javascript("document.getElementById('day-announcement-text').innerText = '';")
        
    update_ui_for_game_state(client)
    print("게임 초기화 완료.")

@ui.page('/')
async def main_page(client: Client):
    ui.html('<div id="background-div" class="background-div"></div>', sanitize=False)
    ui.add_head_html('<link rel="stylesheet" href="/static/styles.css">')
    ui.add_head_html('<script src="/static/scripts.js"></script>')

    with ui.row().classes('w-full h-16 bg-gray-900 text-white items-center justify-between p-4 shadow-md z-30'):
        ui.label().bind_text_from(state, 'day_phase_text').classes('text-xl font-mono')
        next_button = ui.button(on_click=lambda: step_phase_handler(client)).props('color=primary push')
        next_button.bind_text_from(state, 'next_button_text')
        next_button.bind_enabled_from(state, 'is_processing_narrative', backward=lambda v: not v)

    with ui.column().classes('w-full max-w-5xl mx-auto p-8 items-center'):
        with ui.grid(columns=4).classes('w-full gap-8'):
            for i in range(8):
                ui.html(content=create_card_html(i), sanitize=False)
    
    ui.html('<div id="announcement-backdrop" class="announcement-backdrop"></div>', sanitize=False)
    with ui.element('div').classes('day-announcement').props('id="day-announcement"'):
        ui.label().props('id="day-announcement-text"')

    await client.connected()
    ui.run_javascript('initCardHoverEffects();')
    await init_game(client)

# --- App Entrypoint ---
def run_app():
    ui.run(title='Mafia AI', storage_secret='a_very_secret_key_for_demo', reload=False)

if __name__ in {"__main__", "__mp_main__"}:
    run_app()

