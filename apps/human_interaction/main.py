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
    ui_event_queue: Deque = Field(default_factory=deque)
    is_processing_events: bool = False

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
    "CLAIM_SELF_DOCTOR": [ # New entry
        "저는 의사입니다. 밤에는 시민들을 치료하죠.",
        "저를 믿으세요. 제가 바로 의사입니다.",
        "아픈 시민들을 치료하는 의사가 접니다."
    ],
    "CLAIM_OTHER_MAFIA": [
        "아무래도 {target_id}번 플레이어는 마피아 같습니다.",
        "{target_id}번, 정체를 밝히시죠. 당신 마피아잖아!",
        "제 감이 말해주고 있습니다. {target_id}번이 마피아입니다."
    ],
    "VOTE_TARGET": [
        "저는 {target_id}번에게 투표하겠습니다.",
        "의심스러운 {target_id}번에게 한 표 행사합니다.",
        "{target_id}번이 마피아라고 확신합니다."
    ],
    "ABSTAIN": [
        "아직은 잘 모르겠습니다. 기권하겠습니다.",
        "이번 투표는 기권입니다.",
        "확신이 설 때까지 움직이지 않겠습니다."
    ],
    "SILENCE": [
        "흠...",
        "과연..." # Modified entry
    ]
}

def get_random_narrative(event: GameEvent) -> str:
    """이벤트를 기반으로 다양한 내러티브를 반환합니다."""
    # 특정 조건에 맞는 키 생성
    key = None
    if event.event_type == EventType.CLAIM:
        if event.value == Role.POLICE and (event.target_id is None or event.target_id == event.actor_id):
            key = "CLAIM_SELF_POLICE"
        elif event.value == Role.DOCTOR and (event.target_id is None or event.target_id == event.actor_id): # New logic
            key = "CLAIM_SELF_DOCTOR"
        elif event.value == Role.MAFIA and event.target_id is not None and event.target_id != event.actor_id:
            key = "CLAIM_OTHER_MAFIA"
        elif event.value is None:
            key = "SILENCE"
    elif event.event_type == EventType.VOTE:
        if event.target_id is not None:
            key = "VOTE_TARGET"
        else:
            key = "ABSTAIN"

    # 다양한 버전이 있는 경우, 랜덤 선택
    if key and key in narrative_variations:
        template = random.choice(narrative_variations[key])
        # target_id가 없는 경우(ex: 기권) format 에러 방지
        return template.format(target_id=event.target_id) if '{target_id}' in template else template
    
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
    """Dynamically creates and shows a new banner, which auto-removes itself."""
    escaped_text = text.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')
    display_duration = 3000  # ms, how long banner stays fully visible

    js_code = f"""
        const container = document.getElementById('banner-container');
        if (container) {{
            const banner = document.createElement('div');
            banner.className = 'banner-item';
            banner.innerText = '{escaped_text}';
            container.appendChild(banner);
            
            // Add class to trigger entrance animation
            requestAnimationFrame(() => {{
                void banner.offsetWidth; // Force reflow to ensure transition starts
                banner.classList.add('visible');
            }});

            // After display_duration, start the hiding animation
            setTimeout(() => {{
                // Step 1: Fade out (opacity transition)
                banner.classList.remove('visible');
                
                // Step 2: After opacity transition ends, collapse height
                banner.addEventListener('transitionend', function onOpacityEnd(e) {{
                    if (e.propertyName === 'opacity') {{
                        banner.removeEventListener('transitionend', onOpacityEnd);
                        
                        // Get computed height including padding
                        const computedStyle = window.getComputedStyle(banner);
                        const totalHeight = banner.offsetHeight;
                        
                        // Set explicit height before transition
                        banner.style.height = totalHeight + 'px';
                        
                        // Force reflow
                        void banner.offsetHeight;
                        
                        // Add collapsing class and animate to 0
                        requestAnimationFrame(() => {{
                            banner.classList.add('collapsing');
                            banner.style.height = '0px';
                            banner.style.marginBottom = '0px';
                            banner.style.paddingTop = '0px';
                            banner.style.paddingBottom = '0px';
                        }});
                        
                        // Step 3: After height transition ends, remove from DOM
                        banner.addEventListener('transitionend', function onCollapseEnd(e) {{
                            if (e.propertyName === 'height') {{
                                banner.remove();
                            }}
                        }});
                    }}
                }});

            }}, {display_duration});
        }}
    """
    client.run_javascript(js_code)

# --- UI Components ---
def create_card_html(player_id: int) -> str:
    """플레이어 카드의 HTML 구조를 생성합니다. 플레이어별 고유 클래스를 추가합니다."""
    return f"""
    <div class="card-container card-container-{player_id}">
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
    for player in state.game_engine.players:
        client.run_javascript(f"document.getElementById('player-role-{player.id}').innerText = '{player.role.name}';")
        client.run_javascript(f"document.getElementById('player-card-{player.id}').classList.{'add' if not player.alive else 'remove'}('dead');")

    phase_name = state.game_engine.phase.name.replace('_', ' ').title()
    state.day_phase_text = f"Day {state.game_engine.day} | {phase_name}"

    theme = 'night' if state.game_engine.phase == Phase.NIGHT else 'day'
    client.run_javascript(f"set_theme('{theme}')")
    
    if state.game_engine.day > state.previous_day:
        state.previous_day = state.game_engine.day

# --- UI Event Queue Processor ---
def process_ui_events(client: Client):
    """(UI-Safe) 모든 UI 이벤트를 순차적으로 처리하는 마스터 큐 프로세서 (ui.timer 기반)"""
    # 큐가 비었으면 처리를 종료하고 플래그를 해제
    if not state.ui_event_queue:
        state.is_processing_events = False
        print("UI Event processor FINISHED.")
        return

    # 큐의 첫 이벤트를 꺼내서 처리
    event_type, *args = state.ui_event_queue.popleft()
    print(f"Processing UI Event: {event_type}, {args}")

    delay = 0.1  # 다음 이벤트까지의 기본 대기 시간

    if event_type == 'announcement':
        text, = args
        show_announcement(client, text)
        delay = 0.5  # 배너 사이의 간격
    
    elif event_type == 'narrative':
        actor_id, text = args
        hold_duration_ms = 1500  # 단축된 유지 시간
        js_call = f"type_text('player-bubble-text-{actor_id}', '{text}', {hold_duration_ms})"
        client.run_javascript(js_call)
        delay = 0.25  # 내러티브 사이 간격

    # 처리 후, 다음 이벤트를 처리하기 위해 스스로를 다시 스케줄링
    ui.timer(delay, lambda: process_ui_events(client), once=True)

# --- Game Control ---
async def step_phase_handler(client: Client):
    """게임 단계를 진행하고, 그에 따른 UI 이벤트를 큐에 추가"""
    if state.game_over:
        await init_game(client)
        return
    if state.is_processing_events: # 이벤트 처리 중에는 진행 방지
        return

    old_phase = state.game_engine.phase
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
    new_phase = state.game_engine.phase

    # --- UI 이벤트 큐에 추가 ---
    # 1. 단계 전환 알림
    if new_phase != old_phase:
        if new_phase == Phase.NIGHT:
            state.ui_event_queue.append(('announcement', "밤이 되었습니다"))
        elif new_phase == Phase.DAY_DISCUSSION:
            state.ui_event_queue.append(('announcement', f"{state.game_engine.day}일차 낮이 밝았습니다"))
    
    # 2. 게임 이벤트(사망, 발언 등) 처리
    new_events = state.game_engine.history[history_len_before:]
    for event in new_events:
        # log_to_console(client, event) # 임시 비활성화
        if event.event_type == EventType.EXECUTE:
            state.ui_event_queue.append(('announcement', f"투표로 {event.target_id}번 플레이어가 처형되었습니다. (직업: {event.value.name})"))
        elif event.event_type == EventType.KILL:
            state.ui_event_queue.append(('announcement', f"지난 밤 {event.target_id}번 플레이어가 살해당했습니다."))
        elif event.event_type in [EventType.CLAIM, EventType.VOTE] and event.actor_id is not None:
            if event.event_type == EventType.VOTE and event.target_id is not None:
                client.run_javascript(f"shake_card({event.target_id});")
            text = get_random_narrative(event).replace('"', '\\"').replace("'", "\\'")
            state.ui_event_queue.append(('narrative', event.actor_id, text))

    update_ui_for_game_state(client)

    # --- UI 이벤트 처리 시작 ---
    if state.ui_event_queue and not state.is_processing_events:
        state.is_processing_events = True
        print("UI Event processor KICKED OFF.")
        process_ui_events(client)

    # 3. 게임 종료 알림
    if is_over:
        state.game_over = True
        winner = "CITIZEN" if is_win else "MAFIA"
        state.ui_event_queue.append(('announcement', f"{winner} 팀 승리!"))
        state.next_button_text = "PLAY AGAIN"
        if not state.is_processing_events:
            state.is_processing_events = True
            print("UI Event processor KICKED OFF for game over.")
            process_ui_events(client)


async def init_game(client: Client):
    """새 게임을 시작하고 UI를 초기화합니다."""
    print("새로운 게임을 시작합니다...")
    await asyncio.to_thread(state.game_engine.reset)
    
    state.game_over = False
    state.previous_day = 0
    state.next_button_text = "NEXT PHASE"
    state.ui_event_queue.clear()
    state.is_processing_events = False

    client.run_javascript("set_theme('day')")
    # 아래 2줄은 더 이상 존재하지 않는 요소에 접근하므로 삭제 또는 주석 처리합니다.
    # client.run_javascript("document.getElementById('announcement-backdrop').classList.remove('visible');")
    # client.run_javascript("document.getElementById('day-announcement-text').innerText = '';")
        
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
        next_button.bind_enabled_from(state, 'is_processing_events', backward=lambda v: not v)

    with ui.element('div').classes('player-area w-full'):
        for i in range(8):
            ui.html(content=create_card_html(i), sanitize=False)
    
    # Container for dynamically added announcement banners
    ui.element('div').props('id="banner-container"')

    await client.connected()
    
    # JavaScript 함수: 플레이어 발언 기록 가져오기
    ui.run_javascript('''
        window.getPlayerStatements = async function(playerId) {
            const response = await fetch(`/api/player_statements/${playerId}`);
            return await response.json();
        };
    ''')
    
    ui.run_javascript('initCardHoverEffects();')
    await init_game(client)

# --- API Endpoints ---
@app.get('/api/player_statements/{player_id}')
def get_player_statements(player_id: int):
    """특정 플레이어의 발언 기록을 반환합니다."""
    statements = []
    for event in state.game_engine.history:
        if event.event_type in [EventType.CLAIM, EventType.VOTE] and event.actor_id == player_id:
            phase_name = event.phase.name.replace('_', ' ').title() if event.phase else 'Unknown'
            text = get_random_narrative(event)
            statements.append({
                'day': event.day,
                'phase': phase_name,
                'text': text,
                'event_type': event.event_type.name
            })
    return statements

# --- App Entrypoint ---
def run_app():
    ui.run(title='Mafia AI', storage_secret='a_very_secret_key_for_demo', reload=False)

if __name__ in {"__main__", "__mp_main__"}:
    run_app()

