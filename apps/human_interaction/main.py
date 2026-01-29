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
from core.engine.state import Role, Phase, EventType, GameEvent, GameAction
from core.managers.logger import LogManager
from apps.human_interaction.human_agent import HumanAgent

# 정적 파일 경로 설정
app.add_static_files('/static', STATIC_DIR)

# --- Application State ---
class AppState(BaseModel):
    game_engine: MafiaGame
    log_manager: LogManager
    day_phase_text: str = "Day 0 | WAITING"
    game_over: bool = False
    previous_day: int = 0
    previous_phase: Optional[Phase] = None
    next_button_text: str = "NEXT PHASE"
    ui_event_queue: Deque = Field(default_factory=deque)
    is_processing_events: bool = False
    pending_death_announcements: list = Field(default_factory=list)
    human_player_id: int = 0
    human_action_future: Optional[asyncio.Future] = None
    waiting_for_human: bool = False
    selected_target: int = -1  # UI에서 선택된 타겟
    selected_role: Optional[Role] = None  # UI에서 선택된 역할

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
    <div class="card-container card-container-{player_id}" style="position: absolute; pointer-events: auto;">
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
    # 각 플레이어의 역할 및 생존 상태 업데이트
    for player in state.game_engine.players:
        client.run_javascript(f"document.getElementById('player-role-{player.id}').innerText = '{player.role.name}';")
        client.run_javascript(f"document.getElementById('player-card-{player.id}').classList.{'add' if not player.alive else 'remove'}('dead');")
    
    # placeholder 이동 + 카드 애니메이션
    alive_players = [p.id for p in state.game_engine.players if p.alive]
    dead_players = [p.id for p in state.game_engine.players if not p.alive]
    
    # 현재 phase 확인
    current_phase = state.game_engine.phase if state.game_engine else None
    is_voting_phase = current_phase == Phase.DAY_VOTE
    
    client.run_javascript(f"""
        (function() {{
            const aliveDeck = document.getElementById('alive-deck');
            const votedArea = document.getElementById('voted-area');
            const deadArea = document.getElementById('dead-area');
            const aliveIds = {alive_players};
            const deadIds = {dead_players};
            const cardLayer = document.getElementById('card-layer');
            const isVotingPhase = {str(is_voting_phase).lower()};
            
            if (!cardLayer) {{
                console.error('Card layer not found!');
                return;
            }}
            
            // Step 1: placeholder들 이동
            aliveIds.forEach(id => {{
                const placeholder = document.getElementById('placeholder-' + id);
                // 투표 phase가 아니면 모든 alive 카드를 alive-deck으로
                // 투표 phase면 voted에 있는 애들은 그대로 두기
                if (placeholder) {{
                    if (!isVotingPhase || placeholder.parentElement !== votedArea) {{
                        if (aliveDeck && placeholder.parentElement !== aliveDeck) {{
                            aliveDeck.appendChild(placeholder);
                        }}
                    }}
                }}
            }});
            
            deadIds.forEach(id => {{
                const placeholder = document.getElementById('placeholder-' + id);
                if (placeholder && deadArea && placeholder.parentElement !== deadArea) {{
                    deadArea.appendChild(placeholder);
                }}
            }});
            
            // Grid 재배치가 완료되도록 두 번의 requestAnimationFrame 사용
            requestAnimationFrame(() => {{
                requestAnimationFrame(() => {{
                    // Step 2: 각 카드를 해당 placeholder 위치로 애니메이션
                    const layerRect = cardLayer.getBoundingClientRect();
            
            for (let i = 0; i < 8; i++) {{
                const placeholder = document.getElementById('placeholder-' + i);
                const card = document.querySelector('.card-container-' + i);
                
                if (placeholder && card) {{
                    const rect = placeholder.getBoundingClientRect();
                    const targetLeft = rect.left - layerRect.left;
                    const targetTop = rect.top - layerRect.top;
                    
                    card.style.left = targetLeft + 'px';
                    card.style.top = targetTop + 'px';
                }}
            }}
                }});
            }});
        }})()
    """)

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
        hold_duration_ms = 1500
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
    
    # Step A: AI 행동 계산
    async def get_single_action(player):
        player_view = state.game_engine.get_game_status(viewer_id=player.id)
        action = await asyncio.to_thread(player.get_action, player_view)
        return player.id, action
    
    # HumanAgent가 아닌 플레이어들의 행동만 계산
    ai_players = [p for p in living_players if not isinstance(p, HumanAgent)]
    action_tasks = [get_single_action(p) for p in ai_players]
    ai_action_results = await asyncio.gather(*action_tasks)
    actions = dict(ai_action_results)
    
    # Step B: 사람 행동 대기 (생존해 있고 GAME_START가 아닌 경우)
    human_player = state.game_engine.players[state.human_player_id]
    if human_player.alive and old_phase != Phase.GAME_START:
        # UI 입력 대기를 위한 Future 생성
        state.human_action_future = asyncio.Future()
        state.waiting_for_human = True
        
        # UI가 업데이트되도록 트리거 (버튼들이 보이도록)
        ui.update()
        
        # 사람의 행동을 기다림
        human_action = await state.human_action_future
        actions[state.human_player_id] = human_action
        state.waiting_for_human = False
    
    # Step C: 행동 통합 및 엔진 실행

    # 1. 현재 phase 시작 알림 먼저 표시 (phase가 바뀐 경우에만)
    if state.previous_phase != old_phase:
        if old_phase == Phase.NIGHT:
            state.ui_event_queue.append(('announcement', "밤이 되었습니다"))
        elif old_phase == Phase.DAY_DISCUSSION:
            state.ui_event_queue.append(('announcement', f"{state.game_engine.day}일차 낮이 밝았습니다"))
        state.previous_phase = old_phase

    # 2. 이전 턴의 사망 메시지를 그 다음에 표시
    for death_msg in state.pending_death_announcements:
        state.ui_event_queue.append(('announcement', death_msg))
    state.pending_death_announcements.clear()

    # 3. 현재 상태를 UI에 표시 (step 실행 전!)
    update_ui_for_game_state(client)

    # 4. step 실행 (다음 phase로 전환)
    _, is_over, is_win = await asyncio.to_thread(state.game_engine.step_phase, actions)
    new_phase = state.game_engine.phase
    
    # 5. 게임 이벤트 처리 - 사망은 pending에 저장, 발언은 DOM 순서대로 큐에 추가
    new_events = state.game_engine.history[history_len_before:]
    
    # 새 발언 이벤트를 임시로 저장 (actor_id -> text 매핑)
    new_narrative_events = {}
    
    for event in new_events:
        # log_to_console(client, event) # 임시 비활성화
        if event.event_type == EventType.EXECUTE:
            state.pending_death_announcements.append(f"투표로 {event.target_id}번 플레이어가 처형되었습니다. (직업: {event.value.name})")
        elif event.event_type == EventType.KILL:
            state.pending_death_announcements.append(f"지난 밤 {event.target_id}번 플레이어가 살해당했습니다.")
        elif event.event_type in [EventType.CLAIM, EventType.VOTE] and event.actor_id is not None:
            if event.event_type == EventType.VOTE and event.target_id is not None:
                client.run_javascript(f"shake_card({event.target_id});")
            # 발언을 임시로 저장
            text = get_random_narrative(event).replace('"', '\\"').replace("'", "\\'")
            new_narrative_events[event.actor_id] = text
    
    # DOM 순서대로 발언을 큐에 추가: alive-deck → voted-area → dead-area
    dom_order = await client.run_javascript("""
        (function() {
            const aliveDeck = document.getElementById('alive-deck');
            const votedArea = document.getElementById('voted-area');
            const deadArea = document.getElementById('dead-area');
            
            let order = [];
            
            // alive-deck의 placeholder 순서대로
            if (aliveDeck) {
                const aliveChildren = aliveDeck.querySelectorAll('.card-placeholder');
                aliveChildren.forEach(ph => {
                    const id = ph.id.replace('placeholder-', '');
                    order.push(parseInt(id));
                });
            }
            
            // voted-area의 placeholder 순서대로
            if (votedArea) {
                const votedChildren = votedArea.querySelectorAll('.card-placeholder');
                votedChildren.forEach(ph => {
                    const id = ph.id.replace('placeholder-', '');
                    order.push(parseInt(id));
                });
            }
            
            // dead-area의 placeholder 순서대로
            if (deadArea) {
                const deadChildren = deadArea.querySelectorAll('.card-placeholder');
                deadChildren.forEach(ph => {
                    const id = ph.id.replace('placeholder-', '');
                    order.push(parseInt(id));
                });
            }
            
            return order;
        })()
    """, timeout=5.0)
    
    if dom_order:
        for player_id in dom_order:
            if player_id in new_narrative_events:
                state.ui_event_queue.append(('narrative', player_id, new_narrative_events[player_id]))

    # --- UI 이벤트 처리 시작 ---
    if state.ui_event_queue and not state.is_processing_events:
        state.is_processing_events = True
        print("UI Event processor KICKED OFF.")
        process_ui_events(client)

    # 6. 게임 종료 알림
    if is_over:
        state.game_over = True
        winner = "CITIZEN" if is_win else "MAFIA"
        state.ui_event_queue.append(('announcement', f"{winner} 팀 승리!"))
        state.next_button_text = "PLAY AGAIN"
        if not state.is_processing_events:
            state.is_processing_events = True
            print("UI Event processor KICKED OFF for game over.")
            process_ui_events(client)
    
    # 7. execute 후 night으로 넘어가면 바로 다음 단계로 진행
    if new_phase == Phase.NIGHT:
        await step_phase_handler(client)


async def init_game(client: Client):
    """새 게임을 시작하고 UI를 초기화합니다."""
    print("새로운 게임을 시작합니다...")
    
    # 플레이어 초기화: Player 0은 Human, 나머지는 AI
    agents = []
    for i in range(8):
        if i == state.human_player_id:
            agents.append(HumanAgent(i, Role.CITIZEN))
        else:
            agents.append(RuleBaseAgent(i, Role.CITIZEN))
    
    state.game_engine = MafiaGame(agents=agents)
    await asyncio.to_thread(state.game_engine.reset)
    
    state.game_over = False
    state.previous_day = 0
    state.previous_phase = None
    state.next_button_text = "NEXT PHASE"
    state.ui_event_queue.clear()
    state.is_processing_events = False
    state.pending_death_announcements.clear()
    state.human_action_future = None
    state.waiting_for_human = False
    state.selected_target = -1
    state.selected_role = None
    
    # 버튼 딕셔너리 초기화
    target_buttons.clear()
    role_buttons.clear()

    client.run_javascript("set_theme('day')")
    
    # placeholder들을 alive-deck으로 초기화
    client.run_javascript("""
        const aliveDeck = document.getElementById('alive-deck');
        for (let i = 0; i < 8; i++) {
            const placeholder = document.getElementById('placeholder-' + i);
            if (placeholder && aliveDeck) {
                aliveDeck.appendChild(placeholder);
            }
        }
        
        const votedArea = document.getElementById('voted-area');
        const deadArea = document.getElementById('dead-area');
        if (votedArea) votedArea.innerHTML = '';
        if (deadArea) deadArea.innerHTML = '';
    """)
        
    update_ui_for_game_state(client)
    print("게임 초기화 완료.")

def on_human_action(target_id: int, claim_role: Optional[Role] = None):
    """사람 플레이어의 행동을 처리하는 콜백"""
    if state.human_action_future and not state.human_action_future.done():
        action = GameAction(target_id=target_id, claim_role=claim_role)
        state.human_action_future.set_result(action)
        # 선택 초기화
        state.selected_target = -1
        state.selected_role = None

def select_target(target_id: int):
    """타겟을 선택/해제하는 함수"""
    if state.selected_target == target_id:
        state.selected_target = -1  # 같은 버튼을 다시 누르면 선택 해제
    else:
        state.selected_target = target_id
    # UI 업데이트
    update_action_buttons()

def select_role(role: Optional[Role]):
    """역할을 선택/해제하는 함수"""
    if state.selected_role == role:
        state.selected_role = None  # 같은 버튼을 다시 누르면 선택 해제
    else:
        state.selected_role = role
    # UI 업데이트
    update_action_buttons()

def update_action_buttons():
    """행동 버튼들의 스타일을 업데이트"""
    # 타겟 버튼 스타일 업데이트
    for i in range(8):
        if i != state.human_player_id and i in target_buttons:
            btn = target_buttons[i]
            if state.selected_target == i:
                btn.style('background: rgba(100, 200, 100, 0.8); color: white; font-weight: bold; border-radius: 8px; font-family: "Inter", "Noto Sans KR", sans-serif; text-transform: none; transition: all 0.2s;')
            else:
                btn.style('background: rgba(26, 26, 26, 0.08); color: rgba(26, 26, 26, 0.85); font-weight: normal; border-radius: 8px; font-family: "Inter", "Noto Sans KR", sans-serif; text-transform: none; transition: all 0.2s;')
    
    # 역할 버튼 스타일 업데이트
    for role in [Role.POLICE, Role.DOCTOR, Role.MAFIA]:
        if role in role_buttons:
            btn = role_buttons[role]
            if state.selected_role == role:
                btn.style('background: rgba(100, 100, 255, 0.8); color: white; font-weight: bold; border-radius: 8px; font-family: "Inter", "Noto Sans KR", sans-serif; text-transform: none; transition: all 0.2s;')
            else:
                btn.style('background: rgba(100, 100, 255, 0.15); color: rgba(26, 26, 26, 0.85); font-weight: normal; border: 1px solid rgba(100, 100, 255, 0.3); border-radius: 8px; font-family: "Inter", "Noto Sans KR", sans-serif; text-transform: none; transition: all 0.2s;')

# 버튼 참조를 저장할 딕셔너리
target_buttons = {}
role_buttons = {}

def confirm_action():
    """선택된 타겟과 역할로 행동을 확정하는 함수"""
    on_human_action(state.selected_target, state.selected_role)

@ui.page('/')
async def main_page(client: Client):
    ui.html('<div id="background-div" class="background-div"></div>', sanitize=False)
    ui.add_head_html('<link rel="stylesheet" href="/static/styles.css">')
    ui.add_head_html('<script src="/static/scripts.js"></script>')

    with ui.row().classes('w-full h-16 items-center justify-between z-30').style('padding: 1rem 4rem; background: transparent;'):
        ui.label().bind_text_from(state, 'day_phase_text').classes('text-xl').style('color: rgba(26, 26, 26, 0.85); font-weight: 500; letter-spacing: 1px; font-family: "Inter", "Noto Sans KR", sans-serif;')
        next_button = ui.button(on_click=lambda: step_phase_handler(client)).props('flat').classes('px-8 py-2')
        next_button.bind_text_from(state, 'next_button_text')
        # 이벤트 처리 중이거나 사람 입력 대기 중일 때 버튼 비활성화
        next_button.bind_enabled_from(state, 'is_processing_events', 
            backward=lambda v: not v and not state.waiting_for_human)
        next_button.style('background: rgba(26, 26, 26, 0.08); border: 1px solid rgba(26, 26, 26, 0.15); border-radius: 8px; font-weight: 500; letter-spacing: 0.5px; font-family: "Inter", "Noto Sans KR", sans-serif; color: rgba(26, 26, 26, 0.85); transition: all 0.2s ease; text-transform: none;')
    
    # 사람 플레이어 행동 컨트롤 패널
    with ui.column().classes('w-full items-center z-30').style('padding: 1rem 2rem; background: rgba(255, 255, 255, 0.95); border-bottom: 2px solid rgba(26, 26, 26, 0.15); gap: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);') as human_control_panel:
        human_control_panel.bind_visibility_from(state, 'waiting_for_human')
        
        # 헤더
        with ui.row().classes('w-full justify-between items-center'):
            with ui.column().classes('gap-1'):
                ui.label('🎮 당신의 차례입니다! (Player 0)').classes('text-xl font-bold').style('color: rgba(26, 26, 26, 0.9); font-family: "Inter", "Noto Sans KR", sans-serif;')
                phase_label = ui.label().classes('text-sm font-medium').style('color: rgba(26, 26, 26, 0.6); font-family: "Inter", "Noto Sans KR", sans-serif;')
                phase_label.bind_text_from(state.game_engine, 'phase', 
                    backward=lambda p: f"현재 단계: {p.name.replace('_', ' ').title()}" if p else "")
            
            # 현재 역할 표시
            role_label = ui.label().classes('text-base font-semibold px-4 py-2').style('background: rgba(100, 100, 255, 0.15); border-radius: 8px; color: rgba(26, 26, 26, 0.85); font-family: "Inter", "Noto Sans KR", sans-serif;')
            role_label.bind_text_from(state.game_engine, 'players',
                backward=lambda players: f"내 역할: {next((p.role.name for p in players if p.id == state.human_player_id), 'Unknown')}")
        
        ui.separator().classes('w-full').style('background: rgba(26, 26, 26, 0.1);')
        
        # 타겟 선택
        with ui.column().classes('w-full gap-2'):
            ui.label('👤 타겟 선택:').classes('text-base font-semibold').style('color: rgba(26, 26, 26, 0.85); font-family: "Inter", "Noto Sans KR", sans-serif;')
            with ui.row().classes('gap-2 flex-wrap'):
                for i in range(8):
                    if i != state.human_player_id:
                        btn = ui.button(f'Player {i}', on_click=lambda pid=i: select_target(pid)).classes('px-5 py-3')
                        btn.style('background: rgba(26, 26, 26, 0.08); border-radius: 8px; font-family: "Inter", "Noto Sans KR", sans-serif; text-transform: none; transition: all 0.2s;')
                        # 생존 여부에 따라 버튼 활성화/비활성화
                        btn.bind_enabled_from(state.game_engine, 'players', 
                            backward=lambda players, pid=i: any(p.id == pid and p.alive for p in players))
                        # 버튼 참조 저장
                        target_buttons[i] = btn
        
        # 역할 주장
        with ui.column().classes('w-full gap-2'):
            ui.label('🎭 역할 주장 (선택사항):').classes('text-base font-semibold').style('color: rgba(26, 26, 26, 0.85); font-family: "Inter", "Noto Sans KR", sans-serif;')
            with ui.row().classes('gap-2'):
                for role in [Role.POLICE, Role.DOCTOR, Role.MAFIA]:
                    claim_btn = ui.button(f'{role.name}', on_click=lambda r=role: select_role(r)).classes('px-5 py-3')
                    claim_btn.style('background: rgba(100, 100, 255, 0.15); border: 1px solid rgba(100, 100, 255, 0.3); border-radius: 8px; font-family: "Inter", "Noto Sans KR", sans-serif; text-transform: none; transition: all 0.2s;')
                    # 버튼 참조 저장
                    role_buttons[role] = claim_btn
        
        ui.separator().classes('w-full').style('background: rgba(26, 26, 26, 0.1);')
        
        # 행동 확정 버튼들
        with ui.row().classes('gap-3 justify-center w-full'):
            confirm_btn = ui.button('✅ 행동 확정', on_click=confirm_action).classes('px-8 py-3')
            confirm_btn.style('background: rgba(50, 150, 50, 0.9); color: white; border-radius: 8px; font-weight: 600; font-family: "Inter", "Noto Sans KR", sans-serif; text-transform: none; font-size: 1.1em;')
            
            pass_btn = ui.button('⏭️ 기권/패스', on_click=lambda: on_human_action(-1)).classes('px-8 py-3')
            pass_btn.style('background: rgba(200, 200, 200, 0.5); color: rgba(26, 26, 26, 0.85); border-radius: 8px; font-weight: 500; font-family: "Inter", "Noto Sans KR", sans-serif; text-transform: none; font-size: 1.1em;')
        
        # 선택 상태 표시
        selection_info = ui.label().classes('text-sm text-center').style('color: rgba(26, 26, 26, 0.6); font-family: "Inter", "Noto Sans KR", sans-serif; font-style: italic;')
        selection_info.bind_text_from(state, 'selected_target',
            backward=lambda t: f"선택됨: 타겟 Player {t}" + (f", 역할 주장: {state.selected_role.name}" if state.selected_role else "") if t != -1 else 
                              (f"선택됨: 역할 주장만 ({state.selected_role.name})" if state.selected_role else "타겟 또는 역할을 선택하고 확정 버튼을 누르세요"))

    with ui.element('div').classes('player-area w-full'):
        # 영역 컨테이너 (보이는 레이아웃)
        with ui.element('div').props('id="area-container"').style('display: flex; width: 100%; height: 100%; justify-content: space-between;'):
            # 왼쪽: 살아있는 플레이어들
            with ui.element('div').props('id="alive-deck"').style('flex: 0 0 60%;'):
                for i in range(8):
                    ui.element('div').props(f'id="placeholder-{i}" class="card-placeholder"')
            
            # 오른쪽: 투표/죽은 플레이어
            with ui.element('div').props('id="right-side"').style('flex: 0 0 37.5%; display: flex; flex-direction: column; gap: 2rem;'):
                ui.element('div').props('id="voted-area"').style('flex: 1;')
                ui.element('div').props('id="dead-area"').style('flex: 1;')
        
        # 카드 레이어 (absolute overlay)
        with ui.element('div').props('id="card-layer"').style('position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;'):
            for i in range(8):
                ui.html(content=create_card_html(i), sanitize=False)
    
    # Container for dynamically added announcement banners
    ui.element('div').props('id="banner-container"')

    await client.connected()
    
    # 디버깅 및 초기 위치 설정
    client.run_javascript("""
        console.log('Initializing card positions...');
        
        const cardLayer = document.getElementById('card-layer');
        const aliveDeck = document.getElementById('alive-deck');
        
        if (!cardLayer || !aliveDeck) {
            console.error('Required elements not found!', {cardLayer, aliveDeck});
            return;
        }
        
        const layerRect = cardLayer.getBoundingClientRect();
        console.log('Layer rect:', layerRect);
        
        // 각 카드를 placeholder 위치로 설정
        for (let i = 0; i < 8; i++) {
            const placeholder = document.getElementById('placeholder-' + i);
            const card = document.querySelector('.card-container-' + i);
            
            if (placeholder && card) {
                const rect = placeholder.getBoundingClientRect();
                const left = rect.left - layerRect.left;
                const top = rect.top - layerRect.top;
                
                console.log(`Card ${i} positioned at: left=${left}, top=${top}`);
                card.style.left = left + 'px';
                card.style.top = top + 'px';
            } else {
                console.warn(`Card ${i} or placeholder not found!`, {card, placeholder});
            }
        }
    """)
    
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

