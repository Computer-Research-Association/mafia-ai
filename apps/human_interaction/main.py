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
    action_step: int = 0  # 0: 플레이어 선택, 1: 역할 선택
    current_instruction: str = "플레이어를 선택하세요"

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
    
    # Step B: 사람 행동 대기 (생존해 있고 특정 Phase에서만)
    human_player = state.game_engine.players[state.human_player_id]
    
    # Check if human player is alive AND current phase requires human action
    requires_human_action = False
    if human_player.alive:
        if old_phase in [Phase.DAY_DISCUSSION, Phase.DAY_VOTE, Phase.DAY_EXECUTE]:
            requires_human_action = True
        elif old_phase == Phase.NIGHT and human_player.role in [Role.MAFIA, Role.POLICE, Role.DOCTOR]:
            requires_human_action = True

    if requires_human_action:
        state.human_action_future = asyncio.Future()
        
        # 모달 UI를 업데이트하고 표시하여 사용자 행동을 받음
        update_modal_ui()
        client.run_javascript('document.getElementById("human-control-modal").classList.add("visible");')
        state.waiting_for_human = True
        
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
    state.action_step = 0
    state.current_instruction = "플레이어를 선택하세요"

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

def on_execution_vote(agree: bool, target_id: Optional[int]):
    """처형 찬반 투표를 처리하는 콜백"""
    ui.run_javascript('document.getElementById("human-control-modal").classList.remove("visible");')
    if state.human_action_future and not state.human_action_future.done():
        # 찬성하면 후보의 ID를, 반대하면 -1 (기권/반대)를 전송
        action = GameAction(target_id=target_id if agree else -1)
        state.human_action_future.set_result(action)

def on_player_or_role_action(target_id: int, claim_role: Optional[Role] = None):
    """플레이어 선택 또는 역할 주장을 처리하는 콜백"""
    ui.run_javascript('document.getElementById("human-control-modal").classList.remove("visible");')
    if state.human_action_future and not state.human_action_future.done():
        action = GameAction(target_id=target_id, claim_role=claim_role)
        state.human_action_future.set_result(action)
        state.selected_target = -1
        state.selected_role = None
        state.action_step = 0
        state.current_instruction = "플레이어를 선택하세요."

def select_player_action(player_id: int):
    """플레이어를 선택하는 함수 (Phase별로 다르게 처리)"""
    current_phase = state.game_engine.phase
    if current_phase == Phase.DAY_DISCUSSION:
        state.selected_target = player_id
        state.action_step = 1
        state.current_instruction = f"Player {player_id}에 대해 역할을 주장하거나, 주장 없이 의견만 피력합니다."
        update_modal_ui()
    else:
        on_player_or_role_action(player_id, None)

def select_role_action(role: Optional[Role]):
    """역할을 선택하고 행동 확정"""
    on_player_or_role_action(state.selected_target, role)

def go_back():
    """이전 단계로 돌아가기"""
    state.action_step = 0
    state.selected_target = -1
    state.selected_role = None
    state.current_instruction = "플레이어를 선택하세요."
    update_modal_ui()

def update_modal_ui():
    """모달 UI를 현재 게임 상태에 맞게 업데이트합니다."""
    if modal_container:
        modal_container.clear()
        with modal_container:
            current_phase = state.game_engine.phase
            if current_phase == Phase.DAY_EXECUTE:
                render_execution_vote_selection()
            elif state.action_step == 0:
                render_player_selection()
            else:
                render_role_selection()

def render_execution_vote_selection():
    """처형 찬반 투표 UI 렌더링"""
    # HACK: game.py를 수정할 수 없어, UI에서 직접 처형 후보자를 계산합니다.
    # _last_votes는 private 멤버이므로 직접 접근은 권장되지 않습니다.
    votes = state.game_engine._last_votes
    target_id = None
    if votes:
        max_v = max(votes)
        if max_v > 0:
            targets = [i for i, v in enumerate(votes) if v == max_v]
            if len(targets) == 1:
                target_id = targets[0]

    if target_id is not None:
        state.current_instruction = f"플레이어 {target_id}를 처형하시겠습니까?"
    
        with ui.row().classes('w-full gap-4 justify-center items-center'):
            with ui.card().classes('bg-red-800 w-48 h-48 cursor-pointer hover:bg-red-700').on('click', lambda t=target_id: on_execution_vote(True, t)):
                with ui.column().classes('w-full h-full justify-center items-center'):
                    ui.label('찬성').classes('text-white text-4xl font-bold')
            
            with ui.card().classes('bg-blue-800 w-48 h-48 cursor-pointer hover:bg-blue-700').on('click', lambda t=target_id: on_execution_vote(False, t)):
                 with ui.column().classes('w-full h-full justify-center items-center'):
                    ui.label('반대').classes('text-white text-4xl font-bold')
    else:
        # 처형 대상이 없는 경우 (동점자 발생 등)
        state.current_instruction = "처형 대상이 결정되지 않았습니다. 다음 단계로 넘어갑니다."
        # 자동으로 다음 단계로 넘어가는 로직을 위해 future를 즉시 완료시킵니다.
        if state.human_action_future and not state.human_action_future.done():
            # 모달을 먼저 닫고 future를 완료합니다.
            ui.run_javascript('document.getElementById("human-control-modal").classList.remove("visible");')
            # action은 중요하지 않으므로 빈 action을 보냅니다.
            state.human_action_future.set_result(GameAction())

def render_player_selection():
    """플레이어 선택 UI 렌더링 (카드 기반)"""
    current_phase = state.game_engine.phase
    human_player = state.game_engine.players[state.human_player_id]
    
    phase_msg = ""
    if current_phase == Phase.DAY_DISCUSSION:
        phase_msg = "낮 토론: 다른 플레이어(또는 자신)를 지목하고 역할을 주장하세요."
    elif current_phase == Phase.DAY_VOTE:
        phase_msg = "투표: 처형할 플레이어를 선택하세요."
    elif current_phase == Phase.NIGHT:
        if human_player.role == Role.MAFIA:
            phase_msg = f"밤: 제거할 플레이어를 선택하세요. (당신은 {human_player.role.name})"
        elif human_player.role == Role.POLICE:
            phase_msg = f"밤: 조사할 플레이어를 선택하세요. (당신은 {human_player.role.name})"
        elif human_player.role == Role.DOCTOR:
            phase_msg = f"밤: 보호할 플레이어를 선택하세요. (당신은 {human_player.role.name})"

    ui.label(phase_msg).classes('text-sm text-center mb-2 text-gray-400')
    
    with ui.element('div').classes('action-button-grid grid-cols-4'):
        for player in state.game_engine.players:
            is_alive = player.alive
            is_self = player.id == state.human_player_id
            
            card_classes = 'action-player-card'
            if is_self:
                card_classes += ' is-self'
            if not is_alive:
                card_classes += ' is-dead'

            with ui.card().classes(card_classes).on('click', lambda pid=player.id: select_player_action(pid)) as card:
                if not is_alive:
                    card.props('disable')

                ui.label(f'Player {player.id}').classes('action-player-card-title')
                status = "나" if is_self else ("사망" if not is_alive else "생존")
                ui.label(status).classes('action-player-card-status')

    ui.button('기권/패스', on_click=lambda: on_player_or_role_action(-1, None)).classes('w-full mt-4 utility-button back')

def render_role_selection():
    """역할 선택 UI 렌더링"""
    ui.label(f'Player {state.selected_target}를 선택했습니다').classes('text-sm text-center mb-2 text-gray-400')
    
    with ui.element('div').classes('action-button-grid grid-cols-4'):
        for role in [Role.POLICE, Role.DOCTOR, Role.MAFIA, Role.CITIZEN]:
            btn = ui.button(
                f'{role.name}',
                on_click=lambda r=role: select_role_action(r)
            ).classes('action-button role-claim-button')

    with ui.element('div').classes('utility-button-grid'):
        ui.button('역할 주장 안 함', on_click=lambda: select_role_action(None)).classes('utility-button confirm')
        ui.button('← 뒤로', on_click=go_back).classes('utility-button back')

# 모달 컨테이너 참조
modal_container = None

@ui.page('/')
async def main_page(client: Client):
    ui.html('<div id="background-div" class="background-div"></div>', sanitize=False)
    ui.add_head_html('<link rel="stylesheet" href="/static/styles.css">')
    ui.add_head_html('<script src="/static/scripts.js"></script>')

    with ui.row().classes('w-full h-16 items-center justify-between z-30').style('padding: 1rem 4rem; background: transparent;'):
        ui.label().bind_text_from(state, 'day_phase_text').classes('text-xl').style('color: rgba(26, 26, 26, 0.85); font-weight: 500; letter-spacing: 1px; font-family: "Inter", "Noto Sans KR", sans-serif;')
        next_button = ui.button(on_click=lambda: step_phase_handler(client)).props('flat').classes('px-8 py-2')
        next_button.bind_text_from(state, 'next_button_text')
        next_button.bind_enabled_from(state, 'is_processing_events', 
            backward=lambda v: not v and not state.waiting_for_human)
        next_button.style('background: rgba(26, 26, 26, 0.08); border: 1px solid rgba(26, 26, 26, 0.15); border-radius: 8px; font-weight: 500; letter-spacing: 0.5px; font-family: "Inter", "Noto Sans KR", sans-serif; color: rgba(26, 26, 26, 0.85); transition: all 0.2s ease; text-transform: none;')
    
    with ui.element('div').props('id="human-control-modal"').classes('fixed inset-0 flex items-center justify-center z-50') as human_control_modal:
        with ui.column().classes('items-center gap-4 w-full max-w-2xl'):
            ui.label('당신의 차례입니다!').classes('action-modal-header text-3xl font-bold text-center mb-2')
            
            role_label = ui.label().classes('action-modal-role text-lg text-center mb-4')
            role_label.bind_text_from(state.game_engine, 'players',
                backward=lambda players: f"당신의 역할: {next((p.role.name for p in players if p.id == state.human_player_id), 'Unknown')}")
            
            ui.separator().style('background: rgba(255, 255, 255, 0.2); margin: 1rem 0;')
            
            instruction_label = ui.label().classes('action-modal-instruction text-xl font-semibold text-center mb-6')
            instruction_label.bind_text_from(state, 'current_instruction')
            
            global modal_container
            with ui.column().classes('w-full modal-content-wrapper').props('id="modal-content-wrapper"') as modal_container:
                render_player_selection()

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
    ui.run(title='Mafia AI', storage_secret='a_very_secret_key_for_demo', reload=True)

if __name__ in {"__main__", "__mp_main__"}:
    run_app()

