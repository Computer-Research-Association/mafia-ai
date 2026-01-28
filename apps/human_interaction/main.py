"""
마피아 게임 Human-AI Interaction을 위한 NiceGUI 웹 애플리케이션의 메인 파일.
(수정: 내러티브 말풍선, 밤/낮 전환, 페이즈 연출 적용)
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
from core.engine.state import GameStatus, Role, Phase, EventType
from core.managers.logger import LogManager

# --- [스타일링] 카드 및 연출 효과를 위한 CSS 및 JavaScript ---

ENHANCED_CSS = """
:root { --card-height: 180px; --card-width: 130px; }
.card-container { perspective: 1000px; position: relative; }
.card {
    width: var(--card-width); height: var(--card-height);
    background: #1a1a1a; color: white; border-radius: 10px; border: 2px solid #555;
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    font-family: 'Segoe UI', sans-serif; transition: transform 0.1s ease-out;
    position: relative; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}
.card-content { z-index: 2; text-shadow: 0 0 5px black; }
.card::after {
    content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: linear-gradient(110deg, rgba(255,255,255,0) 40%, rgba(255,255,255,0.3) 50%, rgba(255,255,255,0) 60%);
    animation: shine 7s infinite linear; z-index: 1; opacity: 0.3;
}
@keyframes shine { 0% { transform: translateX(-60%) translateY(-10%) rotate(20deg); } 100% { transform: translateX(60%) translateY(10%) rotate(20deg); } }
.card.dead { filter: grayscale(100%); }
.card.dead::before {
    content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: linear-gradient(to top right, rgba(255,255,255,0) 0%, rgba(255,255,255,0) 49.8%, rgba(255,255,255,0.4) 50%, rgba(255,255,255,0) 50.2%),
                linear-gradient(to top left, rgba(255,255,255,0) 0%, rgba(255,255,255,0) 49.8%, rgba(255,255,255,0.4) 50%, rgba(255,255,255,0) 50.2%);
    background-size: 100% 100%; transform: rotate(15deg) scale(1.2); z-index: 3; opacity: 0.6;
}
.speech-bubble {
    position: absolute; bottom: 80%; left: 90%;
    min-width: 150px; max-width: 250px;
    background-color: #fff; color: #000; border-radius: 8px; padding: 10px;
    font-size: 0.9em; box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    opacity: 0; visibility: hidden; transition: opacity 0.3s, visibility 0.3s;
    z-index: 10; white-space: pre-wrap;
}
.speech-bubble.visible { opacity: 1; visibility: visible; }
.speech-bubble::after {
    content: ''; position: absolute;
    bottom: -10px; left: 20px;
    border-width: 10px 10px 0; border-style: solid;
    border-color: #fff transparent transparent transparent;
}
.night-overlay {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background-color: rgba(6, 0, 15, 0.7); z-index: 20;
    display: none; opacity: 0; transition: opacity 1.5s;
}
.night-overlay.visible { display: block; opacity: 1; }
.fog {
    position: absolute; width: 200vw; height: 100vh;
    background: url(https://i.imgur.com/74gbhS9.png) repeat-x;
    background-size: contain; background-position: center;
    animation: fog-move 60s linear infinite;
}
.fog.fog-1 { opacity: 0.1; animation-duration: 120s; }
.fog.fog-2 { opacity: 0.1; animation-duration: 80s; }
@keyframes fog-move { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
.day-announcement {
    position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-size: 10em; color: white; font-weight: bold; text-shadow: 0 0 20px black;
    z-index: 30; opacity: 0; pointer-events: none;
}
.day-announcement.animate { animation: zoom-fade 2.5s ease-out; }
@keyframes zoom-fade {
    0% { transform: translate(-50%, -50%) scale(0.5); opacity: 0; }
    50% { transform: translate(-50%, -50%) scale(1.1); opacity: 1; }
    100% { transform: translate(-50%, -50%) scale(1.5); opacity: 0; }
}
"""

ENHANCED_JS = """
function type_text(elementId, text, duration = 5000) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const bubble = element.parentElement;
    bubble.classList.add('visible');
    
    let i = 0;
    element.innerHTML = '';
    const typing = setInterval(() => {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
        } else {
            clearInterval(typing);
            setTimeout(() => {
                bubble.classList.remove('visible');
            }, duration);
        }
    }, 30);
}

const cards = document.querySelectorAll('.card');
cards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        const rotateX = (y - centerY) / 10;
        const rotateY = (centerX - x) / 10;
        card.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
    });
    card.addEventListener('mouseleave', () => {
        card.style.transform = 'rotateX(0) rotateY(0)';
    });
});
"""

# --- Application State ---
class AppState:
    def __init__(self):
        agents = [RuleBaseAgent(i, Role.CITIZEN) for i in range(8)]
        self.game_engine = MafiaGame(agents=agents)
        self.log_manager = LogManager(experiment_name="narrative_generator", write_mode=False)
app.state = AppState()

# --- Browser Console Logging & JS Interaction ---
def log_to_console(client: Client, data: BaseModel):
    try:
        json_data = json.dumps(data.model_dump(mode='json'))
        client.run_javascript(f'console.log(JSON.parse(String.raw`{json_data}`))')
    except Exception as e:
        print(f"콘솔 로깅 실패: {e}")
        client.run_javascript(f'console.error("Failed to log data: {e}");')

# --- UI Components ---
def create_header():
    with ui.header().classes('bg-primary text-white p-4'):
        ui.label('Mafia AI: Human Interaction').classes('text-2xl font-bold')

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

def update_player_cards(client: Client):
    game_engine: MafiaGame = app.state.game_engine
    for player in game_engine.players:
        client.run_javascript(f"document.getElementById('player-role-{player.id}').innerText = '{player.role.name}';")
        if not player.alive:
            client.run_javascript(f"document.getElementById('player-card-{player.id}').classList.add('dead');")
        else:
            client.run_javascript(f"document.getElementById('player-card-{player.id}').classList.remove('dead');")

# --- Game Loop ---
async def run_game_loop(client: Client):
    game_engine: MafiaGame = app.state.game_engine
    log_manager: LogManager = app.state.log_manager
    
    print("새로운 게임을 시작합니다...")
    await asyncio.to_thread(game_engine.reset)
    update_player_cards(client)

    previous_day, previous_phase = 0, None
    is_over = False
    while not is_over:
        await asyncio.sleep(2)
        
        # Day/Night Transition & Announcement
        if game_engine.day > previous_day:
            client.run_javascript(f"document.getElementById('day-announcement-text').innerText = 'Day {game_engine.day}';")
            client.run_javascript("const el = document.getElementById('day-announcement'); el.classList.remove('animate'); void el.offsetWidth; el.classList.add('animate');")
            previous_day = game_engine.day
        
        if game_engine.phase != previous_phase:
            if game_engine.phase == Phase.NIGHT:
                client.run_javascript("document.getElementById('night-overlay').classList.add('visible');")
            else:
                client.run_javascript("document.getElementById('night-overlay').classList.remove('visible');")
            previous_phase = game_engine.phase
        
        living_players = [p for p in game_engine.players if p.alive]
        history_len_before = len(game_engine.history)

        async def get_single_action(player):
            player_view = game_engine.get_game_status(viewer_id=player.id)
            log_to_console(client, player_view)
            action = await asyncio.to_thread(player.get_action, player_view)
            return player.id, action
        
        action_tasks = [get_single_action(p) for p in living_players]
        action_results = await asyncio.gather(*action_tasks)
        actions = dict(action_results)
        
        _, is_over, is_win = await asyncio.to_thread(game_engine.step_phase, actions)
        
        new_events = game_engine.history[history_len_before:]
        for event in new_events:
            log_to_console(client, event)
            if event.event_type == EventType.CLAIM:
                text = log_manager.interpret_event(event).replace('"', '\\"').replace("'", "\\'")
                if event.actor_id is not None:
                    client.run_javascript(f"type_text('player-bubble-text-{event.actor_id}', '{text}');")

        update_player_cards(client)
    
    winner = "시민" if is_win else "마피아"
    print(f"게임 종료! {winner} 팀의 승리입니다!")
    log_to_console(client, game_engine.get_game_status())

@ui.page('/')
async def main_page(client: Client):
    ui.add_head_html(f'<style>{ENHANCED_CSS}</style>')
    create_header()

    with ui.column().classes('w-full max-w-4xl mx-auto p-8 items-center'):
        with ui.grid(columns=4).classes('w-full gap-8'):
            for i in range(8):
                ui.html(content=create_card_html(i), sanitize=False)

    # Hidden overlays for effects
    with ui.element('div').classes('night-overlay').props('id="night-overlay"'):
        ui.element('div').classes('fog fog-1')
        ui.element('div').classes('fog fog-2')
    with ui.element('div').classes('day-announcement').props('id="day-announcement"') as announcement:
        ui.label().props('id="day-announcement-text"')

    await client.connected()
    ui.run_javascript(ENHANCED_JS)
    asyncio.create_task(run_game_loop(client))

# --- App Entrypoint ---
def run_app():
    ui.run(title='Mafia AI', storage_secret='a_very_secret_key_for_demo', reload=False)

if __name__ in {"__main__", "__mp_main__"}:
    run_app()

