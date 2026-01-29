# Human Interaction - 마피아 게임 사람 플레이 기능

## 개요
NiceGUI로 구현된 마피아 게임 시뮬레이터에 사람이 직접 플레이할 수 있는 기능이 추가되었습니다.

## 주요 기능

### 1. HumanAgent 클래스
- 파일: `apps/human_interaction/human_agent.py`
- `BaseAgent`를 상속받아 사람 플레이어를 표현
- `get_action` 메서드는 더미 값을 반환 (실제 행동은 UI에서 주입)

### 2. UI 기반 행동 입력
- **Player 0**이 Human Player로 설정됨
- 사람의 차례가 되면 행동 컨트롤 패널이 자동으로 표시됨
- 다음 기능 제공:
  - 타겟 선택 (생존한 다른 플레이어)
  - 역할 주장 (POLICE, DOCTOR, MAFIA)
  - 기권/패스
  - 행동 확정

### 3. 비동기 입력 대기
- `asyncio.Future`를 사용하여 사람의 입력을 대기
- AI 플레이어들의 행동은 먼저 계산하고, 사람의 입력만 별도로 대기
- UI가 차단되지 않고 자연스럽게 동작

## 사용 방법

### 게임 시작
```bash
python apps/human_interaction/main.py
```

### 게임 플레이
1. 브라우저에서 게임 화면이 열립니다
2. **NEXT PHASE** 버튼을 클릭하여 게임을 진행합니다
3. Player 0 (Human)의 차례가 되면 행동 컨트롤 패널이 나타납니다
4. 원하는 행동을 선택하고 **행동 확정** 버튼을 클릭합니다
   - 타겟만 선택: 투표/공격/조사/치료 등의 타겟 지정 행동
   - 역할만 주장: 자신의 역할 주장
   - 타겟 + 역할: 다른 플레이어의 역할 추정 (예: "Player 3는 마피아다")
   - 기권/패스: 아무 행동도 하지 않음
5. AI 플레이어들의 행동이 자동으로 진행됩니다

## 구현 세부사항

### AppState 변경사항
```python
class AppState(BaseModel):
    # ... 기존 필드들
    human_player_id: int = 0  # 사람 플레이어 ID
    human_action_future: Optional[asyncio.Future] = None  # 사람 행동 대기용 Future
    waiting_for_human: bool = False  # 사람 입력 대기 중 플래그
    selected_target: int = -1  # UI에서 선택된 타겟
    selected_role: Optional[Role] = None  # UI에서 선택된 역할
```

### init_game 수정
- Player 0을 `HumanAgent`로 초기화
- 나머지 7명은 `RuleBaseAgent`로 초기화
- 게임 초기화 시 사람 플레이어 관련 상태 초기화

### step_phase_handler 수정
```python
# Step A: AI 행동 계산
ai_players = [p for p in living_players if not isinstance(p, HumanAgent)]
# ... AI 행동만 비동기로 수집

# Step B: 사람 행동 대기
if human_player.alive and old_phase != Phase.GAME_START:
    state.human_action_future = asyncio.Future()
    state.waiting_for_human = True
    human_action = await state.human_action_future  # 입력 대기
    actions[state.human_player_id] = human_action

# Step C: 행동 통합 및 엔진 실행
state.game_engine.step_phase(actions)
```

## UI 컴포넌트

### 행동 컨트롤 패널
- 위치: 페이지 상단 (day/phase 라벨과 게임 영역 사이)
- 표시 조건: `state.waiting_for_human == True`
- 구성 요소:
  - 헤더: 플레이어 정보 및 현재 역할 표시
  - 타겟 선택 버튼들 (생존한 플레이어만 활성화)
  - 역할 주장 버튼들
  - 행동 확정/기권 버튼
  - 현재 선택 상태 표시

### 스타일링
- 반투명 배경 (`rgba(255, 255, 255, 0.95)`)
- 선택된 버튼은 강조 표시 (녹색 배경)
- Inter 및 Noto Sans KR 폰트 사용
- 모던하고 깔끔한 디자인

## 주의사항

1. **core 로직 미수정**: 모든 변경사항은 `apps/human_interaction` 폴더 내에서만 이루어짐
2. **Player 0 고정**: 현재는 항상 Player 0이 Human Player로 설정됨
3. **단일 Human Player**: 현재는 1명의 Human Player만 지원
4. **Phase 제한 없음**: 모든 Phase에서 입력 가능 (GAME_START 제외)

## 향후 개선 가능 사항

- [ ] Human Player ID를 UI에서 선택 가능하게 변경
- [ ] 여러 명의 Human Player 동시 지원
- [ ] Phase별로 허용되는 행동 제한
- [ ] 행동 히스토리 표시
- [ ] 튜토리얼 모드 추가
- [ ] 모바일 반응형 UI 개선

## 파일 구조
```
apps/human_interaction/
├── human_agent.py          # HumanAgent 클래스 정의
├── main.py                 # 메인 애플리케이션 (수정됨)
├── static/
│   ├── styles.css         # CSS 스타일
│   └── scripts.js         # JavaScript 유틸리티
└── README_HUMAN_PLAY.md   # 이 문서
```
