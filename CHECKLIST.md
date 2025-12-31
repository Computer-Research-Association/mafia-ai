# 리팩토링 체크리스트

## ✅ 완료된 작업

### 1. 핵심 인프라
- [x] `core/logger.py` - LogManager 클래스 구현
  - JSONL 로깅
  - TensorBoard 통합
  - 내러티브 해석 (`interpret_event()`)
- [x] `core/narrative_templates.yaml` - 이벤트 해석 템플릿

### 2. 에이전트 리팩토링
- [x] `core/agent/llmAgent.py`
  - LogManager 의존성 주입
  - `_create_conversation_log()` 개선
  - 신뢰도 행렬 마크다운 변환 (`_belief_to_markdown()`)

### 3. 학습 루프 슬림화
- [x] `core/runner.py`
  - matplotlib 코드 제거
  - LogManager 통합
  - TensorBoard 메트릭 기록

### 4. 메인 간소화
- [x] `main.py`
  - LogManager 초기화 및 주입
  - 레거시 분석 코드 제거

### 5. GUI 개선
- [x] `gui/tabs/replay.py`
  - JSONL 파일 읽기
  - LogManager 해석 로직 사용
  - 세션 브라우징 기능

### 6. 정리 작업
- [x] 레거시 파일 제거
  - `utils/analysis.py`
  - `utils/log_parser.py`
  - `utils/visualize.py`
  - `gui/tabs/ai_stats.py`
- [x] `requirements.txt` 간소화 및 TensorBoard 추가
- [x] 문서화 (`REFACTORING_GUIDE.md`)

---

## ⚠️ 추가 작업 필요

### 1. MafiaGame 통합 (높은 우선순위)
현재 `core/game.py`의 `MafiaGame` 클래스는 여전히 텍스트 로그를 사용합니다.

**필요한 작업:**
```python
# core/game.py
class MafiaGame:
    def __init__(self, logger: LogManager):
        self.logger = logger
        # ...
    
    def process_turn(self):
        # ...
        event = GameEvent(...)
        self.logger.log_event(event)
        # ...
```

### 2. MafiaEnv 통합 (높은 우선순위)
`core/env.py`의 `MafiaEnv` 역시 LogManager 통합이 필요합니다.

### 3. 에피소드 구분 (중간 우선순위)
JSONL 파일에 에피소드 경계를 명시하는 이벤트 추가:
```json
{"day": 0, "phase": 0, "event_type": 99, "actor_id": -1, "episode": 1, "special": "EPISODE_START"}
{"day": 5, "phase": 3, "event_type": 99, "actor_id": -1, "episode": 1, "special": "EPISODE_END"}
```

### 4. GUI 탭 정리 (낮은 우선순위)
`gui/tabs/ai_stats.py`를 사용하는 코드가 있다면 제거 필요.

---

## 🎯 빠른 시작 가이드

### 설치
```bash
pip install -r requirements.txt
```

### RL 학습 실행
```bash
python main.py --mode train --agent ppo --episodes 1000
```

### TensorBoard 실행
```bash
tensorboard --logdir=./logs
```

### GUI 실행
```bash
python main.py
```

---

## 📁 파일 구조 (변경 후)

```
mafia-ai/
├── core/
│   ├── logger.py                    ✨ 신규
│   ├── narrative_templates.yaml     ✨ 신규
│   ├── agent/
│   │   └── llmAgent.py              🔄 수정
│   ├── runner.py                    🔄 수정
│   ├── game.py                      ⚠️ 통합 필요
│   └── env.py                       ⚠️ 통합 필요
├── gui/
│   └── tabs/
│       ├── replay.py                🔄 수정
│       └── ai_stats.py              ❌ 삭제됨
├── utils/
│   ├── analysis.py                  ❌ 삭제됨
│   ├── log_parser.py                ❌ 삭제됨
│   └── visualize.py                 ❌ 삭제됨
├── main.py                          🔄 수정
├── requirements.txt                 🔄 수정
├── REFACTORING_GUIDE.md            ✨ 신규
└── CHECKLIST.md                    ✨ 신규 (이 파일)
```

---

## 📊 메트릭 모니터링

TensorBoard에서 확인 가능한 메트릭:
- `Reward/Total` - 에피소드별 총 보상
- `Win/IsWin` - 승리 여부
- `Win/Rate` - 최근 100 에피소드 승률
- `Metrics/*` - 커스텀 메트릭

---

**마지막 업데이트**: 2025년 12월 31일
