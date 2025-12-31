# Mafia AI 리팩토링 완료 가이드

## 📋 개요

이 문서는 Mafia AI 프로젝트의 대규모 리팩토링 작업 결과를 요약합니다.

**리팩토링 목표:**
- 데이터와 서사의 분리
- 해석 로직의 중앙화
- 실시간 시각화 (TensorBoard)
- 코드 간소화 및 레거시 제거

---

## 🎯 주요 변경사항

### 1. LogManager 시스템 도입 (core/logger.py)

**핵심 기능:**
- **JSONL 로깅**: `GameEvent` 객체를 `.jsonl` 형식으로 기록
- **TensorBoard 통합**: 학습 메트릭 실시간 모니터링
- **내러티브 해석**: 이벤트를 자연어로 변환 (`interpret_event()`)

**사용 예시:**
```python
from core.logger import LogManager

# LogManager 초기화
logger = LogManager(experiment_name="ppo_mlp_train", log_dir="./logs")

# 게임 이벤트 로깅
event = GameEvent(day=1, phase=Phase.DAY_DISCUSSION, ...)
logger.log_event(event)

# 학습 메트릭 기록
logger.log_metrics(
    episode=100,
    total_reward=15.5,
    is_win=True,
    win_rate=0.65
)

# 내러티브 변환 (GUI/LLM용)
narrative = logger.interpret_event(event)
# -> "Day 1 | Player 3는 자신이 경찰라고 주장했습니다."

# 리소스 정리
logger.close()
```

---

### 2. LLM 에이전트 개선 (core/agent/llmAgent.py)

**변경사항:**
- `LogManager` 인스턴스를 생성자에서 주입받음
- `_create_conversation_log()` 메서드가 `LogManager.interpret_event()` 사용
- 신뢰도 행렬(`belief`)을 마크다운 테이블 형식으로 변환하여 프롬프트에 주입

**신뢰도 행렬 마크다운 예시:**
```markdown
| Player ID | 시민 | 경찰 | 의사 | 마피아 |
|---|---|---|---|---|
| Player 0 | 25.0 | 25.0 | 25.0 | 25.0 |
| Player 1 | 30.0 | 20.0 | 10.0 | 40.0 |
...
```

---

### 3. Runner 슬림화 (core/runner.py)

**제거된 기능:**
- `matplotlib` 기반 그래프 생성
- 로컬 통계 리스트 (`history_rewards`, `history_win_rates`)
- `utils.visualize.plot_results()` 호출

**추가된 기능:**
- `LogManager`를 파라미터로 받아 메트릭 기록
- 학습 완료 후 TensorBoard 접속 안내 메시지 출력

---

### 4. Main 간소화 (main.py)

**변경사항:**
- `run_simulation()` 함수에서 `LogManager` 생성 및 주입
- `utils.analysis.analyze_log_file()` 제거 (TensorBoard 대체)
- 실험 이름 기반 세션 디렉토리 자동 생성

---

### 5. GUI 리플레이 탭 재구성 (gui/tabs/replay.py)

**변경사항:**
- 텍스트 로그 파서 제거
- JSONL 파일 직접 읽기
- `LogManager.interpret_event()`를 통한 내러티브 생성
- 세션 디렉토리 브라우징 기능 추가

**사용법:**
1. GUI 실행 후 "Replay" 탭 선택
2. 왼쪽 목록에서 실험 세션 선택
3. 자동으로 JSONL 파일을 로드하여 이벤트 표시

---

### 6. 레거시 파일 제거

다음 파일들이 삭제되었습니다:
- ❌ `utils/analysis.py` (정규표현식 기반 로그 분석)
- ❌ `utils/log_parser.py` (텍스트 로그 파서)
- ❌ `utils/visualize.py` (matplotlib 그래프 생성)
- ❌ `gui/tabs/ai_stats.py` (자체 그래프 구현)

---

## 🚀 TensorBoard 사용법

### 1. 학습 실행
```bash
python main.py --mode train --agent ppo --episodes 1000
```

### 2. TensorBoard 실행
```bash
tensorboard --logdir=./logs
```

### 3. 브라우저에서 확인
브라우저를 열고 다음 주소로 접속:
```
http://localhost:6006
```

### 4. 주요 메트릭
- **Reward/Total**: 에피소드별 총 보상
- **Win/IsWin**: 승리 여부 (0 또는 1)
- **Win/Rate**: 최근 100 에피소드 승률

---

## 📊 데이터 흐름

```
게임 엔진 (MafiaGame/MafiaEnv)
    ↓
GameEvent 생성
    ↓
LogManager.log_event()
    ├─→ JSONL 파일 저장 (events.jsonl)
    └─→ TensorBoard 메트릭 기록

필요 시점:
    ├─→ GUI 리플레이: LogManager.interpret_event() 호출
    └─→ LLM 에이전트: LogManager.interpret_event() 호출
```

---

## 🔧 의존성 업데이트

`requirements.txt`가 간소화되었습니다. 새로운 의존성 설치:

```bash
pip install -r requirements.txt
```

**주요 의존성:**
- `torch` (딥러닝)
- `tensorboard` (시각화, **신규**)
- `pydantic` (데이터 검증)
- `PyYAML` (템플릿 로드)
- `PyQt6` (GUI)
- `openai` (LLM API)

---

## 📝 내러티브 템플릿 커스터마이징

`core/narrative_templates.yaml` 파일을 수정하여 이벤트 해석 방식을 변경할 수 있습니다:

```yaml
CLAIM_SELF: "Day {day} | Player {actor_id}는 자신이 {role_name}라고 주장했습니다."
VOTE: "Day {day} | Player {actor_id}가 Player {target_id}에게 투표했습니다."
# ... 더 많은 템플릿
```

---

## ⚠️ 주의사항

### 1. 기존 로그 파일과의 호환성
- 이전 `.txt` 형식 로그는 더 이상 지원되지 않습니다
- 새로운 시스템은 `.jsonl` 형식만 지원합니다

### 2. MafiaGame/MafiaEnv 통합 필요
- 현재 `MafiaGame` 클래스는 아직 `LogManager`를 사용하지 않습니다
- LLM 전용 모드를 실행하려면 추가 통합 작업이 필요합니다

### 3. GUI 업데이트
- `gui/tabs/ai_stats.py`가 제거되었으므로, 관련 임포트가 있는 파일은 수정이 필요할 수 있습니다

---

## 🎓 다음 단계

1. **MafiaGame 통합**: `MafiaGame` 클래스에 `LogManager` 통합
2. **에피소드 구분**: JSONL 파일에 에피소드 경계 마커 추가
3. **고급 필터링**: GUI에서 특정 이벤트 타입만 필터링하는 기능 추가
4. **통계 대시보드**: TensorBoard 외에 커스텀 통계 뷰 추가

---

## 📚 참고 자료

- [TensorBoard 공식 문서](https://www.tensorflow.org/tensorboard)
- [Pydantic 공식 문서](https://docs.pydantic.dev/)
- [JSONL 형식 설명](http://jsonlines.org/)

---

## 🙋 문의사항

리팩토링 관련 문의사항은 프로젝트 이슈 트래커에 남겨주세요.

---

**리팩토링 완료일**: 2025년 12월 31일
**주요 변경 파일**:
- ✅ `core/logger.py` (신규)
- ✅ `core/narrative_templates.yaml` (신규)
- ✅ `core/agent/llmAgent.py` (수정)
- ✅ `core/runner.py` (수정)
- ✅ `main.py` (수정)
- ✅ `gui/tabs/replay.py` (수정)
- ✅ `requirements.txt` (수정)
