# Core Modules

마피아 게임 AI 프로젝트의 핵심 로직이 구현된 디렉토리입니다. 게임 엔진, 에이전트 설계, 강화학습 환경 래퍼, 그리고 실험 관리자를 포함하고 있습니다.

## 디렉토리 구조 및 설명

### 1. `agents/` (AI 에이전트)

다양한 전략과 알고리즘을 사용하는 플레이어 에이전트들의 구현체입니다.

* **`base_agent.py`**: 모든 에이전트가 상속받는 추상 기본 클래스(ABC)입니다. 공통 인터페이스인 `get_action(status)`을 정의합니다.
* **`rl_agent.py`**: 강화학습(RL) 기반 에이전트입니다. PPO 및 REINFORCE 알고리즘을 지원하며, `DynamicActorCritic` 모델을 사용하여 행동을 결정합니다.
* **`llm_agent.py`**: 거대 언어 모델(LLM, Solar-pro 등)을 활용한 생성형 에이전트입니다. 자연어 로그를 해석하고 프롬프트(`prompts.yaml`)에 기반하여 추론합니다.
* **`rule_base_agent.py`**: 사전에 정의된 휴리스틱 규칙에 따라 행동하는 에이전트입니다. 직업별(경찰, 의사, 마피아) 정석 플레이를 수행합니다.

### 2. `engine/` (게임 엔진)

강화학습 라이브러리에 의존하지 않는 순수한 마피아 게임 로직입니다.

* **`game.py` (`MafiaGame`)**: 게임의 페이즈(낮/밤), 투표, 처형, 스킬 사용 등 전체 흐름을 제어합니다.
* **`state.py`**: 게임 내 데이터 모델을 정의합니다.
* `GameStatus`: 에이전트에게 제공되는 관측 정보 (생존자 목록, 턴 정보 등).
* `GameAction`: 에이전트의 행동 (Target 지목 + Role 주장).
* `GameEvent`: 게임 내 발생 사건 기록 (Log).



### 3. `envs/` (RL 환경)

`Core Engine`을 강화학습 학습용 인터페이스로 래핑한 모듈입니다.

* **`mafia_env.py` (`MafiaEnv`)**:
* **PettingZoo ParallelEnv**를 상속받아 멀티 에이전트 학습 환경을 제공합니다.
* **Observation Space (286-dim)**: 플레이어 상태, 게임 진행도, 그리고 관계 맵(Vote/Attack/Vouch Map)을 벡터화하여 제공합니다.
* **Action Space**: Multi-Discrete `[9, 5]` (타겟 9개 + 역할 5개) 구조를 가집니다.
* **Reward System**: 승패, 마피아 처형 기여, 직업별 역할 수행에 따른 보상을 계산합니다.



### 4. `managers/` (실험 및 관리)

게임 실행, 로깅, 데이터 관리 등을 담당하는 유틸리티 클래스입니다.

* **`experiment.py` (`ExperimentManager`)**: 설정(`config`)과 인자(`args`)를 바탕으로 게임 환경과 에이전트를 조립(Build)합니다.
* **`runner.py`**:
* `train()`: SuperSuit 기반의 병렬 환경에서 RL 에이전트를 학습시킵니다.
* `test()`: 학습된 모델이나 룰 기반 에이전트로 시뮬레이션을 돌리고 데이터를 수집합니다.


* **`logger.py` (`LogManager`)**: 게임 이벤트를 JSONL 파일로 기록하고, TensorBoard에 학습 메트릭을 실시간으로 시각화합니다.
* **`stats.py` (`StatsManager`)**: 승률, 생존율, 평균 보상 등 통계 지표를 계산합니다.
* **`expert.py` (`ExpertDataManager`)**: 모방 학습(Imitation Learning)을 위해 전문가(승리한 플레이어)의 데이터를 수집하고 데이터셋(`Dataset`)으로 가공합니다.