# AI Modules

강화학습(RL)을 위한 신경망 모델, 학습 알고리즘(PPO, REINFORCE), 그리고 데이터 버퍼를 포함하는 모듈입니다.

## 파일 구성

### 1. `model.py`: DynamicActorCritic
- **역할**: Actor-Critic 구조의 신경망 정의.
- **특징**:
  - `backbone` 옵션에 따라 **MLP**, **LSTM**, **GRU** 중 선택 가능.
  - **Multi-Discrete Action Head**: 타겟 선정($9$)과 직업 주장($5$)을 위한 두 개의 출력 헤드를 가짐.
  - RNN 사용 시 Hidden State를 자동으로 관리하며 배치 처리를 지원.

### 2. `ppo.py`: Proximal Policy Optimization
- **역할**: PPO 알고리즘 구현체.
- **주요 기능**:
  - `RolloutBuffer`에 저장된 데이터를 바탕으로 정책 업데이트.
  - **Imitation Learning (IL)**: 전문가 데이터(`expert_loader`)가 있을 경우 BC(Behavior Cloning) Loss를 추가하여 학습.
  - GAE(Generalized Advantage Estimation) 및 Gradient Clipping 적용.

### 3. `reinforce.py`: REINFORCE
- **역할**: 기본적인 Policy Gradient 알고리즘 구현.
- **주요 기능**:
  - Monte-Carlo 방식으로 에피소드 종료 후 리턴을 계산하여 학습.
  - PPO와의 성능 비교를 위한 베이스라인으로 사용.

### 4. `buffer.py`: RolloutBuffer
- **역할**: PPO 학습을 위한 트랜지션(State, Action, Reward, LogProb 등) 저장소.