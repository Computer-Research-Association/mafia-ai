from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.logger import LogManager


def train(env, agent, args, logger: 'LogManager'):
    """
    학습 모드 실행
    
    LogManager를 통해 JSONL 및 TensorBoard에 로깅합니다.
    matplotlib 기반 통계는 제거되었습니다.
    """
    algorithm_name = getattr(agent, 'algorithm', args.agent).upper()
    backbone_name = getattr(agent, 'backbone', 'mlp').upper()
    print(f"Start Training ({algorithm_name}+{backbone_name}) for {args.episodes} episodes...")

    # 승률 계산을 위한 최근 승리 기록 (슬라이딩 윈도우)
    recent_wins = []
    window_size = 100

    for episode in range(1, args.episodes + 1):
        # RNN 은닉 상태 초기화
        if hasattr(agent, 'reset_hidden'):
            agent.reset_hidden()
        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        is_win = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, _ = env.step(action)

            # RLAgent의 인터페이스 사용
            agent.store_reward(reward, done)

            # 승리 판정
            if done and reward > 5.0:
                is_win = True

            obs = next_obs
            total_reward += reward

        # 에피소드 종료 후 학습
        agent.update()

        # 승률 계산
        recent_wins.append(1 if is_win else 0)
        if len(recent_wins) > window_size:
            recent_wins.pop(0)
        current_win_rate = sum(recent_wins) / len(recent_wins)

        # LogManager를 통한 메트릭 기록
        logger.log_metrics(
            episode=episode,
            total_reward=total_reward,
            is_win=is_win,
            win_rate=current_win_rate
        )

        if episode % 100 == 0:
            print(
                f"Ep {episode:5d} | Score: {total_reward:6.2f} | Win Rate: {current_win_rate*100:3.0f}%"
            )

    print(f"\n학습 완료. TensorBoard로 결과 확인: tensorboard --logdir={logger.session_dir / 'tensorboard'}")


def test(env, agent, args):
    """테스트 모드 실행"""
    print("Start Test Simulation...")
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        env.render()

    print(f"Test Game Over. Total Reward: {total_reward}")
