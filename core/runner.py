import threading
from typing import TYPE_CHECKING, Dict, Any
from core.agent.llmAgent import LLMAgent

if TYPE_CHECKING:
    from core.logger import LogManager


def train(env, agent, args, logger: 'LogManager'):
    """
    학습 모드 실행 - 학습 가능 에이전트가 있을 때만 update() 호출
    
    LogManager를 통해 JSONL 및 TensorBoard에 로깅합니다.
    """
    algorithm_name = getattr(agent, 'algorithm', args.agent).upper()
    backbone_name = getattr(agent, 'backbone', 'mlp').upper()
    
    # 학습 가능 여부 확인
    is_trainable = hasattr(agent, 'update') and callable(getattr(agent, 'update'))
    
    if is_trainable:
        print(f"Start Training ({algorithm_name}+{backbone_name}) for {args.episodes} episodes...")
    else:
        print(f"Start Simulation ({algorithm_name}) for {args.episodes} episodes... (학습 불가)")

    # 승률 계산을 위한 최근 승리 기록
    recent_wins = []
    window_size = 100

    for episode in range(1, args.episodes + 1):
        # RNN 은닉 상태 초기화 (있는 경우만)
        if hasattr(agent, 'reset_hidden'):
            agent.reset_hidden()
        
        obs_dict, _ = env.reset()
        done = False
        total_reward = 0
        is_win = False

        # Helper to convert int ID to string ID
        def id_to_agent(i): return f"player_{i}"

        while not done:
            actions = {}
            
            # 1. RL Agent Action (Player 0)
            agent_key = id_to_agent(agent.id)
            if agent_key in obs_dict:
                action_vector = agent.select_action_vector(obs_dict[agent_key])
                actions[agent_key] = action_vector
            
            # 2. Other Agents (LLM/Bot) - Async Execution
            threads = []
            lock = threading.Lock()
            
            def get_llm_action(player, p_key):
                try:
                    a = player.get_action()
                    with lock:
                        actions[p_key] = a
                except Exception as e:
                    print(f"Error in LLM action: {e}")

            # env.game.players 접근
            for p in env.game.players:
                p_key = id_to_agent(p.id)
                if p_key not in actions and p.alive:
                    # LLMAgent인 경우 비동기 호출
                    if isinstance(p, LLMAgent):
                        t = threading.Thread(target=get_llm_action, args=(p, p_key))
                        threads.append(t)
                        t.start()
                    else:
                        # 그 외(Rule-based Bot 등)는 동기 호출
                        try:
                            actions[p_key] = p.get_action()
                        except Exception as e:
                            print(f"Error in Bot action: {e}")
            
            # 모든 스레드 종료 대기
            for t in threads:
                t.join()

            # 3. Environment Step
            next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)
            
            # Check if game is over
            done = any(terminations.values()) or any(truncations.values())

            # 보상 저장 (RL Agent)
            if agent_key in rewards:
                reward = rewards[agent_key]
                if hasattr(agent, 'store_reward'):
                    agent.store_reward(reward, done)
                total_reward += reward

            # 승리 판정 (Player 0 기준)
            if done:
                # env.step에서 반환된 info를 통해 승리 여부 확인
                is_win = info.get("win", False)

            obs_dict = next_obs_dict

        # 에피소드 종료 후 학습 (학습 가능한 경우만)
        if is_trainable:
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
            mode = "Training" if is_trainable else "Simulation"
            print(
                f"[{mode}] Ep {episode:5d} | Score: {total_reward:6.2f} | Win Rate: {current_win_rate*100:3.0f}%"
            )

    if is_trainable:
        print(f"\n학습 완료. TensorBoard로 결과 확인: tensorboard --logdir={logger.session_dir / 'tensorboard'}")
    else:
        print(f"\n시뮬레이션 완료. 로그 확인: {logger.session_dir}")


def test(env, agent, args):
    """테스트 모드 실행"""
    print("Start Test Simulation...")
    obs_dict, _ = env.reset()
    done = False
    total_reward = 0

    # Helper to convert int ID to string ID
    def id_to_agent(i): return f"player_{i}"

    while not done:
        actions = {}
        
        # RL Agent
        agent_key = id_to_agent(agent.id)
        if agent_key in obs_dict:
            action_vector = agent.select_action_vector(obs_dict[agent_key])
            actions[agent_key] = action_vector
            
        # Other Agents (Sync for test to avoid complexity or Async if needed)
        # For test, we can keep it sync or async. Let's use sync for simplicity or copy async logic.
        # Using sync for now as test speed is less critical or we want deterministic debug.
        for p in env.game.players:
            p_key = id_to_agent(p.id)
            if p_key not in actions and p.alive:
                actions[p_key] = p.get_action()

        next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)
        
        # Check if game is over
        done = any(terminations.values()) or any(truncations.values())
        
        if agent_key in rewards:
            total_reward += rewards[agent_key]
            
        obs_dict = next_obs_dict
        env.render()

    print(f"Test Game Over. Total Reward: {total_reward}")
