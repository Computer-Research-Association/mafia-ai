import threading
from typing import TYPE_CHECKING, Dict, Any, List
from core.agent.llmAgent import LLMAgent

if TYPE_CHECKING:
    from core.logger import LogManager
    from core.agent.rlAgent import RLAgent


def train(env, rl_agents: Dict[int, Any], all_agents: Dict[int, Any], args, logger: 'LogManager'):
    """
    학습 모드 실행 - 다중 에이전트 지원
    
    Args:
        env: MafiaEnv 인스턴스
        rl_agents: {player_id: RLAgent} 형태의 딕셔너리 (학습 대상)
        all_agents: {player_id: Agent} 형태의 딕셔너리 (모든 에이전트)
        args: 실행 인자
        logger: LogManager
    """
    # 첫 번째 에이전트 기준으로 정보 출력 (대표)
    if not rl_agents:
        print("No RL agents to train.")
        return

    first_agent = next(iter(rl_agents.values()))
    algorithm_name = getattr(first_agent, 'algorithm', args.agent).upper()
    backbone_name = getattr(first_agent, 'backbone', 'mlp').upper()
    
    print(f"Start Training ({algorithm_name}+{backbone_name}) for {args.episodes} episodes...")
    print(f"Active RL Agents: {list(rl_agents.keys())}")

    # 승률 계산을 위한 최근 승리 기록 (에이전트별)
    recent_wins = {pid: [] for pid in rl_agents.keys()}
    window_size = 100

    # Helper to convert int ID to string ID
    def id_to_agent(i): return f"player_{i}"

    for episode in range(1, args.episodes + 1):
        # RNN 은닉 상태 초기화
        for agent in rl_agents.values():
            if hasattr(agent, 'reset_hidden'):
                agent.reset_hidden()
        
        obs_dict, _ = env.reset()
        done = False
        
        # 에피소드별 보상 추적
        episode_rewards = {pid: 0.0 for pid in rl_agents.keys()}
        is_wins = {pid: False for pid in rl_agents.keys()}

        while not done:
            actions = {}
            
            # 1. RL Agents Actions
            for pid, agent in rl_agents.items():
                agent_key = id_to_agent(pid)
                if agent_key in obs_dict:
                    action_vector = agent.select_action_vector(obs_dict[agent_key])
                    actions[agent_key] = action_vector
            
            # 2. Other Agents (LLM/Bot) - Async Execution
            threads = []
            lock = threading.Lock()
            
            def get_llm_action(player, p_key):
                try:
                    # LLM 에이전트에게 현재 상태 전달
                    status = env.game.get_game_status(player.id)
                    player.observe(status)
                    
                    a = player.get_action()
                    with lock:
                        actions[p_key] = a
                except Exception as e:
                    print(f"Error in LLM action: {e}")

            # all_agents 딕셔너리를 사용하여 에이전트 순회
            for pid, agent in all_agents.items():
                p_key = id_to_agent(pid)
                
                # RL 에이전트가 아니고, 아직 액션이 없으며, 살아있는 경우
                # (살아있는지 확인하려면 env.game.players를 참조하거나 status를 봐야 함)
                # 여기서는 env.game.players[pid].alive를 참조 (env 캡슐화 위반 최소화)
                # 또는 obs_dict에 키가 있는지 확인? obs_dict는 살아있는 에이전트만 포함됨.
                # 하지만 LLM 에이전트는 obs_dict를 안 쓸 수도 있음.
                # PettingZoo env.agents는 살아있는 에이전트 리스트임.
                
                if p_key in env.agents and pid not in rl_agents and p_key not in actions:
                    if isinstance(agent, LLMAgent):
                        t = threading.Thread(target=get_llm_action, args=(agent, p_key))
                        threads.append(t)
                        t.start()
                    else:
                        try:
                            # 일반 봇도 상태 업데이트 필요할 수 있음
                            if hasattr(agent, 'observe'):
                                status = env.game.get_game_status(pid)
                                agent.observe(status)
                            actions[p_key] = agent.get_action()
                        except Exception as e:
                            print(f"Error in Bot action: {e}")
            
            for t in threads:
                t.join()

            # 3. Environment Step
            next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)
            
            # Check if game is over
            done = any(terminations.values()) or any(truncations.values())

            # 4. Store Rewards & Track Stats
            for pid, agent in rl_agents.items():
                agent_key = id_to_agent(pid)
                if agent_key in rewards:
                    reward = rewards[agent_key]
                    episode_rewards[pid] += reward
                    
                    if hasattr(agent, 'store_reward'):
                        # 종료 여부 전달
                        is_term = terminations.get(agent_key, False) or truncations.get(agent_key, False)
                        agent.store_reward(reward, is_term)
                
                # 승리 여부 확인 (infos에서)
                if agent_key in infos:
                    is_wins[pid] = infos[agent_key].get("win", False)

            obs_dict = next_obs_dict

        # 에피소드 종료 후 학습
        for agent in rl_agents.values():
            if hasattr(agent, 'update'):
                agent.update()

        # 승률 및 로깅 (대표 에이전트 또는 전체)
        for pid in rl_agents.keys():
            recent_wins[pid].append(1 if is_wins[pid] else 0)
            if len(recent_wins[pid]) > window_size:
                recent_wins[pid].pop(0)
        
        # 대표 에이전트 (첫 번째)
        rep_pid = list(rl_agents.keys())[0]
        rep_win_rate = sum(recent_wins[rep_pid]) / len(recent_wins[rep_pid]) if recent_wins[rep_pid] else 0.0
        
        logger.log_metrics(
            episode=episode,
            total_reward=episode_rewards[rep_pid],
            is_win=is_wins[rep_pid],
            win_rate=rep_win_rate
        )

        if episode % 100 == 0:
            print(
                f"[Training] Ep {episode:5d} | Agent {rep_pid} Score: {episode_rewards[rep_pid]:6.2f} | Win Rate: {rep_win_rate*100:3.0f}%"
            )

    print(f"\n학습 완료. TensorBoard로 결과 확인: tensorboard --logdir={logger.session_dir / 'tensorboard'}")


def test(env, all_agents: Dict[int, Any], args):
    """테스트 모드 실행 - 다중 에이전트 지원"""
    print("Start Test Simulation...")
    obs_dict, _ = env.reset()
    done = False
    
    episode_rewards = {pid: 0.0 for pid in all_agents.keys()}

    # Helper to convert int ID to string ID
    def id_to_agent(i): return f"player_{i}"

    while not done:
        actions = {}
        
        # Iterate all agents
        for pid, agent in all_agents.items():
            agent_key = id_to_agent(pid)
            
            # Skip dead agents
            if agent_key not in env.agents:
                continue
                
            # RL Agent
            if hasattr(agent, 'select_action_vector') and agent_key in obs_dict:
                action_vector = agent.select_action_vector(obs_dict[agent_key])
                actions[agent_key] = action_vector
            
            # LLM / Bot Agent
            else:
                # Update status
                if hasattr(agent, 'observe'):
                    status = env.game.get_game_status(pid)
                    agent.observe(status)
                
                actions[agent_key] = agent.get_action()

        next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)
        
        done = any(terminations.values()) or any(truncations.values())
        
        for pid in all_agents.keys():
            agent_key = id_to_agent(pid)
            if agent_key in rewards:
                episode_rewards[pid] += rewards[agent_key]
            
        obs_dict = next_obs_dict
        env.render()

    print("Test Game Over.")
    for pid, score in episode_rewards.items():
        print(f"Agent {pid} Total Reward: {score}")
