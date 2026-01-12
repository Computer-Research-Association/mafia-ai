import numpy as np
import os
from typing import Dict, Any, Optional
import threading
import torch
from tqdm import tqdm

from config import Role, config
from core.managers.stats import StatsManager
from core.managers.logger import LogManager
from core.managers.expert import ExpertDataManager


def train(
    env,
    rl_agents: Dict[int, Any],
    all_agents: Dict[int, Any],
    args,
    logger: "LogManager",
    stop_event: Optional[threading.Event] = None,
):
    """
    SuperSuit VectorEnv 전용 병렬 학습 루프
    - 배치 크기 불일치(Shape Mismatch) 원천 차단
    """

    # 1. 플레이어 수 동적 감지 (하드코딩 방지)
    # env.possible_agents가 있으면 사용, 없으면 config나 rl_agents+all_agents 길이로 추론
    if hasattr(env, "possible_agents"):
        PLAYERS_PER_GAME = len(env.possible_agents)
    else:
        PLAYERS_PER_GAME = 8

    sample_indices = list(range(0, env.num_envs, PLAYERS_PER_GAME))
    actual_num_games = len(sample_indices)
    
    print(f"=== Start SuperSuit Parallel Training ===")
    print(f"  - Actual Parallel Games: {actual_num_games}")

    stats_manager = StatsManager()
    total_episodes = args.episodes

    slot_episode_ids = [i + 1 for i in range(actual_num_games)]
    next_global_episode_id = actual_num_games + 1

    # --- 초기화 ---
    obs, infos = env.reset()

    # RNN Hidden States 초기화 (계산된 actual_num_games 사용)
    for agent in rl_agents.values():
        if hasattr(agent, "reset_hidden"):
            agent.reset_hidden(batch_size=actual_num_games)

    completed_episodes = 0
    current_rewards = {}  # 빈 딕셔너리 생성
    train_metrics = {}    # 학습 메트릭 저장용

    # 에이전트마다 자기 할당량을 직접 계산 (len)
    for pid in rl_agents.keys():
        # 내(pid)가 처리해야 할 데이터의 인덱스 리스트 생성
        # 예: 전체 65개 슬롯, 8명 게임 -> 0번 플레이어는 9개, 1번 플레이어는 8개...
        my_indices = list(range(pid, env.num_envs, PLAYERS_PER_GAME))

        # 실제 내가 받은 데이터 개수 (9 or 8)
        my_batch_size = len(my_indices)

        # 1. RNN Hidden State 초기화 (내 실제 크기에 맞춤)
        if hasattr(rl_agents[pid], "reset_hidden"):
            rl_agents[pid].reset_hidden(batch_size=my_batch_size)

        # 2. 보상 버퍼 초기화 (내 실제 크기에 맞춤)
        # 이렇게 하면 나중에 (9,) 그릇에 (9,) 데이터를 담게 되어 에러가 안 남
        current_rewards[pid] = np.zeros(my_batch_size)

    pbar = tqdm(total=total_episodes, desc="Training", unit="ep")
    recorder_pid = None  # 첫 번째로 발견된 프로세스 ID만 기록

    while True:
        is_target_still_running = any(eid <= total_episodes for eid in slot_episode_ids)
        
        if completed_episodes >= total_episodes and not is_target_still_running:
            break

        if stop_event and stop_event.is_set():
            break

        # --- [1. 행동 결정 (Batch Action)] ---
        all_actions = np.zeros((env.num_envs, 2), dtype=int)

        for pid in range(PLAYERS_PER_GAME):
            # PID별 인덱스 추출
            indices = list(range(pid, env.num_envs, PLAYERS_PER_GAME))

            if pid in rl_agents:
                agent = rl_agents[pid]

                # 관측 데이터 슬라이싱
                if isinstance(obs, dict):
                    agent_obs = {
                        key: val[indices]
                        for key, val in obs.items()
                        if isinstance(val, (np.ndarray, list))
                    }
                else:
                    agent_obs = obs[indices]

                # 행동 선택
                actions = agent.select_action_vector(agent_obs)

                if not isinstance(actions, np.ndarray):
                    actions = np.array(actions)

                # 차원 안전장치
                if len(actions.shape) == 1:
                    pass

                # 배열 할당 (여기도 indices 길이와 actions 길이가 같으므로 안전)
                all_actions[indices] = actions

            elif pid in all_agents:
                # LLM/Rule 기반 스킵
                pass

        # --- [2. 환경 진행 (Step)] ---
        next_obs, rewards, terminations, truncations, infos = env.step(all_actions)

        # 학습 속도 향상을 위해 텍스트 로그 처리 로직 제거
        # 오직 통계와 텐서보드 메트릭만 남김

        # --- [3. 보상 저장 및 버퍼 관리] ---
        for pid, agent in rl_agents.items():
            indices = list(range(pid, env.num_envs, PLAYERS_PER_GAME))

            # 배치 데이터 추출
            p_rewards = rewards[indices]
            p_terms = terminations[indices]
            p_truncs = truncations[indices]

            current_rewards[pid] += p_rewards

            # 학습용 버퍼 저장
            for i, idx in enumerate(indices):
                is_done = p_terms[i] or p_truncs[i]
                agent.store_reward(p_rewards[i], is_done)

        obs = next_obs

        # --- [4. 종료 체크 및 모델 업데이트] ---
        # 0번 플레이어 기준 종료 체크
        p0_indices = list(range(0, env.num_envs, PLAYERS_PER_GAME))
        dones = np.logical_or(terminations[p0_indices], truncations[p0_indices])
        num_finished_now = np.sum(dones)

        if num_finished_now > 0:
            finished_slot_indices = np.where(dones)[0]
            for slot_idx in finished_slot_indices:
                slot_episode_ids[slot_idx] = next_global_episode_id
                next_global_episode_id += 1
            
            completed_episodes += num_finished_now
            finished_indices = np.where(dones)[0]

            # --- [Stats Tracking Logic] ---
            # 1. Collect Completed Infos
            finished_infos = []
            if isinstance(infos, dict) and isinstance(list(infos.values())[0], (list, np.ndarray)):
                 # SuperSuit Stacked Dict: {'player_0': [info0, info1...], ...}
                 for j in finished_indices:
                     # Check Key ("player_0") exists
                     key_0 = "player_0"
                     if key_0 in infos:
                         finished_infos.append(infos[key_0][j])
                     else:
                         # Fallback
                         first_key = list(infos.keys())[0]
                         finished_infos.append(infos[first_key][j])
            
            elif isinstance(infos, (list, tuple)):
                 # Standard Gym Vector Env: infos is list of dicts (length = total agents)
                 for j in finished_indices:
                     # Game j's Player 0 is at index j * PLAYERS_PER_GAME
                     idx = j * PLAYERS_PER_GAME
                     finished_infos.append(infos[idx])
            
            # 2. Collect Rewards & Wins for Legacy Stats
            agent_rewards_stats = {}
            agent_wins_stats = {}
            
            for pid in rl_agents.keys():
                # Reward (Avg of finished games)
                avg_r = np.mean(current_rewards[pid][finished_indices])
                agent_rewards_stats[pid] = avg_r
                
                # Win (Check via infos)
                batch_wins = []
                for j in finished_indices:
                    w = False
                    if isinstance(infos, (list, tuple)):
                        idx = j * PLAYERS_PER_GAME + pid
                        if idx < len(infos):
                            w = infos[idx].get('win', False)
                    elif isinstance(infos, dict):
                        agent_key = f"player_{pid}"
                        if agent_key in infos and j < len(infos[agent_key]):
                             w = infos[agent_key][j].get('win', False)
                    
                    batch_wins.append(w)
                
                agent_wins_stats[pid] = (np.mean(batch_wins) > 0.5) if batch_wins else False

            # 3. Calculate Stats
            metrics = stats_manager.calculate_stats(
                infos=finished_infos,
                rl_agents=rl_agents,
                all_agents=all_agents,
                episode_rewards=agent_rewards_stats,
                is_wins=agent_wins_stats,
                train_metrics=train_metrics
            )

            # 대표 에이전트 평균 보상 로깅
            rep_pid = list(rl_agents.keys())[0]
            avg_reward = agent_rewards_stats[rep_pid]

            logger.set_episode(int(completed_episodes))
            logger.log_metrics(
                episode=int(completed_episodes), 
                total_reward=avg_reward, 
                is_win=False,
                **metrics
            )

            # 끝난 게임의 누적 보상 리셋
            for pid in rl_agents:
                current_rewards[pid][finished_indices] = 0.0

            pbar.update(num_finished_now)

            # 업데이트 주기 체크
            if (
                completed_episodes - num_finished_now
            ) // 100 != completed_episodes // 100:
                pbar.write("[System] Updating Agents...")
                for pid, agent in rl_agents.items():
                    if hasattr(agent, "update"):
                        res = agent.update()
                        if res:
                             # Try to categorize metrics by Role
                             role_key = "Mafia" if all_agents[pid].role == Role.MAFIA else "Citizen"
                             train_metrics[role_key] = res

    # --- 학습 종료 및 저장 ---
    print("\n[System] Saving trained models...")
    model_dir = getattr(config.paths, "MODEL_DIR", "./models")
    os.makedirs(model_dir, exist_ok=True)
    for pid, agent in rl_agents.items():
        save_path = os.path.join(model_dir, f"agent_{pid}_supersuit.pt")
        if hasattr(agent, "save"):
            agent.save(save_path)
            print(f"Saved: {save_path}")


def test(
    env, 
    all_agents: Dict[int, Any], 
    args, 
    logger: "LogManager" = None,
    stop_event: Optional[threading.Event] = None
):
    """
    Runner centrally controls logging. 
    It extracts 'log_events' from env.infos and logs them via the provided logger.
    """
    num_episodes = args.episodes
    print(f"=== Start Data Collection / Test (Target: {num_episodes} eps) ===")

    # Setup Data Manager
    data_manager = None
    if logger and logger.session_dir:
        data_manager = ExpertDataManager(save_dir=logger.session_dir)
        print(f"  - [Data Collection ON] Saved to: {data_manager.save_path}")
    else:
        print("  - [Data Collection OFF] No logger provided.")

    # Helper function to process logs from infos
    def process_logs(info_dict):
        if not logger or not info_dict: return
        for _, info_item in info_dict.items():
            if isinstance(info_item, dict) and "log_events" in info_item:
                for ev_dict in info_item["log_events"]:
                    try:
                        from core.engine.state import GameEvent
                        logger.log_event(GameEvent(**ev_dict))
                    except Exception: pass
                break # Process only once per turn

    completed_episodes = 0
    
    # Initial Reset & Log (Captures Day 0 / Role assignments)
    obs, infos = env.reset()
    process_logs(infos)
    
    # Stats
    win_counts = {Role.CITIZEN: 0, Role.MAFIA: 0}

    # tqdm으로 전체 에피소드 루프
    pbar = tqdm(total=num_episodes, desc="Collecting Data", unit="ep")

    while completed_episodes < num_episodes:
        if stop_event and stop_event.is_set(): break

        # --- [1. 행동 결정] ---
        actions = {}
        for agent_id_str in env.agents:
            p_id = int(agent_id_str.split("_")[1])
            full_obs = obs[agent_id_str]
            obs_vec = full_obs['observation']
            agent = all_agents[p_id]
            
            action_obj = None
            action_vector = [0, 0] 

            try:
                # LLM / Rule / Heuristic
                if hasattr(agent, "get_action"):
                    game_status = env.get_game_status(p_id)
                    action_obj = agent.get_action(game_status)
                    action_vector = action_obj.to_multi_discrete()
                    actions[p_id] = action_obj 

                # RL Agent
                elif hasattr(agent, "select_action_vector"):
                    action_vector = agent.select_action_vector(full_obs)
                    actions[p_id] = action_vector 

                # 데이터 수집 (Runner 주도)
                if data_manager:
                    current_ep_id = completed_episodes + 1
                    action_mask = full_obs.get('action_mask')
                    data_manager.record_turn(current_ep_id, p_id, obs_vec, action_vector, action_mask=action_mask)

            except Exception as e:
                # 에러 발생 시 건너뜀 (데이터 오염 방지)
                pass

        # --- [2. 환경 진행] ---
        env_actions = {f"player_{pid}": act for pid, act in actions.items()}
        next_obs, rewards, terminations, truncations, infos = env.step(env_actions)
        done = not env.agents

        # --- [3. 로그 저장 (Runner 중앙 관리)] ---
        process_logs(infos) # Log events occurring during step

        obs = next_obs

        # --- [4. 종료 체크] ---
        if done:
            completed_episodes += 1
            
            winner = env.game.winner
            if winner:
                win_counts[winner] += 1
            
            # 데이터 파일 저장 (Flush)
            if data_manager:
                data_manager.flush_episode(
                    completed_episodes, 
                    winner_role=winner, 
                    players=env.game.players
                )
            
            # 로그 메트릭 기록
            if logger:
                is_win = (winner == Role.MAFIA) 
                logger.log_metrics(completed_episodes, total_reward=0, is_win=is_win)
                # 다음 에피소드 번호 세팅 (중요: Day 0 로그가 섞이지 않게 함)
                logger.set_episode(completed_episodes + 1)

            pbar.update(1)
            win_rate = (win_counts[Role.MAFIA] / completed_episodes) * 100 if completed_episodes > 0 else 0.0
            pbar.set_postfix(mafia_win_rate=f"{win_rate:.1f}%")
            
            # [중요] 목표를 아직 못 채웠을 때만 리셋 (불필요한 Day 0 로그 방지)
            if completed_episodes < num_episodes:
                obs, infos = env.reset()
                process_logs(infos) # Capture new episode's start logs

    pbar.close()
    print(f"\n=== Test/Collection Finished ===")
    print(f"Results: {win_counts}")
    if data_manager:
        print(f"Expert Data Saved: {data_manager.save_path}")