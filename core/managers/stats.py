from typing import Dict, Any, List, Optional
import numpy as np
from collections import defaultdict, deque
from config import Role, EventType, Phase


class StatsManager:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.recent_wins = defaultdict(deque)
        self.recent_mafia_wins = []
        self.recent_citizen_wins = []
        self.mafia_win_days = []
        self.citizen_win_days = []

    def calculate_stats(
        self,
        infos: Any,
        rewards: List[float],
        rl_agents: Dict[int, Any],
        train_metrics: Dict[str, Dict[str, List[float]]],
    ) -> Dict[str, float]:

        metrics = {}
        mafia_rewards = []
        citizen_rewards = []
        slot_rewards = defaultdict(list)
        slot_wins = defaultdict(list)

        # 1. 모든 플레이어 인스턴스를 순회하며 실제 역할(role) 기준으로 분류
        for i, info in enumerate(infos):
            pid = i % 8  # 플레이어 번호 추출
            role = info.get("role")
            r = rewards[i]
            win = 1 if info.get("win", False) else 0

            # 슬롯(PID)별 보상/승리 데이터 수집
            if pid in rl_agents:
                slot_rewards[pid].append(r)
                slot_wins[pid].append(win)

            # 팀별 데이터 분류 (실제 게임 엔진이 부여한 역할 기준)
            if role == Role.MAFIA:
                mafia_rewards.append(r)
            else:
                citizen_rewards.append(r)

        # 2. 에이전트 슬롯별 지표 계산
        for pid in rl_agents.keys():
            avg_r = np.mean(slot_rewards[pid]) if slot_rewards[pid] else 0.0
            avg_w = np.mean(slot_wins[pid]) if slot_wins[pid] else 0.0

            # 최근 승률 히스토리 업데이트
            self.recent_wins[pid].append(avg_w)
            if len(self.recent_wins[pid]) > self.window_size:
                self.recent_wins[pid].popleft()

            metrics[f"Agent_{pid}/Reward"] = avg_r
            if len(self.recent_wins[pid]) >= self.window_size:
                metrics[f"Agent_{pid}/Win_Rate"] = np.mean(self.recent_wins[pid])

        # 3. 팀별 평균 지표 계산
        metrics["Reward/Total"] = sum(rewards)
        metrics["Reward/Mafia_Avg"] = np.mean(mafia_rewards) if mafia_rewards else 0.0
        metrics["Reward/Citizen_Avg"] = (
            np.mean(citizen_rewards) if citizen_rewards else 0.0
        )

        # Team/Role Training Stats
        # 이제 train_metrics는 pid를 key로 가짐
        for pid, agent_metrics in train_metrics.items():
            if "loss" in agent_metrics:
                metrics[f"Agent_{pid}/Loss"] = agent_metrics["loss"]
            if "entropy" in agent_metrics:
                metrics[f"Agent_{pid}/Entropy"] = agent_metrics["entropy"]
            if "approx_kl" in agent_metrics:
                metrics[f"Agent_{pid}/KL_Divergence"] = agent_metrics["approx_kl"]
            if "clip_frac" in agent_metrics:
                metrics[f"Agent_{pid}/Clip_Fraction"] = agent_metrics["clip_frac"]

        # 이전 role_key 기반 로직 제거 (더 이상 사용 안함)
        for role_key in ["Mafia", "Citizen"]:
            role_metrics = train_metrics.get(role_key, {})
            if "loss" in role_metrics and role_metrics["loss"]:
                metrics[f"Train/{role_key}_Loss"] = np.mean(role_metrics["loss"])
            if "entropy" in role_metrics and role_metrics["entropy"]:
                metrics[f"Train/{role_key}_Entropy"] = np.mean(role_metrics["entropy"])
            if "approx_kl" in role_metrics and role_metrics["approx_kl"]:
                metrics[f"Train/{role_key}_ApproxKL"] = np.mean(
                    role_metrics["approx_kl"]
                )
            if "clip_frac" in role_metrics and role_metrics["clip_frac"]:
                metrics[f"Train/{role_key}_ClipFrac"] = np.mean(
                    role_metrics["clip_frac"]
                )

        # --- 2. Game Stats (via infos) ---
        game_metrics_list = []

        # Helper to extract metrics from a single info dict
        def extract_metrics(info_dict):
            if isinstance(info_dict, dict) and "episode_metrics" in info_dict:
                return info_dict["episode_metrics"]
            return None

        if isinstance(infos, (list, tuple)):
            # Vector environment
            for info in infos:
                m = extract_metrics(info)
                if m:
                    game_metrics_list.append(m)
        elif isinstance(infos, dict):
            # Single environment or dict-based vector info
            # Try iterating values (single env with multiple agents)
            for key, val in infos.items():
                m = extract_metrics(val)
                if m:
                    game_metrics_list.append(m)

            # Fallback: if infos itself has episode_metrics
            m = extract_metrics(infos)
            if m:
                game_metrics_list.append(m)

        # Update History & Calculate Averages
        if game_metrics_list:
            for m in game_metrics_list:
                mafia_won = m.get("Game/Mafia_Win", 0.0) > 0.5
                citizen_won = m.get("Game/Citizen_Win", 0.0) > 0.5
                day = m.get("Game/Duration", 0)

                self.recent_mafia_wins.append(1 if mafia_won else 0)
                self.recent_citizen_wins.append(1 if citizen_won else 0)

                if mafia_won:
                    self.mafia_win_days.append(day)
                if citizen_won:
                    self.citizen_win_days.append(day)

            # Lists Maintenance
            if len(self.recent_mafia_wins) > self.window_size:
                self.recent_mafia_wins = self.recent_mafia_wins[-self.window_size :]
            if len(self.recent_citizen_wins) > self.window_size:
                self.recent_citizen_wins = self.recent_citizen_wins[-self.window_size :]
            if len(self.mafia_win_days) > self.window_size:
                self.mafia_win_days = self.mafia_win_days[-self.window_size :]
            if len(self.citizen_win_days) > self.window_size:
                self.citizen_win_days = self.citizen_win_days[-self.window_size :]

            # Average Scalar Metrics from current batch
            # We filter out Win/Duration related keys to use windowed versions,
            # Or just update 'metrics' with whatever is in game_metrics_list (averaged)
            # EXCEPT calculate win rate from history

            # First, average all raw metrics from the batch
            keys = game_metrics_list[0].keys()
            for k in keys:
                values = [g[k] for g in game_metrics_list if k in g]
                if values:
                    metrics[k] = np.mean(values)

            # Now overwrite Win Rates and Avg Durations with Windowed stats
            metrics["Game/Mafia_WinRate"] = (
                np.mean(self.recent_mafia_wins) if self.recent_mafia_wins else 0.0
            )
            metrics["Game/Citizen_WinRate"] = (
                np.mean(self.recent_citizen_wins) if self.recent_citizen_wins else 0.0
            )
            metrics["Game/Avg_Day_When_Mafia_Wins"] = (
                np.mean(self.mafia_win_days) if self.mafia_win_days else 0.0
            )
            metrics["Game/Avg_Day_When_Citizen_Wins"] = (
                np.mean(self.citizen_win_days) if self.citizen_win_days else 0.0
            )

        return metrics
