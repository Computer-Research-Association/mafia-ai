import sys
import os
import threading
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Any

from PyQt6.QtWidgets import QApplication
from core.managers.runner import train, test
from core.managers.experiment import ExperimentManager
from gui.pages.dashboard import DashBoard
from config import config as cfg

STOP = threading.Event()  # GUI용 종료 신호


def run_simulation(args, stop_event: threading.Event = None):
    """
    AI 학습/테스트 통합 로직 (CLI & GUI 겸용)
    ExperimentManager를 통해 설정 및 객체 생성을 위임합니다.
    """
    # [수정] CLI 모드일 때 player_configs가 없으므로 동적 생성
    if not hasattr(args, "player_configs") or not args.player_configs:
        print("[CLI Mode] Generating player configurations...")
        player_configs: List[Dict[str, Any]] = []
        rl_agent_count = args.rl_count if hasattr(args, "rl_count") else 0

        for i in range(cfg.game.PLAYER_COUNT):
            if i < rl_agent_count:
                # RL 에이전트 설정
                player_configs.append(
                    {
                        "type": "rl",
                        "role": (
                            args.rl_role.upper()
                            if hasattr(args, "rl_role")
                            else "RANDOM"
                        ),
                        "algo": "ppo",  # 기본 알고리즘
                        "backbone": "lstm",  # 기본 백본
                    }
                )
            else:
                # 룰 기반 에이전트 설정
                player_configs.append({"type": "rba", "role": "RANDOM"})

        args.player_configs = player_configs

    # ExperimentManager 초기화
    experiment = ExperimentManager(args)
    print(f"Simulation started: {experiment.logger.experiment_name}")

    try:
        # 환경 및 에이전트 생성
        env = experiment.build_env()
        agents = experiment.build_agents()
        rl_agents = experiment.get_rl_agents(agents)

        print(f"[{args.mode.upper()}] mode with {len(agents)} agents.")

        # 모델 로드 (테스트 모드 또는 지정된 경우)
        # 폴더 경로가 전달된 경우
        if hasattr(args, "model_dir") and args.model_dir:
            model_dir_path = Path(args.model_dir)
            print(f"📂  Loading Models from directory: {model_dir_path.name}")

            for agent in rl_agents.values():
                # 패턴 매칭: agent_{id}_*.pt
                search_pattern = model_dir_path / f"agent_{agent.id}_*.pt"
                found_files = list(glob.glob(str(search_pattern)))

                if found_files:
                    target_file = found_files[0]
                    try:
                        agent.load(target_file)
                        print(
                            f"   ✅ Agent {agent.id} loaded: {Path(target_file).name}"
                        )
                    except Exception as e:
                        print(f"   ❌ Failed to load {target_file}: {e}")
                else:
                    # 해당 에이전트 번호에 맞는 파일이 없을 경우
                    print(f"   ⚠️ No specific model found for Agent {agent.id}")

        print(f"Player Configs: {args.player_configs}")

        # 모드별 실행
        if args.mode == "train":
            # 병렬 환경 생성 시 CPU 코어 수 제한 ( 안정성 )
            num_cpus = min(4, os.cpu_count() or 1)
            env = experiment.build_vec_env(num_envs=8, num_cpus=num_cpus)
            train(
                env,
                rl_agents,
                agents,
                args,
                experiment.logger,
                stop_event=stop_event,
            )
        elif args.mode == "test":
            test(env, agents, args, logger=experiment.logger, stop_event=stop_event)

    finally:
        # 리소스 정리
        experiment.close()
        print("Simulation finished.")


def start_gui():
    """GUI 모드 실행 함수"""
    app = QApplication(sys.argv)
    launcher = DashBoard()

    def on_simulation_start(args):
        STOP.clear()
        sim_thread = threading.Thread(
            target=run_simulation, args=(args, STOP), daemon=True
        )
        sim_thread.start()

    def on_simulation_stop():
        print("Simulation stop signal received.")
        STOP.set()

    # 시그널 연결
    launcher.start_simulation_signal.connect(on_simulation_start)
    launcher.stop_simulation_signal.connect(on_simulation_stop)

    launcher.show()
    sys.exit(app.exec())


def main():
    """메인 엔트리 포인트: CLI 인자 확인 후 GUI 또는 CLI 모드 실행"""
    parser = argparse.ArgumentParser(description="Mafia AI Trainer - GUI and CLI")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        help="Execution mode: train or test",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of episodes to run"
    )
    parser.add_argument(
        "--lambda", type=float, dest="lambda_val", help="Lambda for reward calculation"
    )
    parser.add_argument(
        "--rl_role",
        type=str,
        default="mafia",
        choices=["mafia", "police", "doctor", "citizen"],
        help="Role for RL agents",
    )
    parser.add_argument("--rl_count", type=int, default=0, help="Number of RL agents")
    parser.add_argument("--log_dir", type=str, help="Directory to save logs")
    parser.add_argument(
        "--model_dir", type=str, help="Directory containing .pt files for agents"
    )

    # sys.argv 길이가 1보다 크면 CLI 인자가 있는 것으로 간주
    if len(sys.argv) > 1:
        args = parser.parse_args()
        print("\n--- Running in CLI mode ---")
        run_simulation(args)  # CLI 모드에서는 stop_event가 None
    else:
        print("--- Starting GUI mode ---")
        start_gui()


if __name__ == "__main__":
    main()
