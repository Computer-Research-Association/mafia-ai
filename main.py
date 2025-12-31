import sys
import argparse
import os
import threading
import tkinter as tk

from core.env import MafiaEnv
from core.game import MafiaGame
from core.agent.rlAgent import RLAgent
from core.logger import LogManager
from config import Role
from PyQt6.QtWidgets import QApplication
from core.runner import train, test

from gui.launcher import Launcher

# GUI 모듈 임포트 (gui 패키지가 없어도 에러 안 나게 처리)
try:
    from gui.gui_viewer import MafiaLogViewerApp

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


def run_simulation(args):
    """
    AI 학습/테스트 로직
    LogManager를 통한 통합 로깅 시스템 사용
    새로운 player_configs 구조 지원
    """
    # 하위 호환성: 이전 구조(args.agent) 지원
    if hasattr(args, 'agent'):
        # CLI에서 실행된 경우 (레거시 구조)
        player_configs = None
        experiment_name = f"{args.agent}_{getattr(args, 'backbone', 'mlp')}_{args.mode}"
    else:
        # GUI에서 실행된 경우 (새로운 구조)
        player_configs = args.player_configs
        
        # Player 0의 설정으로 실험 이름 생성
        player0_config = player_configs[0]
        if player0_config['type'] == 'rl':
            experiment_name = f"{player0_config['algo']}_{player0_config['backbone']}_{args.mode}"
        else:
            experiment_name = f"llm_{args.mode}"
    
    # LogManager 초기화
    log_dir = str(getattr(args, 'paths', {}).get('log_dir', 'logs'))
    logger = LogManager(experiment_name=experiment_name, log_dir=log_dir)
    
    print(f"Simulation started: {experiment_name}")
    print(f"Player configurations: {player_configs if player_configs else 'Legacy mode'}")

    try:
        # 레거시 구조 처리
        if player_configs is None:
            # LLM 에이전트 모드
            if args.agent == "llm":
                print("Running LLM-only simulation.")
                # TODO: LLM 전용 시뮬레이션 로직 구현
                print("LLM simulation finished.")

            # RL 에이전트 모드 (PPO, REINFORCE)
            else:
                print(f"[{args.mode.upper()}] mode for {args.agent.upper()} agent.")
                
                # 환경 및 에이전트 초기화
                env = MafiaEnv()
                state_dim = env.observation_space["observation"].shape[0]
                action_dim = env.action_space.n

                agent = RLAgent(
                    player_id=0,
                    role=Role.CITIZEN,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    algorithm=args.agent,
                    backbone=getattr(args, "backbone", "mlp"),
                    use_il=getattr(args, "use_il", False),
                    hidden_dim=getattr(args, "hidden_dim", 128),
                    num_layers=getattr(args, "num_layers", 2),
                )

                # 모드별 실행
                if args.mode == "train":
                    train(env, agent, args, logger)
                elif args.mode == "test":
                    test(env, agent, args)
        
        # 새로운 player_configs 구조 처리
        else:
            print(f"[{args.mode.upper()}] mode with player_configs.")
            
            # Player 0 설정 추출
            player0_config = player_configs[0]
            
            # Player 0이 RL인 경우에만 학습/테스트 진행
            if player0_config['type'] == 'rl':
                env = MafiaEnv()
                state_dim = env.observation_space["observation"].shape[0]
                action_dim = env.action_space.n

                agent = RLAgent(
                    player_id=0,
                    role=Role.CITIZEN,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    algorithm=player0_config['algo'],
                    backbone=player0_config['backbone'],
                    use_il=False,  # GUI에서는 기본적으로 IL 비활성화
                    hidden_dim=player0_config.get('hidden_dim', 128),
                    num_layers=player0_config.get('num_layers', 2),
                )

                # 모드별 실행
                if args.mode == "train":
                    train(env, agent, args, logger)
                elif args.mode == "test":
                    test(env, agent, args)
            else:
                print("Player 0 is LLM. Full game simulation not yet implemented.")
                # TODO: 전체 게임 시뮬레이션 로직 구현
    
    finally:
        # LogManager 리소스 정리
        logger.close()
        print("Simulation finished.")


def start_gui():
    app = QApplication(sys.argv)
    launcher = Launcher()

    def on_simulation_start(args):
        sim_thread = threading.Thread(target=run_simulation, args=(args,), daemon=True)
        sim_thread.start()

    # 시그널 연결
    launcher.start_simulation_signal.connect(on_simulation_start)

    launcher.show()
    sys.exit(app.exec())


def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Mafia AI Training/Testing Script")
        parser.add_argument(
            "--mode", type=str, default="train", choices=["train", "test"]
        )
        parser.add_argument(
            "--agent", type=str, default="ppo", choices=["ppo", "reinforce", "llm"]
        )
        parser.add_argument("--episodes", type=int, default=1000)
        parser.add_argument("--gui", action="store_true")

        # RLAgent 설정
        parser.add_argument(
            "--backbone",
            type=str,
            default="mlp",
            choices=["mlp", "lstm", "gru"],
            help="Neural network backbone",
        )
        parser.add_argument(
            "--use_il", action="store_true", help="Enable Imitation Learning"
        )
        parser.add_argument(
            "--hidden_dim",
            type=int,
            default=128,
            help="Hidden dimension for neural network",
        )
        parser.add_argument(
            "--num_layers", type=int, default=2, help="Number of layers for RNN"
        )

        args = parser.parse_args()

        if args.gui and GUI_AVAILABLE:
            # 기존 Tkinter 뷰어 실행 로직 (유지)
            print("Launching Legacy GUI with Simulation...")
            sim_thread = threading.Thread(
                target=run_simulation, args=(args,), daemon=True
            )
            sim_thread.start()
            root = tk.Tk()
            root.mainloop()
        else:
            run_simulation(args)

    # 인자가 없으면 -> GUI 실행
    else:
        print("Start GUI")
        start_gui()


if __name__ == "__main__":
    main()
