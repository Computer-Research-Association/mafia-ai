import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict

# -----------------------------------------------------------------------------
# 1. 기본 설정 및 경로 정의
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

PYTHON_EXE = sys.executable
MAIN_PY_PATH = PROJECT_ROOT / "main.py"

# RQ1 실험 구조 정의 (폴더명: 역할)
SCENARIOS = {
    "1_mafia": {"rl_role": "mafia", "rl_count": 2},
    "2_police": {"rl_role": "police", "rl_count": 1},
    "3_doctor": {"rl_role": "doctor", "rl_count": 1},
    "4_citizen": {"rl_role": "citizen", "rl_count": 4},
}

# 탐색할 람다 값 목록
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]


# -----------------------------------------------------------------------------
# 2. 모델 수집 함수 (Collect Logic)
# -----------------------------------------------------------------------------
def get_rq1_models(root_dir: Path = None) -> List[Dict]:
    if root_dir is None:
        root_dir = PROJECT_ROOT / "logs" / "rq1"

    found_sessions = []
    print(f"[Search] Scanning for models in: {root_dir}")

    if not root_dir.exists():
        print(f"[Error] Log directory not found: {root_dir}")
        return []

    # 1. 시나리오 폴더 순회 (예: 1_mafia)
    for scenario_folder, config in SCENARIOS.items():
        role_name = config["rl_role"]
        rl_count = config["rl_count"]

        scenario_path = root_dir / scenario_folder
        if not scenario_path.exists():
            continue

        for l_val in LAMBDAS:
            lambda_path = scenario_path / f"lambda_{l_val}"
            if not lambda_path.exists():
                continue

            # 세션 폴더 순회
            for session_dir in lambda_path.iterdir():
                if not session_dir.is_dir():
                    continue

                # models 폴더가 있고, 그 안에 .pt 파일이 하나라도 있는지 확인
                models_dir = session_dir / "models"
                if models_dir.exists() and list(models_dir.glob("*.pt")):
                    session_info = {
                        "models_dir": str(models_dir),
                        "session_name": session_dir.name,
                        "scenario": scenario_folder,
                        "role": role_name,
                        "rl_count": rl_count,
                        "lambda": l_val,
                    }
                    found_sessions.append(session_info)

    return found_sessions


# -----------------------------------------------------------------------------
# 3. 테스트 실행 함수 (Execution Logic)
# -----------------------------------------------------------------------------
def run_batch_tests(episodes: int = 100, runs: int = 100):
    # 1. 세션 수집
    sessions = get_rq1_models()
    total_sessions = len(sessions)

    if total_sessions == 0:
        print("\nNo valid sessions found. Please check 'logs/rq1' directory.")
        return

    print(f"\nFound {total_sessions} sessions. Starting batch evaluation...")
    print(f"Target Episodes: {episodes}")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    start_time = time.time()

    # 2. 각 세션별로 반복 테스트 실행
    for idx, sess in enumerate(sessions):
        role = sess["role"]

        print(f"\n[{idx+1}/{total_sessions}] Session: {sess['session_name']}")
        print(f"Target: {sess['scenario']} | {role.upper()} | Lambda {sess['lambda']}")

        # [핵심] 지정된 횟수만큼 반복 실행 (Run 1 ~ Run N)
        for r in range(1, runs + 1):

            run_folder_name = f"{sess['session_name']}_run_{r}"

            eval_log_dir = (
                PROJECT_ROOT
                / "logs"
                / "rq1_test"
                / sess["scenario"]
                / f"lambda_{sess['lambda']}"
                / run_folder_name
            )
            eval_log_dir.mkdir(parents=True, exist_ok=True)

            print(f"Running {r}/{runs} ... ", end="", flush=True)

            cmd = [
                PYTHON_EXE,
                str(MAIN_PY_PATH),
                "--mode",
                "test",
                "--episodes",
                str(episodes),
                "--rl_role",
                role,
                "--rl_count",
                str(sess["rl_count"]),
                "--lambda",
                str(sess["lambda"]),
                "--log_dir",
                str(eval_log_dir),
                "--model_dir",
                sess["models_dir"],
            ]

            try:
                subprocess.run(cmd, check=True)

                print(f"Done (Saved in .../{run_folder_name})")
                success_count += 1
            except subprocess.CalledProcessError:
                print(f"Failed")
                fail_count += 1
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                return

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"All {runs} runs completed in {elapsed:.1f} seconds.")
    print(f"Success: {success_count} | Failed: {fail_count}")
    print(f"Results saved in: {PROJECT_ROOT / 'logs' / 'rq1_test'}")


# -----------------------------------------------------------------------------
# 4. 메인 실행부
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Test Runner for RQ1 Models")
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes for testing each model",
    )
    parser.add_argument(
        "--runs", type=int, default=100, help="Number of runs per session (e.g., 100)"
    )

    args = parser.parse_args()

    # 실행
    run_batch_tests(episodes=args.episodes, runs=args.runs)
