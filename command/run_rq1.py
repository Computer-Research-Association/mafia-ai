import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# 1. 실험 시나리오 설정 (RQ1 정의 기반)
# 각 시나리오별 RL 에이전트의 역할과 수 설정
SCENARIOS = {
    "1_mafia": {"rl_role": "mafia", "rl_count": 2},
    "2_police": {"rl_role": "police", "rl_count": 1},
    "3_doctor": {"rl_role": "doctor", "rl_count": 1},
    "4_citizen": {"rl_role": "citizen", "rl_count": 4}
}

# 2. 독립 변수 (Lambda) 설정
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

# 3. 공통 학습 설정
EPISODES = 10000  # 실험당 에피소드 수
PYTHON_EXE = sys.executable  # 현재 파이썬 실행 경로

def run_experiment(scenario_id, rl_role, rl_count, lambda_val):
    """
    개별 실험을 subprocess로 실행합니다.
    """
    # 로그 디렉토리 경로 생성 (예: logs/rq1/1_mafia/lambda_0.1)
    log_dir = PROJECT_ROOT / "logs" / "rq1" / scenario_id / f"lambda_{lambda_val}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"▶ 실험 시작: 시나리오={scenario_id}, Lambda={lambda_val}")
    print(f"{'='*60}")

    # main.py 실행 인자 구성
    main_py_path = PROJECT_ROOT / "main.py"

    cmd = [
        PYTHON_EXE, str(main_py_path),
        "--mode", "train",
        "--episodes", str(EPISODES),
        "--lambda", str(lambda_val),
        "--rl_role", rl_role,
        "--rl_count", str(rl_count),
        "--log_dir", str(log_dir)
    ]

    try:
        # 프로세스 실행 및 완료 대기
        # CLI 모드로 실행되므로 GUI를 띄우지 않고 자원을 효율적으로 사용합니다.
        subprocess.run(cmd, check=True)
        print(f"\n✅ 실험 완료: {scenario_id} | Lambda: {lambda_val}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 실험 실패: {scenario_id} (Lambda {lambda_val}) | 에러: {e}")
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 실험이 중단되었습니다.")
        sys.exit(1)

if __name__ == "__main__":
    print("=== Mafia AI RQ1 Automated Experiment Runner ===")
    
    # 총 실험 횟수 계산 (4 시나리오 * 6 람다 = 24회)
    total_runs = len(SCENARIOS) * len(LAMBDAS)
    current_run = 0

    for s_id, s_info in SCENARIOS.items():
        for l_val in LAMBDAS:
            current_run += 1
            print(f"\n[전체 진행도: {current_run}/{total_runs}]")
            
            run_experiment(
                scenario_id=s_id,
                rl_role=s_info["rl_role"],
                rl_count=s_info["rl_count"],
                lambda_val=l_val
            )

    print("\n" + "="*60)
    print("🎉 모든 RQ1 실험이 성공적으로 종료되었습니다.")
    print(f"결과 데이터 확인: {Path('logs/rq1').absolute()}")
    print("="*60)