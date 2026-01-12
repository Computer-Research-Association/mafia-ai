import optuna
import argparse
from argparse import Namespace
import torch
import numpy as np
import os

# 프로젝트 모듈 가져오기
from config import config
from core.managers.experiment import ExperimentManager
from core.managers.runner import train


def objective(trial):
    # ==========================================
    # 1. [하이퍼파라미터 추천 받기]
    # ==========================================

    # 학습률 (Learning Rate): 로그 스케일로 탐색 (0.00001 ~ 0.001)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    # 감마 (Gamma): 미래 보상 중요도 (0.9 ~ 0.999)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)

    # 엔트로피 계수 (Entropy Coef): 탐험 비율 (0.01 ~ 0.1)
    entropy_coef = trial.suggest_float("entropy_coef", 0.01, 0.1)

    # 배치 크기 (Batch Size): 64, 128, 256 중 선택
    batch_size = trial.suggest_categorical("batch_size", [128, 256])

    # 신경망 크기 (Hidden Dim): 모델 복잡도
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])

    print(
        f"\n[Trial {trial.number}] Testing: LR={lr:.5f}, Gamma={gamma:.3f}, Batch={batch_size}, Hidden={hidden_dim}"
    )

    # ==========================================
    # 2. [설정 덮어쓰기]
    # ==========================================
    # config.py의 전역 설정값을 Optuna가 추천한 값으로 임시 교체
    config.train.LR = lr
    config.train.GAMMA = gamma
    config.train.ENTROPY_COEF = entropy_coef
    config.train.BATCH_SIZE = batch_size

    # ==========================================
    # 3. [실험 환경 구축]
    # ==========================================
    # ExperimentManager를 위한 가짜 인자(Namespace) 생성
    # 최적화 때는 에피소드를 적게(예: 300판) 설정해서 빠르게 훑어보는 게 좋습니다.
    args = Namespace(
        mode="train",
        episodes=300,  # 300판만 돌려보고 판단 (너무 길면 오래 걸림)
        player_configs=[
            # 8명 모두 PPO RLAgent로 설정 (모델 크기는 위에서 추천받은 값 적용)
            {
                "type": "rl",
                "algo": "ppo",
                "backbone": "lstm",
                "hidden_dim": hidden_dim,
                "num_layers": 2,
                "role": "random",  # 랜덤 역할 배정
            }
            for _ in range(8)
        ],
        # 로그 경로는 trial 번호별로 분리
        paths={"log_dir": f"logs/optuna/trial_{trial.number}", "model_dir": "models"},
    )

    experiment = ExperimentManager(args)
    final_score = -999.0

    try:
        # 환경 생성 (Optuna 중에는 num_cpus=0 또는 1로 설정하여 안전하게 실행)
        # 이미 Optuna 자체가 무거우므로 병렬 환경을 과하게 쓰면 멈출 수 있음
        env = experiment.build_vec_env(
            num_envs=8, num_cpus=0
        )  # 0 = 메인 프로세스에서 실행

        agents = experiment.build_agents()
        rl_agents = experiment.get_rl_agents(agents)

        # ==========================================
        # 4. [학습 실행]
        # ==========================================
        # runner.py의 train 함수가 '마지막 평균 보상'을 반환해야 함
        final_score = train(
            env=env,
            rl_agents=rl_agents,
            all_agents=agents,
            args=args,
            logger=experiment.logger,
        )

        # NaN(수치 오류)이 나오면 최하점 처리
        if np.isnan(final_score):
            final_score = -999.0

    except Exception as e:
        print(f"[Optuna Error] Trial {trial.number} failed: {e}")
        final_score = -999.0  # 에러 나면 최하점

    finally:
        # 리소스 정리 (파일 닫기 등)
        experiment.close()
        try:
            env.close()
        except:
            pass

    # Optuna에게 점수 보고
    return final_score


if __name__ == "__main__":
    # 1. DB 파일 설정
    db_file_path = "mafia_optuna.db"
    storage_name = f"sqlite:///{db_file_path}"

    # [핵심 수정] 기존 DB 파일이 있다면 삭제 (덮어쓰기 효과)
    if os.path.exists(db_file_path):
        os.remove(db_file_path)
        print(f"기존 DB 파일({db_file_path})을 삭제하고 새로 시작합니다.")

    # 2. 스터디 생성
    # 파일이 삭제되었으므로 load_if_exists는 의미가 없지만,
    # 혹시 모를 상황을 위해 True로 둬도 상관없습니다.
    study = optuna.create_study(
        study_name="mafia-ppo-tuning",
        direction="maximize",
        storage=storage_name,
        load_if_exists=True,  # 없으면 만들고, 있으면 로드함 (방금 지웠으니 무조건 새로 만듦)
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50),
    )

    print("=== Mafia AI Hyperparameter Optimization Start ===")
    print(f"Logs will be saved to: {storage_name}")

    # 3. 최적화 실행
    study.optimize(objective, n_trials=20)  # 20번 시도로 수정

    # 4. 결과 출력
    print("\n==================================")
    print(f"Best Value (Reward): {study.best_value}")
    print(f"Best Params: {study.best_params}")
    print("==================================")

    print(f"\n[Tip] View dashboard: optuna-dashboard {storage_name}")
