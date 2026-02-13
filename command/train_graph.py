import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

# 1. 시각화 스타일 설정 (영어 폰트 사용)
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"  # 기본 폰트 사용

# 2. 실험 설정 복원 (run_rq1.py와 동일한 순서 보장)
SCENARIOS = {
    "1_mafia": {"rl_role": "mafia", "rl_count": 2},
    "2_police": {"rl_role": "police", "rl_count": 1},
    "3_doctor": {"rl_role": "doctor", "rl_count": 1},
    "4_citizen": {"rl_role": "citizen", "rl_count": 4},
}
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

# 실험 순서 리스트 생성
experiment_order = []
for s_id in SCENARIOS.keys():
    for l_val in LAMBDAS:
        experiment_order.append((s_id, l_val))

print(f"Number of experiments defined: {len(experiment_order)}")

# 3. TensorBoard 로그 디렉토리 매핑
# 노트북 파일 위치 기준 상위 폴더(..)로 이동하여 tensorboard 폴더 찾기
PROJECT_ROOT = Path(os.getcwd())
TB_ROOT_DIR = PROJECT_ROOT / "tensorboard"
PLOT_DIR = PROJECT_ROOT / "plots"  # 학습 그래프는 하위 폴더에 분리

# 저장 폴더 생성
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 날짜순 정렬 (먼저 생성된 폴더가 먼저 실행된 실험)
log_dirs = sorted(glob.glob(str(TB_ROOT_DIR / "ppo_mlp_train_*")))

print(f"Number of log folders found: {len(log_dirs)}")

if len(log_dirs) != len(experiment_order):
    print("Warning: Number of log folders does not match number of experiments.")
    if len(log_dirs) > len(experiment_order):
        print(f"-> Using only the last {len(experiment_order)} logs.")
        log_dirs = log_dirs[-len(experiment_order) :]

# 4. 데이터 파싱
data = []

for log_dir, (scenario_id, lambda_val) in zip(log_dirs, experiment_order):
    print(
        f"Parsing: {Path(log_dir).name} -> Scenario: {scenario_id}, Lambda: {lambda_val}"
    )

    event_files = glob.glob(str(Path(log_dir) / "events.out.tfevents.*"))
    if not event_files:
        continue

    event_file = sorted(event_files)[-1]

    try:
        ea = EventAccumulator(event_file)
        ea.Reload()

        tags = ea.Tags()["scalars"]

        for tag in tags:
            events = ea.Scalars(tag)
            for event in events:
                data.append(
                    {
                        "step": event.step,
                        "value": event.value,
                        "metric": tag,
                        "scenario": scenario_id,
                        "lambda": lambda_val,
                        "log_dir": Path(log_dir).name,
                    }
                )
    except Exception as e:
        print(f"Error parsing {event_file}: {e}")

df = pd.DataFrame(data)
print(f"\nTotal data points: {len(df)}")

# 5. 스무딩 및 시각화 함수
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def plot_role_metrics_fast(df, smoothing=0.99, downsample_step=100):
    if df.empty:
        print("No data to plot.")
        return

    # 1. 데이터 정렬
    df_sorted = df.sort_values(by=["scenario", "lambda", "metric", "step"])

    # 2. Warmup 제외 및 다운샘플링
    if downsample_step > 1:
        # 다운샘플링: step을 downsample_step 단위로 묶어서 평균/대표값 사용
        df_sorted["step_bin"] = (df_sorted["step"] // downsample_step) * downsample_step

        # 그룹화하여 평균 계산
        df_resampled = df_sorted.groupby(
            ["scenario", "lambda", "metric", "step_bin"], as_index=False
        )["value"].mean()
        df_resampled.rename(columns={"step_bin": "step"}, inplace=True)
    else:
        df_resampled = df_sorted

    scenarios = df_resampled["scenario"].unique()
    n_scenarios = len(scenarios)

    if n_scenarios == 0:
        print("No scenarios to plot.")
        return

    # 컬러맵 미리 생성
    lambdas = sorted(df_sorted["lambda"].unique())
    cmap = plt.cm.get_cmap("viridis", len(lambdas))
    colors = {l_val: cmap(i) for i, l_val in enumerate(lambdas)}

    # 그래프 설정: 시나리오(행) x 메트릭(열)
    # sharex='col': X축 공유
    fig, axes = plt.subplots(
        n_scenarios, 4, figsize=(20, 4 * n_scenarios), squeeze=False, sharex="col"
    )

    # 메인 타이틀 설정 (크고 굵게, 부가 정보 제거)
    fig.suptitle(
        "Impact of Lambda($\lambda$) on Agent Training",
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )

    subplot_cols = [
        ("Agent_0/KL_Divergence", "KL Divergence"),
        ("Agent_0/Entropy", "Entropy"),
        ("Agent_0/Loss", "Loss"),
        ("Agent_0/Reward", "Reward (Agent 0)"),
    ]

    # 전체 X, Y 라벨 (공통)
    # matplotlib 버전에 따라 supxlabel/supylabel 사용
    try:
        fig.supxlabel("Step", fontsize=18, fontweight="bold", y=0.02)
        # x를 0.02 대신 0.01로 조금 더 왼쪽으로 이동
        fig.supylabel("Value", fontsize=18, fontweight="bold", x=0.01)
    except AttributeError:
        # 구버전 호환용
        fig.text(0.5, 0.02, "Step", ha="center", fontsize=18, fontweight="bold")
        fig.text(
            0.01,
            0.5,
            "Value",
            va="center",
            rotation="vertical",
            fontsize=18,
            fontweight="bold",
        )

    lines_for_legend = []
    labels_for_legend = []

    for i, scenario in enumerate(scenarios):
        scenario_df = df_resampled[df_resampled["scenario"] == scenario]

        for j, (keyword, title) in enumerate(subplot_cols):
            ax = axes[i, j]

            # Title: 맨 윗 행에만 표시
            if i == 0:
                ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

            # Scenario Label: 맨 오른쪽 열의 오른쪽에 표시
            if j == 3:
                ax2 = ax.twinx()
                ax2.set_ylabel(
                    scenario, fontsize=14, fontweight="bold", rotation=270, labelpad=20
                )
                ax2.set_yticks([])  # 눈금 제거
                ax2.spines["right"].set_visible(False)
                ax2.spines["top"].set_visible(False)
                ax2.spines["bottom"].set_visible(False)
                ax2.spines["left"].set_visible(False)

            # 데이터 그리기
            metric_tags = [t for t in scenario_df["metric"].unique() if keyword in t]

            if not metric_tags or scenario_df.empty:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            else:
                subset = scenario_df[scenario_df["metric"].isin(metric_tags)]
                if not subset.empty:
                    for l_val in subset["lambda"].unique():
                        l_subset = subset[subset["lambda"] == l_val].sort_values("step")
                        valid_data = l_subset.dropna(subset=["value"]).copy()
                        if valid_data.empty:
                            continue

                        valid_data["smoothed_value"] = (
                            valid_data["value"]
                            .ewm(alpha=(1 - smoothing), adjust=True)
                            .mean()
                        )

                        (line,) = ax.plot(
                            valid_data["step"],
                            valid_data["smoothed_value"],
                            color=colors[l_val],
                            linewidth=2,
                        )

                        # 범례 수집 (한 번만)
                        if i == 0 and j == 0 and l_val not in labels_for_legend:
                            lines_for_legend.append(line)
                            labels_for_legend.append(l_val)

            # 스타일 설정
            ax.grid(True, alpha=0.3)

            # KL Divergence (첫 번째 열) 과학적 표기법 적용
            if j == 0:
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-2, 2))
                ax.yaxis.set_major_formatter(formatter)

            # Y축 범위 자동 조정
            y_min, y_max = np.inf, -np.inf
            has_data = False
            for line in ax.get_lines():
                y_data = line.get_ydata()
                if len(y_data) > 0:
                    y_valid = y_data[~np.isnan(y_data)]
                    if len(y_valid) > 0:
                        y_min = min(y_min, np.min(y_valid))
                        y_max = max(y_max, np.max(y_valid))
                        has_data = True

            if has_data:
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.5
                ax.set_ylim(y_min - margin, y_max + margin)

    # 하단 통합 범례 설정
    if lines_for_legend:
        # 라벨 순서대로 정렬
        sorted_indices = np.argsort(labels_for_legend)
        sorted_lines = [lines_for_legend[i] for i in sorted_indices]
        sorted_labels = [labels_for_legend[i] for i in sorted_indices]

        fig.legend(
            sorted_lines,
            sorted_labels,
            title="Lambda",
            title_fontsize=12,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),  # 그래프 아래쪽
            ncol=len(sorted_labels),
            frameon=False,
            fontsize=12,
        )

    # 레이아웃 조정 (범례 공간 확보)
    # rect의 첫번째 값(Left margin)을 0.1 -> 0.05로 줄여 좌측 여백 감소
    plt.tight_layout(rect=[0.00, 0.08, 0.95, 0.975])  # Left, Bottom, Right, Top
    plt.subplots_adjust(wspace=0.25, hspace=0.15)  # 간격 미세 조정
    plt.savefig(PLOT_DIR / "0_training_metrics_line.png", dpi=300, bbox_inches="tight")
    plt.close()


if not df.empty:
    plot_role_metrics_fast(df, smoothing=0.99, downsample_step=150)
