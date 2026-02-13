import sys
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from math import pi
from scipy.stats import pearsonr, entropy

# -----------------------------------------------------------------------------
# 1. 경로 및 기본 설정
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(os.getcwd())
SOURCE_DIR = PROJECT_ROOT / "logs" / "rq1_test"
PLOT_DIR = PROJECT_ROOT / "plots"

# 저장 폴더 생성 (없으면 생성)
try:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ 'plots' folder is ready at: {PLOT_DIR}\n")
except Exception as e:
    print(f"❌ Failed to create directory: {e}")
    sys.exit(1)

TARGET_ROLES = {
    "1_mafia": "Mafia",
    "2_police": "Police",
    "3_doctor": "Doctor",
    "4_citizen": "Citizen",
}

# 행동 분석 키워드
KEYWORDS = {
    "Police": ["police", "cop", "경찰"],
    "Doctor": ["doctor", "medic", "healer", "의사"],
    "Citizen": ["citizen", "villager", "시민"],
    "Attack": ["mafia", "killer", "vote", "마피아", "투표"],
}

ROLE_ID_MAP = {0: "Citizen", 1: "Police", 2: "Doctor", 3: "Mafia"}

# 그래프 스타일 설정
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")
print(f"SOURCE_DIR: {SOURCE_DIR}")
print(f"Save Path      : {PLOT_DIR}")

# data 뽑아오기

data = []
print(f"Reading logs from {SOURCE_DIR}")
print("데이터를 추출합니다...")

if SOURCE_DIR.exists():
    for file_path in SOURCE_DIR.glob("**/*.jsonl"):
        if (
            file_path.stat().st_size == 0
            or "1001" in file_path.name
            or "train_set" in file_path.name
        ):
            continue

        try:
            parts = file_path.parts
            lambda_part = next((p for p in parts if p.startswith("lambda_")), None)
            if not lambda_part:
                continue

            scenario_part = parts[parts.index(lambda_part) - 1]
            lambda_val = float(lambda_part.split("_")[1])
            target_role_str = TARGET_ROLES.get(scenario_part)
            if not target_role_str:
                continue

            # --- [Game Scope Variables] ---
            my_agent_ids = []
            game_actions_temp = []

            # 행동 데이터 임시 저장소
            my_self_claims = []  # 나는 누구라고 주장했나?
            my_accusations = []  # 누구를 마피아라고 공격했나?
            my_vote_target = None
            my_night_action_target = None
            defense_utterance = ""

            # 하루 별 데이터 저장소
            day_actions = defaultdict(list)
            max_day = 0

            # 파일 읽기
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    phase = event.get("phase")
                    day = event.get("day")

                    # 최대 날짜
                    if day is not None and day > max_day:
                        max_day = day

                    # [Phase 4] 게임 시작
                    if phase == 4:
                        role_code = event.get("value")
                        assigned_role = ROLE_ID_MAP.get(role_code)

                        # 이 역할이 '내가 찾는 역할'이면, 해당 actor(target_id)를 내 ID로 등록
                        if assigned_role == target_role_str:
                            agent_id = event.get("target_id")
                            if agent_id is not None:
                                my_agent_ids.append(agent_id)

                    # [Phase 0] 낮 토론: 주장(Claim) 및 공격(Accusation) 파싱
                    elif phase == 0:
                        actor = event.get("actor_id")
                        if actor in my_agent_ids:
                            target = event.get("target_id")  # 주장하는 대상
                            val = event.get("value")  # 주장하는 직업

                            action_str = "Silence"

                            # val이 있는 경우 (구조적 행동)
                            if val is not None:
                                role_name = ROLE_ID_MAP.get(val, "Unknown")
                                # 1. Self-Claim
                                if target == actor:
                                    action_str = f"Claim {role_name}"
                                # 2. Accusation
                                else:
                                    if val == 3:
                                        action_str = "Accusation"
                                    else:
                                        action_str = f"Target {role_name}"

                            if action_str.startswith("Claim"):
                                my_self_claims.append(action_str)
                            elif action_str == "Silence":
                                my_self_claims.append("Silence")
                            elif action_str == "Accusation":
                                my_accusations.append("Accusation")

                    # [Phase 1] 투표 (Vote)
                    elif phase == 1:
                        voter = event.get("actor_id")
                        target = event.get("target_id")

                        if voter in my_agent_ids:
                            my_vote_target = target

                    # [Phase 3] 밤 행동
                    elif phase == 3:
                        actor = event.get("actor_id")
                        target = event.get("target_id")

                        if actor in my_agent_ids:
                            my_night_action_target = target

                    # [Phase 2] 처형
                    elif phase == 2:
                        speaker = event.get("actor_id")
                        if speaker in my_agent_ids:
                            defense_utterance = "Defended"

                    # [Phase 5] 게임 종료
                    elif phase == 5:
                        if scenario_part == "1_mafia":
                            is_win = 0 if event.get("value") else 1
                        else:
                            is_win = 1 if event.get("value") else 0

                        # --- 최종 데이터 정리 ---

                        # 1. 모든 주장 기록 (변경 사항)
                        # Counter를 사용하여 가장 빈번한 것 하나만 뽑는 대신,
                        # 리스트 전체를 그대로 저장하거나 빈 경우 Silence 처리
                        if my_self_claims:
                            # 리스트 전체 저장 (예: ['Silence', 'Claim Police', 'Silence', ...])
                            final_claim_history = my_self_claims
                        else:
                            final_claim_history = ["Silence"]

                        # 2. 공격 여부
                        final_target_claim = "Accusation" if my_accusations else "None"

                        # (참고) 단순히 '침묵'만 했는지, 실제로 주장을 한 적이 있는지 판단하기 위해
                        # 주장을 한 번이라도 했으면 Passive가 아닌 Active로 볼 수도 있습니다.
                        # 여기서는 기존 로직 유지하되, final_self_claim 변수가 없어졌으므로
                        # history에 'Claim'으로 시작하는 요소가 있는지 확인하는 로직으로 대체 가능합니다.
                        has_active_claim = any(
                            c.startswith("Claim") for c in final_claim_history
                        )

                        if final_target_claim == "None" and has_active_claim:
                            final_target_claim = "Passive"

                        data.append(
                            {
                                "Scenario": scenario_part,
                                "Lambda": lambda_val,
                                "Win_Value": is_win,
                                "Self_Claim": final_claim_history,
                                "Target_Claim": final_target_claim,
                                "Vote_Target": (
                                    my_vote_target
                                    if my_vote_target is not None
                                    else "No Vote"
                                ),
                                "Night_Target": (
                                    my_night_action_target
                                    if my_night_action_target is not None
                                    else "No Action"
                                ),
                                "Defense_Exist": 1 if defense_utterance else 0,
                            }
                        )

        except Exception as e:
            continue

# -----------------------------------------------------------------------------
# 3. 데이터프레임 생성 및 배치 처리
# -----------------------------------------------------------------------------
df = pd.DataFrame(data)

if not df.empty:
    df = df.sort_values(by=["Scenario", "Lambda"])

    # 배치 ID 생성 (20판 단위)
    df["Batch_ID"] = df.groupby(["Scenario", "Lambda"]).cumcount() // 20

    # 배치별 승률 계산 (행동 데이터는 배치로 묶기 애매하므로 df에 그대로 둠)
    batch_df = (
        df.groupby(["Scenario", "Lambda", "Batch_ID"])["Win_Value"].mean().reset_index()
    )
    batch_df.rename(columns={"Win_Value": "Win_Rate"}, inplace=True)

    print(f"✅ Data Extraction Complete!")
    print(f"Total Games Parsed: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
else:
    print("⚠️ No data loaded.")


# =============================================================================
# [Graph 1] 승률 분포 (KDE Plot)
# =============================================================================
scenarios = sorted(batch_df["Scenario"].unique())
target_lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]  # 비교할 람다 값들

# 2. 2x2 서브플롯 생성
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()  # 1차원으로 펴기

# 3. 반복문으로 4개 그래프 그리기
for i, scenario in enumerate(scenarios):
    ax = axes[i]

    # 해당 시나리오 및 타겟 람다 데이터만 필터링
    subset = batch_df[
        (batch_df["Scenario"] == scenario) & (batch_df["Lambda"].isin(target_lambdas))
    ]

    # KDE Plot 그리기
    sns.kdeplot(
        data=subset,
        x="Win_Rate",
        hue="Lambda",
        palette="Set2",
        fill=True,
        alpha=0.3,
        linewidth=2,
        common_norm=False,
        ax=ax,
    )

    # 그래프 데코레이션
    ax.set_title(f"{scenario.upper()}", fontsize=16, fontweight="bold")
    ax.set_xlim(0, 1.0)  # 승률 범위 고정 (매우 중요! 비교를 위해)
    ax.set_xlabel("Win Rate")
    ax.set_ylabel("Density")
    ax.axvline(
        0.5, color="red", linestyle="--", alpha=0.5, label="Balance (50%)"
    )  # 밸런스 기준선
    ax.grid(axis="y", linestyle=":", alpha=0.5)

# 4. 전체 제목 및 레이아웃 조정
plt.suptitle("Win Rate Distribution Analysis by Role & Lambda", fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / "1_win_rate_kde.png", dpi=300, bbox_inches="tight")
plt.close()


# =============================================================================
# [Graph 2] 승률 박스 플롯 (Box Plot)
# =============================================================================


# 1. 이상치 판별 및 데이터 분리 (Pandas Transform 활용)
def get_inlier_mask(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    return (x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)


# 전체 데이터셋에 대해 마스크 생성 (Scenario/Lambda 그룹별 계산)
mask = batch_df.groupby(["Scenario", "Lambda"])["Win_Rate"].transform(get_inlier_mask)

# 정상 데이터(Inliers)만 따로 저장
inliers_df = batch_df[mask]

# 2. 그래프 그리기 준비
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

scenarios = sorted(batch_df["Scenario"].unique())
palette = sns.color_palette("Blues", n_colors=len(batch_df["Lambda"].unique()))

for i, scenario in enumerate(scenarios):
    ax = axes[i]

    full_subset = batch_df[batch_df["Scenario"] == scenario]  # 박스플롯용 (전체)
    inlier_subset = inliers_df[
        inliers_df["Scenario"] == scenario
    ]  # 스트립플롯용 (정상만)

    local_min = full_subset["Win_Rate"].min()
    local_max = full_subset["Win_Rate"].max()

    # 위아래로 여백(padding) 주기
    padding = (local_max - local_min) * 0.1
    if padding == 0:
        padding = 0.05

    y_limit_min = local_min - padding
    y_limit_max = local_max + padding

    # 1. Box Plot (전체 데이터 -> 이상치는 X로 표시)
    sns.boxplot(
        data=full_subset,
        x="Lambda",
        y="Win_Rate",
        hue="Lambda",
        legend=False,
        ax=ax,
        palette=palette,
        width=0.6,
        linewidth=1.5,
        flierprops={
            "marker": "x",
            "markerfacecolor": "red",
            "markeredgecolor": "red",
            "markersize": 8,
        },
    )

    # 2. Strip Plot (정상 데이터만 -> 점으로 표시)
    sns.stripplot(
        data=inlier_subset,
        x="Lambda",
        y="Win_Rate",
        ax=ax,
        color="darkblue",
        alpha=0.4,
        jitter=True,
        size=4,
    )

    # 3. 꾸미기 및 축 적용
    ax.set_title(f"{scenario.upper()}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Lambda (Entropy Weight)")
    ax.set_ylabel("Win Rate (per 20 games batch)")
    ax.set_ylim(y_limit_min, y_limit_max)
    ax.grid(False)

plt.suptitle("Win Rate Analysis: Dynamic Scale per Role", fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / "2_win_rate_box_strip.png", dpi=300, bbox_inches="tight")
plt.close()


# =============================================================================
# [Graph 3] 발언 분포 (Bar Chart)
# =============================================================================

# -----------------------------------------------------------------------------
# 1. 메모리 효율적인 데이터 집계 (explode 대체)
# -----------------------------------------------------------------------------


def count_claims(series):
    # 리스트들의 리스트를 하나로 합쳐서 카운트
    all_claims = [item for sublist in series for item in sublist]
    return Counter(all_claims)


# 시나리오와 람다별로 발언 빈도 계산
grouped = (
    df.groupby(["Scenario", "Lambda"])["Self_Claim"]
    .apply(count_claims)
    .unstack(fill_value=0)
)

# 전체 발화 수 대비 비율(%) 계산
claim_ratios = grouped.div(grouped.sum(axis=1), axis=0) * 100

print("집계 완료!")

# -----------------------------------------------------------------------------
# 2. 그래프 스타일 및 색상 설정 (기존과 동일)
# -----------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
scenarios = sorted(df["Scenario"].unique())

claim_colors = {
    "Claim Citizen": "#3498db",  # 파랑
    "Claim Doctor": "#2ecc71",  # 초록
    "Claim Mafia": "#9b59b6",  # 보라
    "Claim Police": "#e74c3c",  # 빨강
    "Silence": "#34495e",  # 짙은 남색/회색
}

# -----------------------------------------------------------------------------
# 3. 그래프 그리기 (기존 로직 유지)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, scenario in enumerate(scenarios):
    if i >= len(axes):
        break
    ax = axes[i]

    if scenario in claim_ratios.index.get_level_values(0):
        # xs 대신 loc를 사용하여 안전하게 접근
        subset = claim_ratios.loc[scenario]

        # 색상 리스트 생성
        current_cols = subset.columns
        color_list = [claim_colors.get(c, "#333333") for c in current_cols]

        # 그래프 그리기
        subset.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=color_list,
            width=0.85,
            edgecolor="white",
            linewidth=0.8,
        )

        ax.set_title(f"Scenario: {scenario}", fontsize=15, fontweight="bold", pad=10)
        ax.set_xlabel("Lambda", fontsize=11)
        ax.set_ylabel("Percentage of Total Actions (%)", fontsize=11)
        ax.set_ylim(0, 100)

        plt.setp(ax.get_xticklabels(), rotation=0)
        if ax.get_legend():
            ax.get_legend().remove()

# 남는 subplot 숨기기
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

# 통합 범례 (모든 발언 종류 추출)
unique_claims = sorted(list(claim_colors.keys()))
handles = [
    plt.Rectangle((0, 0), 1, 1, color=claim_colors[label]) for label in unique_claims
]

fig.legend(
    handles,
    unique_claims,
    title="Claim Type (All Utterances)",
    loc="upper right",
    bbox_to_anchor=(0.98, 0.95),
    fontsize=11,
    title_fontsize=12,
    frameon=True,
)

plt.suptitle("Raw Distribution of All Claims", fontsize=20, y=0.98)
plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig(PLOT_DIR / "3_claim_distribution_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================================================
# [Graph 4] 방사형 그래프 (Radar Chart)
# =============================================================================

print("데이터 집계 및 방사형 그래프 전처리 중...")
categories = ["Claim Mafia", "Claim Citizen", "Claim Doctor", "Claim Police", "Silence"]
N = len(categories)


def get_claim_counts(series):
    # 리스트를 풀어서 카운트하고, 없는 카테고리는 0으로 채움
    all_claims = [item for sublist in series for item in sublist]
    counts = Counter(all_claims)
    return pd.Series({cat: counts.get(cat, 0) for cat in categories})


# 그룹별 집계 및 비율(%) 변환
claim_stats = (
    df.groupby(["Scenario", "Lambda"])["Self_Claim"]
    .apply(get_claim_counts)
    .unstack(fill_value=0)
)
claim_ratios = claim_stats.div(claim_stats.sum(axis=1), axis=0) * 100

# -----------------------------------------------------------------------------
# 2. 오각형(방사형) 그래프 설정
# -----------------------------------------------------------------------------
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # 폐곡선 생성

sns.set_theme(style="white")
scenarios = sorted(df["Scenario"].unique())
lambdas = sorted(df["Lambda"].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(lambdas)))

# -----------------------------------------------------------------------------
# 3. 그래프 시각화
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 15), subplot_kw=dict(polar=True))
axes = axes.flatten()

for i, scenario in enumerate(scenarios):
    if i >= len(axes):
        break
    ax = axes[i]

    # 그래프 기본 설정
    ax.set_theta_offset(pi / 2)  # 12시 방향 시작
    ax.set_theta_direction(-1)  # 시계 방향
    ax.set_frame_on(False)  # 외곽 테두리 제거
    ax.grid(False)  # 기본 그리드 제거

    # 맞춤형 배경 그리드 (오각형 가이드라인)
    grid_levels = [20, 40, 60, 80, 100]
    for g in grid_levels:
        ax.plot(
            angles,
            [g] * (N + 1),
            color="lightgrey",
            linewidth=0.7,
            linestyle="--",
            zorder=0,
        )
        ax.text(0, g, f"{g}", color="grey", fontsize=8, ha="center", va="center")

    # Spoke (축) 그리기
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 100], color="lightgrey", linewidth=0.8, zorder=0)

    # 데이터 플로팅
    if scenario in claim_ratios.index.get_level_values(0):
        subset = claim_ratios.loc[scenario]
        for j, lam in enumerate(lambdas):
            if lam in subset.index:
                values = subset.loc[lam].values.tolist()
                values += values[:1]  # 폐곡선

                ax.plot(
                    angles,
                    values,
                    linewidth=2,
                    label=f"λ={lam}",
                    color=colors[j],
                    alpha=0.9,
                )
                ax.fill(angles, values, color=colors[j], alpha=0.05)

    # 라벨 설정
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)  # 여유 공간
    ax.set_title(f"Scenario: {scenario}", size=18, weight="bold", pad=30)

# 범례 설정
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center")
plt.savefig(PLOT_DIR / "4_claim_radar.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================================================
# [Graph 5] Active Claim Stacked Bar
# =============================================================================
print("유효 발언 데이터 집계 중...")
active_categories = ["Claim Citizen", "Claim Doctor", "Claim Mafia", "Claim Police"]


def get_active_claim_counts(series):
    # 리스트를 풀어서 카운트하고, Silence는 제외
    all_claims = [item for sublist in series for item in sublist if item != "Silence"]
    counts = Counter(all_claims)
    # Silence를 제외한 유효 발언들만 시리즈로 반환
    return pd.Series({cat: counts.get(cat, 0) for cat in active_categories})


# 그룹별 집계
claim_stats = (
    df.groupby(["Scenario", "Lambda"])["Self_Claim"]
    .apply(get_active_claim_counts)
    .unstack(fill_value=0)
)

# 비율(%) 변환: 행(Row)의 합이 0인 경우(전부 침묵인 경우)를 대비해 fillna(0)
claim_ratios = claim_stats.div(claim_stats.sum(axis=1), axis=0).fillna(0) * 100

print("✅ 전처리 완료!")

# -----------------------------------------------------------------------------
# 2. 그래프 스타일 및 색상 설정
# -----------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
scenarios = sorted(df["Scenario"].unique())
claim_colors = {
    "Claim Citizen": "#3498db",  # 파랑
    "Claim Doctor": "#2ecc71",  # 초록
    "Claim Mafia": "#9b59b6",  # 보라
    "Claim Police": "#e74c3c",  # 빨강
}

# -----------------------------------------------------------------------------
# 3. 그래프 시각화
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, scenario in enumerate(scenarios):
    if i >= len(axes):
        break
    ax = axes[i]

    if scenario in claim_ratios.index.get_level_values(0):
        subset = claim_ratios.loc[scenario]

        # 실제 데이터에 존재하는 컬럼만 색상 매핑
        current_cols = subset.columns
        color_list = [claim_colors.get(c, "#333333") for c in current_cols]

        # 누적 막대 그래프 생성
        subset.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=color_list,
            width=0.8,
            edgecolor="white",
            linewidth=0.8,
        )

        ax.set_title(f"Scenario: {scenario}", fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("Reward Lambda ($\lambda$)", fontsize=12)
        ax.set_ylabel("Share of Active Claims (%)", fontsize=12)
        ax.set_ylim(0, 105)

        plt.setp(ax.get_xticklabels(), rotation=0)
        if ax.get_legend():
            ax.get_legend().remove()

# 빈 서브플롯 숨기기
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

# 통합 범례 설정
handles = [
    plt.Rectangle((0, 0), 1, 1, color=claim_colors[l]) for l in active_categories
]
fig.legend(
    handles,
    active_categories,
    title="Active Claims",
    loc="upper right",
    bbox_to_anchor=(0.98, 0.95),
)
plt.suptitle("Distribution of Active Role Claims", fontsize=20, y=0.98)
plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig(PLOT_DIR / "5_active_claim_stacked.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================================================
# [Graph 6] Entropy vs WinRate (Dual Axis)
# =============================================================================
print("지표 산출 중...")


def calculate_optimized_metrics(group):
    # A. 승률 계산
    win_rate = group["Win_Value"].mean()

    # B. 리스트 형태의 Self_Claim을 풀어서 엔트로피 계산 (Memory Safe)
    all_claims = [item for sublist in group["Self_Claim"] for item in sublist]
    if not all_claims:
        return pd.Series({"Win_Rate": win_rate, "Raw_Entropy": 0})

    counts = pd.Series(all_claims).value_counts(normalize=True)
    raw_entropy = entropy(counts, base=2)

    return pd.Series({"Win_Rate": win_rate, "Raw_Entropy": raw_entropy})


# 그룹화 수행
stats_df = (
    df.groupby(["Scenario", "Lambda"]).apply(calculate_optimized_metrics).reset_index()
)

# Y축 범위 설정을 위한 관측치 확인
max_ent = stats_df["Raw_Entropy"].max()
ent_limit = max_ent * 1.2 if max_ent > 0 else 1.0

# -----------------------------------------------------------------------------
# 2. 이중 축 시각화 (Dual Axis Line Plot)
# -----------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
scenarios = sorted(stats_df["Scenario"].unique())

fig, axes = plt.subplots(2, 2, figsize=(18, 13))
axes = axes.flatten()

for i, scenario in enumerate(scenarios):
    if i >= len(axes):
        break
    ax1 = axes[i]
    subset = stats_df[stats_df["Scenario"] == scenario].sort_values("Lambda")

    # --- [왼쪽 축: Strategic Entropy (Bits)] ---
    color_ent = "#d35400"  # Burnt Orange
    lns1 = ax1.plot(
        subset["Lambda"],
        subset["Raw_Entropy"],
        marker="s",
        markersize=8,
        color=color_ent,
        linewidth=3,
        label="Strategic Entropy (Bits)",
    )
    ax1.set_ylabel("Entropy (Bits)", color=color_ent, fontsize=14, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_ent)
    ax1.set_ylim(0, ent_limit)

    # --- [오른쪽 축: Win Rate (0.0-1.0)] ---
    ax2 = ax1.twinx()
    color_win = "#2c3e50"  # Midnight Blue
    lns2 = ax2.plot(
        subset["Lambda"],
        subset["Win_Rate"],
        marker="o",
        markersize=8,
        color=color_win,
        linewidth=3,
        label="Win Rate",
    )
    ax2.set_ylabel("Win Rate", color=color_win, fontsize=14, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_win)
    ax2.set_ylim(0, 1.05)  # 승률은 항상 0~1 고정
    ax2.grid(False)

    # --- [공통 설정] ---
    ax1.set_title(f"Scenario: {scenario}", fontsize=18, fontweight="bold", pad=15)
    ax1.set_xlabel("Reward Lambda ($\lambda$)", fontsize=13)

    # 범례 통합
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left", frameon=True, fontsize=11)

# 빈 서브플롯 제거
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.suptitle("Impact of Reward Weight on Entropy and Performance", fontsize=20, y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(PLOT_DIR / "6_entropy_winrate_dual.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================================================
# [Graph 7] Correlation Regression Plot
# =============================================================================
correlation_results = []

scenarios = sorted(stats_df["Scenario"].unique())

for scenario in scenarios:
    subset = stats_df[stats_df["Scenario"] == scenario]

    # 데이터가 2개 미만이면 상관계수 계산 불가
    if len(subset) > 1:
        # 피어슨 상관계수와 p-value 계산
        corr, p_value = pearsonr(subset["Raw_Entropy"], subset["Win_Rate"])
        correlation_results.append(
            {"Scenario": scenario, "Correlation": corr, "P-Value": p_value}
        )

corr_df = pd.DataFrame(correlation_results)

# -----------------------------------------------------------------------------
# 2. 상관관계 시각화 (Regression Plot)
# -----------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for i, scenario in enumerate(scenarios):
    if i >= len(axes):
        break
    ax = axes[i]

    subset = stats_df[stats_df["Scenario"] == scenario]

    # 상관계수 텍스트 추출
    row = corr_df[corr_df["Scenario"] == scenario]
    r_val = row["Correlation"].values[0] if not row.empty else 0
    p_val = row["P-Value"].values[0] if not row.empty else 0

    # 회귀선이 포함된 산점도 (Regression Plot)
    sns.regplot(
        data=subset,
        x="Raw_Entropy",
        y="Win_Rate",
        ax=ax,
        scatter_kws={"s": 100, "alpha": 0.7, "color": "#2c3e50"},
        line_kws={"color": "#e74c3c", "lw": 3},
    )

    # 그래프 내부 주석 (상관계수 및 p-value)
    text_str = f"Pearson $r$ = {r_val:.3f}\n$p$-value = {p_val:.4f}"
    ax.text(
        0.05,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 람다 값을 포인트 옆에 텍스트로 표시 (데이터 레이블링)
    for _, point in subset.iterrows():
        ax.text(
            point["Raw_Entropy"],
            point["Win_Rate"],
            f"$\lambda$={point['Lambda']}",
            fontsize=10,
            ha="right",
            va="bottom",
            alpha=0.8,
        )

    ax.set_title(f"Scenario: {scenario}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Strategic Entropy (Raw Bits)", fontsize=12)
    ax.set_ylabel("Win Rate (0~1)", fontsize=12)

    # ===== 수정된 부분 =====
    # 데이터 범위 기반 자동 스케일링 (여백 10% 추가)
    y_min = subset["Win_Rate"].min()
    y_max = subset["Win_Rate"].max()
    y_margin = (y_max - y_min) * 0.1  # 10% 여백
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    # =======================

# 빈 subplot 정리
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.suptitle("Pearson Correlation: Strategic Entropy vs Win Rate", fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / "7_correlation_scatter.png", dpi=300, bbox_inches="tight")
plt.close()

# =============================================================================
# [Graph 8] Diversity by Action Type
# =============================================================================
print("\n=== Pearson Correlation Statistical Report ===")
print(corr_df.to_string(index=False))


def calculate_diversity(items):
    """리스트 요소들의 섀넌 엔트로피 계산"""
    cleaned = []
    if isinstance(items, (list, pd.Series)):
        for x in items:
            if isinstance(x, list):
                cleaned.extend([str(i) for i in x])
            else:
                cleaned.append(str(x))

    if not cleaned:
        return 0.0

    counts = Counter(cleaned)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return entropy(probs, base=2)


# -----------------------------------------------------------------------------
# 2. 데이터 가공 (Long Format 변환)
# -----------------------------------------------------------------------------
print("데이터 구조 변환 중 (Wide -> Long)...")

df = df.sort_values(by=["Scenario", "Lambda", "Batch_ID"])
grouped = df.groupby(["Scenario", "Lambda", "Batch_ID"])

long_data = []

for (scen, lam, bid), group in grouped:
    if len(group) < 20:
        continue

    win_rate = group["Win_Value"].mean()

    # 3가지 행동에 대해 각각 엔트로피 계산 후 데이터 추가
    actions = {"Speech": "Self_Claim", "Vote": "Vote_Target", "Night": "Night_Target"}

    for action_label, col_name in actions.items():
        if col_name in group.columns:
            ent = calculate_diversity(group[col_name])
            long_data.append(
                {
                    "Scenario": scen,
                    "Lambda": lam,
                    "Win_Rate": win_rate,
                    "Entropy": ent,
                    "Action_Type": action_label,  # 색상 구분을 위한 라벨
                }
            )

melted_df = pd.DataFrame(long_data)

# -----------------------------------------------------------------------------
# 3. 시각화 (Action 통합 그래프)
# -----------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
scenarios = sorted(melted_df["Scenario"].unique())

# 2x2 레이아웃
n_cols = 2
n_rows = (len(scenarios) + 1) // 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7 * n_rows))
axes = axes.flatten()

# 행동별 색상 지정 (가독성: Speech=빨강, Vote=파랑, Night=초록)
action_palette = {"Speech": "#e74c3c", "Vote": "#3498db", "Night": "#2ecc71"}

for i, scenario in enumerate(scenarios):
    ax = axes[i]
    subset = melted_df[melted_df["Scenario"] == scenario]

    if len(subset) < 2:
        ax.text(0.5, 0.5, "Not enough data", ha="center")
        continue

    # -------------------------------------------------------
    # A. 산점도 그리기 (Hue = Action_Type)
    # -------------------------------------------------------
    sns.scatterplot(
        data=subset,
        x="Entropy",
        y="Win_Rate",
        hue="Action_Type",
        palette=action_palette,
        style="Action_Type",  # 모양도 다르게 (동그라미, X, 네모)
        s=80,
        alpha=0.6,
        ax=ax,
    )

    # -------------------------------------------------------
    # B. 행동별 회귀선 (Trend Line) 추가
    # -------------------------------------------------------
    # 각 행동별로 추세선을 따로 그려서 기울기를 비교합니다.
    for action_type, color in action_palette.items():
        act_subset = subset[subset["Action_Type"] == action_type]
        if len(act_subset) > 1 and act_subset["Entropy"].std() > 0:
            sns.regplot(
                data=act_subset,
                x="Entropy",
                y="Win_Rate",
                scatter=False,
                ax=ax,
                line_kws={"color": color, "linestyle": "--", "lw": 2},
                label=f"{action_type} Trend",
            )

            # 상관계수 계산 (선택 사항: 그래프가 너무 복잡하면 주석 처리)
            # r, _ = pearsonr(act_subset['Entropy'], act_subset['Win_Rate'])
            # print(f"[{scenario}] {action_type} r={r:.3f}")

    # -------------------------------------------------------
    # C. 데코레이션
    # -------------------------------------------------------
    ax.set_title(f"Scenario: {scenario}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Batch Entropy (Diversity)", fontsize=12)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_ylim(-0.05, 1.05)

    # 범례 설정
    ax.legend(title="Action Type", loc="lower right")

# 남은 빈칸 처리
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.suptitle("Impact of Diversity by Action Type", fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / "8_action_diversity.png", dpi=300, bbox_inches="tight")
plt.close()
