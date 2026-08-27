"""Location+ 파이프라인의 시각화: 변수 중요도, 카운트별 위치 가치 맵, 구종별 핫존, 안정화 곡선, 최종 분포."""

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import kurtosis, normaltest, skew

BRAND_RED = "#CE0E2D"

SAVANT_COLORS = {
    "FF": "#D22D49", "SI": "#F58A1F", "FC": "#933F2C", "CH": "#1DB954",
    "FS": "#3EAC9A", "FO": "#55C592", "SC": "#54C93F", "CU": "#00D1ED",
    "KC": "#6236CD", "CS": "#005AE0", "SL": "#EAEB29", "ST": "#E7B73A",
    "SV": "#92A7D6", "KN": "#3B44B3", "EP": "#7B8185", "FA": "#999999",
    "IN": "#999999", "PO": "#999999",
}


def _draw_home_plate(ax):
    home_plate = patches.Polygon(
        [[-0.708, 0.0], [-0.708, -0.2], [0.0, -0.4], [0.708, -0.2], [0.708, 0.0]],
        closed=True, facecolor="whitesmoke", edgecolor="black", linewidth=1.5, clip_on=False, zorder=3,
    )
    ax.add_patch(home_plate)


def plot_feature_importance(fi_df: pd.DataFrame, title: str):
    """XGBoost 변수 중요도(gain) 막대그래프."""
    d = fi_df.sort_values("mean", ascending=False).iloc[::-1]
    y = np.arange(len(d))
    means = d["mean"].to_numpy(dtype=float)

    vmin, vmax = float(np.min(means)), float(np.max(means))
    t = (means - vmin) / (vmax - vmin) if vmax > vmin else np.ones_like(means)
    alphas = 0.25 + 0.75 * t
    base_rgb = mcolors.to_rgb(BRAND_RED)
    bar_colors = [(*base_rgb, float(a)) for a in alphas]

    plt.figure(figsize=(10, max(4, 0.35 * len(d) + 2)))
    plt.barh(y, means, color=bar_colors, edgecolor="black", linewidth=1.0)
    plt.yticks(y, d.index.tolist())
    plt.xlabel("importance (mean)")
    plt.title(title)
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return plt.gcf()


def plot_location_value_map_by_count(df: pd.DataFrame, pitch_type: str, p_throws: str, stand: str):
    """볼카운트 12개 구간별로 스트라이크존 타일 평균 위치 가치를 히트맵으로 보여준다."""
    all_counts = ["0-0", "1-0", "2-0", "3-0", "0-1", "1-1", "2-1", "3-1", "0-2", "1-2", "2-2", "3-2"]
    x_bins = np.linspace(-2, 2, 9)
    z_bins = np.linspace(0.0, 4.5, 10)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2

    dff = df.loc[
        (df["pitch_type"].astype(str) == pitch_type) & (df["p_throws"].astype(str) == p_throws) & (df["stand"].astype(str) == stand),
        ["plate_x", "plate_z", "count", "predicted_value_oof"],
    ].copy()
    dff["x_bin"] = pd.cut(dff["plate_x"], bins=x_bins, include_lowest=True, labels=False)
    dff["z_bin"] = pd.cut(dff["plate_z"], bins=z_bins, include_lowest=True, labels=False)
    dff = dff.dropna(subset=["x_bin", "z_bin"]).astype({"x_bin": int, "z_bin": int})

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle(
        f"Empirical OOF Location Value Map by Count ({p_throws}HP vs {stand}HB | {pitch_type})\n"
        f"Red = Good for Pitcher (+), Blue = Bad for Pitcher (-)",
        fontsize=18, fontweight="bold", y=0.98,
    )

    for i, count in enumerate(all_counts):
        ax = axes[i // 4, i % 4]
        d = dff.loc[dff["count"].astype(str) == count, ["x_bin", "z_bin", "predicted_value_oof"]]

        Z = np.full((len(z_centers), len(x_centers)), np.nan, dtype=float)
        g = d.groupby(["z_bin", "x_bin"], as_index=False)["predicted_value_oof"].mean()
        Z[g["z_bin"].to_numpy(), g["x_bin"].to_numpy()] = g["predicted_value_oof"].to_numpy()

        max_abs = float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1e-9
        max_abs = max_abs or 1e-9
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

        mesh = ax.pcolormesh(x_bins, z_bins, Z, cmap="coolwarm", norm=norm, edgecolors="black", linewidth=0.5)

        for r in range(len(z_centers)):
            for c in range(len(x_centers)):
                val = Z[r, c]
                if np.isfinite(val):
                    color = "white" if abs(val) > (max_abs * 0.5) else "black"
                    ax.text(x_centers[c], z_centers[r], f"{val:+.3f}", ha="center", va="center", color=color, fontsize=7.5, fontweight="bold")

        ax.add_patch(patches.Rectangle((-1, 1), 2.0, 2.0, linewidth=2.5, edgecolor="black", facecolor="none", zorder=3))
        _draw_home_plate(ax)
        ax.set_title(f"Count: {count}", fontsize=14, fontweight="bold")
        ax.set_xlim(-2, 2)
        ax.set_ylim(0.0, 4.5)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02).set_label("Mean OOF Value", fontsize=9)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    return fig


def plot_hot_zone_kde(df: pd.DataFrame, target_pitcher: str):
    """특정 투수의 구종별 투구 위치를 KDE로 시각화한 '핫존' 맵."""
    pitcher_df = df.loc[df["player_name"] == target_pitcher].copy()
    pitch_counts = pitcher_df["pitch_type"].dropna().value_counts()
    pitch_types_sorted = pitch_counts.index.to_numpy()
    n_pitches = len(pitch_types_sorted)

    if n_pitches == 0:
        return None

    cols = 3
    rows = (n_pitches + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 7))
    axes = np.array(axes).ravel()

    plot_i = 0
    for pitch_type in pitch_types_sorted:
        pt_df = pitcher_df.loc[pitcher_df["pitch_type"] == pitch_type].copy()
        count = len(pt_df)
        if count < 1:
            continue

        ax = axes[plot_i]
        plot_i += 1

        if "log_density_score" in pt_df.columns:
            s_min, s_max = pt_df["log_density_score"].min(), pt_df["log_density_score"].max()
            if np.isfinite(s_min) and np.isfinite(s_max) and s_max != s_min:
                pt_df["vis_weight"] = (pt_df["log_density_score"] - s_min) / (s_max - s_min) + 0.1
            else:
                pt_df["vis_weight"] = 1.0
        else:
            pt_df["vis_weight"] = 1.0

        if count < 3:
            pt_df["plate_x"] = pt_df["plate_x"].to_numpy() + np.random.normal(0, 0.05, count)
            pt_df["plate_z"] = pt_df["plate_z"].to_numpy() + np.random.normal(0, 0.05, count)

        bw_val = 0.3 if count <= 20 else 0.7

        try:
            sns.kdeplot(
                data=pt_df, x="plate_x", y="plate_z", weights="vis_weight", fill=True, thresh=0.01,
                levels=100, cmap="RdBu_r", alpha=0.75, ax=ax, zorder=2, bw_adjust=bw_val, warn_singular=False,
            )
        except Exception:
            pass

        ax.add_patch(patches.Rectangle((-0.83, 1.5), 1.66, 2.0, linewidth=2.5, edgecolor="black", facecolor="none", zorder=6))
        h1, h2 = 1.5 + 2.0 / 3, 1.5 + 4.0 / 3
        v1, v2 = -0.83 + 1.66 / 3, -0.83 + 3.32 / 3
        for y in (h1, h2):
            ax.plot([-0.83, 0.83], [y, y], color="black", linestyle=":", linewidth=1, alpha=0.5, zorder=6)
        for x in (v1, v2):
            ax.plot([x, x], [1.5, 3.5], color="black", linestyle=":", linewidth=1, alpha=0.5, zorder=6)
        _draw_home_plate(ax)

        ax.set_title(f"{pitch_type} (n = {count})", fontsize=18, fontweight="bold", pad=10)
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-0.6, 4.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    for j in range(plot_i, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    fig.suptitle(f"Continuous Hot Zone Visualization: {target_pitcher}", fontsize=26, fontweight="bold", y=1.03)
    return fig


def plot_zscore_violin(df: pd.DataFrame, min_pitch_count: int = 100):
    """구종별 로케이션 일관성(z-score) 분포를 바이올린 플롯으로 비교한다."""
    pitch_counts = df["pitch_type"].value_counts()
    valid_pitches = pitch_counts[pitch_counts > min_pitch_count].index
    plot_df = df[df["pitch_type"].isin(valid_pitches)].copy()

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    pitch_order = list(valid_pitches)
    palette_map = {p: SAVANT_COLORS.get(p, "#999999") for p in pitch_order}

    sns.violinplot(data=plot_df, x="pitch_type", y="z_score_by_pitch", order=pitch_order, palette=palette_map, inner="quartile", linewidth=1.2, cut=0)
    plt.axhline(y=0, color="black", linestyle="--", linewidth=2, label="Mean 0")
    plt.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, label="Plus 1 Std")
    plt.axhline(y=-1.0, color="gray", linestyle=":", linewidth=1.5, label="Minus 1 Std")
    plt.title("Z Score Distribution of Pitch Location Density by Pitch Type", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Pitch Type", fontsize=14)
    plt.ylabel("Standardized Density Z Score", fontsize=14)
    plt.legend(loc="upper right")
    sns.despine(left=True)
    plt.tight_layout()
    return plt.gcf()


def plot_stabilization_curve(stab_df: pd.DataFrame):
    """최소 투구 수에 따른 반분신뢰도(Spearman-Brown) 곡선. 0.7선을 넘는 지점이 지표 안정화 시점."""
    pitch_order = sorted(stab_df["pitch_type"].unique().tolist())
    palette_map = {p: SAVANT_COLORS.get(p, "#999999") for p in pitch_order}

    plt.figure(figsize=(15, 10))
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=stab_df, x="min_pitches", y="reliability_R", hue="pitch_type", hue_order=pitch_order, palette=palette_map, marker="o", linewidth=2.5, alpha=0.8)
    plt.axhline(y=0.7, color="black", linestyle="--", linewidth=2, label="Stabilization Threshold")
    plt.title("Stabilization of Adjusted Location Value Percentile Based Weighting", fontsize=18, fontweight="bold")
    plt.xlabel("Minimum Pitches per Pitch Type", fontsize=14)
    plt.ylabel("Spearman Brown Reliability", fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Pitch Type")
    plt.tight_layout()
    return plt.gcf()


def plot_location_plus_distribution(leaderboard: pd.DataFrame):
    """Location+ (20-80 스케일) 최종 분포와 정규성 진단 요약."""
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    sns.histplot(leaderboard["location+"], bins=range(20, 86, 5), kde=True, color="#D22D49", alpha=0.6, edgecolor="white")

    plt.axvline(50, color="black", linestyle="--", linewidth=2, label="League Average 50")
    actual_mean = leaderboard["location+"].mean()
    plt.axvline(actual_mean, color="blue", linestyle="-", linewidth=1.5, label=f"Calculated Mean {actual_mean:.1f}")

    plt.title("Location Plus 20 80 Scale Distribution", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Location Plus Grade 20 80 Scale", fontsize=14)
    plt.ylabel("Number of Pitchers", fontsize=14)
    plt.xticks(range(20, 85, 5))
    plt.legend(fontsize=12)

    stats_text = (
        f"Total Pitchers: {len(leaderboard)}\n"
        f"Mean: {actual_mean:.1f}\n"
        f"Std Dev: {leaderboard['location+'].std():.1f}\n"
        f"Max: {leaderboard['location+'].max()}\n"
        f"Min: {leaderboard['location+'].min()}"
    )
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
    plt.tight_layout()
    return plt.gcf()


def print_normality_diagnostics(leaderboard: pd.DataFrame):
    """왜도/첨도/D'Agostino 정규성 검정 결과를 출력한다."""
    data = leaderboard["location+"].dropna()
    print(f"표본 수 {len(data)}")
    print(f"왜도 {float(skew(data)):.3f}")
    print(f"첨도 {float(kurtosis(data)):.3f}")
    _, p_value = normaltest(data)
    print(f"D Agostino K squared p value {p_value:.4e}")


def plot_qq(leaderboard: pd.DataFrame):
    """Location+ 분포의 정규성을 Q-Q Plot으로 확인한다."""
    data = leaderboard["location+"].dropna()
    plt.figure(figsize=(10, 6))
    sm.qqplot(data, line="s", ax=plt.gca())
    plt.title("Q Q Plot for location+", fontsize=15)
    plt.tight_layout()
    return plt.gcf()
