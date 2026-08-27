"""강공 vs 번트 기대득점/득점확률 차이를 wOBA x wOBA 그리드 히트맵으로 시각화한다."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_decision_heatmap(decision_grid: pd.DataFrame):
    """current_woba(타석 타자) x next_woba(다음 타자) 기준으로
    강공이 유리한 구간(빨강)과 번트가 유리한 구간(파랑)을 함께 보여준다.
    """
    ev_matrix = decision_grid.pivot(index="next_woba", columns="current_woba", values="ev_diff")
    score_prob_matrix = decision_grid.pivot(index="next_woba", columns="current_woba", values="score_prob_diff")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ev_abs_max = np.abs(ev_matrix.values).max()
    sp_abs_max = np.abs(score_prob_matrix.values).max()

    x_pos = np.arange(len(ev_matrix.columns))
    y_pos = np.arange(len(ev_matrix.index))

    x_tick_idx = [i for i, x in enumerate(ev_matrix.columns) if np.isclose((x * 1000) % 50, 0)]
    y_tick_idx = [i for i, y in enumerate(ev_matrix.index) if np.isclose((y * 1000) % 50, 0)]

    im1 = axes[0].imshow(ev_matrix.values, origin="lower", aspect="auto", cmap="coolwarm", vmin=-ev_abs_max, vmax=ev_abs_max)
    axes[0].contour(x_pos, y_pos, ev_matrix.values, levels=[0], colors="black", linewidths=1.5, linestyles="dashed")
    axes[0].set_title("Swing - Bunt: Expected Runs")
    axes[0].set_xlabel("Current wOBA")
    axes[0].set_ylabel("Next wOBA")
    axes[0].set_xticks(x_tick_idx)
    axes[0].set_xticklabels([f"{ev_matrix.columns[i]:.3f}" for i in x_tick_idx], rotation=45)
    axes[0].set_yticks(y_tick_idx)
    axes[0].set_yticklabels([f"{ev_matrix.index[i]:.3f}" for i in y_tick_idx])
    plt.colorbar(im1, ax=axes[0]).set_label("EV Difference")

    im2 = axes[1].imshow(
        score_prob_matrix.values, origin="lower", aspect="auto", cmap="coolwarm", vmin=-sp_abs_max, vmax=sp_abs_max
    )
    axes[1].contour(x_pos, y_pos, score_prob_matrix.values, levels=[0], colors="black", linewidths=1.5, linestyles="dashed")
    axes[1].set_title("Swing - Bunt: Score Probability")
    axes[1].set_xlabel("Current wOBA")
    axes[1].set_ylabel("Next wOBA")
    axes[1].set_xticks(x_tick_idx)
    axes[1].set_xticklabels([f"{score_prob_matrix.columns[i]:.3f}" for i in x_tick_idx], rotation=45)
    axes[1].set_yticks(y_tick_idx)
    axes[1].set_yticklabels([f"{score_prob_matrix.index[i]:.3f}" for i in y_tick_idx])
    plt.colorbar(im2, ax=axes[1]).set_label("Score Probability Difference")

    plt.tight_layout()
    return fig
