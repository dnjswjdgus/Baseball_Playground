"""Figures for the swing-versus-bunt analysis (seaborn style, every mark outlined in black)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from src.config import BUNT_STATE, EVENT_CLASSES, SWING_STATE
from src.modeling import DecisionGrid

EDGE = "black"
BASES_ORDER = ["000", "100", "010", "001", "110", "101", "011", "111"]
BASES_LABEL = ["Empty", "1B", "2B", "3B", "1B-2B", "1B-3B", "2B-3B", "Loaded"]
EVENT_LABEL = {
    "BB_HBP": "BB/HBP",
    "SINGLE": "1B",
    "DOUBLE": "2B",
    "TRIPLE": "3B",
    "HOME_RUN": "HR",
    "STRIKEOUT": "K",
    "BIP_OUT": "In-play out",
    "GIDP": "GIDP",
}
DIVERGING = sns.color_palette("vlag", as_cmap=True)  # red: swing is better, blue: bunt is better


def set_style() -> None:
    """Apply the shared seaborn theme."""
    sns.set_theme(style="white", context="notebook", font_scale=1.05)
    plt.rcParams.update(
        {
            "axes.edgecolor": EDGE,
            "axes.linewidth": 1.2,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
        }
    )


def _style_colorbar(ax, label: str) -> None:
    """Outline the colorbar and label it."""
    cbar = ax.collections[0].colorbar
    cbar.outline.set_edgecolor(EDGE)
    cbar.outline.set_linewidth(1.2)
    cbar.set_label(label)


def _save(fig: plt.Figure, path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor="white")


def plot_decision_map(grid: DecisionGrid, path: Path | None = None, step: int = 10) -> plt.Figure:
    """Swing minus bunt over current x next batter wOBA.

    Cells show every `step`-th grid point; the dashed black line is the exact break-even
    contour from the full-resolution grid (red cells favor swinging, blue cells favor bunting).
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    panels = [
        (grid.ev_diff, "Expected runs (swing - bunt)", "Runs", 1.0),
        (grid.score_prob_diff, "Score probability (swing - bunt)", "Percentage points", 100.0),
    ]

    for ax, (matrix, title, cbar_label, scale) in zip(axes, panels):
        coarse = matrix.iloc[::step, ::step].sort_index(ascending=False) * scale
        limit = np.abs(coarse.to_numpy()).max()
        sns.heatmap(
            coarse,
            ax=ax,
            cmap=DIVERGING,
            center=0,
            vmin=-limit,
            vmax=limit,
            linewidths=0.8,
            linecolor=EDGE,
            square=True,
            xticklabels=[f"{v:.2f}" for v in coarse.columns],
            yticklabels=[f"{v:.2f}" for v in coarse.index],
            cbar_kws={"shrink": 0.8, "pad": 0.03},
        )
        _style_colorbar(ax, cbar_label)

        # Exact zero contour from the fine grid, mapped to heatmap cell coordinates
        fine = matrix.sort_index(ascending=False)
        cell = coarse.columns[1] - coarse.columns[0]
        x = (fine.columns.to_numpy() - coarse.columns[0]) / cell + 0.5
        y = (coarse.index[0] - fine.index.to_numpy()) / cell + 0.5
        ax.contour(x, y, fine.to_numpy() * scale, levels=[0], colors=EDGE, linewidths=2.2, linestyles="--")

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel("Current batter wOBA")
        ax.set_ylabel("Next batter wOBA")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_state_values(re_table: pd.DataFrame, score_prob_table: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """The 24 base-out states: run expectancy (top) and probability of scoring (bottom)."""
    set_style()
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2))
    panels = [
        (re_table, "RE", "Run expectancy", "{:.2f}", "YlOrRd"),
        (score_prob_table.assign(score_prob=score_prob_table["score_prob"] * 100), "score_prob", "Probability of scoring (%)", "{:.0f}", "BuPu"),
    ]

    for ax, (table, value, title, fmt, cmap) in zip(axes, panels):
        matrix = table.pivot(index="start_outs", columns="start_bases", values=value)[BASES_ORDER].astype(float)
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            annot=matrix.map(fmt.format),
            fmt="",
            annot_kws={"fontsize": 10},
            linewidths=1.0,
            linecolor=EDGE,
            xticklabels=BASES_LABEL,
            yticklabels=[f"{o} out" for o in matrix.index],
            cbar_kws={"shrink": 0.9, "pad": 0.02},
        )
        _style_colorbar(ax, "")
        ax.set_title(title, fontweight="bold", loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="y", rotation=0)

        # Outline the two states being compared
        for outs, bases in (SWING_STATE, BUNT_STATE):
            ax.add_patch(
                Rectangle((BASES_ORDER.index(bases), outs), 1, 1, fill=False, edgecolor=EDGE, linewidth=3.5, clip_on=False)
            )

    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_event_profile(event_probs: pd.DataFrame, woba_levels, path: Path | None = None) -> plt.Figure:
    """Outcome mix of hitters at several wOBA levels (100% stacked bars)."""
    set_style()
    levels = [round(w, 3) for w in woba_levels]
    mix = event_probs.loc[levels, EVENT_CLASSES] * 100
    palette = sns.color_palette("Set2", len(EVENT_CLASSES))

    fig, ax = plt.subplots(figsize=(11, 4.6))
    left = np.zeros(len(levels))
    y = np.arange(len(levels))
    for event, color in zip(EVENT_CLASSES, palette):
        ax.barh(y, mix[event].to_numpy(), left=left, color=color, edgecolor=EDGE, linewidth=1.2, label=EVENT_LABEL[event], height=0.72)
        left += mix[event].to_numpy()

    ax.set_yticks(y, [f".{int(round(w * 1000)):03d}" for w in levels])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of plate appearances (%)")
    ax.set_ylabel("Batter wOBA")
    ax.set_title("Outcome mix by batter wOBA", fontweight="bold", loc="left")
    ax.invert_yaxis()
    sns.despine(ax=ax, left=False, bottom=False, top=False, right=False)
    ax.legend(ncol=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=True, edgecolor=EDGE, columnspacing=1.0, handlelength=1.2)

    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_break_even(grid: DecisionGrid, current_levels, path: Path | None = None, marker_step: int = 10) -> plt.Figure:
    """Swing minus bunt versus the next batter's wOBA for several current batters."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
    panels = [
        (grid.ev_diff, "Expected runs (swing - bunt)", "Runs", 1.0),
        (grid.score_prob_diff, "Score probability (swing - bunt)", "Percentage points", 100.0),
    ]
    palette = sns.color_palette("flare", len(current_levels))

    for ax, (matrix, title, ylabel, scale) in zip(axes, panels):
        for level, color in zip(current_levels, palette):
            series = matrix[round(level, 3)] * scale
            sns.lineplot(x=series.index, y=series.to_numpy(), ax=ax, color=color, linewidth=2.2, label=f".{int(round(level * 1000)):03d}")
            marked = series.iloc[::marker_step]
            ax.scatter(marked.index, marked.to_numpy(), s=70, color=color, edgecolor=EDGE, linewidth=1.2, zorder=3)
        ax.axhline(0, color=EDGE, linewidth=1.6, linestyle="--", zorder=1)
        ax.set_title(title, fontweight="bold", loc="left")
        ax.set_xlabel("Next batter wOBA")
        ax.set_ylabel(ylabel)
        ax.get_legend().set_title("Current wOBA")
        ax.get_legend().get_frame().set_edgecolor(EDGE)
        sns.despine(ax=ax, left=False, bottom=False, top=False, right=False)

    fig.tight_layout()
    _save(fig, path)
    return fig
