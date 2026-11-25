"""Figures for Location+ (seaborn style, every mark outlined in black)."""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import RELIABILITY_CUTOFF, SAVANT_COLORS, ZONE_X, ZONE_Z

EDGE = "black"
BRAND_RED = "#CE0E2D"
PITCH_NAME = {
    "FF": "Four-seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider", "ST": "Sweeper", "CU": "Curveball",
    "CH": "Changeup", "FS": "Splitter", "KC": "Knuckle curve", "SV": "Slurve", "KN": "Knuckleball",
}


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


def _save(fig: plt.Figure, path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor="white")


def _draw_zone(ax, linewidth: float = 2.4) -> None:
    """Strike zone and home plate from the catcher's view."""
    ax.add_patch(
        patches.Rectangle(
            (ZONE_X[0], ZONE_Z[0]), ZONE_X[1] - ZONE_X[0], ZONE_Z[1] - ZONE_Z[0],
            linewidth=linewidth, edgecolor=EDGE, facecolor="none", zorder=6,
        )
    )
    ax.add_patch(
        patches.Polygon(
            [[-0.708, 0.0], [-0.708, -0.2], [0.0, -0.4], [0.708, -0.2], [0.708, 0.0]],
            closed=True, facecolor="whitesmoke", edgecolor=EDGE, linewidth=1.4, clip_on=False, zorder=6,
        )
    )


def plot_location_value_maps(df: pd.DataFrame, pitch_types=("FF", "SL", "CH"), path: Path | None = None, min_count: int = 40) -> plt.Figure:
    """Mean predicted location value by pitch location, right-handed pitcher vs right-handed batter.

    Red cells are good for the pitcher and blue cells are bad (run value per pitch).
    """
    set_style()
    x_bins = np.arange(-2.0, 2.0001, 0.25)
    z_bins = np.arange(0.0, 4.5001, 0.25)

    fig, axes = plt.subplots(1, len(pitch_types), figsize=(5.4 * len(pitch_types), 6.4))
    for ax, pitch_type in zip(np.atleast_1d(axes), pitch_types):
        d = df.loc[
            (df["pitch_type"].astype(str) == pitch_type)
            & (df["p_throws"].astype(str) == "R")
            & (df["stand"].astype(str) == "R"),
            ["plate_x", "plate_z", "predicted_value_oof"],
        ].copy()
        d["x_bin"] = pd.cut(d["plate_x"], x_bins, labels=False, include_lowest=True)
        d["z_bin"] = pd.cut(d["plate_z"], z_bins, labels=False, include_lowest=True)
        grouped = d.dropna(subset=["x_bin", "z_bin"]).groupby(["z_bin", "x_bin"])["predicted_value_oof"].agg(["mean", "size"]).reset_index()
        grouped = grouped[grouped["size"] >= min_count]

        grid = np.full((len(z_bins) - 1, len(x_bins) - 1), np.nan)
        grid[grouped["z_bin"].astype(int), grouped["x_bin"].astype(int)] = grouped["mean"]
        limit = np.nanmax(np.abs(grid))

        mesh = ax.pcolormesh(
            x_bins, z_bins, grid, cmap=sns.color_palette("vlag", as_cmap=True),
            norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), edgecolors=EDGE, linewidth=0.5,
        )
        _draw_zone(ax)
        ax.set_facecolor("#eeeeee")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, 4.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(PITCH_NAME.get(pitch_type, pitch_type), fontweight="bold", pad=8)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        cbar.outline.set_edgecolor(EDGE)
        cbar.outline.set_linewidth(1.2)
        cbar.set_label("Run value for the pitcher")

    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_hot_zones(df: pd.DataFrame, leaderboard: pd.DataFrame, pitch_type: str = "FF", min_pitches: int = 400, path: Path | None = None) -> plt.Figure:
    """Where the best and the worst Location+ pitchers throw the same pitch type (density contours)."""
    set_style()
    counts = df.loc[df["pitch_type"].astype(str) == pitch_type].groupby("player_name", observed=True).size()
    eligible = leaderboard[leaderboard["player_name"].isin(counts[counts >= min_pitches].index)]
    picks = [("Highest Location+", eligible.iloc[0]), ("Lowest Location+", eligible.iloc[-1])]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.6), sharex=True, sharey=True)
    for ax, (label, row) in zip(axes, picks):
        d = df[(df["player_name"] == row["player_name"]) & (df["pitch_type"].astype(str) == pitch_type)]
        sns.kdeplot(
            data=d, x="plate_x", y="plate_z", fill=True, levels=8, thresh=0.04, cmap="Reds", alpha=0.9,
            linewidths=1.0, ax=ax, warn_singular=False,
        )
        sns.kdeplot(data=d, x="plate_x", y="plate_z", levels=8, thresh=0.04, color=EDGE, linewidths=0.9, ax=ax, warn_singular=False)
        _draw_zone(ax)
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-0.6, 4.8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"{label}\n{_display_name(row['player_name'])}  ({PITCH_NAME.get(pitch_type, pitch_type)}, n = {len(d)})", fontweight="bold")

    fig.tight_layout()
    _save(fig, path)
    return fig


def _display_name(name: str) -> str:
    """Statcast stores 'Last, First'."""
    return " ".join(reversed([part.strip() for part in name.split(",")]))


def plot_consistency_by_pitch(df: pd.DataFrame, min_pitch_count: int = 20000, path: Path | None = None) -> plt.Figure:
    """Distribution of the standardized location-density score by pitch type."""
    set_style()
    data = df[["pitch_type", "z_score_by_pitch"]].assign(pitch_type=df["pitch_type"].astype(str))
    counts = data["pitch_type"].value_counts()
    order = [p for p in counts.index if counts[p] > min_pitch_count]
    palette = {p: SAVANT_COLORS.get(p, "#999999") for p in order}

    fig, ax = plt.subplots(figsize=(11, 5.6))
    sns.violinplot(
        data=data[data["pitch_type"].isin(order)], x="pitch_type", y="z_score_by_pitch", hue="pitch_type", order=order,
        palette=palette, legend=False, inner="quartile", linewidth=1.3, linecolor=EDGE, cut=0, ax=ax,
    )
    ax.axhline(0, color=EDGE, linestyle="--", linewidth=1.6)
    ax.set_xlabel("")
    ax.set_xticks(range(len(order)), [PITCH_NAME.get(p, p) for p in order])
    ax.set_ylabel("Location consistency (z-score)")
    ax.set_title("How tightly pitchers repeat a location, by pitch type", fontweight="bold", loc="left")
    sns.despine(ax=ax, top=False, right=False)

    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_stabilization_curve(stab_df: pd.DataFrame, path: Path | None = None, top: int = 6) -> plt.Figure:
    """Split-half reliability against the minimum pitch count."""
    set_style()
    sizes = stab_df.groupby("pitch_type")["min_pitches"].count().sort_values(ascending=False)
    order = sizes.index[:top].tolist()
    palette = {p: SAVANT_COLORS.get(p, "#999999") for p in order}

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for pitch_type in order:
        d = stab_df[stab_df["pitch_type"] == pitch_type]
        ax.plot(d["min_pitches"], d["reliability_R"], color=palette[pitch_type], linewidth=2.3, label=PITCH_NAME.get(pitch_type, pitch_type))
        ax.scatter(d["min_pitches"], d["reliability_R"], s=60, color=palette[pitch_type], edgecolor=EDGE, linewidth=1.1, zorder=3)
    ax.axhline(RELIABILITY_CUTOFF, color=EDGE, linestyle="--", linewidth=1.6)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Minimum pitches per pitch type")
    ax.set_ylabel("Split-half reliability (Spearman-Brown)")
    ax.set_title("Location+ stabilizes with sample size", fontweight="bold", loc="left")
    sns.despine(ax=ax, top=False, right=False)
    legend = ax.legend(frameon=True, edgecolor=EDGE, loc="lower right", ncol=2)
    legend.get_frame().set_linewidth(1.2)

    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_leaderboard(leaderboard: pd.DataFrame, top_n: int = 10, path: Path | None = None) -> plt.Figure:
    """Distribution of Location+ grades (left) and the top pitchers (right)."""
    set_style()
    fig, (ax_hist, ax_top) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1.1]})

    grades = leaderboard["location+"].value_counts().reindex(range(20, 85, 5), fill_value=0)
    ax_hist.bar(grades.index, grades.to_numpy(), width=4.2, color=BRAND_RED, edgecolor=EDGE, linewidth=1.3)
    for grade, count in grades.items():
        ax_hist.text(
            grade, count + grades.max() * 0.015, f"{count}", ha="center", fontsize=9, zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
        )
    ax_hist.axvline(50, color=EDGE, linestyle="--", linewidth=1.6)
    ax_hist.set_xticks(grades.index)
    ax_hist.set_xlabel("Location+ (20-80 scale)")
    ax_hist.set_ylabel("Pitchers")
    ax_hist.set_title(f"Distribution ({len(leaderboard)} pitchers)", fontweight="bold", loc="left")
    sns.despine(ax=ax_hist, top=False, right=False)

    top = leaderboard.head(top_n).iloc[::-1]
    ax_top.barh(
        [_display_name(n) for n in top["player_name"]], top["weighted_rv_above_avg"] * 1000,
        color=sns.color_palette("flare", len(top)), edgecolor=EDGE, linewidth=1.3,
    )
    for i, (_, row) in enumerate(top.iterrows()):
        ax_top.text(row["weighted_rv_above_avg"] * 1000, i, f"  {row['location+']}", va="center", fontsize=10, fontweight="bold")
    ax_top.set_xlim(0, top["weighted_rv_above_avg"].max() * 1000 * 1.15)
    ax_top.set_xlabel("Weighted run value above average (x 1,000 per pitch)")
    ax_top.set_title("Top 10 (bar label: Location+)", fontweight="bold", loc="left")
    sns.despine(ax=ax_top, top=False, right=False)

    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_feature_importance(fi_df: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """XGBoost gain importance averaged over folds (error bars: standard deviation)."""
    set_style()
    d = fi_df.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(d.index, d["mean"], xerr=d["std"], color=sns.color_palette("crest", len(d)), edgecolor=EDGE, linewidth=1.3,
            error_kw={"ecolor": EDGE, "capsize": 4, "elinewidth": 1.2})
    ax.set_xlabel("Gain importance (mean over folds)")
    ax.set_title("What drives the location value model", fontweight="bold", loc="left")
    sns.despine(ax=ax, top=False, right=False)

    fig.tight_layout()
    _save(fig, path)
    return fig
