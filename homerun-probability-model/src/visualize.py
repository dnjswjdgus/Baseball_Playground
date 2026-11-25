"""Figures for the home run model (seaborn style, every mark outlined in black)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

EDGE = "black"
MODEL_ORDER = ["Logistic Regression", "LightGBM", "TabPFN"]
FEATURE_LABEL = {
    "launch_speed": "Launch speed",
    "launch_angle": "Launch angle",
    "spray_angle": "Spray angle",
    "home_team": "Home team",
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


def _model_palette(models) -> dict:
    colors = sns.color_palette("Set2", len(MODEL_ORDER))
    return {m: colors[MODEL_ORDER.index(m)] for m in models}


def _rate_table(df: pd.DataFrame, x: str, y: str, x_step: float, y_step: float, min_count: int) -> pd.DataFrame:
    """Home run rate (%) per (y, x) cell; cells with fewer than `min_count` balls are left empty."""
    cells = pd.DataFrame(
        {
            "x": np.floor(df[x] / x_step) * x_step,
            "y": np.floor(df[y] / y_step) * y_step,
            "hr": df["events"],
        }
    )
    grouped = cells.groupby(["y", "x"])["hr"].agg(["mean", "size"]).reset_index()
    grouped.loc[grouped["size"] < min_count, "mean"] = np.nan
    return grouped.pivot(index="y", columns="x", values="mean").sort_index(ascending=False) * 100


def plot_home_run_rate_map(df: pd.DataFrame, path: Path | None = None, min_count: int = 20) -> plt.Figure:
    """Observed home run rate of home run candidate balls by batted-ball features."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))

    panels = [
        (_rate_table(df, "launch_speed", "launch_angle", 2.0, 3.0, min_count), "Launch speed (mph)", "Launch angle (deg)"),
        (_rate_table(df, "spray_angle", "launch_speed", 6.0, 2.0, min_count), "Spray angle (deg)", "Launch speed (mph)"),
    ]
    for ax, (table, xlabel, ylabel) in zip(axes, panels):
        sns.heatmap(
            table,
            ax=ax,
            cmap="rocket_r",
            vmin=0,
            vmax=100,
            linewidths=0.8,
            linecolor=EDGE,
            xticklabels=[f"{v:.0f}" for v in table.columns],
            yticklabels=[f"{v:.0f}" for v in table.index],
            cbar_kws={"shrink": 0.85, "pad": 0.03},
        )
        cbar = ax.collections[0].colorbar
        cbar.outline.set_edgecolor(EDGE)
        cbar.outline.set_linewidth(1.2)
        cbar.set_label("Home run rate (%)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)
        ax.set_facecolor("#eeeeee")  # empty cells: too few balls

    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_model_comparison(results: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Test F1, precision and recall for each model (grouped bars)."""
    set_style()
    long = results.melt(id_vars="model", value_vars=["test_f1", "precision", "recall"], var_name="metric", value_name="score")
    long["metric"] = long["metric"].map({"test_f1": "F1", "precision": "Precision", "recall": "Recall"})
    order = [m for m in MODEL_ORDER if m in set(results["model"])]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    sns.barplot(
        data=long, x="metric", y="score", hue="model", hue_order=order, palette=_model_palette(order),
        edgecolor=EDGE, linewidth=1.3, ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Test score")
    ax.set_title("Model comparison on the test set", fontweight="bold", loc="left")
    sns.despine(ax=ax, top=False, right=False)
    legend = ax.legend(title=None, frameon=True, edgecolor=EDGE, loc="upper right")
    legend.get_frame().set_linewidth(1.2)

    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_calibration_curves(y_test, proba_by_model: dict, path: Path | None = None) -> plt.Figure:
    """Predicted home run probability against the observed rate, per model."""
    set_style()
    palette = _model_palette(proba_by_model)

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.plot([0, 1], [0, 1], color=EDGE, linestyle="--", linewidth=1.6, zorder=1)
    for name, proba in proba_by_model.items():
        observed, predicted = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, color=palette[name], linewidth=2.2, zorder=2, label=name)
        ax.scatter(predicted, observed, s=85, color=palette[name], edgecolor=EDGE, linewidth=1.2, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted home run probability")
    ax.set_ylabel("Observed home run rate")
    ax.set_title("Calibration", fontweight="bold", loc="left")
    sns.despine(ax=ax, top=False, right=False)
    legend = ax.legend(frameon=True, edgecolor=EDGE, loc="upper left")
    legend.get_frame().set_linewidth(1.2)

    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_feature_importance(importance: pd.DataFrame, model_name: str, path: Path | None = None) -> plt.Figure:
    """Permutation importance (increase in Brier score) per feature."""
    set_style()
    data = importance.assign(label=importance["feature"].map(FEATURE_LABEL)).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(
        data=data, x="importance", y="label", hue="label", palette="crest", legend=False,
        edgecolor=EDGE, linewidth=1.3, ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=4, fontsize=10)
    ax.set_xlim(0, data["importance"].max() * 1.18)
    ax.set_xlabel("Increase in Brier score when shuffled")
    ax.set_ylabel("")
    ax.set_title(f"Feature importance ({model_name})", fontweight="bold", loc="left")
    sns.despine(ax=ax, top=False, right=False)

    fig.tight_layout()
    _save(fig, path)
    return fig
