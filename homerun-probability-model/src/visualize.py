"""모델 성능 비교(Calibration Curve)와 변수 중요도 시각화."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve


def plot_calibration_curves(y_test, proba_by_model: dict[str, "pd.Series"]):
    """모델별 예측확률이 실제 발생률과 얼마나 일치하는지(calibration) 비교한다."""
    plt.figure(figsize=(7, 7))

    for model_name, proba in proba_by_model.items():
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        plt.plot(prob_pred, prob_true, marker="o", label=model_name)

    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    plt.xlabel("Predicted Home Run Probability")
    plt.ylabel("Actual Home Run Rate")
    plt.title("Calibration Curve Comparison")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_feature_importance(importance_df: pd.DataFrame):
    """Permutation(Calibration 기반) 변수 중요도 막대그래프."""
    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.barh(importance_df["feature"], importance_df["importance"], edgecolor="black", linewidth=0.8, alpha=0.85)
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, f" {width:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Increase in Brier Score")
    ax.set_ylabel("Feature")
    ax.set_title("TabPFN Calibration-based Feature Importance")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig
