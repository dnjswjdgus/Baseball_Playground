"""Run the Location+ pipeline and write the figures.

    python main.py
"""

import matplotlib

matplotlib.use("Agg")

from scipy.stats import kurtosis, normaltest, skew

from src.config import FEATURES, FIGURE_DIR, MIN_STAB_PITCHES, SEASON_END, SEASON_START
from src.modeling import (
    build_feature_importance_table,
    build_location_plus_leaderboard,
    compute_density_scores,
    compute_location_run_value,
    compute_stabilization_reliability,
    compute_zscore_by_pitch,
    train_oof_location_value,
)
from src.preprocessing import build_dataset, load_statcast
from src.visualize import (
    plot_consistency_by_pitch,
    plot_feature_importance,
    plot_hot_zones,
    plot_leaderboard,
    plot_location_value_maps,
    plot_stabilization_curve,
)


def main() -> None:
    # 1. Data: regular-season pitches with a location, pitch type, count and run value
    df = build_dataset(load_statcast(SEASON_START, SEASON_END))
    print(f"{len(df):,} pitches")

    # 2. Location value: out-of-fold XGBoost
    df, models = train_oof_location_value(df, FEATURES)
    importance = build_feature_importance_table(models, FEATURES, importance_type="gain")

    # 3. Consistency: KDE density per pitcher x pitch type x batter side, weighted into the value
    df = compute_zscore_by_pitch(compute_density_scores(df))
    df = compute_location_run_value(df)

    # 4. Reliability and the Location+ leaderboard
    stabilization = compute_stabilization_reliability(df)
    leaderboard = build_location_plus_leaderboard(df, min_stab_pitches=MIN_STAB_PITCHES)
    print(leaderboard.head(10).to_string(index=False))

    grades = leaderboard["location+"]
    print(
        f"\n{len(leaderboard)} pitchers | mean {grades.mean():.1f} | std {grades.std():.1f} | "
        f"skew {skew(grades):.3f} | kurtosis {kurtosis(grades):.3f} | D'Agostino p = {normaltest(grades)[1]:.4f}"
    )

    # 5. Figures
    FIGURE_DIR.mkdir(exist_ok=True)
    plot_location_value_maps(df, path=FIGURE_DIR / "01_location_value_maps.png")
    plot_hot_zones(df, leaderboard, path=FIGURE_DIR / "02_hot_zones.png")
    plot_consistency_by_pitch(df, path=FIGURE_DIR / "03_consistency_by_pitch.png")
    plot_stabilization_curve(stabilization, path=FIGURE_DIR / "04_stabilization.png")
    plot_leaderboard(leaderboard, path=FIGURE_DIR / "05_leaderboard.png")
    plot_feature_importance(importance, path=FIGURE_DIR / "06_feature_importance.png")
    print(f"Figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
