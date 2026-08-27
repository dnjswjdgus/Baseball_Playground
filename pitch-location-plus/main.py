"""Location+ 파이프라인 실행 스크립트.

python main.py 로 실행하면 2025 시즌 Statcast 데이터를 수집해
투구 위치 가치를 OOF XGBoost로 예측하고, KDE 기반 일관성 점수로 보정한 뒤
안정화된 구종만으로 투수별 Location+ (20-80 스케일) 리더보드를 산출한다.
"""

import matplotlib.pyplot as plt

from src.preprocessing import load_statcast, build_dataset, FEATURES
from src.modeling import (
    train_oof_location_value,
    build_feature_importance_table,
    compute_density_scores,
    compute_zscore_by_pitch,
    compute_location_run_value,
    compute_stabilization_reliability,
    build_location_plus_leaderboard,
)
from src.visualize import (
    plot_feature_importance,
    plot_zscore_violin,
    plot_stabilization_curve,
    plot_location_plus_distribution,
    print_normality_diagnostics,
    plot_qq,
)


def main():
    df_origin = load_statcast(start_dt="2025-03-18", end_dt="2025-09-28")
    df = build_dataset(df_origin)

    df, models = train_oof_location_value(df, FEATURES)
    fi_table = build_feature_importance_table(models, FEATURES, importance_type="gain")
    plot_feature_importance(fi_table, "XGBoost Feature Importance (gain): mean over folds")

    df = compute_density_scores(df)
    df = compute_zscore_by_pitch(df)
    plot_zscore_violin(df)

    df = compute_location_run_value(df)
    stab_df = compute_stabilization_reliability(df)
    plot_stabilization_curve(stab_df)

    leaderboard = build_location_plus_leaderboard(df, min_stab_pitches=60)
    print(leaderboard.head(50))

    plot_location_plus_distribution(leaderboard)
    print_normality_diagnostics(leaderboard)
    plot_qq(leaderboard)

    plt.show()


if __name__ == "__main__":
    main()
