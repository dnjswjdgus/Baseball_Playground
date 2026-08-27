"""강공 vs 번트 의사결정 분석 파이프라인 실행 스크립트.

python main.py 로 실행하면 2025 시즌 Statcast 데이터를 수집해
Run Expectancy / Score Probability 테이블을 만들고, wOBA 기반 기대득점 비교 히트맵을 출력한다.
"""

import numpy as np

from src.preprocessing import (
    load_statcast,
    build_pa_table,
    build_transition_table,
    build_run_expectancy_table,
    build_score_probability_table,
)
from src.modeling import (
    estimate_batter_woba,
    build_batter_event_profile,
    build_event_state_ev,
    build_event_state_score_prob,
    build_decision_grid,
)
from src.visualize import plot_decision_heatmap

import matplotlib.pyplot as plt


def main():
    df_raw = load_statcast(start_dt="2025-03-18", end_dt="2025-09-28")
    pa_all = build_pa_table(df_raw)

    transition_counts = build_transition_table(pa_all)
    re_table = build_run_expectancy_table(pa_all)
    score_prob_table = build_score_probability_table(pa_all)

    batter_woba = estimate_batter_woba(pa_all)
    batter_event_profile = build_batter_event_profile(pa_all, batter_woba)

    event_state_ev = build_event_state_ev(transition_counts, re_table)
    event_state_score_prob = build_event_state_score_prob(transition_counts, score_prob_table)

    woba_grid = np.round(np.arange(0.200, 0.500, 0.002), 3)
    decision_grid = build_decision_grid(
        batter_event_profile, event_state_ev, event_state_score_prob, woba_grid
    )

    plot_decision_heatmap(decision_grid)
    plt.show()


if __name__ == "__main__":
    main()
