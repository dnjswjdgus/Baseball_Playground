"""Run the full swing-versus-bunt pipeline and write the figures.

    python main.py                      # 2025 regular season, figures saved to ./figures
    python main.py --start 2025-04-01 --end 2025-04-30
"""

import argparse

import matplotlib

matplotlib.use("Agg")

from src.config import FIGURE_DIR, SEASON_END, SEASON_START, WOBA_GRID
from src.modeling import (
    build_batter_event_profile,
    build_decision_grid,
    build_event_state_value,
    compare_swing_vs_bunt,
    estimate_batter_woba,
    estimate_event_probs,
)
from src.preprocessing import (
    build_pa_table,
    build_run_expectancy_table,
    build_score_probability_table,
    build_transition_table,
    load_statcast,
)
from src.visualize import plot_break_even, plot_decision_map, plot_event_profile, plot_state_values


def main(start: str, end: str) -> None:
    # 1. Data: pitches -> plate appearances -> state tables
    df_raw = load_statcast(start_dt=start, end_dt=end)
    pa_all = build_pa_table(df_raw)
    transitions = build_transition_table(pa_all)
    re_table = build_run_expectancy_table(pa_all)
    score_prob_table = build_score_probability_table(pa_all)
    print(f"{df_raw['game_pk'].nunique():,} games, {len(pa_all):,} plate appearances")

    # 2. Model: outcome probabilities by wOBA and the value of each outcome
    batter_woba = estimate_batter_woba(pa_all)
    profile = build_batter_event_profile(pa_all, batter_woba)
    event_state_ev = build_event_state_value(transitions, re_table.rename(columns={"RE": "value"}), "value")
    event_state_sp = build_event_state_value(
        transitions, score_prob_table.rename(columns={"score_prob": "value"}), "value", scored_value=1.0
    )
    grid = build_decision_grid(profile, event_state_ev, event_state_sp, WOBA_GRID)

    # 3. Headline numbers
    case = compare_swing_vs_bunt(profile, event_state_ev, event_state_sp, current_woba=0.299, next_woba=0.480)
    print(f"Swing better on expected runs in {(grid.ev_diff > 0).to_numpy().mean() * 100:.1f}% of combinations")
    print(f"Bunt better on score probability in {(grid.score_prob_diff < 0).to_numpy().mean() * 100:.1f}% of combinations")
    print(
        f"Current .299 / next .480: swing - bunt = {case['ev_diff']:+.3f} runs, "
        f"{case['score_prob_diff'] * 100:+.2f} pp score probability"
    )

    # 4. Figures
    FIGURE_DIR.mkdir(exist_ok=True)
    plot_decision_map(grid, FIGURE_DIR / "01_decision_map.png")
    plot_break_even(grid, [0.250, 0.300, 0.350, 0.400], FIGURE_DIR / "02_break_even.png")
    plot_state_values(re_table, score_prob_table, FIGURE_DIR / "03_state_values.png")
    levels = [0.250, 0.300, 0.350, 0.400, 0.450]
    plot_event_profile(estimate_event_probs(profile, levels), levels, FIGURE_DIR / "04_event_profile.png")
    print(f"Figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=SEASON_START, help="first date (YYYY-MM-DD)")
    parser.add_argument("--end", default=SEASON_END, help="last date (YYYY-MM-DD)")
    args = parser.parse_args()
    main(args.start, args.end)
