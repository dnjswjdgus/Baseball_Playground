"""Turn raw Statcast pitches into plate-appearance states and transition tables.

Outputs the 24 base-out Run Expectancy table, the probability of scoring at least
one run from each state, and the (state, event) -> end-state transition table.
"""

import pandas as pd
from pybaseball import cache, statcast

from src.config import EVENT_CLASS_MAP

PA_ID = ["game_pk", "at_bat_number"]


def load_statcast(start_dt: str, end_dt: str) -> pd.DataFrame:
    """Download Statcast pitch-level data and keep regular-season games only."""
    cache.enable()
    df = statcast(start_dt=start_dt, end_dt=end_dt)
    return df[df["game_type"] == "R"].copy()


def map_event_class(event: str) -> str:
    """Collapse a Statcast event into one of the eight model classes."""
    return EVENT_CLASS_MAP.get(event, "EXCLUDED")


def build_pa_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse pitches into one row per plate appearance with start and end states."""
    raw = df_raw.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()

    # Start state comes from the first pitch of each plate appearance
    pa_start = raw.drop_duplicates(subset=PA_ID, keep="first")[
        PA_ID + ["outs_when_up", "on_1b", "on_2b", "on_3b"]
    ].copy()
    pa_start["start_outs"] = pa_start["outs_when_up"].astype(int)
    pa_start["start_bases"] = (
        pa_start["on_1b"].notna().astype(int).astype(str)
        + pa_start["on_2b"].notna().astype(int).astype(str)
        + pa_start["on_3b"].notna().astype(int).astype(str)
    )
    pa_start = pa_start[PA_ID + ["start_outs", "start_bases"]]

    # Result comes from the last pitch that carries an event
    pa_result = (
        raw.dropna(subset=["events"])
        .drop_duplicates(subset=PA_ID, keep="last")
        .copy()
    )
    pa_all = pa_result[
        PA_ID
        + ["game_date", "inning", "inning_topbot", "batter", "player_name", "events", "bat_score", "post_bat_score"]
    ].merge(pa_start, on=PA_ID, how="left")

    pa_all["runs_on_play"] = pa_all["post_bat_score"] - pa_all["bat_score"]
    pa_all["event_class"] = pa_all["events"].apply(map_event_class)

    # The end state of a plate appearance is the start state of the next one in the same half-inning
    pa_all["offense_id"] = (
        pa_all["game_pk"].astype(str) + "_" + pa_all["inning"].astype(str) + "_" + pa_all["inning_topbot"].astype(str)
    )
    pa_all = pa_all.sort_values(PA_ID).copy()
    grouped = pa_all.groupby("offense_id")
    pa_all["end_outs"] = grouped["start_outs"].shift(-1).fillna(3).astype(int)
    pa_all["end_bases"] = grouped["start_bases"].shift(-1).fillna("000")

    return pa_all


def build_transition_table(pa_all: pd.DataFrame) -> pd.DataFrame:
    """Estimate P(runs scored, end state | start state, event class)."""
    pa = pa_all[pa_all["event_class"] != "EXCLUDED"]

    keys = ["start_outs", "start_bases", "event_class"]
    transitions = (
        pa.groupby(keys + ["runs_on_play", "end_outs", "end_bases"]).size().reset_index(name="n")
    )
    transitions["total"] = transitions.groupby(keys)["n"].transform("sum")
    transitions["prob"] = transitions["n"] / transitions["total"]
    return transitions


def _runs_to_end_of_inning(pa_all: pd.DataFrame) -> pd.DataFrame:
    """Attach the runs scored from each plate appearance to the end of the half-inning."""
    inning_end_score = pa_all.groupby("offense_id")["post_bat_score"].last().rename("inning_end_score")
    pa = pa_all.merge(inning_end_score, on="offense_id", how="left")
    pa["runs_to_end_inning"] = pa["inning_end_score"] - pa["bat_score"]
    return pa


def build_run_expectancy_table(pa_all: pd.DataFrame) -> pd.DataFrame:
    """Average runs scored to the end of the inning from each of the 24 base-out states."""
    pa = _runs_to_end_of_inning(pa_all)
    table = pa.groupby(["start_outs", "start_bases"])["runs_to_end_inning"].mean().reset_index(name="RE")
    table["RE"] = table["RE"].round(4)
    return table.sort_values(["start_outs", "start_bases"]).reset_index(drop=True)


def build_score_probability_table(pa_all: pd.DataFrame) -> pd.DataFrame:
    """Probability of scoring at least one more run in the inning from each base-out state."""
    pa = _runs_to_end_of_inning(pa_all)
    pa["scored"] = (pa["runs_to_end_inning"] > 0).astype(int)
    table = pa.groupby(["start_outs", "start_bases"])["scored"].mean().reset_index(name="score_prob")
    return table.sort_values(["start_outs", "start_bases"]).reset_index(drop=True)
