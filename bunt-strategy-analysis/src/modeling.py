"""wOBA-based outcome probabilities and the swing-versus-bunt comparison.

Approach:
1. Estimate each batter's season wOBA and use a Gaussian kernel over wOBA to smooth the
   outcome mix (walk, single, ..., double play) of league hitters with a similar wOBA.
2. Weight each outcome by the value it creates from a base-out state (expected runs or the
   probability of scoring) and compare swinging away with a successful sacrifice bunt.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    BUNT_STATE,
    EVENT_CLASSES,
    KERNEL_BANDWIDTH,
    MIN_PA,
    SWING_STATE,
    WOBA_EXCLUDED_EVENTS,
    WOBA_WEIGHTS,
)


def estimate_batter_woba(pa_all: pd.DataFrame) -> pd.DataFrame:
    """Season wOBA per batter from FanGraphs weights."""
    pa = pa_all.copy()
    pa["woba_value"] = pa["events"].map(WOBA_WEIGHTS).fillna(0)
    pa["woba_denom"] = (~pa["events"].isin(WOBA_EXCLUDED_EVENTS)).astype(int)

    batter_woba = pa.groupby("batter").agg(woba_num=("woba_value", "sum"), woba_denom=("woba_denom", "sum")).reset_index()
    batter_woba["woba_est"] = batter_woba["woba_num"] / batter_woba["woba_denom"]
    return batter_woba.dropna(subset=["woba_est"]).copy()


def build_batter_event_profile(pa_all: pd.DataFrame, batter_woba: pd.DataFrame) -> pd.DataFrame:
    """Batter x outcome-class counts joined with each batter's wOBA."""
    pa = pa_all[pa_all["event_class"] != "EXCLUDED"]

    counts = pa.groupby(["batter", "event_class"]).size().reset_index(name="count")
    profile = pa[["batter"]].drop_duplicates().merge(pd.DataFrame({"event_class": EVENT_CLASSES}), how="cross")
    profile = profile.merge(counts, on=["batter", "event_class"], how="left")
    profile["count"] = profile["count"].fillna(0)

    model_pa = pa.groupby("batter").size().reset_index(name="model_pa")
    profile = profile.merge(model_pa, on="batter", how="left")
    profile = profile.merge(batter_woba[["batter", "woba_est", "woba_denom"]], on="batter", how="left")
    return profile.dropna(subset=["woba_est"]).copy()


def estimate_event_probs(
    profile: pd.DataFrame, woba_values, bandwidth: float = KERNEL_BANDWIDTH, min_pa: int = MIN_PA
) -> pd.DataFrame:
    """Outcome probabilities for hitters of a given wOBA (rows: wOBA, columns: outcome class).

    Each batter with at least `min_pa` plate appearances contributes his outcome counts with a
    Gaussian weight centered on the target wOBA, so similar hitters count the most.
    """
    data = profile[profile["woba_denom"] >= min_pa]
    counts = data.pivot(index="batter", columns="event_class", values="count")[EVENT_CLASSES]
    batter_woba = data.drop_duplicates("batter").set_index("batter").loc[counts.index, "woba_est"].to_numpy()

    targets = np.asarray(woba_values, dtype=float)
    weights = np.exp(-0.5 * ((batter_woba[None, :] - targets[:, None]) / bandwidth) ** 2)
    weighted = weights @ counts.to_numpy()
    probs = weighted / weighted.sum(axis=1, keepdims=True)
    return pd.DataFrame(probs, index=pd.Index(targets, name="woba"), columns=EVENT_CLASSES)


def build_event_state_value(
    transitions: pd.DataFrame, end_state_value: pd.DataFrame, value_col: str, scored_value: float | None = None
) -> pd.DataFrame:
    """Value created by each (start state, outcome class) pair.

    For expected runs the value is runs on the play plus the run expectancy of the end state.
    For the scoring probability, any run on the play makes the value 1 and otherwise it is the
    scoring probability of the end state. An end state of three outs is worth zero.
    """
    merged = transitions.merge(
        end_state_value.rename(columns={"start_outs": "end_outs", "start_bases": "end_bases", value_col: "end_value"}),
        on=["end_outs", "end_bases"],
        how="left",
    )
    merged["end_value"] = merged["end_value"].fillna(0)

    if scored_value is None:
        merged["branch_value"] = merged["runs_on_play"] + merged["end_value"]
    else:
        merged["branch_value"] = np.where(merged["runs_on_play"] > 0, scored_value, merged["end_value"])

    merged["weighted"] = merged["prob"] * merged["branch_value"]
    return merged.groupby(["start_outs", "start_bases", "event_class"])["weighted"].sum().reset_index(name="event_value")


def expected_value(event_probs: pd.DataFrame, event_state_value: pd.DataFrame, state: tuple) -> pd.Series:
    """Expected value of batting from `state` for every wOBA row of `event_probs`."""
    outs, bases = state
    subset = event_state_value[(event_state_value["start_outs"] == outs) & (event_state_value["start_bases"] == bases)]
    value_by_event = subset.set_index("event_class")["event_value"].reindex(EVENT_CLASSES).fillna(0)
    values = event_probs[EVENT_CLASSES].to_numpy() @ value_by_event.to_numpy(dtype=float)
    return pd.Series(values, index=event_probs.index)


@dataclass
class DecisionGrid:
    """Swing-minus-bunt differences over every (current, next) wOBA pair.

    Matrices are indexed by next-batter wOBA (rows) and current-batter wOBA (columns).
    A positive value favors swinging away and a negative value favors bunting.
    """

    ev_diff: pd.DataFrame
    score_prob_diff: pd.DataFrame
    swing_ev: pd.Series
    bunt_ev: pd.Series
    swing_score_prob: pd.Series
    bunt_score_prob: pd.Series


def compare_swing_vs_bunt(
    profile: pd.DataFrame,
    event_state_ev: pd.DataFrame,
    event_state_score_prob: pd.DataFrame,
    current_woba: float,
    next_woba: float,
    bandwidth: float = KERNEL_BANDWIDTH,
    min_pa: int = MIN_PA,
) -> dict:
    """Swing and bunt values for one specific (current, next) wOBA pair."""
    probs = estimate_event_probs(profile, [current_woba, next_woba], bandwidth, min_pa)
    current, following = probs.iloc[[0]], probs.iloc[[1]]

    result = {
        "swing_ev": expected_value(current, event_state_ev, SWING_STATE).iloc[0],
        "bunt_ev": expected_value(following, event_state_ev, BUNT_STATE).iloc[0],
        "swing_score_prob": expected_value(current, event_state_score_prob, SWING_STATE).iloc[0],
        "bunt_score_prob": expected_value(following, event_state_score_prob, BUNT_STATE).iloc[0],
    }
    result["ev_diff"] = result["swing_ev"] - result["bunt_ev"]
    result["score_prob_diff"] = result["swing_score_prob"] - result["bunt_score_prob"]
    return result


def build_decision_grid(
    profile: pd.DataFrame,
    event_state_ev: pd.DataFrame,
    event_state_score_prob: pd.DataFrame,
    woba_grid,
    bandwidth: float = KERNEL_BANDWIDTH,
    min_pa: int = MIN_PA,
) -> DecisionGrid:
    """Compare swinging with a successful bunt for every current/next wOBA combination.

    Swinging uses the current batter from the swing state. Bunting hands the next batter the
    bunt state, so the value of bunting depends only on the next batter's wOBA.
    """
    probs = estimate_event_probs(profile, woba_grid, bandwidth, min_pa)

    swing_ev = expected_value(probs, event_state_ev, SWING_STATE)
    bunt_ev = expected_value(probs, event_state_ev, BUNT_STATE)
    swing_sp = expected_value(probs, event_state_score_prob, SWING_STATE)
    bunt_sp = expected_value(probs, event_state_score_prob, BUNT_STATE)

    def difference(swing: pd.Series, bunt: pd.Series) -> pd.DataFrame:
        matrix = swing.to_numpy()[None, :] - bunt.to_numpy()[:, None]
        return pd.DataFrame(
            matrix,
            index=pd.Index(bunt.index, name="next_woba"),
            columns=pd.Index(swing.index, name="current_woba"),
        )

    return DecisionGrid(
        ev_diff=difference(swing_ev, bunt_ev),
        score_prob_diff=difference(swing_sp, bunt_sp),
        swing_ev=swing_ev,
        bunt_ev=bunt_ev,
        swing_score_prob=swing_sp,
        bunt_score_prob=bunt_sp,
    )
