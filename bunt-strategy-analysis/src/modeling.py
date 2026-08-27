"""타자 wOBA 기반 이벤트 발생확률 추정, 그리고 강공/번트의 기대득점·득점확률 비교 모델.

핵심 아이디어:
1. 타자의 시즌 wOBA를 커널(가우시안) 가중치로 사용해, 비슷한 wOBA를 가진 리그 전체 타자들의
   이벤트 분포(BB_HBP/SINGLE/.../GIDP)를 스무딩하여 "해당 wOBA 수준 타자의 기대 이벤트 확률"을 추정한다.
2. 이벤트별 확률 x (해당 이벤트가 상황(outs, bases)에서 만드는 기대득점 or 득점확률)을 가중합하여
   강공(swing)과 번트(bunt) 각각의 기대값을 계산하고 비교한다.
"""

import numpy as np
import pandas as pd

WOBA_WEIGHTS = {
    "walk": 0.691,
    "hit_by_pitch": 0.722,
    "single": 0.882,
    "double": 1.252,
    "triple": 1.584,
    "home_run": 2.037,
}

EVENT_CLASSES = ["BB_HBP", "SINGLE", "DOUBLE", "TRIPLE", "HOME_RUN", "STRIKEOUT", "BIP_OUT", "GIDP"]


def estimate_batter_woba(pa_all: pd.DataFrame) -> pd.DataFrame:
    """FanGraphs 가중치를 이용해 타자별 시즌 wOBA를 추정한다."""

    def woba_value(event):
        return WOBA_WEIGHTS.get(event, 0)

    def woba_denom(event):
        if event in ["intent_walk", "catcher_interf", "truncated_pa", "sac_bunt", "sac_bunt_double_play"]:
            return 0
        return 1

    pa_all = pa_all.copy()
    pa_all["woba_value"] = pa_all["events"].apply(woba_value)
    pa_all["woba_denom"] = pa_all["events"].apply(woba_denom)

    batter_woba = (
        pa_all.groupby("batter").agg(woba_num=("woba_value", "sum"), woba_denom=("woba_denom", "sum")).reset_index()
    )
    batter_woba["woba_est"] = batter_woba["woba_num"] / batter_woba["woba_denom"]
    return batter_woba.dropna(subset=["woba_est"]).copy()


def build_batter_event_profile(pa_all: pd.DataFrame, batter_woba: pd.DataFrame) -> pd.DataFrame:
    """타자 x 이벤트클래스 조합별 발생 횟수와 타자의 wOBA를 결합한 프로필 테이블을 만든다."""
    pa = pa_all[pa_all["event_class"] != "EXCLUDED"].copy()

    batter_event_counts = pa.groupby(["batter", "event_class"]).size().reset_index(name="count")

    batters = pa[["batter"]].drop_duplicates()
    event_grid = pd.DataFrame({"event_class": EVENT_CLASSES})
    profile = batters.merge(event_grid, how="cross")
    profile = profile.merge(batter_event_counts, on=["batter", "event_class"], how="left")
    profile["count"] = profile["count"].fillna(0)

    batter_model_pa = pa.groupby("batter").size().reset_index(name="model_pa")
    profile = profile.merge(batter_model_pa, on="batter", how="left")
    profile = profile.merge(batter_woba[["batter", "woba_est", "woba_denom"]], on="batter", how="left")

    return profile.dropna(subset=["woba_est"]).copy()


def get_event_probs_from_woba(
    batter_event_profile: pd.DataFrame, target_woba: float, bandwidth: float = 0.015, min_pa: int = 300
) -> pd.DataFrame:
    """target_woba 주변 타자들을 가우시안 가중으로 스무딩해 이벤트클래스별 발생확률을 추정한다."""
    event_grid = pd.DataFrame({"event_class": EVENT_CLASSES})
    data = batter_event_profile[batter_event_profile["woba_denom"] >= min_pa].copy()

    data["woba_weight"] = np.exp(-0.5 * ((data["woba_est"] - target_woba) / bandwidth) ** 2)
    data["weighted_count"] = data["count"] * data["woba_weight"]

    event_probs = data.groupby("event_class")["weighted_count"].sum().reset_index()
    event_probs = event_grid.merge(event_probs, on="event_class", how="left")
    event_probs["weighted_count"] = event_probs["weighted_count"].fillna(0)
    event_probs["prob"] = event_probs["weighted_count"] / event_probs["weighted_count"].sum()

    return event_probs[["event_class", "prob"]]


def build_event_state_ev(transition_counts: pd.DataFrame, re_table: pd.DataFrame) -> pd.DataFrame:
    """(시작상태, 이벤트클래스)가 만드는 기대득점(현재 타석 득점 + 종료상태 RE)을 계산한다."""
    transition_ev = transition_counts.merge(
        re_table.rename(columns={"start_outs": "end_outs", "start_bases": "end_bases", "RE": "end_RE"}),
        on=["end_outs", "end_bases"],
        how="left",
    )
    transition_ev["end_RE"] = transition_ev["end_RE"].fillna(0)
    transition_ev["branch_ev"] = transition_ev["runs_on_play"] + transition_ev["end_RE"]

    return (
        transition_ev.assign(weighted_ev=transition_ev["prob"] * transition_ev["branch_ev"])
        .groupby(["start_outs", "start_bases", "event_class"])["weighted_ev"]
        .sum()
        .reset_index(name="event_ev")
    )


def build_event_state_score_prob(transition_counts: pd.DataFrame, score_prob_table: pd.DataFrame) -> pd.DataFrame:
    """(시작상태, 이벤트클래스)가 만드는 이닝 내 득점확률을 계산한다."""
    transition_sp = transition_counts.merge(
        score_prob_table.rename(columns={"start_outs": "end_outs", "start_bases": "end_bases", "score_prob": "end_score_prob"}),
        on=["end_outs", "end_bases"],
        how="left",
    )
    transition_sp["end_score_prob"] = transition_sp["end_score_prob"].fillna(0)
    transition_sp["branch_score_prob"] = np.where(
        transition_sp["runs_on_play"] > 0, 1, transition_sp["end_score_prob"]
    )

    return (
        transition_sp.assign(weighted_sp=transition_sp["prob"] * transition_sp["branch_score_prob"])
        .groupby(["start_outs", "start_bases", "event_class"])["weighted_sp"]
        .sum()
        .reset_index(name="event_score_prob")
    )


def calc_expected_value(
    batter_event_profile: pd.DataFrame,
    event_state_ev: pd.DataFrame,
    target_woba: float,
    start_outs: int,
    start_bases: str,
    bandwidth: float = 0.015,
    min_pa: int = 300,
) -> float:
    """특정 wOBA 타자가 특정 상황에서 타격했을 때의 기대득점."""
    event_probs = get_event_probs_from_woba(batter_event_profile, target_woba, bandwidth, min_pa)
    state_ev = event_state_ev[(event_state_ev["start_outs"] == start_outs) & (event_state_ev["start_bases"] == start_bases)]

    result = event_probs.merge(state_ev, on="event_class", how="left")
    result["event_ev"] = result["event_ev"].fillna(0)
    return (result["prob"] * result["event_ev"]).sum()


def calc_score_probability(
    batter_event_profile: pd.DataFrame,
    event_state_score_prob: pd.DataFrame,
    target_woba: float,
    start_outs: int,
    start_bases: str,
    bandwidth: float = 0.015,
    min_pa: int = 300,
) -> float:
    """특정 wOBA 타자가 특정 상황에서 타격했을 때 이닝 내 득점할 확률."""
    event_probs = get_event_probs_from_woba(batter_event_profile, target_woba, bandwidth, min_pa)
    state_sp = event_state_score_prob[
        (event_state_score_prob["start_outs"] == start_outs) & (event_state_score_prob["start_bases"] == start_bases)
    ]

    result = event_probs.merge(state_sp, on="event_class", how="left")
    result["event_score_prob"] = result["event_score_prob"].fillna(0)
    return (result["prob"] * result["event_score_prob"]).sum()


def compare_swing_vs_bunt(
    batter_event_profile: pd.DataFrame,
    event_state_ev: pd.DataFrame,
    event_state_score_prob: pd.DataFrame,
    current_woba: float,
    next_woba: float,
    bandwidth: float = 0.015,
    min_pa: int = 300,
) -> pd.DataFrame:
    """무사 1루(강공 시도)와 1사 2루(번트 성공 가정) 상황을 비교해
    강공이 유리한지 번트가 유리한지 기대득점/득점확률 기준으로 판단한다.
    """
    swing_ev = calc_expected_value(batter_event_profile, event_state_ev, current_woba, 0, "100", bandwidth, min_pa)
    bunt_ev = calc_expected_value(batter_event_profile, event_state_ev, next_woba, 1, "010", bandwidth, min_pa)
    swing_sp = calc_score_probability(batter_event_profile, event_state_score_prob, current_woba, 0, "100", bandwidth, min_pa)
    bunt_sp = calc_score_probability(batter_event_profile, event_state_score_prob, next_woba, 1, "010", bandwidth, min_pa)

    return pd.DataFrame(
        [
            {
                "current_woba": current_woba,
                "next_woba": next_woba,
                "swing_ev": swing_ev,
                "bunt_ev": bunt_ev,
                "ev_diff_swing_minus_bunt": swing_ev - bunt_ev,
                "swing_score_prob": swing_sp,
                "bunt_score_prob": bunt_sp,
                "score_prob_diff_swing_minus_bunt": swing_sp - bunt_sp,
            }
        ]
    )


def build_decision_grid(
    batter_event_profile: pd.DataFrame,
    event_state_ev: pd.DataFrame,
    event_state_score_prob: pd.DataFrame,
    woba_grid: np.ndarray,
    bandwidth: float = 0.015,
    min_pa: int = 300,
) -> pd.DataFrame:
    """타석에 들어선 타자(current_woba) x 다음 타자(next_woba) 조합 전체에 대해
    강공/번트 기대득점·득점확률 차이를 계산한다 (히트맵 원본 데이터).
    """
    rows = []
    for current_woba in woba_grid:
        for next_woba in woba_grid:
            result = compare_swing_vs_bunt(
                batter_event_profile, event_state_ev, event_state_score_prob, current_woba, next_woba, bandwidth, min_pa
            ).iloc[0]
            rows.append(
                {
                    "current_woba": current_woba,
                    "next_woba": next_woba,
                    "ev_diff": result["ev_diff_swing_minus_bunt"],
                    "score_prob_diff": result["score_prob_diff_swing_minus_bunt"],
                }
            )
    return pd.DataFrame(rows)
