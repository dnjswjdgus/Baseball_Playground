"""Statcast 원자료를 PA(타석) 단위 상태-전이 데이터로 변환하고,
Run Expectancy(기대득점) 및 Score Probability(득점확률) 테이블을 생성한다.
"""

import pandas as pd
from pybaseball import statcast

PA_ID = ["game_pk", "at_bat_number"]

EVENT_CLASS_MAP = {
    "walk": "BB_HBP",
    "hit_by_pitch": "BB_HBP",
    "single": "SINGLE",
    "double": "DOUBLE",
    "triple": "TRIPLE",
    "home_run": "HOME_RUN",
    "strikeout": "STRIKEOUT",
    "grounded_into_double_play": "GIDP",
    "field_out": "BIP_OUT",
    "force_out": "BIP_OUT",
    "fielders_choice_out": "BIP_OUT",
    "fielders_choice": "BIP_OUT",
    "double_play": "BIP_OUT",
    "triple_play": "BIP_OUT",
    "sac_fly": "BIP_OUT",
    "sac_fly_double_play": "BIP_OUT",
    "strikeout_double_play": "BIP_OUT",
}


def load_statcast(start_dt: str, end_dt: str) -> pd.DataFrame:
    """지정 기간의 Statcast 투구 자료 중 정규시즌(game_type == 'R')만 반환한다."""
    df = statcast(start_dt=start_dt, end_dt=end_dt)
    return df[df["game_type"] == "R"].copy()


def map_event_class(event: str) -> str:
    """이벤트를 분석에 사용할 8개 클래스(BB_HBP/SINGLE/.../GIDP)로 축약한다."""
    return EVENT_CLASS_MAP.get(event, "EXCLUDED")


def build_pa_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """pitch-level 데이터를 PA(타석) 단위로 압축하고 시작/종료 상태(outs, bases)를 부여한다."""
    raw = df_raw.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()

    pa_start = raw.drop_duplicates(subset=PA_ID, keep="first").copy()
    pa_start = pa_start[["game_pk", "at_bat_number", "outs_when_up", "on_1b", "on_2b", "on_3b"]].copy()
    pa_start["start_outs"] = pa_start["outs_when_up"].astype(int)
    pa_start["start_bases"] = (
        pa_start["on_1b"].notna().astype(int).astype(str)
        + pa_start["on_2b"].notna().astype(int).astype(str)
        + pa_start["on_3b"].notna().astype(int).astype(str)
    )
    pa_start = pa_start[["game_pk", "at_bat_number", "start_outs", "start_bases"]].copy()

    pa_result = (
        raw.dropna(subset=["events"])
        .sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .drop_duplicates(subset=PA_ID, keep="last")
        .copy()
    )

    pa_all = pa_result[
        [
            "game_pk", "game_date", "inning", "inning_topbot", "at_bat_number",
            "batter", "player_name", "events", "bat_score", "post_bat_score",
        ]
    ].copy()

    pa_all = pa_all.merge(pa_start, on=["game_pk", "at_bat_number"], how="left")
    pa_all["runs_on_play"] = pa_all["post_bat_score"] - pa_all["bat_score"]
    pa_all["event_class"] = pa_all["events"].apply(map_event_class)

    pa_all["offense_id"] = (
        pa_all["game_pk"].astype(str) + "_" + pa_all["inning"].astype(str) + "_" + pa_all["inning_topbot"].astype(str)
    )
    pa_all = pa_all.sort_values(["game_pk", "at_bat_number"]).copy()
    pa_all["end_outs"] = pa_all.groupby("offense_id")["start_outs"].shift(-1).fillna(3).astype(int)
    pa_all["end_bases"] = pa_all.groupby("offense_id")["start_bases"].shift(-1).fillna("000")

    return pa_all


def build_transition_table(pa_all: pd.DataFrame) -> pd.DataFrame:
    """(시작상태, 이벤트) -> (종료상태) 전이확률표를 생성한다."""
    pa = pa_all[pa_all["event_class"] != "EXCLUDED"].copy()

    transition_counts = (
        pa.groupby(["start_outs", "start_bases", "event_class", "runs_on_play", "end_outs", "end_bases"])
        .size()
        .reset_index(name="n")
    )
    transition_counts["total"] = transition_counts.groupby(["start_outs", "start_bases", "event_class"])["n"].transform("sum")
    transition_counts["prob"] = transition_counts["n"] / transition_counts["total"]
    return transition_counts


def build_run_expectancy_table(pa_all: pd.DataFrame) -> pd.DataFrame:
    """(outs, bases) 상태별 이닝 종료까지의 평균 득점(Run Expectancy)을 계산한다."""
    inning_end_score = pa_all.groupby("offense_id")["post_bat_score"].last().reset_index(name="inning_end_score")
    pa_re = pa_all.merge(inning_end_score, on="offense_id", how="left")
    pa_re["runs_to_end_inning"] = pa_re["inning_end_score"] - pa_re["bat_score"]

    re_table = pa_re.groupby(["start_outs", "start_bases"])["runs_to_end_inning"].mean().reset_index(name="RE")
    re_table["RE"] = re_table["RE"].round(4)
    return re_table.sort_values(["start_outs", "start_bases"]).reset_index(drop=True)


def build_score_probability_table(pa_all: pd.DataFrame) -> pd.DataFrame:
    """(outs, bases) 상태별 이닝 내 최소 1점 이상 득점할 확률을 계산한다."""
    inning_end_score = pa_all.groupby("offense_id")["post_bat_score"].last().reset_index(name="inning_end_score")
    pa_re = pa_all.merge(inning_end_score, on="offense_id", how="left")
    pa_re["runs_to_end_inning"] = pa_re["inning_end_score"] - pa_re["bat_score"]

    score_prob_table = (
        pa_re.assign(scored=(pa_re["runs_to_end_inning"] > 0).astype(int))
        .groupby(["start_outs", "start_bases"])["scored"]
        .mean()
        .reset_index(name="score_prob")
    )
    return score_prob_table.sort_values(["start_outs", "start_bases"]).reset_index(drop=True)
