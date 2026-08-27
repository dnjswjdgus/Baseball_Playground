"""Statcast 투구 데이터에서 위치 가치(Location Value) 모델링에 필요한 컬럼을 정제한다."""

import pandas as pd
from pybaseball import statcast

REQUIRED_COLS = [
    "plate_x",
    "plate_z",
    "delta_pitcher_run_exp",
    "pitch_type",
    "stand",
    "p_throws",
    "balls",
    "strikes",
]

CATEGORICAL_COLS = ["stand", "p_throws", "pitch_type", "count"]
FEATURES = ["plate_x", "plate_z", "stand", "p_throws", "pitch_type", "count"]


def load_statcast(start_dt: str, end_dt: str) -> pd.DataFrame:
    """지정 기간의 Statcast pitch-level 데이터를 수집한다."""
    return statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)


def build_dataset(df_origin: pd.DataFrame) -> pd.DataFrame:
    """투수 관점 득점기대값(delta_pitcher_run_exp)을 타깃으로, 볼카운트 문자열과
    범주형 타입을 정리한 모델 입력 데이터셋을 만든다.
    """
    df = df_origin.copy()
    df = df.dropna(subset=REQUIRED_COLS).copy()

    df["count"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
    df["pitcher_run_value"] = df["delta_pitcher_run_exp"].astype(float)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    return df
