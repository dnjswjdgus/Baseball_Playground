"""Statcast 타구 데이터에서 홈런 후보군을 추출하고 모델 입력 피처를 생성한다."""

import copy

import numpy as np
import pandas as pd
from pybaseball import statcast
from sklearn.preprocessing import LabelEncoder

# 홈플레이트 기준 타구 방향 좌표계 원점 (Statcast hc_x/hc_y 좌표계 보정값)
HC_X_ORIGIN = 125.42
HC_Y_ORIGIN = 198.27

FEATURES = ["launch_speed", "launch_angle", "spray_angle", "home_team"]
TARGET = "events"


def load_statcast(start_dt: str, end_dt: str) -> pd.DataFrame:
    """지정 기간의 Statcast pitch-level 데이터를 수집한다."""
    return statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)


def build_dataset(df_origin: pd.DataFrame) -> pd.DataFrame:
    """타구 속도/각도/방향 기준으로 '홈런이 될 수 있었던 타구'만 남기고,
    launch_speed / launch_angle / spray_angle / home_team -> events(홈런 여부) 데이터셋을 만든다.
    """
    df = copy.deepcopy(df_origin)
    df = df[["hc_x", "hc_y", "launch_speed", "launch_angle", "events", "home_team"]].copy()

    # 실제 홈런이 나온 타구의 최소 속도/각도를 물리적 하한선으로 사용해 후보군을 필터링
    min_hr_speed = df.loc[df["events"] == "home_run", "launch_speed"].min()
    min_hr_angle = df.loc[df["events"] == "home_run", "launch_angle"].min()
    df = df[(df["launch_angle"] >= min_hr_angle) & (df["launch_speed"] >= min_hr_speed)].copy()

    df["events"] = df["events"].map({"home_run": 1}).fillna(0).astype(int)

    df[["hc_x", "hc_y"]] = df[["hc_x", "hc_y"]].astype(float)
    dx = df["hc_x"] - HC_X_ORIGIN
    dy = HC_Y_ORIGIN - df["hc_y"]
    df["spray_angle"] = np.degrees(np.arctan(dx / dy)) * 0.75
    df = df.drop(columns=["hc_x", "hc_y"])

    df = df.dropna(how="any").reset_index(drop=True)

    le = LabelEncoder()
    df["home_team"] = le.fit_transform(df["home_team"])

    return df[["launch_speed", "launch_angle", "spray_angle", "home_team", "events"]]
