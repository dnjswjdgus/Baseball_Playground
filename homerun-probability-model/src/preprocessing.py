"""Extract home run candidates from Statcast batted balls and build the model features."""

import numpy as np
import pandas as pd
from pybaseball import cache, statcast
from sklearn.preprocessing import LabelEncoder

from src.config import HC_X_ORIGIN, HC_Y_ORIGIN, SPRAY_ANGLE_SCALE


def load_statcast(start_dt: str, end_dt: str) -> pd.DataFrame:
    """Download Statcast pitch-level data for the given period (cached by pybaseball)."""
    cache.enable()
    return statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)


def build_dataset(df_origin: pd.DataFrame) -> pd.DataFrame:
    """Keep only batted balls that could have been home runs and build the model table.

    A ball is a candidate when its launch speed and angle are at least as large as the
    smallest values seen on an actual home run. Output columns: launch_speed, launch_angle,
    spray_angle, home_team (label encoded) and events (1 = home run).
    """
    df = df_origin[["hc_x", "hc_y", "launch_speed", "launch_angle", "events", "home_team"]].copy()

    # Physical lower bounds taken from actual home runs
    is_home_run = df["events"] == "home_run"
    min_hr_speed = df.loc[is_home_run, "launch_speed"].min()
    min_hr_angle = df.loc[is_home_run, "launch_angle"].min()
    df = df[(df["launch_angle"] >= min_hr_angle) & (df["launch_speed"] >= min_hr_speed)].copy()

    df["events"] = (df["events"] == "home_run").astype(int)

    # Spray angle from the field coordinates: 0 is straightaway center
    dx = df["hc_x"].astype(float) - HC_X_ORIGIN
    dy = HC_Y_ORIGIN - df["hc_y"].astype(float)
    df["spray_angle"] = np.degrees(np.arctan(dx / dy)) * SPRAY_ANGLE_SCALE
    df = df.drop(columns=["hc_x", "hc_y"]).dropna(how="any").reset_index(drop=True)

    df["home_team"] = LabelEncoder().fit_transform(df["home_team"])
    return df[["launch_speed", "launch_angle", "spray_angle", "home_team", "events"]]
