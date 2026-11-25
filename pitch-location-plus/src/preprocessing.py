"""Clean Statcast pitches into the table used to model pitch location value."""

import pandas as pd
from pybaseball import cache, statcast

from src.config import CATEGORICAL_COLS, REQUIRED_COLS


def load_statcast(start_dt: str, end_dt: str) -> pd.DataFrame:
    """Download Statcast pitch-level data (cached by pybaseball) and keep regular-season games only."""
    cache.enable()
    df = statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
    return df[df["game_type"] == "R"].copy()


def build_dataset(df_origin: pd.DataFrame) -> pd.DataFrame:
    """Model table: pitcher-perspective run value as the target, count string and categorical types."""
    df = df_origin.dropna(subset=REQUIRED_COLS).reset_index(drop=True)

    df["count"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
    df["pitcher_run_value"] = df["delta_pitcher_run_exp"].astype(float)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df
