"""Location+ modeling.

1. Out-of-fold XGBoost predicts how valuable each pitch location is for the pitcher (run value).
2. A 2D kernel density per pitcher x pitch type x batter side scores how tightly the pitcher
   repeats a location (log density).
3. Split-half reliability (Spearman-Brown) finds the sample size at which the metric stabilizes;
   stable pitch types are compared with the league average and rescaled to a 20-80 grade.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.stats import norm as stats_norm
from sklearn.model_selection import KFold

from src.config import (
    MIN_COVERAGE,
    MIN_STAB_PITCHES,
    N_FOLDS,
    STABILIZATION_THRESHOLDS,
    TARGET,
    XGB_PARAMS,
)


def train_oof_location_value(df: pd.DataFrame, features: list[str], target_col: str = TARGET, n_splits: int = N_FOLDS):
    """Predict each pitch's location run value with K-fold out-of-fold models (no leakage)."""
    X = df[features]
    y = df[target_col].to_numpy()

    oof_pred = np.full(len(df), np.nan, dtype=float)
    models = []
    for train_idx, valid_idx in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X):
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_pred[valid_idx] = model.predict(X.iloc[valid_idx])
        models.append(model)

    df = df.copy()
    df["predicted_value_oof"] = oof_pred
    return df, models


def build_feature_importance_table(models: list, features: list[str], importance_type: str = "gain") -> pd.DataFrame:
    """Per-fold XGBoost importance summarized as mean and standard deviation."""
    folds = []
    for model in models:
        score = model.get_booster().get_score(importance_type=importance_type)
        folds.append(pd.Series({f: float(score.get(f, 0.0)) for f in features}, dtype=float))

    table = pd.concat(folds, axis=1)
    table.columns = [f"fold_{i + 1}" for i in range(len(models))]
    table["mean"] = table.mean(axis=1)
    table["std"] = table.std(axis=1)
    return table.sort_values("mean", ascending=False)


def compute_density_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Log kernel density of each pitch within its (pitcher, pitch type, batter side) group.

    Groups with fewer than 3 pitches get no score; they are filled with the pitch-type mean and
    then the overall mean.
    """
    df = df.copy()
    x = df["plate_x"].to_numpy()
    z = df["plate_z"].to_numpy()
    scores = np.full(len(df), np.nan, dtype=float)

    for positions in df.groupby(["player_name", "pitch_type", "stand"], observed=True).indices.values():
        if len(positions) < 3:
            continue
        try:
            kde = gaussian_kde(np.vstack([x[positions], z[positions]]), bw_method="scott")
            scores[positions] = np.log(np.maximum(kde(np.vstack([x[positions], z[positions]])), np.finfo(float).tiny))
        except Exception:
            continue  # singular groups (e.g. identical coordinates) stay empty

    df["log_density_score"] = scores
    pitch_default = df.groupby("pitch_type", observed=False)["log_density_score"].mean()
    df["log_density_score"] = df["log_density_score"].fillna(df["pitch_type"].map(pitch_default).astype(float))
    df["log_density_score"] = df["log_density_score"].fillna(df["log_density_score"].mean())
    return df


def compute_zscore_by_pitch(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the log density within each pitch type."""
    df = df.copy()
    by_pitch = df.groupby("pitch_type", observed=True)["log_density_score"].agg(pitch_mean="mean", pitch_std="std").reset_index()
    df = df.merge(by_pitch, on="pitch_type", how="left")
    df["z_score_by_pitch"] = np.where(
        df["pitch_std"].fillna(0) == 0, 0, (df["log_density_score"] - df["pitch_mean"]) / df["pitch_std"]
    )
    return df.drop(columns=["pitch_mean", "pitch_std"])


def compute_location_run_value(df: pd.DataFrame) -> pd.DataFrame:
    """Weight the OOF run value by consistency: weight = 2 x CDF(z), clipped to 0.1-2.0.

    Repeating a good location is rewarded, and repeating a poor one is penalized more.
    """
    df = df.copy()
    df["consistency_percentile"] = stats_norm.cdf(df["z_score_by_pitch"])
    df["location_weight"] = (df["consistency_percentile"] * 2).clip(0.1, 2.0)
    df["location_run_value"] = df["predicted_value_oof"] * df["location_weight"]
    return df


def compute_stabilization_reliability(df: pd.DataFrame, thresholds=STABILIZATION_THRESHOLDS, min_pairs: int = 8) -> pd.DataFrame:
    """Split-half reliability by pitch type as the minimum pitch count grows.

    Pitches are split into odd and even sequence numbers per pitcher and pitch type; the two
    means are correlated and corrected with the Spearman-Brown formula.
    """
    work = df.copy()
    work["pitch_seq"] = work.groupby(["player_name", "pitch_type"], observed=True).cumcount()
    work["split_half"] = np.where(work["pitch_seq"] % 2 == 0, "even", "odd")

    halves = (
        work.groupby(["player_name", "pitch_type", "split_half"], observed=True)["location_run_value"]
        .mean()
        .unstack()
        .reset_index()
    )
    counts = work.groupby(["player_name", "pitch_type"], observed=True).size().reset_index(name="total_pitches")
    halves = halves.merge(counts, on=["player_name", "pitch_type"], how="left").dropna(subset=["even", "odd"])

    rows = []
    for pitch_type in halves["pitch_type"].dropna().unique().tolist():
        subset_pt = halves[halves["pitch_type"] == pitch_type]
        for threshold in thresholds:
            subset = subset_pt[subset_pt["total_pitches"] >= threshold]
            if len(subset) < min_pairs:
                continue
            r, _ = stats.pearsonr(subset["even"], subset["odd"])
            reliability = (2 * r) / (1 + r) if (1 + r) != 0 else 0
            rows.append({"pitch_type": pitch_type, "min_pitches": threshold, "reliability_R": reliability})
    return pd.DataFrame(rows)


def build_location_plus_leaderboard(
    df: pd.DataFrame, min_stab_pitches: int = MIN_STAB_PITCHES, min_coverage: float = MIN_COVERAGE
) -> pd.DataFrame:
    """Pitcher leaderboard with Location+ on a 20-80 scale (mean 50, standard deviation 10).

    Only pitch types with at least `min_stab_pitches` count, and only pitchers for whom at least
    `min_coverage` of their pitch types qualify. Each pitch type's value is compared with the
    league average, averaged by pitch count, ranked into a percentile, mapped to a normal
    quantile and rounded to the nearest 5.
    """
    counts = df.dropna(subset=["pitch_type"]).groupby(["player_name", "pitch_type"], observed=True).size().reset_index(name="pitch_count")
    total_types = counts.groupby("player_name", observed=True)["pitch_type"].nunique().rename("total_types")
    stable_types = counts.assign(is_stable=counts["pitch_count"] >= min_stab_pitches).groupby("player_name", observed=True)["is_stable"].sum().rename("stable_types")

    coverage = pd.concat([total_types, stable_types], axis=1).fillna(0)
    valid_pitchers = coverage.index[coverage["stable_types"] / coverage["total_types"] >= min_coverage]
    df_valid = df[df["player_name"].isin(valid_pitchers)].dropna(subset=["pitch_type"])

    by_pitch = (
        df_valid.groupby(["player_name", "pitch_type"], observed=True)
        .agg(pitch_count=("location_run_value", "count"), rv_mean=("location_run_value", "mean"))
        .reset_index()
    )
    by_pitch = by_pitch[by_pitch["pitch_count"] >= min_stab_pitches].copy()
    by_pitch["total_rv"] = by_pitch["rv_mean"] * by_pitch["pitch_count"]

    league = by_pitch.groupby("pitch_type", observed=True)[["total_rv", "pitch_count"]].sum()
    league["league_rv_mean"] = league["total_rv"] / league["pitch_count"]

    by_pitch = by_pitch.merge(league[["league_rv_mean"]], left_on="pitch_type", right_index=True, how="left")
    by_pitch["weighted_val_sum"] = (by_pitch["rv_mean"] - by_pitch["league_rv_mean"]) * by_pitch["pitch_count"]

    leaderboard = (
        by_pitch.groupby("player_name", observed=True)
        .agg(total_pitches=("pitch_count", "sum"), total_weighted_val=("weighted_val_sum", "sum"))
        .reset_index()
    )
    leaderboard["weighted_rv_above_avg"] = leaderboard["total_weighted_val"] / leaderboard["total_pitches"]

    rank_p = leaderboard["weighted_rv_above_avg"].rank(method="average", ascending=True).sub(0.5).div(len(leaderboard))
    z_norm = stats_norm.ppf(rank_p.clip(1e-6, 1 - 1e-6))
    leaderboard["location+"] = (((50 + 10 * z_norm) / 5).round() * 5).clip(20, 80).astype(int)

    return leaderboard[["player_name", "total_pitches", "weighted_rv_above_avg", "location+"]].sort_values(
        by=["location+", "weighted_rv_above_avg"], ascending=False
    ).reset_index(drop=True)
