"""투구 위치 가치(Location+) 모델링.

1. OOF(Out-of-Fold) XGBoost로 각 투구의 위치가 투수에게 유리한 정도(run value)를 예측
2. 구종 x 좌·우타자 그룹별 KDE로 "얼마나 일관된 위치에 던졌는지" 로그 밀도 점수화
3. 반분신뢰도(Spearman-Brown)로 지표가 안정화되는 최소 표본 수를 판정하고,
   안정 구종만 골라 리그 평균 대비 가중 위치 가치를 20-80 스케일 점수로 변환
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.stats import norm as stats_norm
from sklearn.model_selection import KFold

XGB_PARAMS = dict(
    enable_categorical=True,
    tree_method="hist",
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=30,
    subsample=0.8,
    random_state=42,
    n_jobs=1,
)


def train_oof_location_value(df: pd.DataFrame, features: list[str], target_col: str = "pitcher_run_value", n_splits: int = 5):
    """5-Fold OOF 예측으로 각 투구의 위치 run value를 추정한다 (검증셋 누수 없이 전체 데이터에 예측값 부여)."""
    X = df[features]
    y = df[target_col].to_numpy()

    oof_pred = np.full(shape=len(df), fill_value=np.nan, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []

    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y[tr_idx]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr)

        oof_pred[va_idx] = model.predict(X_va)
        models.append(model)

    df = df.copy()
    df["predicted_value_oof"] = oof_pred
    return df, models


def get_fold_importance(model, feature_names: list[str], importance_type: str) -> pd.Series:
    score = model.get_booster().get_score(importance_type=importance_type)
    return pd.Series({f: float(score.get(f, 0.0)) for f in feature_names}, dtype=float)


def build_feature_importance_table(models: list, features: list[str], importance_type: str = "gain") -> pd.DataFrame:
    """폴드별 XGBoost 변수 중요도를 평균/표준편차로 요약한다."""
    fold_imps = [get_fold_importance(m, features, importance_type) for m in models]
    fi = pd.concat(fold_imps, axis=1)
    fi.columns = [f"fold_{i + 1}" for i in range(len(models))]
    fi["mean"] = fi.mean(axis=1)
    fi["std"] = fi.std(axis=1)
    return fi.sort_values("mean", ascending=False)


def apply_pure_kde(group: pd.DataFrame) -> pd.DataFrame:
    """(투수, 구종, 타자손) 그룹 내 투구 좌표에 2D KDE를 적용해 로그 밀도 점수를 매긴다.
    표본이 적으면(<3) 점수 없이 NaN을 반환한다 (이후 구종 평균으로 보정).
    """
    group = group.copy()
    group["log_density_score"] = np.nan
    n = len(group)

    if n < 3:
        return group

    coords = np.vstack([group["plate_x"].to_numpy(), group["plate_z"].to_numpy()])
    try:
        kde = gaussian_kde(coords, bw_method="scott")
        densities = kde(coords)
        densities = np.maximum(densities, np.finfo(float).tiny)
        group["log_density_score"] = np.log(densities)
    except (np.linalg.LinAlgError, Exception):
        return group

    return group


def compute_density_scores(df: pd.DataFrame) -> pd.DataFrame:
    """투수 x 구종 x 타자손 단위로 KDE 로그 밀도 점수를 계산하고, 결측은 구종 평균 -> 전체 평균 순으로 보정한다."""
    df = df.groupby(["player_name", "pitch_type", "stand"], group_keys=False).apply(apply_pure_kde)

    pitch_default = df.loc[df["log_density_score"].notna()].groupby("pitch_type", observed=False)["log_density_score"].mean()
    df["log_density_score"] = df["log_density_score"].fillna(df["pitch_type"].map(pitch_default))
    df["log_density_score"] = df["log_density_score"].fillna(df["log_density_score"].mean())

    return df


def compute_zscore_by_pitch(df: pd.DataFrame) -> pd.DataFrame:
    """구종별 로그 밀도 점수의 평균/표준편차 기준으로 표준화(z-score)한다."""
    df = df.copy()
    stats_by_pitch = (
        df.groupby("pitch_type")["log_density_score"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "pitch_mean", "std": "pitch_std"})
    )
    df = df.merge(stats_by_pitch, on="pitch_type", how="left")
    df["z_score_by_pitch"] = np.where(
        df["pitch_std"].fillna(0) == 0, 0, (df["log_density_score"] - df["pitch_mean"]) / df["pitch_std"]
    )
    return df.drop(columns=["pitch_mean", "pitch_std"])


def compute_location_run_value(df: pd.DataFrame) -> pd.DataFrame:
    """z-score를 백분위 기반 가중치(0.1~2.0)로 바꿔 OOF run value에 곱한 최종 위치 가치를 만든다."""
    df = df.copy()
    df["consistency_percentile"] = stats_norm.cdf(df["z_score_by_pitch"])
    df["location_weight"] = (df["consistency_percentile"] * 2).clip(0.1, 2.0)
    df["location_run_value"] = df["predicted_value_oof"] * df["location_weight"]
    return df


def compute_stabilization_reliability(df: pd.DataFrame, thresholds=range(10, 410, 10), min_pairs: int = 8) -> pd.DataFrame:
    """구종별로 최소 투구 수 기준을 늘려가며 반분신뢰도(Spearman-Brown)를 계산해
    지표가 몇 개 투구부터 안정화되는지 판정한다.
    """
    df_stab = df.copy()
    df_stab["pitch_seq"] = df_stab.groupby(["player_name", "pitch_type"]).cumcount()
    df_stab["split_half"] = np.where(df_stab["pitch_seq"] % 2 == 0, "even", "odd")

    split_stats = (
        df_stab.groupby(["player_name", "pitch_type", "split_half"], observed=False)["location_run_value"]
        .mean()
        .unstack()
        .reset_index()
    )
    pitch_counts = df_stab.groupby(["player_name", "pitch_type"], observed=False).size().reset_index(name="total_pitches")
    split_stats = split_stats.merge(pitch_counts, on=["player_name", "pitch_type"], how="left").dropna(subset=["even", "odd"])

    target_pitches = split_stats["pitch_type"].dropna().unique().tolist()
    results = []

    for pt in target_pitches:
        df_pt = split_stats[split_stats["pitch_type"] == pt]
        for t in thresholds:
            subset = df_pt[df_pt["total_pitches"] >= t]
            if len(subset) < min_pairs:
                continue
            r, _ = stats.pearsonr(subset["even"], subset["odd"])
            reliability = (2 * r) / (1 + r) if (1 + r) != 0 else 0
            results.append({"pitch_type": pt, "min_pitches": t, "reliability_R": reliability})

    return pd.DataFrame(results)


def build_location_plus_leaderboard(df: pd.DataFrame, min_stab_pitches: int = 60, min_coverage: float = 0.5) -> pd.DataFrame:
    """안정화 기준(min_stab_pitches) 이상 던진 구종만으로 투수별 리그 평균 대비 가중 위치 가치를 계산하고
    20-80 스케일(평균 50, 표준편차 10)의 Location+ 점수로 변환한다.
    """
    player_pitch_counts = (
        df.dropna(subset=["pitch_type"]).groupby(["player_name", "pitch_type"], observed=True).size().reset_index(name="pitch_count")
    )
    pitcher_total_types = player_pitch_counts.groupby("player_name", observed=True)["pitch_type"].nunique().reset_index(name="total_types")
    pitcher_stable_types = (
        player_pitch_counts.assign(is_stable=player_pitch_counts["pitch_count"] >= min_stab_pitches)
        .groupby("player_name", observed=True)["is_stable"]
        .sum()
        .reset_index(name="stable_types")
    )

    pitcher_coverage = pitcher_total_types.merge(pitcher_stable_types, on="player_name", how="left")
    pitcher_coverage["stable_types"] = pitcher_coverage["stable_types"].fillna(0).astype(int)
    pitcher_coverage["coverage_ratio"] = pitcher_coverage["stable_types"] / pitcher_coverage["total_types"]

    valid_pitchers = pitcher_coverage.loc[pitcher_coverage["coverage_ratio"] >= min_coverage, "player_name"].unique()
    df_valid = df[df["player_name"].isin(valid_pitchers)].dropna(subset=["pitch_type"]).copy()

    player_pitch_rv = (
        df_valid.groupby(["player_name", "pitch_type"], observed=True)
        .agg(pitch_count=("location_run_value", "count"), rv_mean=("location_run_value", "mean"))
        .reset_index()
    )
    player_pitch_rv = player_pitch_rv[player_pitch_rv["pitch_count"] >= min_stab_pitches].copy()
    player_pitch_rv["total_rv"] = player_pitch_rv["rv_mean"] * player_pitch_rv["pitch_count"]

    league_avg = player_pitch_rv.groupby("pitch_type", observed=True)[["total_rv", "pitch_count"]].sum().reset_index()
    league_avg["league_rv_mean"] = league_avg["total_rv"] / league_avg["pitch_count"]
    league_avg = league_avg[["pitch_type", "league_rv_mean"]]

    df_q = player_pitch_rv.merge(league_avg, on="pitch_type", how="left")
    df_q["rv_above_avg"] = df_q["rv_mean"] - df_q["league_rv_mean"]
    df_q["weighted_val_sum"] = df_q["rv_above_avg"] * df_q["pitch_count"]

    player_final = (
        df_q.groupby("player_name", observed=True)
        .agg(total_pitches=("pitch_count", "sum"), total_weighted_val=("weighted_val_sum", "sum"))
        .reset_index()
    )
    player_final["weighted_rv_above_avg"] = player_final["total_weighted_val"] / player_final["total_pitches"]

    n_players = len(player_final)
    player_final["rank_p"] = player_final["weighted_rv_above_avg"].rank(method="average", ascending=True).sub(0.5).div(n_players)
    player_final["rank_p"] = player_final["rank_p"].clip(1e-6, 1 - 1e-6)
    player_final["z_norm"] = stats_norm.ppf(player_final["rank_p"])
    player_final["raw_location+"] = 50 + 10 * player_final["z_norm"]
    player_final["location+"] = ((player_final["raw_location+"] / 5).round() * 5).clip(20, 80).astype(int)

    player_final = player_final.drop(columns=["rank_p", "z_norm"])
    return player_final[["player_name", "total_pitches", "weighted_rv_above_avg", "location+"]].sort_values(
        by=["location+", "weighted_rv_above_avg"], ascending=False
    )
