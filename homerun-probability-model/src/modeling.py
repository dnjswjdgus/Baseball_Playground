"""홈런 확률 예측 모델 (Logistic Regression / LightGBM / TabPFN) 학습, 평가, 변수 중요도 산출."""

import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["launch_speed", "launch_angle", "spray_angle"]
CATEGORICAL_FEATURES = ["home_team"]


def split_dataset(df: pd.DataFrame, features: list[str], target: str = "events"):
    """Train(60%) / Validation(20%) / Test(20%) 로 층화 분할한다."""
    X = df[features].copy()
    y = df[target].copy()
    X["home_team"] = X["home_team"].astype(str)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def find_best_threshold(y_true, y_proba):
    """Validation F1 기준 최적 분류 임계값을 탐색한다."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold, best_f1 = 0.5, -1

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, threshold

    return best_threshold, best_f1


def evaluate_probability_model(model_name, y_val, val_proba, y_test, test_proba):
    """Validation에서 고른 임계값을 Test에 적용해 F1/precision/recall을 산출한다."""
    best_threshold, _ = find_best_threshold(y_val, val_proba)
    test_pred = (test_proba >= best_threshold).astype(int)

    result = {
        "model": model_name,
        "threshold": best_threshold,
        "test_f1": f1_score(y_test, test_pred, zero_division=0),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall_score(y_test, test_pred, zero_division=0),
    }
    return result, test_pred


def train_logistic_regression(X_train, y_train):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    X_train = X_train.copy()
    X_train["home_team"] = X_train["home_team"].astype("category")

    model = LGBMClassifier(objective="binary", class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, categorical_feature=["home_team"])
    return model


def train_tabpfn(X_train, y_train, device: str = "cuda"):
    """TabPFN 파운데이션 모델 학습. TABPFN_TOKEN 환경변수가 필요하다."""
    from tabpfn import TabPFNClassifier

    if "TABPFN_TOKEN" not in os.environ:
        raise RuntimeError("환경변수 TABPFN_TOKEN 을 설정한 뒤 실행하세요.")

    X_train = X_train.copy()
    X_train["home_team"] = X_train["home_team"].astype("category")

    model = TabPFNClassifier(device=device)
    model.fit(X_train, y_train)
    return model


def predict_proba_in_chunks(model, X_data: pd.DataFrame, chunk_size: int = 2000) -> np.ndarray:
    """TabPFN 등 메모리 제약이 있는 모델을 위한 청크 단위 예측."""
    probas = []
    for start_idx in range(0, len(X_data), chunk_size):
        chunk = X_data.iloc[start_idx : start_idx + chunk_size]
        probas.append(model.predict_proba(chunk)[:, 1])
    return np.concatenate(probas)


def calculate_calibration_importance(
    model, X_data: pd.DataFrame, y_true, baseline_brier: float, n_repeats: int = 5, chunk_size: int = 2000, random_state: int = 42
) -> pd.DataFrame:
    """Permutation importance: 변수를 셔플했을 때 Brier Score(보정 오차)가 얼마나 나빠지는지로 변수 중요도를 측정한다."""
    results = []
    rng = np.random.default_rng(random_state)

    for feature in X_data.columns:
        scores = []
        for _ in range(n_repeats):
            X_permuted = X_data.copy()
            X_permuted[feature] = rng.permutation(X_permuted[feature].values)
            permuted_proba = predict_proba_in_chunks(model, X_permuted, chunk_size)
            scores.append(brier_score_loss(y_true, permuted_proba))

        mean_brier = np.mean(scores)
        results.append(
            {
                "feature": feature,
                "importance": mean_brier - baseline_brier,
                "baseline_brier": baseline_brier,
                "permuted_brier": mean_brier,
            }
        )

    return pd.DataFrame(results).sort_values("importance", ascending=False).reset_index(drop=True)
