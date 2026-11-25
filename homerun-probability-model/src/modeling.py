"""Home run probability models (Logistic Regression, LightGBM, TabPFN): training, evaluation, importance."""

import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    VALIDATION_SHARE_OF_TRAINVAL,
)


def split_dataset(df: pd.DataFrame, features: list[str], target: str = TARGET):
    """Stratified train (60%) / validation (20%) / test (20%) split."""
    X = df[features].copy()
    y = df[target].copy()
    X["home_team"] = X["home_team"].astype(str)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VALIDATION_SHARE_OF_TRAINVAL, random_state=RANDOM_STATE, stratify=y_trainval
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def as_category(X: pd.DataFrame) -> pd.DataFrame:
    """Copy of X with home_team as a categorical column (LightGBM and TabPFN input)."""
    X = X.copy()
    X["home_team"] = X["home_team"].astype("category")
    return X


def find_best_threshold(y_true, y_proba):
    """Search the classification threshold that maximizes F1 on the validation set."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        score = f1_score(y_true, (y_proba >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = threshold, score
    return best_threshold, best_f1


def evaluate_probability_model(model_name, y_val, val_proba, y_test, test_proba) -> dict:
    """Pick the threshold on validation, then report test F1, precision and recall."""
    threshold, _ = find_best_threshold(y_val, val_proba)
    test_pred = (test_proba >= threshold).astype(int)
    return {
        "model": model_name,
        "threshold": threshold,
        "test_f1": f1_score(y_test, test_pred, zero_division=0),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall_score(y_test, test_pred, zero_division=0),
        "test_brier": brier_score_loss(y_test, test_proba),
    }


def train_logistic_regression(X_train, y_train):
    """Baseline: standardized numeric features plus one-hot home team."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    return model.fit(X_train, y_train)


def train_lightgbm(X_train, y_train):
    """Gradient boosted trees with a native categorical home team."""
    model = LGBMClassifier(
        objective="binary", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    return model.fit(as_category(X_train), y_train, categorical_feature=["home_team"])


def train_tabpfn(X_train, y_train, device: str = "cuda"):
    """TabPFN foundation model. Requires the TABPFN_TOKEN environment variable."""
    from tabpfn import TabPFNClassifier

    if "TABPFN_TOKEN" not in os.environ:
        raise RuntimeError("Set the TABPFN_TOKEN environment variable before training TabPFN.")

    model = TabPFNClassifier(device=device)
    return model.fit(as_category(X_train), y_train)


def predict_proba_in_chunks(model, X_data: pd.DataFrame, chunk_size: int = 2000) -> np.ndarray:
    """Predict in chunks for models with memory limits such as TabPFN."""
    probas = [
        model.predict_proba(X_data.iloc[start : start + chunk_size])[:, 1]
        for start in range(0, len(X_data), chunk_size)
    ]
    return np.concatenate(probas)


def calculate_calibration_importance(
    model,
    X_data: pd.DataFrame,
    y_true,
    n_repeats: int = 5,
    chunk_size: int = 2000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Permutation importance by Brier score.

    Each feature is shuffled in turn; the increase in Brier score over the unshuffled data
    measures how much the model's probabilities rely on that feature.
    """
    rng = np.random.default_rng(random_state)
    baseline = brier_score_loss(y_true, predict_proba_in_chunks(model, X_data, chunk_size))

    rows = []
    for feature in X_data.columns:
        scores = []
        for _ in range(n_repeats):
            shuffled = X_data.copy()
            # Shuffle by position so categorical columns keep their dtype
            shuffled[feature] = shuffled[feature].iloc[rng.permutation(len(shuffled))].array
            scores.append(brier_score_loss(y_true, predict_proba_in_chunks(model, shuffled, chunk_size)))
        rows.append(
            {
                "feature": feature,
                "importance": np.mean(scores) - baseline,
                "baseline_brier": baseline,
                "permuted_brier": np.mean(scores),
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
