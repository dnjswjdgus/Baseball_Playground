"""홈런 확률 예측 모델 파이프라인 실행 스크립트.

python main.py 로 실행하면 2025 시즌 Statcast 데이터를 수집해
Logistic Regression / LightGBM 을 학습, 비교하고 Calibration Curve를 출력한다.
(TabPFN은 TABPFN_TOKEN 환경변수가 있을 때만 함께 학습한다.)
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing import load_statcast, build_dataset, FEATURES, TARGET
from src.modeling import (
    split_dataset,
    train_logistic_regression,
    train_lightgbm,
    train_tabpfn,
    predict_proba_in_chunks,
    evaluate_probability_model,
    calculate_calibration_importance,
)
from src.visualize import plot_calibration_curves, plot_feature_importance
from sklearn.metrics import brier_score_loss


def main():
    df_origin = load_statcast(start_dt="2025-03-18", end_dt="2025-09-28")
    df = build_dataset(df_origin)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, FEATURES, TARGET)

    logit_model = train_logistic_regression(X_train, y_train)
    logit_val_proba = logit_model.predict_proba(X_val)[:, 1]
    logit_test_proba = logit_model.predict_proba(X_test)[:, 1]
    logit_result, _ = evaluate_probability_model("Logistic Regression", y_val, logit_val_proba, y_test, logit_test_proba)

    lgbm_model = train_lightgbm(X_train, y_train)
    X_val_lgbm = X_val.copy()
    X_val_lgbm["home_team"] = X_val_lgbm["home_team"].astype("category")
    X_test_lgbm = X_test.copy()
    X_test_lgbm["home_team"] = X_test_lgbm["home_team"].astype("category")
    lgbm_val_proba = lgbm_model.predict_proba(X_val_lgbm)[:, 1]
    lgbm_test_proba = lgbm_model.predict_proba(X_test_lgbm)[:, 1]
    lgbm_result, _ = evaluate_probability_model("LightGBM", y_val, lgbm_val_proba, y_test, lgbm_test_proba)

    results = [logit_result, lgbm_result]
    proba_by_model = {"Logistic Regression": logit_test_proba, "LightGBM": lgbm_test_proba}

    if "TABPFN_TOKEN" in os.environ:
        tabpfn_model = train_tabpfn(X_train, y_train)
        X_val_tabpfn = X_val.copy()
        X_val_tabpfn["home_team"] = X_val_tabpfn["home_team"].astype("category")
        X_test_tabpfn = X_test.copy()
        X_test_tabpfn["home_team"] = X_test_tabpfn["home_team"].astype("category")

        tabpfn_val_proba = predict_proba_in_chunks(tabpfn_model, X_val_tabpfn)
        tabpfn_test_proba = predict_proba_in_chunks(tabpfn_model, X_test_tabpfn)
        tabpfn_result, _ = evaluate_probability_model("TabPFN", y_val, tabpfn_val_proba, y_test, tabpfn_test_proba)

        results.append(tabpfn_result)
        proba_by_model["TabPFN"] = tabpfn_test_proba

        baseline_brier = brier_score_loss(y_val, tabpfn_val_proba)
        importance_df = calculate_calibration_importance(tabpfn_model, X_val_tabpfn, y_val, baseline_brier)
        plot_feature_importance(importance_df)

    print(pd.DataFrame(results).sort_values("test_f1", ascending=False).reset_index(drop=True))

    plot_calibration_curves(y_test, proba_by_model)
    plt.show()


if __name__ == "__main__":
    main()
