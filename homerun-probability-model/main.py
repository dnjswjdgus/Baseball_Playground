"""Run the home run probability pipeline and write the figures.

    python main.py                   # Logistic Regression and LightGBM
    TABPFN_TOKEN=... python main.py  # also trains TabPFN when the token is set
"""

import os

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from src.config import FEATURES, FIGURE_DIR, SEASON_END, SEASON_START, TARGET
from src.modeling import (
    as_category,
    calculate_calibration_importance,
    evaluate_probability_model,
    predict_proba_in_chunks,
    split_dataset,
    train_lightgbm,
    train_logistic_regression,
    train_tabpfn,
)
from src.preprocessing import build_dataset, load_statcast
from src.visualize import (
    plot_calibration_curves,
    plot_feature_importance,
    plot_home_run_rate_map,
    plot_model_comparison,
)


def main() -> None:
    # 1. Data: Statcast batted balls -> home run candidates
    df = build_dataset(load_statcast(SEASON_START, SEASON_END))
    print(f"{len(df):,} candidate batted balls, {df[TARGET].sum():,} home runs ({df[TARGET].mean():.1%})")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, FEATURES, TARGET)

    # 2. Models: probabilities on validation and test sets
    models = {
        "Logistic Regression": (train_logistic_regression(X_train, y_train), lambda X: X),
        "LightGBM": (train_lightgbm(X_train, y_train), as_category),
    }
    if "TABPFN_TOKEN" in os.environ:
        models["TabPFN"] = (train_tabpfn(X_train, y_train), as_category)

    results, test_proba = [], {}
    for name, (model, prepare) in models.items():
        val_proba = predict_proba_in_chunks(model, prepare(X_val))
        test_proba[name] = predict_proba_in_chunks(model, prepare(X_test))
        results.append(evaluate_probability_model(name, y_val, val_proba, y_test, test_proba[name]))
    results = pd.DataFrame(results).sort_values("test_f1", ascending=False).reset_index(drop=True)
    print(results.round(4).to_string(index=False))

    # 3. Importance: TabPFN when available, otherwise LightGBM
    importance_name = "TabPFN" if "TabPFN" in models else "LightGBM"
    model, prepare = models[importance_name]
    importance = calculate_calibration_importance(model, prepare(X_val), y_val)
    print(f"\n{importance_name} permutation importance\n{importance[['feature', 'importance']].round(4).to_string(index=False)}")

    # 4. Figures
    FIGURE_DIR.mkdir(exist_ok=True)
    plot_home_run_rate_map(df, FIGURE_DIR / "01_home_run_rate_map.png")
    plot_model_comparison(results, FIGURE_DIR / "02_model_comparison.png")
    plot_calibration_curves(y_test, test_proba, FIGURE_DIR / "03_calibration.png")
    plot_feature_importance(importance, importance_name, FIGURE_DIR / "04_feature_importance.png")
    print(f"Figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
