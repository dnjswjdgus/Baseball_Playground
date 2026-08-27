# Baseball Playground

Aspiring Data Scientist interested in baseball analytics, with experience developing performance metrics and evaluating players using Baseball Savant tracking data and statistical models.

A collection of baseball analytics projects built on 2025 MLB Statcast data, each covering preprocessing, modeling, and visualization.

## Projects

### [bunt-strategy-analysis](bunt-strategy-analysis) — Bunt Decision Analysis by Batting Order
Comparing swing vs. bunt decisions by expected runs and score probability, based on base-out state transition probabilities and batter wOBA.

### [homerun-probability-model](homerun-probability-model) — Feature Importance Analysis via Home Run Prediction Model
Comparing Logistic Regression, LightGBM, and TabPFN to predict home run probability from launch speed and angle, with calibration-based feature importance analysis.

### [pitch-location-plus](pitch-location-plus) — Location+: A Pitcher Command Metric
A custom 20-80 scale metric combining OOF XGBoost pitch-location run value with KDE-based command consistency.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, TabPFN, SciPy, statsmodels, Matplotlib/Seaborn, [pybaseball](https://github.com/jldbc/pybaseball)
