# What Makes a Home Run?

**A home run probability model built on 2025 MLB Statcast batted balls, and what its feature importance says about launch speed, launch angle, spray angle and ballpark.**

<p align="center">
  <img src="figures/01_home_run_rate_map.png" alt="Observed home run rate by launch speed, launch angle and spray angle" width="100%">
</p>

<p align="center"><sub>Observed home run rate of home run candidate balls (40,637 balls, 5,765 home runs). Gray cells have fewer than 20 balls.</sub></p>

## Key findings

| Model | Test F1 | Precision | Recall |
|---|---|---|---|
| LightGBM | **0.780** | 0.760 | 0.801 |
| TabPFN* | 0.779 | 0.723 | 0.845 |
| Logistic Regression | 0.505 | 0.409 | 0.661 |

<sub>*TabPFN numbers come from the original run on a GPU with a TabPFN token. The other two rows are reproduced by `python main.py`.</sub>

- **Non-linear models separate home runs far better.** LightGBM and TabPFN reach F1 near 0.78, while Logistic Regression stays at 0.51.
- **Speed and angle decide it.** Shuffling launch speed or launch angle hurts the predicted probabilities most, spray angle comes next, and **home team adds little** — how hard and at what angle a ball is hit tells far more than which park it is hit in.
- **The ranking is stable across models.** In the original TabPFN analysis the importances were launch speed 0.1067, launch angle 0.0973, spray angle 0.0337 and home team 0.0043; LightGBM gives the same order (0.1156, 0.1098, 0.0401, 0.0078).

<p align="center">
  <img src="figures/04_feature_importance.png" alt="Permutation feature importance" width="70%">
</p>

## Why this project

After reading Leo Breiman's *Statistical Modeling: The Two Cultures* in an introductory data science course, I wanted to see whether a high-performing algorithmic model can reveal the structure of data, not just predict it. I trained models to predict home runs from batted-ball features, then asked the best-calibrated model which features it relies on.

<p align="center">
  <img src="figures/02_model_comparison.png" alt="Test F1, precision and recall by model" width="62%">
  <img src="figures/03_calibration.png" alt="Calibration curves" width="34%">
</p>

<p align="center"><sub>Models trained with balanced class weights predict higher probabilities than the observed rate, so their curves sit below the diagonal. In the original run TabPFN's probabilities were the closest to the actual home run rate, which is why it was used for the importance analysis.</sub></p>

## How it works

1. **Data.** Statcast batted balls from 2025-03-18 to 2025-09-28 via `pybaseball`. A ball is a home run candidate when its launch speed and angle are at least the smallest values seen on an actual home run.
2. **Features.** Launch speed, launch angle, spray angle (computed from the field coordinates) and home team as a categorical variable for park effects.
3. **Models.** Logistic Regression as the baseline, LightGBM for non-linear relationships and TabPFN, a Transformer-based foundation model for tabular data. The data is split 60/20/20 with the class ratio preserved, the F1-optimal threshold is chosen on the validation set, and the test set gives the final scores. Calibration curves check how well probabilities match actual rates.
4. **Importance.** Permutation importance: shuffle one feature at a time and measure the increase in Brier score. This works for models such as TabPFN that have no built-in importance.

## Run it

```bash
pip install -r requirements.txt
python main.py
```

Pitch data are downloaded once with `pybaseball` and cached, and the four figures are written to `figures/`. Without a token, Logistic Regression and LightGBM run and the importance uses LightGBM. To add TabPFN, install `tabpfn` and `torch`, then set the token (a GPU is recommended):

```bash
export TABPFN_TOKEN=your_token_here   # Windows PowerShell: $env:TABPFN_TOKEN="your_token_here"
```

```
homerun-probability-model/
├── main.py                 # Pipeline entry point
├── src/
│   ├── config.py           # Constants: dates, features, split settings
│   ├── preprocessing.py    # Statcast download, home run candidates, feature engineering
│   ├── modeling.py         # Models, evaluation, permutation importance
│   └── visualize.py        # Figures
└── figures/
```

## Limitations and next steps

- Park effects are reduced to one categorical variable (home team), which does not capture fence distance and height by direction, or temperature, humidity and wind.
- Park dimensions matter in practice: a study at the 2025 SABR Analytics Conference examined how changing ballpark dimensions could change a team's run scoring. Next, I plan to combine fence distance and height by direction with the batted-ball direction and add environmental variables, to model how the same batted ball's home run probability changes by park and conditions.
- The Statcast window is the original one (2025-03-18 to 2025-09-28), which also includes a few spring training games played at the start; keeping only `game_type == "R"` is a one-line change.

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, TabPFN, Matplotlib, Seaborn, [pybaseball](https://github.com/jldbc/pybaseball)
