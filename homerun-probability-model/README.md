# Feature Importance Analysis via Home Run Prediction Model

Building a model that predicts whether a batted ball results in a home run, and analyzing which batted-ball features the model relies on most.

## Background

I built a model to predict whether a batted ball results in a home run, and analyzed which batted-ball features the model relies on most. During the Introduction to Data Science course in the first semester of 2025, reading Leo Breiman's *"Statistical Modeling: The Two Cultures"* got me interested in the idea that, beyond explanation-focused statistical models, high-performing algorithmic models can also be used to explore the underlying structure of data. Motivated by this, I used MLB Statcast data to predict home runs and then examined the feature importance of the best-performing model to identify which batted-ball characteristics most determine home run probability.

## Data & Methods

I collected 2025 MLB Statcast batted-ball data via `pybaseball` and used launch speed, launch angle, spray angle, and home team as the main features. Spray angle was computed separately from Statcast's batted-ball coordinates, and home team was included as a categorical variable to capture park-specific effects.

I compared three models: Logistic Regression, LightGBM, and TabPFN. Logistic Regression served as the baseline, while LightGBM (able to learn non-linear relationships) and TabPFN (a Transformer-based foundation model for tabular data) were applied alongside it. The data was split into train/validation/test sets (60/20/20) while preserving the class ratio. The classification threshold that maximized F1 score on the validation set was selected, and final performance was compared on the test set. Beyond simple classification accuracy, I also compared Calibration Curves to check how closely predicted probabilities matched actual home run rates.

## Results

F1 scores were nearly identical between LightGBM (0.7799) and TabPFN (0.7792), while Logistic Regression scored notably lower at 0.5053. This confirmed that models capable of capturing non-linear relationships between features are better suited for distinguishing home runs.

| Model | Test F1 | Precision | Recall |
|---|---|---|---|
| LightGBM | 0.780 | 0.760 | 0.801 |
| TabPFN | 0.779 | 0.723 | 0.845 |
| Logistic Regression | 0.505 | 0.409 | 0.661 |

While LightGBM narrowly led in classification performance, TabPFN's predicted probabilities were the closest match to actual home run rates on the Calibration Curve. Based on this, I performed additional feature importance analysis using TabPFN. Since TabPFN, unlike tree-based models, does not provide standard feature importance directly, I used **Permutation Importance** — measuring how much the Brier Score increases when each feature is randomly shuffled.

The analysis showed launch speed had the highest importance (0.1067), followed by launch angle (0.0973) and spray angle (0.0337). Home team importance was relatively low at 0.0043. In other words, how fast and at what angle a ball was hit provided far more information for predicting home run probability than ballpark information.

## Limitations & Future Work

This analysis simplified park effects into a single categorical variable (home team), which does not fully capture actual ballpark characteristics. The same batted ball could produce a different outcome depending on fence distance and height by direction, as well as environmental factors like temperature, humidity, and wind — none of which were included here.

At the 2025 SABR Analytics Conference, a study was presented analyzing how adjusting ballpark dimensions could affect a team's run-scoring output. This suggests that treating park effects as a simple categorical variable, as done here, has clear limitations. Going forward, I plan to combine fence distance/height by direction with batted-ball direction, and add environmental variables such as temperature and humidity, to model park effects in finer detail — extending this into a model that can capture how home run probability for the same batted ball changes by ballpark and environment.

## Run It

```bash
pip install -r requirements.txt
python main.py
```

TabPFN requires an API token. To use it, set the `TABPFN_TOKEN` environment variable before running (otherwise only Logistic Regression and LightGBM will run).

```bash
export TABPFN_TOKEN=your_token_here   # Windows PowerShell: $env:TABPFN_TOKEN="your_token_here"
```

`notebooks/homerun_expectation.ipynb` contains the original analysis notebook (built for Colab, requires GPU).

## Folder Structure

```
homerun-probability-model/
├── main.py                 # Pipeline entry point
├── src/
│   ├── preprocessing.py    # Statcast collection, home run candidate filtering, feature engineering
│   ├── modeling.py         # Model training/evaluation, calibration-based feature importance
│   └── visualize.py        # Calibration curve, feature importance chart
└── notebooks/
    └── homerun_expectation.ipynb
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, LightGBM, TabPFN, PyTorch, Matplotlib, [pybaseball](https://github.com/jldbc/pybaseball)
