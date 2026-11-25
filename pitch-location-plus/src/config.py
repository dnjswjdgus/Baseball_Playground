"""Project-wide constants shared by the pipeline stages."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_DIR / "figures"

# Statcast window covering the 2025 MLB regular season
SEASON_START = "2025-03-18"
SEASON_END = "2025-09-28"

REQUIRED_COLS = ["plate_x", "plate_z", "delta_pitcher_run_exp", "pitch_type", "stand", "p_throws", "balls", "strikes"]
CATEGORICAL_COLS = ["stand", "p_throws", "pitch_type", "count"]
FEATURES = ["plate_x", "plate_z", "stand", "p_throws", "pitch_type", "count"]
TARGET = "pitcher_run_value"

# Out-of-fold XGBoost location value model
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
N_FOLDS = 5

# A pitch type counts toward Location+ only with at least this many pitches, and a pitcher
# needs at least this share of his pitch types to qualify
MIN_STAB_PITCHES = 60
MIN_COVERAGE = 0.5

# Split-half reliability curve
STABILIZATION_THRESHOLDS = range(10, 410, 10)
RELIABILITY_CUTOFF = 0.7

# Strike zone in feet (catcher's view): width 1.66 ft, from 1.5 ft to 3.5 ft
ZONE_X = (-0.83, 0.83)
ZONE_Z = (1.5, 3.5)

SAVANT_COLORS = {
    "FF": "#D22D49", "SI": "#F58A1F", "FC": "#933F2C", "CH": "#1DB954",
    "FS": "#3EAC9A", "FO": "#55C592", "SC": "#54C93F", "CU": "#00D1ED",
    "KC": "#6236CD", "CS": "#005AE0", "SL": "#EAEB29", "ST": "#E7B73A",
    "SV": "#92A7D6", "KN": "#3B44B3", "EP": "#7B8185", "FA": "#999999",
    "IN": "#999999", "PO": "#999999",
}
