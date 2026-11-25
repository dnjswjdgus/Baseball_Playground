"""Project-wide constants shared by the pipeline stages."""

from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_DIR / "figures"

# Statcast window covering the 2025 MLB regular season
SEASON_START = "2025-03-18"
SEASON_END = "2025-09-28"

# Plate-appearance outcome classes used by the model
EVENT_CLASSES = ["BB_HBP", "SINGLE", "DOUBLE", "TRIPLE", "HOME_RUN", "STRIKEOUT", "BIP_OUT", "GIDP"]

# Mapping from Statcast `events` to the outcome classes above
EVENT_CLASS_MAP = {
    "walk": "BB_HBP",
    "hit_by_pitch": "BB_HBP",
    "single": "SINGLE",
    "double": "DOUBLE",
    "triple": "TRIPLE",
    "home_run": "HOME_RUN",
    "strikeout": "STRIKEOUT",
    "grounded_into_double_play": "GIDP",
    "field_out": "BIP_OUT",
    "force_out": "BIP_OUT",
    "fielders_choice_out": "BIP_OUT",
    "fielders_choice": "BIP_OUT",
    "double_play": "BIP_OUT",
    "triple_play": "BIP_OUT",
    "sac_fly": "BIP_OUT",
    "sac_fly_double_play": "BIP_OUT",
    "strikeout_double_play": "BIP_OUT",
}

# 2025 FanGraphs wOBA weights
WOBA_WEIGHTS = {
    "walk": 0.691,
    "hit_by_pitch": 0.722,
    "single": 0.882,
    "double": 1.252,
    "triple": 1.584,
    "home_run": 2.037,
}

# Events excluded from the wOBA denominator (not an AB, BB, HBP or SF)
WOBA_EXCLUDED_EVENTS = ["intent_walk", "catcher_interf", "truncated_pa", "sac_bunt", "sac_bunt_double_play"]

# Kernel smoothing used to estimate outcome probabilities for a given wOBA
KERNEL_BANDWIDTH = 0.015
MIN_PA = 300

# Decision states: swing away with a runner on 1B and no outs, versus a successful
# sacrifice bunt that leaves a runner on 2B with one out (bases string = 1B/2B/3B)
SWING_STATE = (0, "100")
BUNT_STATE = (1, "010")

# wOBA grid for the current and next batter
WOBA_GRID = np.round(np.arange(0.200, 0.500, 0.002), 3)
