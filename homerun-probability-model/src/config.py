"""Project-wide constants shared by the pipeline stages."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_DIR / "figures"

# Statcast window of the original analysis (2025 MLB season)
SEASON_START = "2025-03-18"
SEASON_END = "2025-09-28"

# Origin of the batted-ball coordinate system (Statcast hc_x / hc_y, home plate position)
HC_X_ORIGIN = 125.42
HC_Y_ORIGIN = 198.27
# Shrinks the raw angle so that foul lines sit at about +/-34 degrees
SPRAY_ANGLE_SCALE = 0.75

FEATURES = ["launch_speed", "launch_angle", "spray_angle", "home_team"]
NUMERIC_FEATURES = ["launch_speed", "launch_angle", "spray_angle"]
CATEGORICAL_FEATURES = ["home_team"]
TARGET = "events"

# 60 / 20 / 20 stratified split
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SHARE_OF_TRAINVAL = 0.25
