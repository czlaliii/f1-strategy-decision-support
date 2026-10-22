"""Central configuration for the dashboard application."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "hungary_2026"
DATASET_PATH = DATA_DIR / "ml_dataset_clean.parquet"
RACE_STATE_PATH = DATA_DIR / "race_state_with_conditions.parquet"
YEARS = list(range(2023, 2027))
PIT_LOSS_SECONDS = 22.0
SESSION_CODES = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SQ",
    "Sprint": "S",
    "Qualifying": "Q",
    "Race": "R",
}
