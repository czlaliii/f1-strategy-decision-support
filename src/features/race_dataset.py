"""Build the lap-level model dataset for any locally stored OpenF1 race.

Run from the project root to prebuild the cached datasets:
    python -m src.features.race_dataset --all
    python -m src.features.race_dataset --race hungarian_2025
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.features.add_race_conditions import (
    add_conditions,
    build_race_condition_intervals,
)
from src.features.build_features import add_lap_features
from src.preprocessing.build_lap_data import build_lap_data
from src.preprocessing.build_race_state import build_race_state

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
CATALOG_PATH = PROJECT_ROOT / "data" / "metadata" / "race_catalog.csv"
# Bump when the dataset logic changes so stale caches are not reused.
DATASET_VERSION = 1

HISTORY_COLUMNS = [
    "previous_lap_time",
    "rolling_3_lap_avg",
    "rolling_5_lap_avg",
    "lap_time_delta",
    "position_change",
]


class RaceDataError(ValueError):
    """Raised when a race lacks the raw data needed for the model dataset."""


def _read_session_json(race_dir: Path, filename: str, session_key: int) -> pd.DataFrame:
    path = race_dir / filename
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as file:
        df = pd.DataFrame(json.load(file))
    if "session_key" in df.columns:
        df = df[pd.to_numeric(df["session_key"], errors="coerce") == session_key]
    return df.reset_index(drop=True)


def resolve_session_key(race_folder: str) -> int:
    """Return the race session_key stored in a local race folder."""
    race_dir = RAW_ROOT / race_folder
    drivers = pd.read_json(race_dir / "drivers.json")
    keys = pd.to_numeric(drivers.get("session_key"), errors="coerce").dropna().unique()
    if len(keys) == 1:
        return int(keys[0])
    if len(keys) == 0:
        raise RaceDataError(f"No session_key in {race_dir / 'drivers.json'}.")
    # Several sessions: the race is the one with the most laps.
    laps = pd.read_json(race_dir / "laps.json")
    max_laps = laps.groupby("session_key")["lap_number"].max()
    return int(max_laps.loc[max_laps.index.isin(keys)].idxmax())


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing-value flags and per-driver fills that use past laps only.

    Unlike handle_missing_values.py, interval gaps are not filled with a
    race-wide median (that would use later laps); the model's imputer fills
    them with training-set medians instead.
    """
    df = df.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
    by_driver = df.groupby("driver_number")

    df["compound"] = by_driver["compound"].ffill().fillna("UNKNOWN")

    df["position_missing"] = df["position"].isna().astype(int)
    df["position"] = by_driver["position"].ffill()

    df["interval_missing"] = df["interval"].isna().astype(int)
    df["interval_change_missing"] = df["interval_change"].isna().astype(int)

    df["has_previous_pit"] = df["laps_since_pit"].notna().astype(int)
    df["laps_since_pit"] = df["laps_since_pit"].fillna(df["tyre_age"])

    df["gap_to_leader_missing"] = df["gap_to_leader"].isna().astype(int)
    df["gap_change_missing"] = df["gap_change"].isna().astype(int)

    for column in HISTORY_COLUMNS:
        df[f"{column}_missing"] = df[column].isna().astype(int)

    return df


def build_race_dataset(race_folder: str, session_key: int) -> pd.DataFrame:
    """Run the pandas pipeline on one race and return every lap row.

    The last lap of each driver is kept (its next_lap_time is NaN) because it
    is still a valid race state for the simulator.
    """
    race_dir = RAW_ROOT / race_folder
    raw = {
        name: _read_session_json(race_dir, f"{name}.json", session_key)
        for name in ["laps", "stints", "pit", "weather", "position", "intervals", "race_control"]
    }
    required = {
        "laps": {"driver_number", "lap_number", "date_start", "lap_duration"},
        "stints": {"driver_number", "stint_number", "lap_start", "lap_end", "compound"},
        "weather": {"date", "air_temperature", "track_temperature"},
        "position": {"date", "driver_number", "position"},
        "intervals": {"date", "driver_number", "gap_to_leader", "interval"},
    }
    for name, columns in required.items():
        if raw[name].empty or not columns.issubset(raw[name].columns):
            raise RaceDataError(f"{race_folder}: {name}.json is missing or incomplete.")

    lap_data = build_lap_data(raw["laps"], raw["stints"], raw["pit"])
    race_state = build_race_state(lap_data, raw["weather"], raw["position"], raw["intervals"])
    if race_state.empty:
        raise RaceDataError(f"{race_folder}: no laps with a valid duration.")

    race_control = raw["race_control"]
    if {"date", "message"}.issubset(race_control.columns):
        race_control = race_control.copy()
        race_control["date"] = pd.to_datetime(race_control["date"], utc=True, format="mixed")
        vsc_intervals = build_race_condition_intervals(race_control)
    else:
        vsc_intervals = []
    race_state = add_conditions(race_state, vsc_intervals)

    dataset = add_missing_indicators(add_lap_features(race_state))
    dataset["race"] = race_folder
    dataset["session_key"] = session_key
    return dataset


def dataset_path(race_folder: str, session_key: int) -> Path:
    return (
        PROCESSED_ROOT / race_folder / "pandas"
        / f"race_dataset_v{DATASET_VERSION}_{session_key}.parquet"
    )


def load_race_dataset(race_folder: str, session_key: int, rebuild: bool = False) -> pd.DataFrame:
    """Load the cached race dataset, building it from raw data if needed."""
    path = dataset_path(race_folder, session_key)
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    dataset = build_race_dataset(race_folder, session_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(path, index=False)
    return dataset


@lru_cache(maxsize=1)
def list_local_races() -> pd.DataFrame:
    """Return local race folders with session_key and race date, oldest first."""
    catalog = pd.read_csv(CATALOG_PATH) if CATALOG_PATH.exists() else pd.DataFrame()
    race_dates = {}
    if {"session_key", "race_date"}.issubset(catalog.columns):
        race_dates = dict(zip(
            pd.to_numeric(catalog["session_key"], errors="coerce"),
            pd.to_datetime(catalog["race_date"], utc=True, errors="coerce"),
        ))

    rows = []
    for race_dir in sorted(RAW_ROOT.iterdir()):
        if not (race_dir / "drivers.json").exists() or not (race_dir / "laps.json").exists():
            continue
        try:
            session_key = resolve_session_key(race_dir.name)
        except (RaceDataError, ValueError, KeyError):
            continue
        race_date = race_dates.get(session_key)
        if race_date is None or pd.isna(race_date):
            laps = _read_session_json(race_dir, "laps.json", session_key)
            if laps.empty or "date_start" not in laps.columns:
                continue
            race_date = pd.to_datetime(laps["date_start"], utc=True, format="mixed").min()
        rows.append({"race": race_dir.name, "session_key": session_key, "race_date": race_date})

    races = pd.DataFrame(rows, columns=["race", "session_key", "race_date"])
    return races.sort_values("race_date").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--race", help="Local race folder name, e.g. hungarian_2025")
    group.add_argument("--all", action="store_true", help="Build every local race")
    parser.add_argument("--rebuild", action="store_true", help="Ignore cached datasets")
    args = parser.parse_args()

    races = list_local_races()
    if args.race:
        races = races[races["race"] == args.race]
        if races.empty:
            parser.error(f"Unknown local race folder: {args.race}")

    for race in races.itertuples():
        try:
            dataset = load_race_dataset(race.race, race.session_key, rebuild=args.rebuild)
            print(f"{race.race}: {len(dataset)} rows")
        except RaceDataError as error:
            print(f"{race.race}: skipped ({error})")


if __name__ == "__main__":
    main()
