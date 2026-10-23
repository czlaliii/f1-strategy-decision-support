"""Official track-map rotation per circuit, cached locally for offline replay.

The rotation comes from the MultiViewer circuit API, the same source FastF1
uses (Session.get_circuit_info). OpenF1 meetings map a meeting to its
circuit_key. Results are cached in data/metadata/circuit_info.json.

Prefetch every local race (run from the project root):
    python -m src.dashboard.circuit_info --all
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import requests

from src.dashboard.config import PROJECT_ROOT

CACHE_PATH = PROJECT_ROOT / "data" / "metadata" / "circuit_info.json"
OPENF1_MEETINGS_URL = "https://api.openf1.org/v1/meetings"
MULTIVIEWER_CIRCUIT_URL = "https://api.multiviewer.app/api/v1/circuits/{circuit_key}/{year}"
REQUEST_TIMEOUT_SECONDS = 5
# OpenF1 allows 30 requests per minute.
OPENF1_MIN_INTERVAL_SECONDS = 2.1
RATE_LIMIT_RETRIES = 2


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _get_meeting(meeting_key: int) -> requests.Response:
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        response = requests.get(
            OPENF1_MEETINGS_URL, params={"meeting_key": meeting_key}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        if response.status_code != 429 or attempt == RATE_LIMIT_RETRIES:
            return response
        time.sleep(min(float(response.headers.get("Retry-After", 5)), 10))
    return response


def _fetch(meeting_key: int, year: int) -> dict:
    meetings = _get_meeting(meeting_key)
    meetings.raise_for_status()
    circuit_key = int(meetings.json()[0]["circuit_key"])
    circuit = requests.get(
        MULTIVIEWER_CIRCUIT_URL.format(circuit_key=circuit_key, year=year),
        headers={"User-Agent": "f1-strategy-decision-support"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    circuit.raise_for_status()
    data = circuit.json()
    return {
        "circuit_key": circuit_key,
        "circuit_name": data.get("circuitName"),
        "rotation": float(data.get("rotation", 0.0)),
    }


def circuit_rotation(meeting_key, year) -> float:
    """Track-map rotation in degrees; 0 if it is neither cached nor downloadable."""
    if meeting_key is None or pd.isna(meeting_key):
        return 0.0
    key = str(int(meeting_key))
    cache = _load_cache()
    if key not in cache:
        try:
            cache[key] = _fetch(int(meeting_key), int(year))
        except (requests.RequestException, ValueError, KeyError, IndexError):
            # Offline or unknown circuit: draw the track unrotated, retry next time.
            return 0.0
        _save_cache(cache)
    return float(cache[key]["rotation"])


def rotate_xy(x, y, degrees):
    """Rotate coordinates counter-clockwise, as in FastF1's track-map examples."""
    angle = np.deg2rad(degrees)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", required=True, help="Prefetch every local race")
    parser.parse_args()
    from src.features.race_dataset import RAW_ROOT, list_local_races

    for race in list_local_races().itertuples():
        drivers = pd.read_json(RAW_ROOT / race.race / "drivers.json")
        meeting_keys = drivers.loc[drivers["session_key"] == race.session_key, "meeting_key"]
        if meeting_keys.empty:
            print(f"{race.race}: skipped (no meeting_key)")
            continue
        cached = str(int(meeting_keys.iloc[0])) in _load_cache()
        rotation = circuit_rotation(meeting_keys.iloc[0], race.race_date.year)
        if not cached:
            time.sleep(OPENF1_MIN_INTERVAL_SECONDS)
        print(f"{race.race}: rotation {rotation:g}°")


if __name__ == "__main__":
    main()
