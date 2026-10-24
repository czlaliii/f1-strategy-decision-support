"""Download the official race classification and starting grid for local races.

Saves session_result.json (race session) and starting_grid.json (meeting) next
to the other raw files. Existing valid files are kept, so the run can resume.

Run from the project root:
    python -m src.ingestion.download_results
"""
import time

import pandas as pd

from src.features.race_dataset import RAW_ROOT, list_local_races
from src.ingestion.api_client import OpenF1Client
from src.ingestion.download_race import get_or_download, read_valid_json

# OpenF1 allows 30 requests per minute.
MIN_REQUEST_INTERVAL_SECONDS = 2.1


def main():
    client = OpenF1Client()
    for race in list_local_races().itertuples():
        race_dir = RAW_ROOT / race.race
        drivers = pd.read_json(race_dir / "drivers.json")
        meeting_keys = drivers.loc[drivers["session_key"] == race.session_key, "meeting_key"]
        if meeting_keys.empty:
            print(f"{race.race}: skipped (no meeting_key)")
            continue
        requests_made = [
            (filename, endpoint, params)
            for filename, endpoint, params in [
                ("session_result.json", "session_result", {"session_key": race.session_key}),
                ("starting_grid.json", "starting_grid", {"meeting_key": int(meeting_keys.iloc[0])}),
            ]
            if read_valid_json(race_dir / filename) is None
        ]
        for filename, endpoint, params in requests_made:
            get_or_download(client, race_dir, filename, endpoint, params, optional_404=True)
            time.sleep(MIN_REQUEST_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
