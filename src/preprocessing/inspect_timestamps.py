import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw/hungary_2026")


def load_json(filename: str) -> pd.DataFrame:

    filepath = DATA_DIR / filename

    with filepath.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return pd.DataFrame(data)


def inspect():
    laps = load_json("laps.json")
    position = load_json("position.json")
    intervals = load_json("intervals.json")

    # Convert timestamps
    laps["date_start"] = pd.to_datetime(
        laps["date_start"],
        format="mixed",
        utc=True
    )

    position["date"] = pd.to_datetime(
        position["date"],
        format="mixed",
        utc=True
    )

    intervals["date"] = pd.to_datetime(
        intervals["date"],
        format="mixed",
        utc=True
    )

    # Position data may include several sessions; keep the race only.
    if "session_key" in position.columns:
        position = position[
            position["session_key"] == 11342
        ].copy()

    print("\n=== LAPS ===")
    print(f"Min: {laps['date_start'].min()}")
    print(f"Max: {laps['date_start'].max()}")

    print("\n=== POSITION ===")
    print(f"Min: {position['date'].min()}")
    print(f"Max: {position['date'].max()}")
    print(f"Rows: {len(position)}")

    print("\n=== INTERVALS ===")
    print(f"Min: {intervals['date'].min()}")
    print(f"Max: {intervals['date'].max()}")

    # Select driver 81, lap 21
    sample = laps[
        (laps["driver_number"] == 81) &
        (laps["lap_number"] == 21)
    ].iloc[0]

    sample_time = sample["date_start"]

    print("\n=== SAMPLE LAP ===")
    print(
        sample[
            [
                "driver_number",
                "lap_number",
                "date_start",
                "lap_duration"
            ]
        ]
    )

    print("\n=== POSITION NEAR SAMPLE ===")

    position_sample = position[
        position["driver_number"] == 81
    ].copy()

    position_sample["time_difference"] = (
        position_sample["date"] - sample_time
    ).abs()

    position_sample = position_sample.sort_values(
        "time_difference"
    ).head(10)

    print(
        position_sample[
            [
                "date",
                "position",
                "time_difference"
            ]
        ].to_string(index=False)
    )

    print("\n=== INTERVAL NEAR SAMPLE ===")

    interval_sample = intervals[
        intervals["driver_number"] == 81
    ].copy()

    interval_sample["time_difference"] = (
        interval_sample["date"] - sample_time
    ).abs()

    interval_sample = interval_sample.sort_values(
        "time_difference"
    ).head(10)

    print(
        interval_sample[
            [
                "date",
                "interval",
                "gap_to_leader",
                "time_difference"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    inspect()