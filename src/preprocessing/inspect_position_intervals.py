import json
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/hungary_2026")


def load_json(filename):
    with open(RAW_PATH / filename, "r", encoding="utf-8") as file:
        return pd.DataFrame(json.load(file))


def main():
    print("Loading position data...")
    position = load_json("position.json")

    print("Loading interval data...")
    intervals = load_json("intervals.json")

    # Timestamp conversion
    position["date"] = pd.to_datetime(
        position["date"],
        utc=True,
        format="mixed"
    )

    intervals["date"] = pd.to_datetime(
        intervals["date"],
        utc=True,
        format="mixed"
    )

    # Load lap data to identify the actual race period
    laps = pd.read_parquet(
        "data/processed/hungary_2026/lap_data.parquet"
    )

    laps["date_start"] = pd.to_datetime(
        laps["date_start"],
        utc=True,
        format="mixed"
    )

    race_start = laps["date_start"].min()
    race_end = laps["date_start"].max()

    print("\n=== BASIC INFORMATION ===")

    print(f"Position rows: {len(position)}")
    print(f"Interval rows: {len(intervals)}")

    print(f"\nRace start: {race_start}")
    print(f"Race end:   {race_end}")

    # Filter position data to race period
    position_race = position[
        (position["date"] >= race_start) &
        (position["date"] <= race_end)
    ].copy()

    # Filter intervals to race period
    intervals_race = intervals[
        (intervals["date"] >= race_start) &
        (intervals["date"] <= race_end)
    ].copy()

    print("\n=== RACE PERIOD ===")

    print(f"Position rows in race period: {len(position_race)}")
    print(f"Interval rows in race period: {len(intervals_race)}")

    print("\n=== POSITION COLUMNS ===")
    for column in position.columns:
        print(f"- {column}")

    print("\n=== INTERVAL COLUMNS ===")
    for column in intervals.columns:
        print(f"- {column}")

    print("\n=== POSITION FIRST 30 RACE ROWS ===")

    print(
        position_race[
            [
                "date",
                "driver_number",
                "position",
            ]
        ]
        .sort_values("date")
        .head(30)
        .to_string(index=False)
    )

    print("\n=== INTERVAL FIRST 30 RACE ROWS ===")

    print(
        intervals_race[
            [
                "date",
                "driver_number",
                "gap_to_leader",
                "interval",
            ]
        ]
        .sort_values("date")
        .head(30)
        .to_string(index=False)
    )

    print("\n=== POSITION COUNTS BY DRIVER ===")

    print(
        position_race["driver_number"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n=== INTERVAL COUNTS BY DRIVER ===")

    print(
        intervals_race["driver_number"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()