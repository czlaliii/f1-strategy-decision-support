import json
from pathlib import Path

import pandas as pd

PROCESSED_PATH = Path(
    "data/processed/hungary_2026/lap_data.parquet"
)

RAW_PATH = Path(
    "data/raw/hungary_2026"
)

OUTPUT_PATH = Path(
    "data/processed/hungary_2026/race_state.parquet"
)

def load_json(filename):
    with open(
        RAW_PATH / filename,
        "r",
        encoding="utf-8"
    ) as file:
        return pd.DataFrame(json.load(file))

def _to_utc(series):
    return pd.to_datetime(
        series,
        utc=True,
        format="mixed"
    ).astype("datetime64[ns, UTC]")


def build_race_state(laps, weather, position, intervals):
    """Attach weather, position and gap data known at each lap end."""

    laps = laps.copy()

    # 1. LAP DATA

    laps["date_start"] = _to_utc(laps["date_start"])

    # Lap end timestamp
    laps["date_end"] = (
        laps["date_start"]
        + pd.to_timedelta(
            laps["lap_duration"],
            unit="s"
        )
    )

    laps = laps.dropna(
        subset=["date_end"]
    ).copy()

    race_start = laps["date_start"].min()
    race_end = laps["date_end"].max()

    # 2. WEATHER

    weather_columns = [
        "wind_direction",
        "rainfall",
        "air_temperature",
        "humidity",
        "pressure",
        "track_temperature",
        "wind_speed",
    ]

    weather = weather.copy()
    weather["date"] = _to_utc(weather["date"])

    # Pre-race samples are kept so early laps get the latest known weather
    # without back-filling from later samples.
    weather = weather[
        weather["date"] <= race_end
    ].copy()

    weather = weather[
        ["date"] + weather_columns
    ].copy()

    weather = weather.sort_values("date")

    laps = laps.sort_values("date_end")

    race_state = pd.merge_asof(
        laps,
        weather,
        left_on="date_end",
        right_on="date",
        direction="backward"
    )

    race_state = race_state.drop(
        columns=["date"]
    )

    # 3. POSITION

    position = position.copy()
    position["date"] = _to_utc(position["date"])

    position = position[
        (position["date"] >= race_start) &
        (position["date"] <= race_end)
    ].copy()

    position = position[
        [
            "date",
            "driver_number",
            "position",
        ]
    ].copy()

    position = position.sort_values(
        ["date", "driver_number"]
    )

    # Use the driver's latest known position at lap end
    race_state = pd.merge_asof(
        race_state.sort_values(
            ["date_end", "driver_number"]
        ),
        position,
        left_on="date_end",
        right_on="date",
        by="driver_number",
        direction="backward"
    )

    race_state = race_state.drop(
        columns=["date"]
    )

    # 4. INTERVALS

    intervals = intervals.copy()
    intervals["date"] = _to_utc(intervals["date"])

    intervals = intervals[
        (intervals["date"] >= race_start) &
        (intervals["date"] <= race_end)
    ].copy()

    intervals["gap_to_leader_raw"] = intervals["gap_to_leader"].astype(str)

    intervals["laps_down"] = (
        intervals["gap_to_leader_raw"]
        .str.extract(r"\+(\d+)\s+LAPS?", expand=False)
        .astype(float)
        .fillna(0)
    )

    intervals["gap_to_leader"] = pd.to_numeric(
        intervals["gap_to_leader"],
        errors="coerce"
    )

    intervals["interval"] = pd.to_numeric(
        intervals["interval"],
        errors="coerce"
    )

    intervals = intervals[
        [
            "date",
            "driver_number",
            "gap_to_leader",
            "interval",
            "laps_down",
        ]
    ].copy()

    race_state = pd.merge_asof(
        race_state.sort_values(
            ["date_end", "driver_number"]
        ),
        intervals,
        left_on="date_end",
        right_on="date",
        by="driver_number",
        direction="backward"
    )

    race_state = race_state.drop(
        columns=["date"]
    )

    # 5. FINAL SORTING

    return race_state.sort_values(
        ["lap_number", "driver_number"]
    ).reset_index(drop=True)


def main():
    print("Loading lap data...")

    laps = pd.read_parquet(PROCESSED_PATH)

    print(f"Lap data shape: {laps.shape}")

    race_state = build_race_state(
        laps,
        load_json("weather.json"),
        load_json("position.json"),
        load_json("intervals.json"),
    )

    # 6. SAVE

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    race_state.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    # 7. INSPECTION

    print("\n=== RACE STATE ===")

    print(f"Shape: {race_state.shape}")

    print("\nColumns:")

    for column in race_state.columns:
        print(f"- {column}")

    print("\nFirst rows:")

    print(
        race_state[
            [
                "driver_number",
                "lap_number",
                "date_start",
                "date_end",
                "lap_duration",
                "compound",
                "tyre_age",
                "pit_stop",
                "position",
                "gap_to_leader",
                "interval",
                "air_temperature",
                "track_temperature",
                "rainfall",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nMissing values:")

    print(
        race_state[
            [
                "position",
                "gap_to_leader",
                "interval",
            ]
        ]
        .isna()
        .sum()
        .to_string()
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )

if __name__ == "__main__":
    main()