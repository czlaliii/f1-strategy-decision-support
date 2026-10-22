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

def main():
    print("Loading lap data...")

    laps = pd.read_parquet(PROCESSED_PATH)

    print(f"Lap data shape: {laps.shape}")

    ### LAP DATA

    laps["date_start"] = pd.to_datetime(
        laps["date_start"],
        utc=True,
        format="mixed"
    ).astype("datetime64[ns, UTC]")

    # The date of the end of the lap
    laps["date_end"] = (
        laps["date_start"]
        + pd.to_timedelta(
            laps["lap_duration"],
            unit="s"
        )
    )

    missing_date_end = laps["date_end"].isna().sum()

    print(
        f"Rows without valid date_end: {missing_date_end}"
    )

    laps = laps.dropna(
        subset=["date_end"]
    ).copy()

    race_start = laps["date_start"].min()
    race_end = laps["date_end"].max()

    ### WEATHER

    weather = load_json("weather.json")

    weather["date"] = pd.to_datetime(
        weather["date"],
        utc=True,
        format="mixed"
    ).astype("datetime64[ns, UTC]")

    weather = weather[
        (weather["date"] >= race_start) &
        (weather["date"] <= race_end)
    ].copy()

    print(f"Weather rows used: {len(weather)}")

    weather = weather[
        [
            "date",
            "wind_direction",
            "rainfall",
            "air_temperature",
            "humidity",
            "pressure",
            "track_temperature",
            "wind_speed",
        ]
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

    # Could be that the weather is NaN at the start
    weather_columns = [
        "wind_direction",
        "rainfall",
        "air_temperature",
        "humidity",
        "pressure",
        "track_temperature",
        "wind_speed",
    ]

    for column in weather_columns:
        race_state[column] = (
            race_state[column]
            .ffill()
            .bfill()
        )

    ### POSITION

    print("Loading position data...")

    position = load_json("position.json")

    position["date"] = pd.to_datetime(
        position["date"],
        utc=True,
        format="mixed"
    ).astype("datetime64[ns, UTC]")

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

    # Align the position data with the end of the lap based on
    # the latest known state associated with the given driver
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

    print("Loading interval data...")

    intervals = load_json("intervals.json")

    intervals["date"] = pd.to_datetime(
        intervals["date"],
        utc=True,
        format="mixed"
    ).astype("datetime64[ns, UTC]")

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

    print(
        intervals[
            (intervals["driver_number"] == 44) &
            (intervals["date"] <= pd.Timestamp("2026-07-26 13:04:51.255", tz="UTC"))
        ][
            [
                "date",
                "gap_to_leader_raw",
                "gap_to_leader",
                "interval",
                "laps_down"
            ]
        ]
        .tail(10)
        .to_string(index=False)
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

    race_state = race_state.sort_values(
        ["lap_number", "driver_number"]
    ).reset_index(drop=True)

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