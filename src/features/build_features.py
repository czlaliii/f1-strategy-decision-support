from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(
    "data/processed/hungary_2026/race_state_with_conditions.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/hungary_2026/features.parquet"
)

def main():
    print("Loading race state...")

    df = pd.read_parquet(INPUT_PATH)

    print(f"Race state shape: {df.shape}")

    # 1. SORT DATA

    df = df.sort_values(
        ["driver_number", "lap_number"]
    ).reset_index(drop=True)

    # 2. PREVIOUS LAP TIME
    # Only the previous lap; the current lap time would leak the target.

    df["previous_lap_time"] = (
        df.groupby("driver_number")["lap_duration"]
        .shift(1)
    )

    # 3. ROLLING LAP TIME

    # Only past laps are used.

    df["rolling_3_lap_avg"] = (
        df.groupby("driver_number")["lap_duration"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
        )
    )

    df["rolling_5_lap_avg"] = (
        df.groupby("driver_number")["lap_duration"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
        )
    )

    # 4. LAP TIME DELTA

    # Change between the previous two laps.

    df["lap_time_delta"] = (
        df["previous_lap_time"]
        - df.groupby("driver_number")[
            "lap_duration"
        ].shift(2)
    )

    # 5. POSITION CHANGE

    # Position change versus the previous lap.

    df["position_change"] = (
        df.groupby("driver_number")["position"]
        .shift(1)
        - df["position"]
    )

    # Positive = gained places, negative = lost, 0 = held.

    # 6. GAP CHANGE

    previous_gap = (
        df.groupby("driver_number")[
            "gap_to_leader"
        ].shift(1)
    )

    df["gap_change"] = (
        df["gap_to_leader"]
        - previous_gap
    )

    # Negative = closer to the leader, positive = further away.

    # 7. INTERVAL CHANGE

    previous_interval = (
        df.groupby("driver_number")[
            "interval"
        ].shift(1)
    )

    df["interval_change"] = (
        df["interval"]
        - previous_interval
    )

    # Negative = gap to the car ahead shrank, positive = grew.

    # 8. LAPS SINCE PIT STOP

    pit_lap = np.where(
        df["pit_stop"] == True,
        df["lap_number"],
        np.nan
    )

    df["last_pit_lap"] = (
        pd.Series(pit_lap, index=df.index)
        .groupby(df["driver_number"])
        .ffill()
    )

    df["laps_since_pit"] = (
        df["lap_number"]
        - df["last_pit_lap"]
    )

    # No earlier pit stop in the first stint; left as NaN (tyre age covers it).

    # 9. TRACK / AIR TEMPERATURE DIFFERENCE

    df["track_air_temp_difference"] = (
        df["track_temperature"]
        - df["air_temperature"]
    )

    # 10. TARGET: NEXT LAP TIME
    # Shift the next lap's lap_duration back onto the current row.

    df["next_lap_time"] = (
        df.groupby("driver_number")["lap_duration"]
        .shift(-1)
    )

    # 11. REMOVE HELPER COLUMN

    df = df.drop(
        columns=["last_pit_lap"]
    )

    # 12. SAVE

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    # 13. INSPECTION

    print("\n=== FEATURES ===")

    print(f"Shape: {df.shape}")

    feature_columns = [
        "driver_number",
        "lap_number",
        "compound",
        "tyre_age",
        "position",
        "gap_to_leader",
        "interval",
        "previous_lap_time",
        "rolling_3_lap_avg",
        "rolling_5_lap_avg",
        "lap_time_delta",
        "position_change",
        "gap_change",
        "interval_change",
        "laps_since_pit",
        "track_air_temp_difference",
        "next_lap_time",
        "laps_down",
        "vsc_duration_in_lap",
        "vsc_in_lap",
        "vsc_lap_fraction",
        "vsc_active_at_end",
    ]

    print("\nFirst rows:")

    print(
        df[feature_columns]
        .head(20)
        .to_string(index=False)
    )

    print("\nMissing values:")

    print(
        df[feature_columns]
        .isna()
        .sum()
        .to_string()
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()