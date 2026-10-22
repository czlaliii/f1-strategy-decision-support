from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/hungary_2026/ml_dataset.parquet")
OUTPUT_PATH = Path("data/processed/hungary_2026/ml_dataset_clean.parquet")

def main():
    print("Loading ML dataset...")

    df = pd.read_parquet(INPUT_PATH)

    print(f"Initial shape: {df.shape}")

    # 1. Compound
    # Constant within a stint, so forward-fill per driver.
    df["compound"] = (
        df.groupby("driver_number")["compound"]
        .ffill()
    )

    # Remaining gaps are before the driver's first known stint.
    df["compound"] = df["compound"].fillna("UNKNOWN")

    # 2. Position
    # Continuous race state, so forward-fill per driver.

    df["position_missing"] = df["position"].isna().astype(int)

    df["position"] = (
        df.groupby("driver_number")["position"]
        .ffill()
    )

    # 3. Interval
    # Keep a missing indicator for the original gaps.
    df["interval_missing"] = df["interval"].isna().astype(int)

    interval_median = df["interval"].median()

    df["interval"] = df["interval"].fillna(interval_median)

    # 4. Interval change
    df["interval_change_missing"] = (
        df["interval_change"].isna().astype(int)
    )

    interval_change_median = df["interval_change"].median()

    df["interval_change"] = (
        df["interval_change"]
        .fillna(interval_change_median)
    )

    # 5. Previous lap / rolling features
    # Naturally missing early in the race; not imputed here.

    # 6. laps_since_pit
    # NaN means no pit stop yet (not zero laps), so add a flag.
    df["has_previous_pit"] = (
        df["laps_since_pit"].notna().astype(int)
    )

    # Numeric fill for the model; tyre_age approximates stint progress in the first stint.
    df["laps_since_pit"] = df["laps_since_pit"].fillna(
        df["tyre_age"]
    )

    # 7. Gap features
    # Not imputed: NaN may mean +1 LAP, so keep a missing flag.
    df["gap_to_leader_missing"] = (
        df["gap_to_leader"].isna().astype(int)
    )

    df["gap_change_missing"] = (
        df["gap_change"].isna().astype(int)
    )

    # 8. Lap history features
    history_columns = [
        "previous_lap_time",
        "rolling_3_lap_avg",
        "rolling_5_lap_avg",
        "lap_time_delta",
        "position_change",
    ]

    for column in history_columns:
        df[f"{column}_missing"] = (
            df[column].isna().astype(int)
        )

    # 9. Missing value report
    print("\n=== MISSING VALUES AFTER CLEANING ===")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        print("\n=== LAPS DOWN ===")

        print(
            df["laps_down"]
            .value_counts(dropna=False)
            .sort_index()
        )

        print(
            f"Missing laps_down: "
            f"{df['laps_down'].isna().sum()}"
        )
    else:
        print(missing)

    # 10. Save
    df.to_parquet(OUTPUT_PATH, index=False)

    print("\n=== FINAL DATASET ===")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()