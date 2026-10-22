from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed/multi_race")

TRAIN_PATH = DATA_DIR / "train.parquet"
VALIDATION_PATH = DATA_DIR / "validation.parquet"
TEST_PATH = DATA_DIR / "test.parquet"


def describe_race(df, race):
    data = df[df["race"] == race].copy()

    target = data["next_lap_time"]

    print("\n" + "=" * 70)
    print(race)
    print("=" * 70)

    print(f"Rows:    {len(data)}")
    print(f"Drivers: {data['driver_number'].nunique()}")
    print(f"Laps:    {data['lap_number'].min()} - {data['lap_number'].max()}")

    print("\nTarget: next_lap_time")
    print(f"Mean:    {target.mean():.3f}")
    print(f"Median:  {target.median():.3f}")
    print(f"Std:     {target.std():.3f}")
    print(f"Min:     {target.min():.3f}")
    print(f"Max:     {target.max():.3f}")

    print("\nTarget quantiles:")
    print(
        target.quantile(
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )
    )

    print("\nExtreme target rows:")

    extreme = data[
        data["next_lap_time"] > 130
    ][
        [
            "driver_number",
            "lap_number",
            "lap_duration",
            "next_lap_time",
            "compound",
            "tyre_age",
            "pit_stop",
            "position",
        ]
    ].sort_values(
        "next_lap_time",
        ascending=False,
    )

    print(f"Rows with next_lap_time > 130s: {len(extreme)}")

    if len(extreme) > 0:
        print(extreme.head(30).to_string(index=False))

    print("\nPit-related target rows:")

    pit_rows = data[
        data["pit_stop"] == True
    ][
        [
            "driver_number",
            "lap_number",
            "lap_duration",
            "next_lap_time",
            "compound",
            "tyre_age",
        ]
    ]

    print(f"Pit-stop rows: {len(pit_rows)}")

    if len(pit_rows) > 0:
        print(
            pit_rows.head(30).to_string(index=False)
        )

    print("\nCompound distribution:")
    print(
        data["compound"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nRainfall distribution:")
    print(
        data["rainfall"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )


def main():
    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)
    test = pd.read_parquet(TEST_PATH)

    print("=" * 70)
    print("MULTI-RACE DATASET DIAGNOSTICS")
    print("=" * 70)

    print("\nTRAIN RACES")

    for race in sorted(train["race"].unique()):
        describe_race(train, race)

    print("\nVALIDATION")

    for race in sorted(validation["race"].unique()):
        describe_race(validation, race)

    print("\nTEST")

    for race in sorted(test["race"].unique()):
        describe_race(test, race)


if __name__ == "__main__":
    main()