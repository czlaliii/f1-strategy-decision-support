from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "multi_race"
    / "train_context.parquet"
)


def main():
    print("=" * 78)
    print("SLOW LAP / TARGET QUALITY ANALYSIS")
    print("=" * 78)

    df = pd.read_parquet(DATA_PATH)

    slow = df[
        df["next_lap_time"].notna()
        & (df["next_lap_time"] > 130)
    ].copy()

    print(f"\nRows with next_lap_time > 130s: {len(slow)}")

    # --- Distribution by race ---

    print("\n" + "-" * 78)
    print("SLOW TARGETS BY RACE")
    print("-" * 78)

    print(
        slow.groupby("race")["next_lap_time"]
        .agg(
            rows="count",
            mean="mean",
            min="min",
            max="max",
        )
        .sort_values("rows", ascending=False)
        .to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --- Distribution by lap ---

    print("\n" + "-" * 78)
    print("SLOW TARGETS BY LAP")
    print("-" * 78)

    lap_stats = (
        slow.groupby(
            ["race", "lap_number"]
        )
        .agg(
            rows=("next_lap_time", "count"),
            mean_target=("next_lap_time", "mean"),
            min_target=("next_lap_time", "min"),
            max_target=("next_lap_time", "max"),
        )
        .sort_values(
            ["race", "lap_number"]
        )
    )

    print(
        lap_stats.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --- Pit stop / pit out ---

    print("\n" + "-" * 78)
    print("SLOW TARGETS VS PIT STATUS")
    print("-" * 78)

    for column in [
        "pit_stop",
        "is_pit_out_lap",
    ]:
        if column in slow.columns:
            print(f"\n{column}:")
            print(
                slow[column]
                .fillna(False)
                .value_counts(dropna=False)
                .to_string()
            )

    # --- Current lap vs target lap ---

    print("\n" + "-" * 78)
    print("CURRENT LAP TIME VS NEXT LAP TIME")
    print("-" * 78)

    columns = [
        "race",
        "lap_number",
        "driver_number",
        "compound",
        "lap_duration",
        "previous_lap_time",
        "next_lap_time",
        "tyre_age",
        "pit_stop",
        "is_pit_out_lap",
        "rainfall",
        "humidity",
        "air_temperature",
        "track_temperature",
    ]

    available = [
        column
        for column in columns
        if column in slow.columns
    ]

    print(
        slow[
            available
        ]
        .sort_values(
            "next_lap_time",
            ascending=False,
        )
        .head(50)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # --- Difference between current and next lap ---

    print("\n" + "-" * 78)
    print("CURRENT → NEXT LAP TIME CHANGE")
    print("-" * 78)

    if (
        "lap_duration" in slow.columns
        and "next_lap_time" in slow.columns
    ):
        temp = slow.dropna(
            subset=[
                "lap_duration",
                "next_lap_time",
            ]
        ).copy()

        temp["lap_time_change"] = (
            temp["next_lap_time"]
            - temp["lap_duration"]
        )

        print(
            temp["lap_time_change"]
            .describe()
            .to_string(
                float_format=lambda x: f"{x:.3f}"
            )
        )

    # --- How many drivers are affected on the same lap? ---

    print("\n" + "-" * 78)
    print("MULTI-CAR SLOW LAPS")
    print("-" * 78)

    multi_car = (
        slow.groupby(
            ["race", "lap_number"]
        )["driver_number"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    print(
        multi_car.head(30).to_string()
    )

    # --- Full slow-lap summary ---

    print("\n" + "-" * 78)
    print("SLOW TARGETS BY COMPOUND")
    print("-" * 78)

    print(
        slow["compound"]
        .fillna("UNKNOWN")
        .value_counts()
        .to_string()
    )

    print("\n" + "-" * 78)
    print("SLOW TARGETS BY RAINFALL")
    print("-" * 78)

    print(
        slow["rainfall"]
        .fillna(0)
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()