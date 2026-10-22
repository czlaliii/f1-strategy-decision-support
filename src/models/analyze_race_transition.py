from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "multi_race"
)

VALIDATION_PATH = DATA_DIR / "validation_context.parquet"
TRAIN_PATH = DATA_DIR / "train_context.parquet"


def analyze_race(df, race_name):
    df = df[df["race"] == race_name].copy()

    print("\n" + "=" * 78)
    print(race_name)
    print("=" * 78)

    # Race-level statistics per current lap
    lap_stats = (
        df.groupby("lap_number")
        .agg(
            cars=("driver_number", "nunique"),
            median_current_lap=("lap_duration", "median"),
            mean_current_lap=("lap_duration", "mean"),
            median_next_lap=("next_lap_time", "median"),
            mean_next_lap=("next_lap_time", "mean"),
            intermediate_cars=(
                "compound",
                lambda x: (x == "INTERMEDIATE").sum(),
            ),
            wet_cars=(
                "compound",
                lambda x: (x == "WET").sum(),
            ),
            rainfall=("rainfall", "max"),
        )
        .reset_index()
    )

    # Lap-to-lap change in median pace
    lap_stats["median_current_change"] = (
        lap_stats["median_current_lap"]
        - lap_stats["median_current_lap"].shift(1)
    )

    lap_stats["median_next_change"] = (
        lap_stats["median_next_lap"]
        - lap_stats["median_current_lap"]
    )

    lap_stats["intermediate_fraction"] = (
        lap_stats["intermediate_cars"]
        / lap_stats["cars"]
    )

    lap_stats["wet_fraction"] = (
        lap_stats["wet_cars"]
        / lap_stats["cars"]
    )

    # Show the most extreme transitions
    print("\nMOST EXTREME RACE-WIDE PACE CHANGES")
    print("-" * 78)

    print(
        lap_stats[
            [
                "lap_number",
                "cars",
                "median_current_lap",
                "median_next_lap",
                "median_next_change",
                "intermediate_fraction",
                "wet_fraction",
                "rainfall",
            ]
        ]
        .sort_values(
            "median_next_change",
            ascending=False,
        )
        .head(20)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # Show all laps around the largest transition
    if len(lap_stats) > 0:
        largest_idx = (
            lap_stats["median_next_change"]
            .idxmax()
        )

        largest_lap = int(
            lap_stats.loc[
                largest_idx,
                "lap_number",
            ]
        )

        print(
            f"\nLARGEST TRANSITION AROUND LAP "
            f"{largest_lap}"
        )

        print("-" * 78)

        around = lap_stats[
            lap_stats["lap_number"].between(
                largest_lap - 3,
                largest_lap + 3,
            )
        ]

        print(
            around.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    return lap_stats


def main():
    print("=" * 78)
    print("RACE-WIDE TRANSITION ANALYSIS")
    print("=" * 78)

    validation = pd.read_parquet(
        VALIDATION_PATH
    )

    train = pd.read_parquet(
        TRAIN_PATH
    )

    # The problematic validation race
    analyze_race(
        validation,
        "silverstone_2025",
    )

    # A few representative wet/intermediate
    # training races
    for race in [
        "australia_2025",
        "zandvoort_2023",
        "sao_paulo_2024",
    ]:
        analyze_race(
            train,
            race,
        )


if __name__ == "__main__":
    main()