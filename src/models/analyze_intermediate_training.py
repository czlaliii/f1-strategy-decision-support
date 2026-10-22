from pathlib import Path

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
    print("INTERMEDIATE TRAINING DATA ANALYSIS")
    print("=" * 78)

    df = pd.read_parquet(DATA_PATH)

    intermediate = df[
        df["compound"] == "INTERMEDIATE"
    ].copy()

    print(f"\nTotal intermediate rows: {len(intermediate)}")

    # --- Target distribution by race ---

    print("\n" + "-" * 78)
    print("NEXT LAP TIME BY RACE")
    print("-" * 78)

    race_stats = (
        intermediate
        .groupby("race")["next_lap_time"]
        .agg(
            rows="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
        .sort_values("mean")
    )

    print(
        race_stats.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --- Rainfall + race ---

    print("\n" + "-" * 78)
    print("INTERMEDIATE BY RACE + RAINFALL")
    print("-" * 78)

    race_rain = (
        intermediate
        .assign(
            rainfall=pd.to_numeric(
                intermediate["rainfall"],
                errors="coerce",
            ).fillna(0)
        )
        .groupby(
            ["race", "rainfall"]
        )["next_lap_time"]
        .agg(
            rows="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
    )

    print(
        race_rain.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --- Lap phase ---

    print("\n" + "-" * 78)
    print("INTERMEDIATE BY RACE PHASE")
    print("-" * 78)

    intermediate = intermediate.copy()

    max_lap = intermediate.groupby(
        "race"
    )["lap_number"].transform("max")

    ratio = (
        intermediate["lap_number"]
        / max_lap
    )

    intermediate["lap_phase"] = pd.cut(
        ratio,
        bins=[0, 0.33, 0.66, 1.0],
        labels=["EARLY", "MIDDLE", "LATE"],
        include_lowest=True,
    )

    phase_stats = (
        intermediate
        .groupby(
            ["race", "lap_phase"],
            observed=True,
        )["next_lap_time"]
        .agg(
            rows="count",
            mean="mean",
            median="median",
            std="std",
        )
    )

    print(
        phase_stats.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --- Weather values ---

    print("\n" + "-" * 78)
    print("WEATHER DURING INTERMEDIATE LAPS")
    print("-" * 78)

    weather_columns = [
        "air_temperature",
        "track_temperature",
        "rainfall",
        "humidity",
        "wind_speed",
    ]

    available = [
        column
        for column in weather_columns
        if column in intermediate.columns
    ]

    print(
        intermediate[available]
        .describe()
        .to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --- Extremely slow laps ---

    print("\n" + "-" * 78)
    print("INTERMEDIATE LAPS ABOVE 130 SECONDS")
    print("-" * 78)

    slow = intermediate[
        intermediate["next_lap_time"] > 130
    ]

    print(f"Rows >130s: {len(slow)}")

    print(
        slow[
            [
                "race",
                "lap_number",
                "driver_number",
                "next_lap_time",
                "rainfall",
                "humidity",
                "air_temperature",
                "track_temperature",
            ]
        ]
        .sort_values(
            "next_lap_time",
            ascending=False,
        )
        .head(30)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


if __name__ == "__main__":
    main()