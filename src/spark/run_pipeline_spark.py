import time
from pathlib import Path

from pyspark.sql import SparkSession

from src.spark.add_race_conditions_spark import (
    add_vsc_features,
    build_vsc_intervals,
)
from src.spark.build_features_spark import (
    build_features,
)
from src.spark.build_lap_data_spark import (
    build_lap_data,
)
from src.spark.build_lap_data_spark import (
    load_json as load_lap_json,
)
from src.spark.build_race_state_spark import (
    build_race_state,
)
from src.spark.build_race_state_spark import (
    load_json as load_race_json,
)

RACES = [
    "bahrain_2023",
    "monaco_2023",
    "zandvoort_2023",
    "imola_2024",
    "canada_2024",
    "silverstone_2024",
    "sao_paulo_2024",
    "australia_2025",
    "silverstone_2025",
    "hungary_2026",
]


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("F1MultiRacePipeline")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def process_race(
    spark: SparkSession,
    race: str,
) -> dict:

    start_time = time.perf_counter()

    print()
    print("=" * 70)
    print(f"PROCESSING RACE: {race}")
    print("=" * 70)

    data_dir = Path("data/raw") / race
    processed_dir = Path("data/processed") / race / "spark"

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # INPUT PATHS
    # ---------------------------------------------------------

    laps_path = data_dir / "laps.json"
    stints_path = data_dir / "stints.json"
    pit_path = data_dir / "pit.json"

    position_path = data_dir / "position.json"
    intervals_path = data_dir / "intervals.json"
    weather_path = data_dir / "weather.json"
    race_control_path = data_dir / "race_control.json"

    # ---------------------------------------------------------
    # OUTPUT PATHS
    # ---------------------------------------------------------

    lap_data_path = (
        processed_dir / "lap_data.parquet"
    )

    race_state_path = (
        processed_dir / "race_state.parquet"
    )

    conditions_path = (
        processed_dir / "race_state_with_conditions.parquet"
    )

    features_path = (
        processed_dir / "features.parquet"
    )

    # ---------------------------------------------------------
    # LOAD RAW DATA
    # ---------------------------------------------------------

    print("\nLoading raw data...")

    laps = load_lap_json(
        spark,
        laps_path,
    )

    stints = load_lap_json(
        spark,
        stints_path,
    )

    pit = load_lap_json(
        spark,
        pit_path,
    )

    position = load_race_json(
        spark,
        position_path,
    )

    intervals = load_race_json(
        spark,
        intervals_path,
    )

    weather = load_race_json(
        spark,
        weather_path,
    )

    race_control = load_race_json(
        spark,
        race_control_path,
    )

    print(f"Laps:           {laps.count()}")
    print(f"Stints:         {stints.count()}")
    print(f"Pit stops:      {pit.count()}")
    print(f"Position:       {position.count()}")
    print(f"Intervals:      {intervals.count()}")
    print(f"Weather:        {weather.count()}")
    print(f"Race control:   {race_control.count()}")

    # ---------------------------------------------------------
    # 1. LAP DATA
    # ---------------------------------------------------------

    print("\n[1/4] Building lap data...")

    lap_data = build_lap_data(
        laps=laps,
        stints=stints,
        pit=pit,
    )

    lap_data = lap_data.cache()

    lap_count = lap_data.count()

    print(
        f"Lap data rows: {lap_count}"
    )

    (
        lap_data
        .write
        .mode("overwrite")
        .parquet(
            str(lap_data_path)
        )
    )

    print(
        f"Saved: {lap_data_path}"
    )

    # ---------------------------------------------------------
    # 2. RACE STATE
    # ---------------------------------------------------------

    print("\n[2/4] Building race state...")

    race_state = build_race_state(
        lap_data=lap_data,
        position=position,
        intervals=intervals,
        weather=weather,
    )

    race_state = race_state.cache()

    race_state_count = race_state.count()

    print(
        f"Race state rows: {race_state_count}"
    )

    (
        race_state
        .write
        .mode("overwrite")
        .parquet(
            str(race_state_path)
        )
    )

    print(
        f"Saved: {race_state_path}"
    )

    # ---------------------------------------------------------
    # 3. RACE CONDITIONS / VSC
    # ---------------------------------------------------------

    print("\n[3/4] Adding race conditions...")

    vsc_intervals = build_vsc_intervals(
        race_control
    )

    print(
        f"VSC intervals: {len(vsc_intervals)}"
    )

    race_state_with_conditions = add_vsc_features(
        race_state,
        vsc_intervals,
    )

    race_state_with_conditions = (
        race_state_with_conditions.cache()
    )

    conditions_count = (
        race_state_with_conditions.count()
    )

    print(
        f"Condition rows: {conditions_count}"
    )

    (
        race_state_with_conditions
        .write
        .mode("overwrite")
        .parquet(
            str(conditions_path)
        )
    )

    print(
        f"Saved: {conditions_path}"
    )

    # ---------------------------------------------------------
    # 4. FEATURE ENGINEERING
    # ---------------------------------------------------------

    print("\n[4/4] Building features...")

    features = build_features(
        race_state_with_conditions
    )

    features = features.cache()

    feature_count = features.count()

    print(
        f"Feature rows: {feature_count}"
    )

    print(
        f"Feature columns: {len(features.columns)}"
    )

    (
        features
        .write
        .mode("overwrite")
        .parquet(
            str(features_path)
        )
    )

    print(
        f"Saved: {features_path}"
    )

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    features.unpersist()
    race_state_with_conditions.unpersist()
    race_state.unpersist()
    lap_data.unpersist()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        f"FINISHED: {race}"
    )
    print(
        f"Total time: {elapsed:.2f} s"
    )

    return {
        "race": race,
        "lap_data_rows": lap_count,
        "race_state_rows": race_state_count,
        "conditions_rows": conditions_count,
        "features_rows": feature_count,
        "feature_columns": len(features.columns),
        "vsc_intervals": len(vsc_intervals),
        "elapsed_seconds": elapsed,
    }


def main() -> None:

    print("=== F1 MULTI-RACE SPARK PIPELINE ===")

    spark = create_spark_session()

    results = []

    try:

        for race in RACES:

            try:

                result = process_race(
                    spark,
                    race,
                )

                results.append(result)

            except Exception as exc:

                print()
                print(
                    f"ERROR processing {race}:"
                )
                print(exc)

                results.append(
                    {
                        "race": race,
                        "error": str(exc),
                    }
                )

    finally:
        spark.stop()

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    for result in results:

        race = result["race"]

        if "error" in result:

            print(
                f"{race:<20} FAILED"
            )
            continue

        print(
            f"{race:<20} "
            f"features={result['features_rows']:<5} "
            f"VSC={result['vsc_intervals']:<3} "
            f"time={result['elapsed_seconds']:.2f}s"
        )

    successful = sum(
        1
        for result in results
        if "error" not in result
    )

    failed = len(results) - successful

    print()
    print(
        f"Successful: {successful}/{len(RACES)}"
    )
    print(
        f"Failed:     {failed}/{len(RACES)}"
    )


if __name__ == "__main__":
    main()