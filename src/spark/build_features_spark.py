import argparse
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

MAX_VALID_LAP_TIME = 200.0


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("F1FeaturesSpark")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def build_features(race_state):
    """
    Spark feature engineering.

    A múltbeli feature-ek kizárólag korábbi körökből
    származnak. A next_lap_time célváltozó.

    A 200 másodpercnél hosszabb, nem nullás köridők
    extrém értéknek számítanak, ezért ezeket még a
    feature engineering előtt kiszűrjük.
    """

    # ---------------------------------------------------------
    # EXTREME LAP TIME FILTER
    # ---------------------------------------------------------

    race_state = race_state.filter(
        F.col("lap_duration").isNull()
        | (F.col("lap_duration") <= MAX_VALID_LAP_TIME)
    )

    driver_window = (
        Window
        .partitionBy("driver_number")
        .orderBy("lap_number")
    )

    previous_3_window = (
        driver_window
        .rowsBetween(-3, -1)
    )

    previous_5_window = (
        driver_window
        .rowsBetween(-5, -1)
    )

    # ---------------------------------------------------------
    # PREVIOUS LAP
    # ---------------------------------------------------------

    result = (
        race_state
        .withColumn(
            "previous_lap_time",
            F.lag("lap_duration", 1).over(
                driver_window
            ),
        )
    )

    # ---------------------------------------------------------
    # ROLLING AVERAGES
    # ---------------------------------------------------------

    result = (
        result
        .withColumn(
            "rolling_3_lap_avg",
            F.avg("lap_duration").over(
                previous_3_window
            ),
        )
        .withColumn(
            "rolling_5_lap_avg",
            F.avg("lap_duration").over(
                previous_5_window
            ),
        )
    )

    # ---------------------------------------------------------
    # LAP TIME DELTA
    # ---------------------------------------------------------

    previous_lap = F.lag(
        "lap_duration",
        1,
    ).over(driver_window)

    lap_before_previous = F.lag(
        "lap_duration",
        2,
    ).over(driver_window)

    result = result.withColumn(
        "lap_time_delta",
        previous_lap - lap_before_previous,
    )

    # ---------------------------------------------------------
    # POSITION CHANGE
    # ---------------------------------------------------------

    result = result.withColumn(
        "position_change",
        F.lag(
            "position",
            1,
        ).over(driver_window)
        - F.col("position"),
    )

    # ---------------------------------------------------------
    # GAP CHANGE
    # ---------------------------------------------------------

    result = result.withColumn(
        "gap_change",
        F.col("gap_to_leader")
        - F.lag("gap_to_leader", 1).over(
            driver_window
        ),
    )

    # ---------------------------------------------------------
    # INTERVAL CHANGE
    # ---------------------------------------------------------

    result = result.withColumn(
        "interval_change",
        F.col("interval")
        - F.lag("interval", 1).over(
            driver_window
        ),
    )

    # ---------------------------------------------------------
    # LAPS SINCE PIT
    # ---------------------------------------------------------

    last_pit_lap = F.max(
        F.when(
            F.col("pit_stop") == True,
            F.col("lap_number"),
        )
    ).over(
        driver_window.rowsBetween(
            Window.unboundedPreceding,
            Window.currentRow,
        )
    )

    result = result.withColumn(
        "laps_since_pit",
        F.when(
            last_pit_lap.isNull(),
            F.lit(None).cast("long"),
        ).otherwise(
            F.col("lap_number")
            - last_pit_lap
        ),
    )

    # ---------------------------------------------------------
    # TRACK / AIR TEMPERATURE DIFFERENCE
    # ---------------------------------------------------------

    result = result.withColumn(
        "track_air_temp_difference",
        F.col("track_temperature")
        - F.col("air_temperature"),
    )

    # ---------------------------------------------------------
    # NEXT LAP TIME
    # ---------------------------------------------------------

    result = result.withColumn(
        "next_lap_time",
        F.lead(
            "lap_duration",
            1,
        ).over(
            driver_window
        ),
    )

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    result = result.drop(
        "_pit_count",
        "_lap_count",
    )

    return result


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "race",
        nargs="?",
        default="hungary_2026",
    )

    args = parser.parse_args()

    race = args.race

    input_path = (
        Path("data/processed")
        / race
        / "spark"
        / "race_state_with_conditions.parquet"
    )

    output_path = (
        Path("data/processed")
        / race
        / "spark"
        / "features.parquet"
    )

    print("=== SPARK FEATURE ENGINEERING ===")

    spark = create_spark_session()

    try:

        print("\nLoading race state...")

        race_state = spark.read.parquet(
            str(input_path)
        )

        input_row_count = race_state.count()

        print(
            f"Input rows: "
            f"{input_row_count}"
        )

        print(
            f"Input columns: "
            f"{len(race_state.columns)}"
        )

        # -----------------------------------------------------
        # COUNT EXTREME LAP TIMES
        # -----------------------------------------------------

        extreme_count = race_state.filter(
            F.col("lap_duration").isNotNull()
            & (F.col("lap_duration") > MAX_VALID_LAP_TIME)
        ).count()

        print(
            f"Extreme lap times (> "
            f"{MAX_VALID_LAP_TIME} s): "
            f"{extreme_count}"
        )

        print("\nBuilding features...")

        features = build_features(
            race_state
        )

        features.cache()

        output_row_count = features.count()

        print(
            f"\nOutput rows: "
            f"{output_row_count}"
        )

        print(
            f"Rows removed: "
            f"{input_row_count - output_row_count}"
        )

        print(
            f"Output columns: "
            f"{len(features.columns)}"
        )

        print("\n=== FEATURE COLUMNS ===")

        feature_columns = [
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
        ]

        for column in feature_columns:
            if column in features.columns:
                print(f"- {column}")

        print("\n=== SAMPLE ===")

        features.select(
            "driver_number",
            "lap_number",
            "lap_duration",
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
        ).orderBy(
            "driver_number",
            "lap_number",
        ).show(
            30,
            truncate=False,
        )

        print("\n=== NULL COUNTS ===")

        for column in feature_columns:
            if column not in features.columns:
                continue

            null_count = features.filter(
                F.col(column).isNull()
            ).count()

            print(
                f"{column:<30}"
                f"{null_count}"
            )

        print(
            f"\nSaving to: {output_path}"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            features
            .write
            .mode("overwrite")
            .parquet(
                str(output_path)
            )
        )

        print("\n=== DONE ===")
        print(
            f"Saved: {output_path}"
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()