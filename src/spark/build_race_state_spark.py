import argparse
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("F1RaceStateSpark")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def load_parquet(
    spark: SparkSession,
    path: Path,
):
    return spark.read.parquet(str(path))


def load_json(
    spark: SparkSession,
    path: Path,
):
    return (
        spark.read
        .option("multiLine", "true")
        .json(str(path))
    )


def add_date_end(lap_data):
    """Compute lap end from lap start and lap duration."""

    result = (
        lap_data
        .withColumn(
            "date_start",
            F.to_timestamp("date_start"),
        )
        .withColumn(
            "lap_duration",
            F.col("lap_duration").cast("double"),
        )
        .filter(
            F.col("lap_duration").isNotNull()
        )
        .withColumn(
            "date_end",
            F.timestamp_micros(
                (
                    F.unix_micros(
                        F.col("date_start")
                    )
                    + F.col("lap_duration") * 1_000_000
                ).cast("long")
            ),
        )
    )

    return result


def asof_join_driver(
    laps,
    events,
    event_time_column,
    value_columns,
):
    """Per-driver backward as-of join.

    Each lap gets the latest event value not later than its date_end.
    """

    lap_columns = laps.columns

    lap_rows = (
        laps
        .select(
            *lap_columns,
        )
        .withColumn(
            "_event_time",
            F.col("date_end"),
        )
        .withColumn(
            "_row_type",
            F.lit(1),
        )
    )

    event_rows = (
        events
        .withColumn(
            "_event_time",
            F.col(event_time_column),
        )
        .withColumn(
            "_row_type",
            F.lit(0),
        )
        .select(
            "driver_number",
            "_event_time",
            "_row_type",
            *[
                F.col(column)
                for column in value_columns
            ],
        )
    )

    # Events and laps share one timeline.
    combined = (
        lap_rows
        .select(
            *lap_columns,
            F.col("_event_time"),
            F.col("_row_type"),
            *[
                F.lit(None).cast(
                    events.schema[column].dataType
                ).alias(column)
                for column in value_columns
            ],
        )
        .unionByName(
            event_rows,
            allowMissingColumns=True,
        )
    )

    window = (
        Window
        .partitionBy("driver_number")
        .orderBy(
            F.col("_event_time"),
            F.col("_row_type"),
        )
        .rowsBetween(
            Window.unboundedPreceding,
            Window.currentRow,
        )
    )

    result = combined

    for column in value_columns:

        last_event = F.last(
            F.when(
                F.col("_row_type") == 0,
                F.struct(
                    F.col("_event_time").alias("event_time"),
                    F.col(column).alias("value"),
                ),
            ),
            ignorenulls=True,
        ).over(window)

        result = result.withColumn(
            column,
            last_event.getField("value"),
        )

    return (
        result
        .filter(
            F.col("_row_type") == 1
        )
        .select(
            *lap_columns,
            *value_columns,
        )
    )


def asof_join_weather(
    laps,
    weather,
    value_columns,
):
    """Backward as-of join of weather data.

    Weather is not per driver, so a global timeline is used.
    """

    lap_columns = laps.columns

    lap_rows = (
        laps
        .withColumn(
            "_event_time",
            F.col("date_end"),
        )
        .withColumn(
            "_row_type",
            F.lit(1),
        )
    )

    weather_rows = (
        weather
        .withColumn(
            "_event_time",
            F.col("date"),
        )
        .withColumn(
            "_row_type",
            F.lit(0),
        )
    )

    combined = (
        lap_rows
        .select(
            *lap_columns,
            "_event_time",
            "_row_type",
            *[
                F.lit(None).cast(
                    weather.schema[column].dataType
                ).alias(column)
                for column in value_columns
            ],
        )
        .unionByName(
            weather_rows.select(
                F.lit(None).cast(
                    laps.schema["driver_number"].dataType
                ).alias("driver_number"),
                *[
                    F.lit(None).cast(
                        laps.schema[column].dataType
                    ).alias(column)
                    for column in lap_columns
                    if column != "driver_number"
                ],
                "_event_time",
                "_row_type",
                *value_columns,
            ),
            allowMissingColumns=True,
        )
    )

    window = (
        Window
        .orderBy(
            F.col("_event_time"),
            F.col("_row_type"),
        )
        .rowsBetween(
            Window.unboundedPreceding,
            Window.currentRow,
        )
    )

    result = combined

    for column in value_columns:
        result = result.withColumn(
            column,
            F.last(
                F.col(column),
                ignorenulls=True,
            ).over(window),
        )

    return (
        result
        .filter(
            F.col("_row_type") == 1
        )
        .select(
            *lap_columns,
            *value_columns,
        )
    )


def build_race_state(
    lap_data,
    position,
    intervals,
    weather,
):
    """Spark equivalent of the pandas build_race_state."""

    # --- DATE END ---

    race_state = add_date_end(
        lap_data
    )

    # --- POSITION ---

    position = (
        position
        .withColumn(
            "date",
            F.to_timestamp("date"),
        )
        .select(
            "driver_number",
            "date",
            F.col("position").cast("double").alias(
                "position"
            ),
        )
    )

    # Use only position data within the actual race window.
    race_start = race_state.select(
        F.min("date_start")
    ).first()[0]

    race_end = race_state.select(
        F.max("date_end")
    ).first()[0]

    position = position.filter(
        (F.col("date") >= F.lit(race_start))
        & (F.col("date") <= F.lit(race_end))
    )

    race_state = asof_join_driver(
        laps=race_state,
        events=position,
        event_time_column="date",
        value_columns=["position"],
    )

    # --- INTERVALS ---

    intervals = (
        intervals
        .withColumn(
            "date",
            F.to_timestamp("date"),
        )
        .withColumn(
            "_gap_to_leader_raw",
            F.upper(
                F.col("gap_to_leader").cast("string")
            ),
        )
        .withColumn(
            "laps_down",
            F.when(
                F.col("_gap_to_leader_raw").rlike(
                    r"\+[0-9]+\s+LAP"
                ),
                F.regexp_extract(
                    F.col("_gap_to_leader_raw"),
                    r"\+([0-9]+)\s+LAP",
                    1,
                ).cast("double"),
            ).otherwise(
                F.lit(0.0)
            ),
        )
        .select(
            "driver_number",
            "date",
            F.col("gap_to_leader")
            .cast("double")
            .alias("gap_to_leader"),
            F.col("interval")
            .cast("double")
            .alias("interval"),
            "laps_down",
        )
    )

    race_state = asof_join_driver(
        laps=race_state,
        events=intervals,
        event_time_column="date",
        value_columns=[
            "gap_to_leader",
            "interval",
            "laps_down",
        ],
    )

    # --- WEATHER ---

    weather = (
        weather
        .withColumn(
            "date",
            F.to_timestamp("date"),
        )
    )

    weather_columns = [
        column
        for column in [
            "air_temperature",
            "track_temperature",
            "humidity",
            "pressure",
            "rainfall",
            "wind_speed",
            "wind_direction",
        ]
        if column in weather.columns
    ]

    if weather_columns:
        race_state = asof_join_weather(
            laps=race_state,
            weather=weather,
            value_columns=weather_columns,
        )

    # --- SORT ---

    race_state = race_state.orderBy(
        "driver_number",
        "lap_number",
    )

    return race_state


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "race",
        nargs="?",
        default="hungary_2026",
    )

    args = parser.parse_args()

    race = args.race

    data_dir = (
        Path("data/raw")
        / race
    )

    processed_dir = (
        Path("data/processed")
        / race
        / "spark"
    )

    lap_data_path = (
        processed_dir
        / "lap_data.parquet"
    )

    position_path = (
        data_dir
        / "position.json"
    )

    intervals_path = (
        data_dir
        / "intervals.json"
    )

    weather_path = (
        data_dir
        / "weather.json"
    )

    output_path = (
        processed_dir
        / "race_state.parquet"
    )

    print(
        "=== SPARK RACE STATE PROCESSING ==="
    )

    spark = create_spark_session()

    try:

        print("\nLoading data...")

        lap_data = load_parquet(
            spark,
            lap_data_path,
        )

        position = load_json(
            spark,
            position_path,
        )

        intervals = load_json(
            spark,
            intervals_path,
        )

        weather = load_json(
            spark,
            weather_path,
        )

        print(
            f"Lap data: {lap_data.count()}"
        )

        print(
            f"Position records: "
            f"{position.count()}"
        )

        print(
            f"Interval records: "
            f"{intervals.count()}"
        )

        print(
            f"Weather records: "
            f"{weather.count()}"
        )

        print(
            "\nBuilding race state..."
        )

        race_state = build_race_state(
            lap_data=lap_data,
            position=position,
            intervals=intervals,
            weather=weather,
        )

        race_state.cache()

        row_count = race_state.count()

        print(
            f"\nRows: {row_count}"
        )

        print(
            f"Columns: "
            f"{len(race_state.columns)}"
        )

        print(
            "\n=== RESULT ==="
        )

        race_state.show(
            20,
            truncate=False,
        )

        print(
            "\n=== RESULT SCHEMA ==="
        )

        race_state.printSchema()

        print(
            f"\nSaving to: {output_path}"
        )

        processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            race_state
            .write
            .mode("overwrite")
            .parquet(
                str(output_path)
            )
        )

        print(
            "\n=== DONE ==="
        )

        print(
            f"Saved: {output_path}"
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()