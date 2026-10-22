"""Aggregate OpenF1 car and location telemetry to race-lap level with Spark."""

import time
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType

from src.spark.session import get_spark_session

RAW_DIR = Path("data/raw/hungary_2026")
OUTPUT_PATH = Path("data/processed/hungary_2026/spark/telemetry_lap_aggregates.parquet")

CAR_SCHEMA = StructType(
    [
        StructField("date", StringType(), False),
        StructField("session_key", LongType(), False),
        StructField("meeting_key", LongType(), False),
        StructField("driver_number", IntegerType(), False),
        StructField("brake", IntegerType(), True),
        StructField("rpm", IntegerType(), True),
        StructField("throttle", IntegerType(), True),
        StructField("drs", IntegerType(), True),
        StructField("speed", IntegerType(), True),
        StructField("n_gear", IntegerType(), True),
    ]
)
LOCATION_SCHEMA = StructType(
    [
        StructField("date", StringType(), False),
        StructField("session_key", LongType(), False),
        StructField("meeting_key", LongType(), False),
        StructField("driver_number", IntegerType(), False),
        StructField("x", IntegerType(), True),
        StructField("y", IntegerType(), True),
        StructField("z", IntegerType(), True),
    ]
)
LAP_SCHEMA = StructType(
    [
        StructField("session_key", LongType(), False),
        StructField("driver_number", IntegerType(), False),
        StructField("lap_number", IntegerType(), False),
        StructField("date_start", StringType(), True),
        StructField("lap_duration", StringType(), True),
    ]
)


def load_telemetry(spark) -> DataFrame:
    """Load both telemetry sources with explicit schemas and normalized columns."""
    car = (
        spark.read.schema(CAR_SCHEMA).json(str(RAW_DIR / "car_data_driver_*.json"))
        .select(
            "date", "session_key", "meeting_key", "driver_number", "brake", "rpm",
            "throttle", "drs", "speed", "n_gear",
        )
        .withColumn("source", F.lit("car"))
        .withColumn("x", F.lit(None).cast("double"))
        .withColumn("y", F.lit(None).cast("double"))
        .withColumn("z", F.lit(None).cast("double"))
    )
    location = (
        spark.read.schema(LOCATION_SCHEMA).json(str(RAW_DIR / "location_driver_*.json"))
        .select("date", "session_key", "meeting_key", "driver_number", "x", "y", "z")
        .withColumn("source", F.lit("location"))
        .withColumn("brake", F.lit(None).cast("int"))
        .withColumn("rpm", F.lit(None).cast("int"))
        .withColumn("throttle", F.lit(None).cast("int"))
        .withColumn("drs", F.lit(None).cast("int"))
        .withColumn("speed", F.lit(None).cast("int"))
        .withColumn("n_gear", F.lit(None).cast("int"))
    )
    return (
        car.unionByName(location)
        .withColumn("timestamp", F.to_timestamp("date"))
        .drop("date")
        .withColumn("speed", F.col("speed").cast("double"))
        .withColumn("throttle", F.col("throttle").cast("double"))
        .withColumn("brake", F.col("brake").cast("double"))
        .withColumn("rpm", F.col("rpm").cast("double"))
        .withColumn("x", F.col("x").cast("double"))
        .withColumn("y", F.col("y").cast("double"))
        .withColumn("z", F.col("z").cast("double"))
    )


def load_race_laps(spark) -> DataFrame:
    """Load valid race-lap intervals used only to assign telemetry samples to laps."""
    laps = spark.read.schema(LAP_SCHEMA).json(str(RAW_DIR / "laps.json"))
    return (
        laps.withColumn("lap_timestamp_start", F.to_timestamp("date_start"))
        .withColumn("lap_duration_seconds", F.col("lap_duration").cast("double"))
        .drop("date_start", "lap_duration")
        .filter(F.col("lap_timestamp_start").isNotNull() & F.col("lap_duration_seconds").isNotNull())
        .withColumn(
            "lap_timestamp_end",
            F.expr("lap_timestamp_start + INTERVAL 1 MICROSECOND * CAST(lap_duration_seconds * 1000000 AS BIGINT)"),
        )
    )


def aggregate_to_laps(telemetry: DataFrame, laps: DataFrame) -> DataFrame:
    """Assign telemetry to lap intervals and calculate source-appropriate aggregates."""
    telemetry_in_race = telemetry.join(
        F.broadcast(laps),
        (
            (telemetry.driver_number == laps.driver_number)
            & (telemetry.session_key == laps.session_key)
            & (telemetry.timestamp >= laps.lap_timestamp_start)
            & (telemetry.timestamp < laps.lap_timestamp_end)
        ),
        "inner",
    ).select(
        telemetry["*"],
        laps["lap_number"],
        laps["lap_timestamp_start"],
        laps["lap_timestamp_end"],
        laps["lap_duration_seconds"],
    )

    return telemetry_in_race.groupBy(
        "driver_number", "session_key", "lap_number", "lap_timestamp_start",
        "lap_timestamp_end", "lap_duration_seconds",
    ).agg(
        F.min("timestamp").alias("telemetry_timestamp_start"),
        F.max("timestamp").alias("telemetry_timestamp_end"),
        F.sum(F.when(F.col("source") == "car", 1).otherwise(0)).alias("car_sample_count"),
        F.sum(F.when(F.col("source") == "location", 1).otherwise(0)).alias("location_sample_count"),
        F.avg("speed").alias("avg_speed"),
        F.max("speed").alias("max_speed"),
        F.avg("throttle").alias("avg_throttle"),
        F.avg("brake").alias("avg_brake"),
        F.avg("rpm").alias("avg_rpm"),
        F.avg("x").alias("avg_x"),
        F.avg("y").alias("avg_y"),
        F.avg("z").alias("avg_z"),
    )


def main():
    started_at = time.perf_counter()
    spark = get_spark_session("F1TelemetryLapAggregation")
    telemetry = load_telemetry(spark).cache()
    input_rows = telemetry.count()
    laps = load_race_laps(spark)
    aggregated = aggregate_to_laps(telemetry, laps)
    output_rows = aggregated.count()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    aggregated.write.mode("overwrite").parquet(str(OUTPUT_PATH))
    elapsed = time.perf_counter() - started_at
    print("=== SPARK PROCESSING ===")
    print(f"Input rows: {input_rows}")
    print(f"Output rows: {output_rows}")
    print(f"Processing time: {elapsed:.3f} seconds")
    print(f"Output: {OUTPUT_PATH}")
    telemetry.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
