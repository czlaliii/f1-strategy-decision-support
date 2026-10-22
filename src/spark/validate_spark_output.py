"""Validate Spark telemetry aggregates against the existing pandas race-state keys."""

from pathlib import Path

import pandas as pd

from src.spark.process_telemetry import OUTPUT_PATH

PANDAS_REFERENCE_PATH = Path("data/processed/hungary_2026/race_state.parquet")


def status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def numeric_summary(series: pd.Series) -> dict:
    return {
        "count": int(series.count()),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def main():
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Spark output not found at {OUTPUT_PATH}. Run process_telemetry first."
        )

    spark_output = pd.read_parquet(OUTPUT_PATH)
    pandas_reference = pd.read_parquet(PANDAS_REFERENCE_PATH)
    spark_keys = set(zip(spark_output.driver_number, spark_output.lap_number))
    pandas_keys = set(zip(pandas_reference.driver_number, pandas_reference.lap_number))
    spark_start = pd.to_datetime(spark_output["lap_timestamp_start"], utc=True)
    spark_end = pd.to_datetime(spark_output["lap_timestamp_end"], utc=True)
    pandas_start = pd.to_datetime(pandas_reference["date_start"], utc=True)
    pandas_end = pd.to_datetime(pandas_reference["date_end"], utc=True)

    rows_match = len(spark_output) == len(pandas_reference)
    drivers_match = set(spark_output.driver_number) == set(pandas_reference.driver_number)
    lap_range_match = (
        spark_output.lap_number.min() == pandas_reference.lap_number.min()
        and spark_output.lap_number.max() == pandas_reference.lap_number.max()
    )
    timestamps_match = (
        spark_start.min() == pandas_start.min()
        and spark_end.max() == pandas_end.max()
    )
    keys_match = spark_keys == pandas_keys
    duration_match = numeric_summary(spark_output["lap_duration_seconds"]) == numeric_summary(
        pandas_reference["lap_duration"]
    )

    print("=== SPARK VALIDATION ===")
    print(f"Spark rows: {len(spark_output)}")
    print(f"Pandas race-state rows: {len(pandas_reference)}")
    print(f"Unique drivers: {spark_output.driver_number.nunique()}")
    print(f"Lap range: {spark_output.lap_number.min()} - {spark_output.lap_number.max()}")
    print("\nTimestamp range:")
    print(f"Spark lap start/end: {spark_start.min()} -> {spark_end.max()}")
    print(f"Pandas start/end:    {pandas_start.min()} -> {pandas_end.max()}")
    print("\nComparison with pandas race_state:")
    print(f"Rows: {status(rows_match)}")
    print(f"Drivers: {status(drivers_match)}")
    print(f"Lap range: {status(lap_range_match)}")
    print(f"Timestamp range: {status(timestamps_match)}")
    print(f"Driver/lap keys: {status(keys_match)}")
    print(f"Lap-duration summary: {status(duration_match)}")
    print("\nNumeric aggregates:")
    print("lap_duration_seconds (Spark):", numeric_summary(spark_output["lap_duration_seconds"]))
    print("lap_duration (pandas):", numeric_summary(pandas_reference["lap_duration"]))
    print("avg_speed (Spark):", numeric_summary(spark_output["avg_speed"]))
    print("car_sample_count (Spark):", numeric_summary(spark_output["car_sample_count"]))


if __name__ == "__main__":
    main()
