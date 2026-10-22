"""Add VSC and Safety Car condition features to a race-state Parquet file.

Usage from the repository root:
    python -m src.spark.add_race_conditions_spark silverstone_2025

Timestamps are treated as UTC. Output is written via pandas, so winutils.exe
is not needed on Windows.
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StructField,
    StructType,
    TimestampType,
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

RACE = sys.argv[1] if len(sys.argv) > 1 else "hungary_2026"

RACE_STATE_PATH = (
    DATA_DIR / "processed" / RACE / "spark" / "race_state.parquet"
)
RACE_CONTROL_PATH = DATA_DIR / "raw" / RACE / "race_control.json"
OUTPUT_PATH = (
    DATA_DIR / "processed" / RACE / "spark" / "race_state_with_conditions.parquet"
)

KEY_COLUMNS = ["driver_number", "lap_number"]
TIME_COLUMNS = ["date_start", "date_end"]
FEATURE_COLUMNS = [
    "vsc_duration_in_lap",
    "vsc_in_lap",
    "vsc_lap_fraction",
    "vsc_active_at_end",
    "sc_duration_in_lap",
    "sc_in_lap",
    "sc_lap_fraction",
    "sc_active_at_end",
]


def parse_timestamp(value):
    """Parse a race-control timestamp and return an aware UTC datetime."""
    if not value:
        return None

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_utc_datetime(value):
    """Interpret naive timestamps as UTC and convert aware timestamps to UTC."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def utc_naive(value):
    """Convert a datetime to UTC and remove tzinfo for Spark TimestampType."""
    if value is None:
        return None
    return to_utc_datetime(value).replace(tzinfo=None)


def load_race_control():
    with open(RACE_CONTROL_PATH, "r", encoding="utf-8") as file:
        events = json.load(file)
    if not isinstance(events, list):
        raise ValueError(f"Expected a JSON list in {RACE_CONTROL_PATH}")
    return events


def resolve_sc_end_timestamp(event_timestamp, race_state_rows):
    """Resolve 'SAFETY CAR IN THIS LAP' to the earliest end of the target lap.

    The target lap is the most common lap among drivers whose lap contains the
    event; its earliest date_end is the interval endpoint.
    """
    active_rows = []

    for row in race_state_rows:
        start = to_utc_datetime(row["date_start"])
        end = to_utc_datetime(row["date_end"])
        if (
            start is not None
            and end is not None
            and start <= event_timestamp < end
        ):
            active_rows.append((int(row["lap_number"]), end))

    if not active_rows:
        print(
            "WARNING: Could not match SC IN THIS LAP event to a lap: "
            f"{event_timestamp}. Using the original event timestamp."
        )
        return event_timestamp

    lap_counts = {}
    for lap_number, _ in active_rows:
        lap_counts[lap_number] = lap_counts.get(lap_number, 0) + 1

    target_lap = max(lap_counts, key=lap_counts.get)
    lap_end_candidates = [
        end for lap_number, end in active_rows if lap_number == target_lap
    ]
    resolved_end = min(lap_end_candidates)

    if resolved_end <= event_timestamp:
        print(
            "WARNING: Resolved SC end is not after event timestamp; "
            "using the original event timestamp."
        )
        return event_timestamp

    print(
        f"SC event at {event_timestamp}: lap {target_lap}, "
        f"resolved end {resolved_end}"
    )
    return resolved_end


def build_intervals(events, race_state_rows):
    """Pair VSC/SC deployment messages with their corresponding end messages."""
    vsc_intervals = []
    sc_intervals = []
    vsc_start = None
    sc_start = None

    for event in sorted(events, key=lambda row: row.get("date", "")):
        message = (event.get("message") or "").strip().upper()
        timestamp = parse_timestamp(event.get("date"))
        if timestamp is None:
            continue

        if message == "VIRTUAL SAFETY CAR DEPLOYED":
            if vsc_start is None:
                vsc_start = timestamp
        elif message == "VIRTUAL SAFETY CAR ENDING":
            if vsc_start is not None:
                if timestamp > vsc_start:
                    vsc_intervals.append((vsc_start, timestamp))
                else:
                    print(f"WARNING: Invalid VSC interval: {vsc_start} -> {timestamp}")
                vsc_start = None

        elif message == "SAFETY CAR DEPLOYED":
            if sc_start is None:
                sc_start = timestamp
        elif message == "SAFETY CAR IN THIS LAP":
            if sc_start is not None:
                sc_end = resolve_sc_end_timestamp(timestamp, race_state_rows)
                if sc_end > sc_start:
                    sc_intervals.append((sc_start, sc_end))
                else:
                    print(f"WARNING: Invalid SC interval: {sc_start} -> {sc_end}")
                sc_start = None

    if vsc_start is not None:
        print(f"WARNING: Unclosed VSC interval starting at {vsc_start}")
    if sc_start is not None:
        print(f"WARNING: Unclosed Safety Car interval starting at {sc_start}")

    return vsc_intervals, sc_intervals


def build_interval_dataframe(spark, intervals, condition_name):
    """Create a Spark interval DataFrame with consistent UTC-naive timestamps."""
    schema = StructType(
        [
            StructField("condition_start", TimestampType(), nullable=False),
            StructField("condition_end", TimestampType(), nullable=False),
        ]
    )
    rows = [(utc_naive(start), utc_naive(end)) for start, end in intervals]
    return spark.createDataFrame(rows, schema=schema).withColumn(
        "condition", F.lit(condition_name)
    )


def add_condition_features(race_state, intervals_df, prefix):
    """Add condition overlap duration, indicator, fraction, and active-at-end."""
    joined = (
        race_state.alias("r")
        .crossJoin(intervals_df.alias("c"))
        .filter(
            (F.col("c.condition_start") < F.col("r.date_end"))
            & (F.col("c.condition_end") > F.col("r.date_start"))
        )
        .withColumn(
            "_overlap_start",
            F.greatest(F.col("r.date_start"), F.col("c.condition_start")),
        )
        .withColumn(
            "_overlap_end",
            F.least(F.col("r.date_end"), F.col("c.condition_end")),
        )
        .withColumn(
            "_overlap_seconds",
            F.when(
                F.col("_overlap_end") > F.col("_overlap_start"),
                F.col("_overlap_end").cast("double")
                - F.col("_overlap_start").cast("double"),
            ).otherwise(F.lit(0.0)),
        )
    )

    condition_duration = joined.groupBy(
        F.col("r.driver_number").alias("driver_number"),
        F.col("r.lap_number").alias("lap_number"),
    ).agg(
        F.sum("_overlap_seconds").alias(f"{prefix}_duration_in_lap"),
        F.max(
            F.when(F.col("c.condition_end") > F.col("r.date_end"), 1).otherwise(0)
        ).alias(f"{prefix}_active_at_end"),
    )

    return (
        race_state.join(condition_duration, on=KEY_COLUMNS, how="left")
        .withColumn(
            f"{prefix}_duration_in_lap",
            F.coalesce(F.col(f"{prefix}_duration_in_lap"), F.lit(0.0)),
        )
        .withColumn(
            f"{prefix}_in_lap",
            F.when(F.col(f"{prefix}_duration_in_lap") > 0, 1)
            .otherwise(0)
            .cast(IntegerType()),
        )
        .withColumn(
            f"{prefix}_lap_fraction",
            F.when(
                F.col("lap_duration") > 0,
                F.least(
                    F.lit(1.0),
                    F.greatest(
                        F.lit(0.0),
                        F.col(f"{prefix}_duration_in_lap")
                        / F.col("lap_duration"),
                    ),
                ),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            f"{prefix}_active_at_end",
            F.coalesce(F.col(f"{prefix}_active_at_end"), F.lit(0)).cast(IntegerType()),
        )
    )


def add_zero_features(race_state, prefix):
    """Add correctly typed zero-valued features if no intervals were found."""
    return (
        race_state.withColumn(f"{prefix}_duration_in_lap", F.lit(0.0))
        .withColumn(f"{prefix}_in_lap", F.lit(0).cast(IntegerType()))
        .withColumn(f"{prefix}_lap_fraction", F.lit(0.0))
        .withColumn(f"{prefix}_active_at_end", F.lit(0).cast(IntegerType()))
    )


def main():
    print("=" * 78)
    print("SPARK RACE CONDITIONS")
    print("=" * 78)
    print(f"\nRace: {RACE}")
    print(f"Race state:    {RACE_STATE_PATH}")
    print(f"Race control:  {RACE_CONTROL_PATH}")
    print(f"Output:        {OUTPUT_PATH}")

    if not RACE_STATE_PATH.exists():
        raise FileNotFoundError(f"Race-state Parquet not found: {RACE_STATE_PATH}")
    if not RACE_CONTROL_PATH.exists():
        raise FileNotFoundError(f"Race-control JSON not found: {RACE_CONTROL_PATH}")

    # Keep the complete original data in pandas so no original columns are lost.
    race_state_pd_full = pd.read_parquet(RACE_STATE_PATH)
    missing = [
        col for col in [*KEY_COLUMNS, *TIME_COLUMNS, "lap_duration"]
        if col not in race_state_pd_full.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns in race_state: {missing}")

    race_state_pd_full = race_state_pd_full.copy()
    race_state_pd_full["_row_order_for_merge"] = range(len(race_state_pd_full))

    # Only pass the simple columns needed by Spark; the full pandas data stays intact.
    race_state_pd = race_state_pd_full[
        [*KEY_COLUMNS, *TIME_COLUMNS, "lap_duration"]
    ].copy()
    for column in TIME_COLUMNS:
        # The source file has timezone-naive timestamps representing UTC.
        race_state_pd[column] = (
            pd.to_datetime(race_state_pd[column], utc=True).dt.tz_localize(None)
        )

    spark = (
        SparkSession.builder.appName(f"RaceConditions_{RACE}")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        race_state = spark.createDataFrame(race_state_pd)
        print(f"\nRace state rows: {race_state.count()}")
        print(f"Race control events: {len(load_race_control())}")

        # These collected timestamps are naive UTC because Spark session TZ is UTC.
        race_state_rows = race_state.select(
            "lap_number", "date_start", "date_end"
        ).collect()

        events = load_race_control()
        vsc_intervals, sc_intervals = build_intervals(events, race_state_rows)

        print("\nVSC intervals found:", len(vsc_intervals))
        for start, end in vsc_intervals:
            print(f"VSC: {start} -> {end}")
        print("\nSafety Car intervals found:", len(sc_intervals))
        for start, end in sc_intervals:
            print(f"SC:  {start} -> {end}")

        if vsc_intervals:
            vsc_df = build_interval_dataframe(spark, vsc_intervals, "VSC")
            race_state = add_condition_features(race_state, vsc_df, "vsc")
        else:
            race_state = add_zero_features(race_state, "vsc")

        if sc_intervals:
            sc_df = build_interval_dataframe(spark, sc_intervals, "SC")
            race_state = add_condition_features(race_state, sc_df, "sc")
        else:
            race_state = add_zero_features(race_state, "sc")

        print(f"\nRows: {race_state.count()}")
        print(f"Columns in Spark calculation frame: {len(race_state.columns)}")
        print("\n=== CONDITION SUMMARY ===")
        race_state.select(
            F.sum("vsc_in_lap").alias("vsc_rows"),
            F.sum("sc_in_lap").alias("sc_rows"),
            F.max("vsc_duration_in_lap").alias("max_vsc_duration"),
            F.max("sc_duration_in_lap").alias("max_sc_duration"),
            F.max("vsc_lap_fraction").alias("max_vsc_fraction"),
            F.max("sc_lap_fraction").alias("max_sc_fraction"),
        ).show(truncate=False)

        print("\n=== VSC LAPS ===")
        race_state.filter(F.col("vsc_in_lap") == 1).select(
            "driver_number", "lap_number", "lap_duration",
            "vsc_duration_in_lap", "vsc_lap_fraction", "vsc_active_at_end",
        ).orderBy("lap_number", "driver_number").show(50, truncate=False)

        print("\n=== SAFETY CAR LAPS ===")
        race_state.filter(F.col("sc_in_lap") == 1).select(
            "driver_number", "lap_number", "lap_duration",
            "sc_duration_in_lap", "sc_lap_fraction", "sc_active_at_end",
        ).orderBy("lap_number", "driver_number").show(100, truncate=False)

        # Return only keys + new features, then merge into the full dataset to keep all columns.
        feature_pd = race_state.select(*KEY_COLUMNS, *FEATURE_COLUMNS).toPandas()
        feature_pd = feature_pd.drop_duplicates(subset=KEY_COLUMNS, keep="last")

        # Remove old generated columns if this script is rerun, then replace them.
        original = race_state_pd_full.drop(
            columns=[c for c in [*FEATURE_COLUMNS] if c in race_state_pd_full.columns]
        )
        result = original.merge(
            feature_pd,
            on=KEY_COLUMNS,
            how="left",
            validate="many_to_one",
        )
        for column in FEATURE_COLUMNS:
            result[column] = result[column].fillna(0)
        result = result.sort_values("_row_order_for_merge").drop(
            columns=["_row_order_for_merge"]
        )

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        # pandas/pyarrow write avoids the Windows winutils dependency.
        # Remove any previous output, whether file or directory.
        if OUTPUT_PATH.exists():
            if OUTPUT_PATH.is_dir():
                shutil.rmtree(OUTPUT_PATH)
            else:
                OUTPUT_PATH.unlink()

        result.to_parquet(OUTPUT_PATH, index=False)

        print(f"Saved: {OUTPUT_PATH}")

        print("\n=== DONE ===")
        print(f"Saved: {OUTPUT_PATH}")
        print(f"Output rows: {len(result)}")
        print(f"Original columns preserved: {len(original.columns) - 1}")
        print(f"Output columns: {len(result.columns)}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
