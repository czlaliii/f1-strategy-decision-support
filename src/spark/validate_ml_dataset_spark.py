from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATASET_PATH = Path(
    "data/processed/multi_race/ml_dataset.parquet"
)


def create_spark_session():
    return (
        SparkSession.builder
        .appName("ValidateMultiRaceMLDataset")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def validate_dataset(df):
    print("=" * 70)
    print("MULTI-RACE ML DATASET VALIDATION")
    print("=" * 70)

    total_rows = df.count()

    print(f"\nTotal rows: {total_rows}")
    print(f"Columns:    {len(df.columns)}")

    # ---------------------------------------------------------
    # 1. Duplicate race + driver + lap keys
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("1. DUPLICATE RACE / DRIVER / LAP KEYS")
    print("=" * 70)

    duplicates = (
        df.groupBy(
            "race",
            "driver_number",
            "lap_number",
        )
        .count()
        .filter(F.col("count") > 1)
        .orderBy(
            "race",
            "driver_number",
            "lap_number",
        )
    )

    duplicate_key_count = duplicates.count()

    print(
        f"Duplicate keys: {duplicate_key_count}"
    )

    if duplicate_key_count > 0:
        print("\nFirst duplicate keys:")
        duplicates.show(30, truncate=False)

        print("\nDuplicate rows by race:")
        (
            duplicates.groupBy("race")
            .agg(
                F.count("*").alias("duplicate_keys")
            )
            .orderBy("race")
            .show(20, truncate=False)
        )
    else:
        print("PASS: No duplicate race/driver/lap keys.")

    # ---------------------------------------------------------
    # 2. Row counts by race
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("2. ROW COUNTS BY RACE")
    print("=" * 70)

    (
        df.groupBy("race")
        .count()
        .orderBy("race")
        .show(20, truncate=False)
    )

    # ---------------------------------------------------------
    # 3. Null target check
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("3. TARGET CHECK")
    print("=" * 70)

    null_targets = df.filter(
        F.col("next_lap_time").isNull()
    ).count()

    print(
        f"Rows with null next_lap_time: {null_targets}"
    )

    if null_targets == 0:
        print("PASS: No null target values.")
    else:
        print("WARNING: Null target values found.")

    # ---------------------------------------------------------
    # 4. VSC summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("4. VSC FEATURE SUMMARY")
    print("=" * 70)

    vsc_rows = df.filter(
        (F.col("vsc_duration_in_lap") > 0)
        | (F.col("vsc_in_lap") > 0)
        | (F.col("vsc_lap_fraction") > 0)
        | (F.col("vsc_active_at_end") > 0)
    )

    total_vsc_rows = vsc_rows.count()

    print(
        f"Rows affected by VSC features: {total_vsc_rows}"
    )

    print("\nVSC rows by race:")

    (
        vsc_rows.groupBy("race")
        .agg(
            F.count("*").alias("vsc_rows"),
            F.max("vsc_duration_in_lap").alias(
                "max_vsc_duration"
            ),
            F.max("vsc_lap_fraction").alias(
                "max_vsc_fraction"
            ),
        )
        .orderBy("race")
        .show(20, truncate=False)
    )

    # ---------------------------------------------------------
    # 5. Basic feature sanity checks
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("5. BASIC SANITY CHECKS")
    print("=" * 70)

    negative_lap_times = df.filter(
        F.col("lap_duration") <= 0
    ).count()

    negative_tyre_age = df.filter(
        F.col("tyre_age") < 0
    ).count()

    invalid_positions = df.filter(
        (F.col("position") < 1)
        | (F.col("position") > 22)
    ).count()

    print(
        f"Invalid lap_duration: {negative_lap_times}"
    )
    print(
        f"Negative tyre_age:    {negative_tyre_age}"
    )
    print(
        f"Invalid positions:    {invalid_positions}"
    )

    print("\n" + "=" * 70)
    print("VALIDATION FINISHED")
    print("=" * 70)


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    spark = create_spark_session()

    try:
        df = spark.read.parquet(
            str(DATASET_PATH)
        )

        validate_dataset(df)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()