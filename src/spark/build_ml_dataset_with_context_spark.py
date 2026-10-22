from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE_DIR = Path(__file__).resolve().parents[2]

ML_DATASET_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "multi_race"
    / "ml_dataset.parquet"
)

CONTEXT_PATH = (
    BASE_DIR
    / "data"
    / "external"
    / "combined_race_context.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "multi_race"
    / "ml_dataset_with_context.parquet"
)


KEY_COLUMNS = [
    "race",
    "driver_number",
]


# A combined context metadata oszlopai.
# Ezeket nem használjuk ML feature-ként:
# - race
# - year
# - event
# - driver
# - driver_number
# - full_name
#
# A team viszont használható feature-ként.
CONTEXT_METADATA_COLUMNS = {
    "race",
    "year",
    "event",
    "driver",
    "driver_number",
    "full_name",
}


def main() -> None:
    print("=" * 78)
    print("BUILD ML DATASET WITH HISTORICAL + PRE-RACE CONTEXT")
    print("=" * 78)

    spark = (
        SparkSession.builder
        .appName("BuildMLDatasetWithContext")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    # ------------------------------------------------------------------
    # 1. Load base ML dataset
    # ------------------------------------------------------------------

    print("\nLoading base ML dataset...")
    print(f"Path: {ML_DATASET_PATH}")

    if not ML_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"ML dataset not found: {ML_DATASET_PATH}"
        )

    ml = spark.read.parquet(str(ML_DATASET_PATH))

    print(f"Rows:    {ml.count()}")
    print(f"Columns: {len(ml.columns)}")

    # ------------------------------------------------------------------
    # 2. Load combined context
    # ------------------------------------------------------------------

    print("\nLoading combined race context...")
    print(f"Path: {CONTEXT_PATH}")

    if not CONTEXT_PATH.exists():
        raise FileNotFoundError(
            f"Combined context not found: {CONTEXT_PATH}"
        )

    context = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(CONTEXT_PATH))
    )

    print(f"Rows:    {context.count()}")
    print(f"Columns: {len(context.columns)}")

    # ------------------------------------------------------------------
    # 3. Normalize join keys
    # ------------------------------------------------------------------

    ml = ml.withColumn(
        "driver_number",
        F.col("driver_number").cast("long")
    )

    context = context.withColumn(
        "driver_number",
        F.col("driver_number").cast("long")
    )

    context = context.withColumn(
        "race",
        F.col("race").cast("string")
    )

    # ------------------------------------------------------------------
    # 4. Validate required columns
    # ------------------------------------------------------------------

    for column in KEY_COLUMNS:
        if column not in ml.columns:
            raise ValueError(
                f"Missing key column from ML dataset: {column}"
            )

        if column not in context.columns:
            raise ValueError(
                f"Missing key column from context: {column}"
            )

    # ------------------------------------------------------------------
    # 5. Context duplicate validation
    # ------------------------------------------------------------------

    context_duplicates = (
        context
        .groupBy(*KEY_COLUMNS)
        .count()
        .filter(F.col("count") > 1)
    )

    duplicate_count = context_duplicates.count()

    print("\n--- CONTEXT KEY VALIDATION ---")

    if duplicate_count > 0:
        context_duplicates.show(20, truncate=False)
        raise ValueError(
            "Duplicate race + driver_number keys found in context."
        )

    print("✅ No race + driver_number duplicates in context.")

    # ------------------------------------------------------------------
    # 6. Check ML keys against context
    # ------------------------------------------------------------------

    ml_keys = (
        ml
        .select(*KEY_COLUMNS)
        .dropDuplicates()
    )

    context_keys = (
        context
        .select(*KEY_COLUMNS)
        .dropDuplicates()
    )

    missing_context = (
        ml_keys
        .join(
            context_keys,
            on=KEY_COLUMNS,
            how="left_anti",
        )
    )

    missing_context_count = missing_context.count()

    print("\n--- KEY MATCH VALIDATION ---")
    print(
        f"ML race+driver keys:      {ml_keys.count()}"
    )
    print(
        f"Context race+driver keys: {context_keys.count()}"
    )
    print(
        f"Missing context keys:     {missing_context_count}"
    )

    if missing_context_count > 0:
        print("\n❌ These ML keys have no context:")
        missing_context.orderBy(
            "race",
            "driver_number",
        ).show(100, truncate=False)

        raise ValueError(
            "Not every ML race + driver_number key has context."
        )

    print("✅ Every ML key has matching context.")

    # ------------------------------------------------------------------
    # 7. Select only context columns that can become features
    # ------------------------------------------------------------------

    context_feature_columns = [
        column
        for column in context.columns
        if column not in CONTEXT_METADATA_COLUMNS
    ]

    print("\n--- CONTEXT FEATURES ---")
    print(
        f"Context feature columns: {len(context_feature_columns)}"
    )

    for column in context_feature_columns:
        print(f"- {column}")

    # Keep only keys + usable context features.
    context_for_join = context.select(
        *KEY_COLUMNS,
        *context_feature_columns,
    )

    # ------------------------------------------------------------------
    # 8. Join
    # ------------------------------------------------------------------

    print("\n--- JOIN ---")

    base_row_count = ml.count()

    enriched = (
        ml
        .join(
            context_for_join,
            on=KEY_COLUMNS,
            how="left",
        )
    )

    enriched_row_count = enriched.count()

    print(f"Rows before join: {base_row_count}")
    print(f"Rows after join:  {enriched_row_count}")

    if enriched_row_count != base_row_count:
        raise ValueError(
            "Row count changed after context join. "
            "This indicates a broken join or duplicate context keys."
        )

    print("✅ Join preserved row count.")

    # ------------------------------------------------------------------
    # 9. Context completeness check
    # ------------------------------------------------------------------

    print("\n--- CONTEXT COMPLETENESS ---")

    # team should always be available.
    if "team" in enriched.columns:
        missing_team = enriched.filter(
            F.col("team").isNull()
        ).count()

        print(f"Missing team: {missing_team}")

        if missing_team > 0:
            raise ValueError(
                "Some joined rows have missing team values."
            )

    # ------------------------------------------------------------------
    # 10. Final summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("FINAL DATASET")
    print("=" * 78)

    print(f"Rows:    {enriched_row_count}")
    print(f"Columns: {len(enriched.columns)}")

    print("\nSchema:")
    enriched.printSchema()

    print("\nSample:")
    enriched.select(
        "race",
        "driver_number",
        "lap_number",
        "compound",
        "previous_lap_time",
        "qualifying_position",
        "driver_previous_race_pace_index",
        "driver_recent_3_race_pace_index",
        "team_recent_3_race_pace_index",
        "fp1_best_lap",
        "next_lap_time",
    ).show(20, truncate=False)

    print("\nSaving...")
    print(f"Path: {OUTPUT_PATH}")

    (
        enriched
        .write
        .mode("overwrite")
        .parquet(str(OUTPUT_PATH))
    )

    print("\n" + "=" * 78)
    print("SUCCESS")
    print("=" * 78)
    print(f"Saved to: {OUTPUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()