from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

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

OUTPUT_DIR = Path("data/processed/multi_race")
OUTPUT_PATH = OUTPUT_DIR / "ml_dataset.parquet"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("F1MultiRaceMLDataset")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def build_ml_dataset(spark):
    datasets = []

    print("=" * 70)
    print("BUILDING MULTI-RACE ML DATASET")
    print("=" * 70)

    for race in RACES:
        input_path = (
            Path("data/processed")
            / race
            / "spark"
            / "features.parquet"
        )

        print(f"\nLoading: {race}")
        print(f"Path:    {input_path}")

        if not input_path.exists():
            raise FileNotFoundError(
                f"Features file not found: {input_path}"
            )

        df = spark.read.parquet(str(input_path))

        # A verseny azonosítójának hozzáadása.
        df = df.withColumn("race", F.lit(race))

        datasets.append(df)

        print(f"Rows:    {df.count()}")
        print(f"Columns: {len(df.columns)}")

    print("\n" + "=" * 70)
    print("UNION")
    print("=" * 70)

    # A 10 verseny összefűzése.
    result = datasets[0]

    for df in datasets[1:]:
        result = result.unionByName(
            df,
            allowMissingColumns=True,
        )

    before_filter = result.count()

    print(
        f"Combined rows before target filtering: "
        f"{before_filter}"
    )

    # Csak normál versenyköröket használunk a modellhez.
    # A 200 másodpercnél hosszabb jelenlegi vagy következő
    # köridők valószínűleg versenymegszakítást / újraindítást
    # tartalmazó mérési értékek, ezért kizárjuk őket.
    result = result.filter(
        F.col("next_lap_time").isNotNull()
    )

    after_null_filter = result.count()

    MAX_VALID_LAP_TIME = 200.0

    result = result.filter(
        (F.col("lap_duration") <= MAX_VALID_LAP_TIME)
        & (F.col("next_lap_time") <= MAX_VALID_LAP_TIME)
    )

    after_target_filter = result.count()

    print(
        f"Rows after null target filter:       "
        f"{after_null_filter}"
    )

    print(
        f"Rows after lap-time filter:           "
        f"{after_target_filter}"
    )

    print(
        f"Removed rows:                         "
        f"{before_filter - after_target_filter}"
    )

    # Rendezés az átláthatóság kedvéért.
    result = result.orderBy(
        "race",
        "driver_number",
        "lap_number",
    )

    return result


def main():
    spark = create_spark_session()

    try:
        dataset = build_ml_dataset(spark)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("FINAL DATASET")
        print("=" * 70)

        print(f"Rows:    {dataset.count()}")
        print(f"Columns: {len(dataset.columns)}")

        print("\nSchema:")
        dataset.printSchema()

        print("\nRows per race:")
        (
            dataset.groupBy("race")
            .count()
            .orderBy("race")
            .show(20, truncate=False)
        )

        print("\nSample:")
        dataset.show(10, truncate=False)

        dataset.write.mode("overwrite").parquet(
            str(OUTPUT_PATH)
        )

        print("\n" + "=" * 70)
        print("SUCCESS")
        print("=" * 70)
        print(f"Saved to: {OUTPUT_PATH}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()