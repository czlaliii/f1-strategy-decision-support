import argparse
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

DATA_DIR = None
OUTPUT_DIR = None

LAPS_PATH = None
STINTS_PATH = None
PIT_PATH = None

OUTPUT_PATH = None

def create_spark_session() -> SparkSession:
    """Létrehozza a helyi SparkSessiont."""

    return (
        SparkSession.builder
        .appName("F1LapDataSpark")
        .master("local[*]")
        .getOrCreate()
    )


def load_json(
    spark: SparkSession,
    filepath: Path,
):
    """JSON fájl betöltése Spark DataFrame-be."""

    return (
        spark.read
        .option("multiLine", "true")
        .json(str(filepath))
    )


def build_stint_mapping(stints):
    """
    Egy stintből driver/lap szintű mapping készítése.

    Például:

    lap_start = 20
    lap_end = 23

    →

    20
    21
    22
    23
    """

    return (
        stints
        .withColumn(
            "lap_start",
            F.col("lap_start").cast("int"),
        )
        .withColumn(
            "lap_end",
            F.col("lap_end").cast("int"),
        )
        .withColumn(
            "lap_number",
            F.explode(
                F.sequence(
                    F.col("lap_start"),
                    F.col("lap_end"),
                )
            ),
        )
        .withColumn(
            "tyre_age",
            (
                F.col("tyre_age_at_start")
                + F.col("lap_number")
                - F.col("lap_start")
            ),
        )
        .select(
            "driver_number",
            "lap_number",
            "stint_number",
            "compound",
            "tyre_age",
        )
    )


def build_lap_data(
    laps,
    stints,
    pit,
):
    """A pandasos build_lap_data Spark megfelelője."""

    # ---------------------------------------------------------
    # STINT MAPPING
    # ---------------------------------------------------------

    stint_mapping = build_stint_mapping(stints)

    stint_window = Window.partitionBy(
        "driver_number",
        "lap_number",
    ).orderBy(
        F.col("stint_number").desc()
    )

    stint_mapping = (
        stint_mapping
        .withColumn(
            "_stint_rank",
            F.row_number().over(stint_window)
        )
        .filter(F.col("_stint_rank") == 1)
        .drop("_stint_rank")
    )

    # ---------------------------------------------------------
    # LAP + STINT
    # ---------------------------------------------------------

    lap_data = laps.join(
        stint_mapping,
        on=[
            "driver_number",
            "lap_number",
        ],
        how="left",
    )

    # ---------------------------------------------------------
    # PIT STOP FLAG
    # ---------------------------------------------------------

    if "driver_number" in pit.columns and "lap_number" in pit.columns:

        pit_flags = (
            pit
            .select(
                "driver_number",
                "lap_number",
            )
            .dropDuplicates()
            .withColumn(
                "pit_stop",
                F.lit(True),
            )
        )

    else:
        # Ha nincs pit adat, létrehozunk egy üres,
        # de megfelelő sémájú DataFrame-et.
        pit_flags = (
            laps
            .select(
                "driver_number",
                "lap_number",
            )
            .limit(0)
            .withColumn(
                "pit_stop",
                F.lit(True),
            )
        )

    lap_data = (
        lap_data
        .join(
            pit_flags,
            on=[
                "driver_number",
                "lap_number",
            ],
            how="left",
        )
        .withColumn(
            "pit_stop",
            F.coalesce(
                F.col("pit_stop"),
                F.lit(False),
            ),
        )
    )

    # ---------------------------------------------------------
    # SORT
    # ---------------------------------------------------------

    lap_data = lap_data.orderBy(
        "driver_number",
        "lap_number",
    )

    return lap_data


def main() -> None:
    global DATA_DIR, OUTPUT_DIR
    global LAPS_PATH, STINTS_PATH, PIT_PATH, OUTPUT_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("race", nargs="?", default="hungary_2026")
    args = parser.parse_args()

    DATA_DIR = Path("data/raw") / args.race
    OUTPUT_DIR = Path("data/processed") / args.race / "spark"

    LAPS_PATH = DATA_DIR / "laps.json"
    STINTS_PATH = DATA_DIR / "stints.json"
    PIT_PATH = DATA_DIR / "pit.json"

    OUTPUT_PATH = OUTPUT_DIR / "lap_data.parquet"

    print("=== SPARK LAP DATA PROCESSING ===")

    spark = create_spark_session()

    try:
        print("\nLoading data...")

        laps = load_json(
            spark,
            LAPS_PATH,
        )

        stints = load_json(
            spark,
            STINTS_PATH,
        )

        pit = load_json(
            spark,
            PIT_PATH,
        )

        print(f"Laps: {laps.count()}")
        print(f"Stints: {stints.count()}")
        print(f"Pit stops: {pit.count()}")

        # -----------------------------------------------------
        # SCHEMAS
        # -----------------------------------------------------

        print("\n=== LAPS SCHEMA ===")
        laps.printSchema()

        print("\n=== STINTS SCHEMA ===")
        stints.printSchema()

        print("\n=== PIT SCHEMA ===")
        pit.printSchema()

        # -----------------------------------------------------
        # PROCESSING
        # -----------------------------------------------------

        print("\nBuilding lap data...")

        lap_data = build_lap_data(
            laps=laps,
            stints=stints,
            pit=pit,
        )

        # Cache, mert többször használjuk.
        lap_data.cache()

        row_count = lap_data.count()

        print(
            f"\nProcessed rows: {row_count}"
        )

        print("\n=== RESULT ===")
        lap_data.show(
            20,
            truncate=False,
        )

        print("\n=== RESULT SCHEMA ===")
        lap_data.printSchema()

        print("\n=== RESULT SHAPE ===")
        print(
            f"Rows: {row_count}"
        )
        print(
            f"Columns: {len(lap_data.columns)}"
        )

        # -----------------------------------------------------
        # SAVE
        # -----------------------------------------------------

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"\nSaving to: {OUTPUT_PATH}"
        )

        (
            lap_data
            .write
            .mode("overwrite")
            .parquet(str(OUTPUT_PATH))
        )

        print("\n=== DONE ===")
        print(
            f"Saved: {OUTPUT_PATH}"
        )

    finally:
        spark.stop()

if __name__ == "__main__":
    main()