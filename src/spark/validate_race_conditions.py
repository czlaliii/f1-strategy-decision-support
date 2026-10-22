from pathlib import Path

import pandas as pd

RACE = "hungary_2026"

PANDAS_PATH = (
    Path("data/processed")
    / RACE
    / "race_state_with_conditions.parquet"
)

SPARK_PATH = (
    Path("data/processed")
    / RACE
    / "spark"
    / "race_state_with_conditions.parquet"
)


VSC_COLUMNS = [
    "vsc_duration_in_lap",
    "vsc_in_lap",
    "vsc_lap_fraction",
    "vsc_active_at_end",
]


def main() -> None:

    print("=== RACE CONDITIONS VALIDATION ===")

    pandas_df = pd.read_parquet(PANDAS_PATH)
    spark_df = pd.read_parquet(SPARK_PATH)

    print(f"Pandas shape: {pandas_df.shape}")
    print(f"Spark shape:  {spark_df.shape}")

    print("\n=== COLUMN CHECK ===")

    missing_from_spark = (
        set(pandas_df.columns)
        - set(spark_df.columns)
    )

    extra_in_spark = (
        set(spark_df.columns)
        - set(pandas_df.columns)
    )

    print(
        "Missing from Spark:",
        missing_from_spark,
    )

    print(
        "Extra in Spark:",
        extra_in_spark,
    )

    print("\n=== VSC FEATURE CHECK ===")

    pandas_df = pandas_df.sort_values(
        ["driver_number", "lap_number"]
    ).reset_index(drop=True)

    spark_df = spark_df.sort_values(
        ["driver_number", "lap_number"]
    ).reset_index(drop=True)

    all_match = True

    for column in VSC_COLUMNS:

        pandas_values = pd.to_numeric(
            pandas_df[column],
            errors="coerce",
        )

        spark_values = pd.to_numeric(
            spark_df[column],
            errors="coerce",
        )

        both_null = (
            pandas_values.isna()
            & spark_values.isna()
        )

        difference = (
            pandas_values
            - spark_values
        ).abs()

        difference = difference[
            ~both_null
        ]

        max_difference = (
            difference.max()
            if len(difference) > 0
            else 0.0
        )

        match = (
            max_difference <= 1e-9
        )

        if not match:
            all_match = False

        print(
            f"{column:<25}"
            f"max difference: "
            f"{max_difference}"
        )

    print("\n=== VSC ROW CHECK ===")

    pandas_vsc = pandas_df[
        pandas_df["vsc_in_lap"] == 1
    ][
        [
            "driver_number",
            "lap_number",
        ]
    ]

    spark_vsc = spark_df[
        spark_df["vsc_in_lap"] == 1
    ][
        [
            "driver_number",
            "lap_number",
        ]
    ]

    pandas_vsc = (
        pandas_vsc
        .sort_values(
            ["lap_number", "driver_number"]
        )
        .reset_index(drop=True)
    )

    spark_vsc = (
        spark_vsc
        .sort_values(
            ["lap_number", "driver_number"]
        )
        .reset_index(drop=True)
    )

    same_vsc_rows = pandas_vsc.equals(
        spark_vsc
    )

    print(
        "Same VSC rows:",
        same_vsc_rows,
    )

    print(
        f"Pandas VSC rows: {len(pandas_vsc)}"
    )

    print(
        f"Spark VSC rows:  {len(spark_vsc)}"
    )

    print("\n=== FINAL RESULT ===")

    if (
        all_match
        and same_vsc_rows
        and not missing_from_spark
        and not extra_in_spark
    ):
        print("PASS")
        print(
            "Spark race conditions "
            "match Pandas."
        )
    else:
        print("FAIL")


if __name__ == "__main__":
    main()