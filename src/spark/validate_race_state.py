from pathlib import Path

import pandas as pd

RACE = "hungary_2026"

PANDAS_PATH = (
    Path("data/processed")
    / RACE
    / "race_state.parquet"
)

SPARK_PATH = (
    Path("data/processed")
    / RACE
    / "spark"
    / "race_state.parquet"
)


def main() -> None:

    print("=== RACE STATE VALIDATION ===")

    pandas_df = pd.read_parquet(
        PANDAS_PATH
    )

    spark_df = pd.read_parquet(
        SPARK_PATH
    )

    print(
        f"Pandas shape: {pandas_df.shape}"
    )

    print(
        f"Spark shape:  {spark_df.shape}"
    )

    # --- COLUMN CHECK ---

    pandas_columns = set(
        pandas_df.columns
    )

    spark_columns = set(
        spark_df.columns
    )

    print("\n=== COLUMN CHECK ===")

    print(
        "Missing from Spark:",
        pandas_columns - spark_columns,
    )

    print(
        "Extra in Spark:",
        spark_columns - pandas_columns,
    )

    # --- COMMON COLUMNS ---

    common_columns = [
        column
        for column in pandas_df.columns
        if column in spark_df.columns
    ]

    # --- SORT ---

    keys = [
        "driver_number",
        "lap_number",
    ]

    pandas_df = pandas_df.sort_values(
        keys
    ).reset_index(drop=True)

    spark_df = spark_df.sort_values(
        keys
    ).reset_index(drop=True)

    # --- ROW CHECK ---

    print("\n=== ROW CHECK ===")

    print(
        "Same row count:",
        len(pandas_df)
        == len(spark_df),
    )

    print(
        "Same keys:",
        pandas_df[keys].equals(
            spark_df[keys]
        ),
    )

    # --- VALUE CHECK ---

    print("\n=== VALUE CHECK ===")

    differences = []

    for column in common_columns:

        pandas_series = pandas_df[column]
        spark_series = spark_df[column]

        # Numeric columns
        if (
            pd.api.types.is_numeric_dtype(
                pandas_series
            )
            and
            pd.api.types.is_numeric_dtype(
                spark_series
            )
        ):

            same = (
                pandas_series
                .fillna(float("nan"))
                .equals(
                    spark_series
                    .fillna(float("nan"))
                )
            )

        else:

            same = pandas_series.equals(
                spark_series
            )

        if not same:
            differences.append(
                column
            )

    if differences:

        print(
            "Different columns:"
        )

        for column in differences:
            print(
                f"  - {column}"
            )

    else:

        print(
            "All common columns match."
        )

    # --- IMPORTANT NUMERIC CHECK ---

    print(
        "\n=== IMPORTANT FIELDS ==="
    )

    important_columns = [
        "position",
        "laps_down",
        "gap_to_leader",
        "interval",
        "air_temperature",
        "track_temperature",
        "humidity",
        "pressure",
        "rainfall",
        "wind_speed",
        "wind_direction",
    ]

    for column in important_columns:

        if (
            column not in pandas_df.columns
            or column not in spark_df.columns
        ):
            continue

        pandas_values = pd.to_numeric(
            pandas_df[column],
            errors="coerce",
        )

        spark_values = pd.to_numeric(
            spark_df[column],
            errors="coerce",
        )

        mask = ~(
            pandas_values.isna()
            & spark_values.isna()
        )

        if mask.any():

            max_difference = (
                pandas_values[mask]
                - spark_values[mask]
            ).abs().max()

        else:

            max_difference = 0.0

        print(
            f"{column:<25}"
            f"max difference: "
            f"{max_difference}"
        )


if __name__ == "__main__":
    main()