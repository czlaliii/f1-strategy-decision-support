from pathlib import Path

import numpy as np
import pandas as pd

RACE = "hungary_2026"

PANDAS_PATH = (
    Path("data/processed")
    / RACE
    / "features.parquet"
)

SPARK_PATH = (
    Path("data/processed")
    / RACE
    / "spark"
    / "features.parquet"
)


FEATURE_COLUMNS = [
    "previous_lap_time",
    "rolling_3_lap_avg",
    "rolling_5_lap_avg",
    "lap_time_delta",
    "position_change",
    "gap_change",
    "interval_change",
    "laps_since_pit",
    "track_air_temp_difference",
    "next_lap_time",
]


def compare_numeric(
    pandas_series: pd.Series,
    spark_series: pd.Series,
):
    pandas_values = pd.to_numeric(
        pandas_series,
        errors="coerce",
    )

    spark_values = pd.to_numeric(
        spark_series,
        errors="coerce",
    )

    pandas_null = pandas_values.isna()
    spark_null = spark_values.isna()

    nulls_match = pandas_null.equals(
        spark_null
    )

    valid = (
        ~pandas_null
        & ~spark_null
    )

    if valid.any():
        max_difference = (
            pandas_values[valid]
            - spark_values[valid]
        ).abs().max()
    else:
        max_difference = 0.0

    values_match = (
        max_difference <= 1e-9
    )

    return (
        nulls_match,
        values_match,
        max_difference,
    )


def main() -> None:

    print("=== FEATURE VALIDATION ===")

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

    # ---------------------------------------------------------
    # COLUMN CHECK
    # ---------------------------------------------------------

    print("\n=== COLUMN CHECK ===")

    missing = (
        set(pandas_df.columns)
        - set(spark_df.columns)
    )

    extra = (
        set(spark_df.columns)
        - set(pandas_df.columns)
    )

    print(
        "Missing from Spark:",
        missing,
    )

    print(
        "Extra in Spark:",
        extra,
    )

    # ---------------------------------------------------------
    # ROW / KEY CHECK
    # ---------------------------------------------------------

    pandas_df = (
        pandas_df
        .sort_values(
            ["driver_number", "lap_number"]
        )
        .reset_index(drop=True)
    )

    spark_df = (
        spark_df
        .sort_values(
            ["driver_number", "lap_number"]
        )
        .reset_index(drop=True)
    )

    print("\n=== ROW CHECK ===")

    print(
        "Same row count:",
        len(pandas_df) == len(spark_df),
    )

    print(
        "Same keys:",
        pandas_df[
            ["driver_number", "lap_number"]
        ].equals(
            spark_df[
                ["driver_number", "lap_number"]
            ]
        ),
    )

    # ---------------------------------------------------------
    # FEATURE CHECK
    # ---------------------------------------------------------

    print("\n=== FEATURE CHECK ===")

    all_match = True

    for column in FEATURE_COLUMNS:

        if (
            column not in pandas_df.columns
            or column not in spark_df.columns
        ):
            print(
                f"{column:<30} MISSING"
            )
            all_match = False
            continue

        (
            nulls_match,
            values_match,
            max_difference,
        ) = compare_numeric(
            pandas_df[column],
            spark_df[column],
        )

        match = (
            nulls_match
            and values_match
        )

        if not match:
            all_match = False

        print(
            f"{column:<30}"
            f"match={match}   "
            f"max_difference={max_difference}"
        )

        if not nulls_match:
            print(
                " " * 30
                + "NULL pattern differs"
            )

        pandas_values = pd.to_numeric(
                pandas_df[column],
                errors="coerce",
            )

        spark_values = pd.to_numeric(
            spark_df[column],
            errors="coerce",
        )

        pandas_missing = pandas_values.isna().to_numpy()
        spark_missing = spark_values.isna().to_numpy()

        mismatch_indices = np.where(
            pandas_missing != spark_missing
        )[0]

        print(
            " " * 30
            + f"Mismatch count: {len(mismatch_indices)}"
        )

        for i in mismatch_indices[:20]:

            print(
                " " * 30
                + f"index={i}, "
                + f"driver={pandas_df.iloc[i]['driver_number']}, "
                + f"lap={pandas_df.iloc[i]['lap_number']}, "
                + f"pandas={pandas_df.iloc[i][column]!r}, "
                + f"spark={spark_df.iloc[i][column]!r}"
            )

    # ---------------------------------------------------------
    # VSC CHECK
    # ---------------------------------------------------------

    print("\n=== VSC CHECK ===")

    vsc_columns = [
        "vsc_duration_in_lap",
        "vsc_in_lap",
        "vsc_lap_fraction",
        "vsc_active_at_end",
    ]

    for column in vsc_columns:

        (
            nulls_match,
            values_match,
            max_difference,
        ) = compare_numeric(
            pandas_df[column],
            spark_df[column],
        )

        match = (
            nulls_match
            and values_match
        )

        if not match:
            all_match = False

        print(
            f"{column:<30}"
            f"match={match}   "
            f"max_difference={max_difference}"
        )

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------

    print("\n=== FINAL RESULT ===")

    if (
        not missing
        and not extra
        and len(pandas_df) == len(spark_df)
        and pandas_df[
            ["driver_number", "lap_number"]
        ].equals(
            spark_df[
                ["driver_number", "lap_number"]
            ]
        )
        and all_match
    ):
        print("PASS")
        print(
            "Spark features match Pandas features."
        )
    else:
        print("FAIL")


if __name__ == "__main__":
    main()