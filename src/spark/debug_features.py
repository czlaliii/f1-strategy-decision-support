from pathlib import Path

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


COLUMNS = [
    "lap_time_delta",
    "position_change",
    "gap_change",
    "interval_change",
    "laps_since_pit",
]


def main():
    pandas_df = pd.read_parquet(
        PANDAS_PATH
    )

    spark_df = pd.read_parquet(
        SPARK_PATH
    )

    keys = [
        "driver_number",
        "lap_number",
    ]

    pandas_df = (
        pandas_df
        .sort_values(keys)
        .reset_index(drop=True)
    )

    spark_df = (
        spark_df
        .sort_values(keys)
        .reset_index(drop=True)
    )

    print("=== FEATURE DIFFERENCE DEBUG ===")

    for column in COLUMNS:

        p = pd.to_numeric(
            pandas_df[column],
            errors="coerce",
        )

        s = pd.to_numeric(
            spark_df[column],
            errors="coerce",
        )

        both_null = (
            p.isna()
            & s.isna()
        )

        value_difference = (
            p - s
        ).abs()

        mismatch = (
            ~both_null
            &
            (
                p.isna()
                != s.isna()
                |
                (
                    value_difference
                    > 1e-9
                )
            )
        )

        debug = pandas_df.loc[
            mismatch,
            keys,
        ].copy()

        debug["pandas"] = p[mismatch].values
        debug["spark"] = s[mismatch].values

        print()
        print(f"=== {column} ===")
        print(
            f"Mismatch rows: "
            f"{len(debug)}"
        )

        if len(debug) > 0:
            print(
                debug.head(15).to_string(
                    index=False
                )
            )


if __name__ == "__main__":
    main()