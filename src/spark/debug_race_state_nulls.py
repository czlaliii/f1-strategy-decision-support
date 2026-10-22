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


COLUMNS = [
    "position",
    "gap_to_leader",
    "interval",
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

    print("=== RACE STATE NULL DEBUG ===")

    for column in COLUMNS:

        p = pandas_df[column]
        s = spark_df[column]

        p_null = p.isna()
        s_null = s.isna()

        mismatch = p_null != s_null

        print()
        print(f"=== {column} ===")
        print(
            f"NULL mismatches: "
            f"{mismatch.sum()}"
        )

        if mismatch.any():

            debug = pandas_df.loc[
                mismatch,
                keys,
            ].copy()

            debug["pandas"] = p[mismatch].values
            debug["spark"] = s[mismatch].values

            print(
                debug.head(20).to_string(
                    index=False
                )
            )


if __name__ == "__main__":
    main()