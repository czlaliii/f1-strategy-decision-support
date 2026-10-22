from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "multi_race"
    / "ml_dataset_with_context.parquet"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "multi_race"
)


TRAIN_RACES = [
    "bahrain_2023",
    "monaco_2023",
    "zandvoort_2023",
    "imola_2024",
    "canada_2024",
    "silverstone_2024",
    "sao_paulo_2024",
    "australia_2025",
]

VALIDATION_RACES = [
    "silverstone_2025",
]

TEST_RACES = [
    "hungary_2026",
]


def main() -> None:
    print("=" * 70)
    print("MULTI-RACE CONTEXT TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_parquet(INPUT_PATH)

    print(f"\nFull dataset: {df.shape}")

    # --------------------------------------------------------------
    # Race existence check
    # --------------------------------------------------------------

    expected_races = (
        TRAIN_RACES
        + VALIDATION_RACES
        + TEST_RACES
    )

    actual_races = set(df["race"].dropna().unique())

    missing_races = [
        race
        for race in expected_races
        if race not in actual_races
    ]

    if missing_races:
        raise ValueError(
            f"Missing races: {missing_races}"
        )

    # --------------------------------------------------------------
    # Temporal split
    # --------------------------------------------------------------

    train_df = df[
        df["race"].isin(TRAIN_RACES)
    ].copy()

    validation_df = df[
        df["race"].isin(VALIDATION_RACES)
    ].copy()

    test_df = df[
        df["race"].isin(TEST_RACES)
    ].copy()

    # --------------------------------------------------------------
    # Row-count validation
    # --------------------------------------------------------------

    total_split_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    if total_split_rows != len(df):
        raise ValueError(
            "Split does not cover exactly the full dataset."
        )

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------

    train_path = (
        OUTPUT_DIR
        / "train_context.parquet"
    )

    validation_path = (
        OUTPUT_DIR
        / "validation_context.parquet"
    )

    test_path = (
        OUTPUT_DIR
        / "test_context.parquet"
    )

    train_df.to_parquet(
        train_path,
        index=False,
    )

    validation_df.to_parquet(
        validation_path,
        index=False,
    )

    test_df.to_parquet(
        test_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    print(f"\nTrain:       {train_df.shape}")
    print(f"Validation:  {validation_df.shape}")
    print(f"Test:        {test_df.shape}")

    print("\nTrain races:")
    print(
        train_df["race"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nValidation races:")
    print(
        validation_df["race"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTest races:")
    print(
        test_df["race"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSaved:")
    print(f"  {train_path}")
    print(f"  {validation_path}")
    print(f"  {test_path}")

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()