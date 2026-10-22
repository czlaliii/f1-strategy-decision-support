from pathlib import Path

import pandas as pd

DATASET_PATH = Path(
    "data/processed/multi_race/ml_dataset.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/multi_race"
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


def main():
    print("=" * 70)
    print("MULTI-RACE TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_parquet(DATASET_PATH)

    print(f"\nFull dataset: {df.shape}")

    # --- TRAIN ---

    train_df = df[
        df["race"].isin(TRAIN_RACES)
    ].copy()

    # --- VALIDATION ---

    validation_df = df[
        df["race"].isin(VALIDATION_RACES)
    ].copy()

    # --- TEST ---

    test_df = df[
        df["race"].isin(TEST_RACES)
    ].copy()

    # --- Check that every row belongs to exactly one split ---

    total_split_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    if total_split_rows != len(df):
        raise ValueError(
            "Split error: some rows are missing or appear in "
            "multiple splits."
        )

    # --- Save ---

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = OUTPUT_DIR / "train.parquet"
    validation_path = OUTPUT_DIR / "validation.parquet"
    test_path = OUTPUT_DIR / "test.parquet"

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

    # --- Summary ---

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    print(f"\nTrain:       {len(train_df)} rows")
    print(f"Validation:  {len(validation_df)} rows")
    print(f"Test:        {len(test_df)} rows")

    print("\nTrain races:")
    for race in TRAIN_RACES:
        count = (train_df["race"] == race).sum()
        print(f"  {race:<20} {count}")

    print("\nValidation races:")
    for race in VALIDATION_RACES:
        count = (validation_df["race"] == race).sum()
        print(f"  {race:<20} {count}")

    print("\nTest races:")
    for race in TEST_RACES:
        count = (test_df["race"] == race).sum()
        print(f"  {race:<20} {count}")

    print("\nSaved:")
    print(f"  {train_path}")
    print(f"  {validation_path}")
    print(f"  {test_path}")

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()