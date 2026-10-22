from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "multi_race"
)

FILES = {
    "TRAIN": DATA_DIR / "train_context.parquet",
    "VALIDATION": DATA_DIR / "validation_context.parquet",
    "TEST": DATA_DIR / "test_context.parquet",
}


def print_section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def analyze_split(name, path):
    df = pd.read_parquet(path)

    print_section(f"{name} DATASET")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # --- Compound distribution ---

    print("\nTYRE COMPOUND DISTRIBUTION")
    print("-" * 50)

    compound = (
        df["compound"]
        .fillna("UNKNOWN")
        .value_counts(dropna=False)
    )

    compound_pct = (
        compound / len(df) * 100
    ).round(2)

    compound_table = pd.DataFrame({
        "rows": compound,
        "percentage": compound_pct,
    })

    print(compound_table.to_string())

    # --- Rainfall distribution ---

    print("\nRAINFALL DISTRIBUTION")
    print("-" * 50)

    if "rainfall" in df.columns:
        print(df["rainfall"].value_counts(dropna=False).to_string())

        rainfall_numeric = pd.to_numeric(
            df["rainfall"],
            errors="coerce",
        )

        rainfall_rows = rainfall_numeric.fillna(0).gt(0).sum()

        print(
            f"\nRows with rainfall > 0: "
            f"{rainfall_rows}"
        )

        print(
            f"Percentage with rainfall > 0: "
            f"{rainfall_rows / len(df) * 100:.2f}%"
        )

    # --- Race distribution ---

    print("\nROWS BY RACE")
    print("-" * 50)

    race_counts = (
        df["race"]
        .value_counts()
        .sort_index()
    )

    print(race_counts.to_string())

    # --- Race + compound ---

    print("\nRACE + COMPOUND DISTRIBUTION")
    print("-" * 50)

    race_compound = pd.crosstab(
        df["race"],
        df["compound"].fillna("UNKNOWN"),
    )

    print(race_compound.to_string())

    # --- Intermediate / wet races ---

    print("\nINTERMEDIATE / WET BY RACE")
    print("-" * 50)

    wet_compounds = {"INTERMEDIATE", "WET"}

    wet_rows = df[
        df["compound"]
        .fillna("")
        .isin(wet_compounds)
    ].copy()

    if len(wet_rows) == 0:
        print("No INTERMEDIATE or WET rows.")
    else:
        wet_by_race = (
            wet_rows["race"]
            .value_counts()
            .sort_index()
        )

        print(wet_by_race.to_string())

        wet_by_compound = (
            wet_rows["compound"]
            .value_counts()
        )

        print("\nBy compound:")
        print(wet_by_compound.to_string())

    # --- Rain + compound ---

    print("\nRAINFALL + COMPOUND")
    print("-" * 50)

    if "rainfall" in df.columns:
        temp = df.copy()

        temp["_rainfall_numeric"] = pd.to_numeric(
            temp["rainfall"],
            errors="coerce",
        ).fillna(0)

        temp["_rain"] = (
            temp["_rainfall_numeric"] > 0
        ).map({
            True: "RAIN",
            False: "NO_RAIN",
        })

        rainfall_compound = pd.crosstab(
            temp["_rain"],
            temp["compound"].fillna("UNKNOWN"),
        )

        print(
            rainfall_compound.to_string()
        )

    return df


def main():
    print("=" * 78)
    print("WEATHER AND TYRE DISTRIBUTION ANALYSIS")
    print("=" * 78)

    datasets = {}

    for name, path in FILES.items():
        datasets[name] = analyze_split(
            name,
            path,
        )

    # --- Combined summary ---

    print_section("COMBINED SUMMARY")

    combined = pd.concat(
        [
            df.assign(split=name)
            for name, df in datasets.items()
        ],
        ignore_index=True,
    )

    print(f"Total rows: {len(combined)}")

    print("\nCOMPOUND BY SPLIT")
    print("-" * 50)

    split_compound = pd.crosstab(
        combined["split"],
        combined["compound"].fillna("UNKNOWN"),
    )

    print(
        split_compound.to_string()
    )

    print("\nRACE + SPLIT")
    print("-" * 50)

    split_race = pd.crosstab(
        combined["race"],
        combined["split"],
    )

    print(
        split_race.to_string()
    )


if __name__ == "__main__":
    main()