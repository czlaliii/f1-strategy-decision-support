from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/processed/hungary_2026/lap_data.parquet")


def main():
    df = pd.read_parquet(DATA_PATH)

    print("\n=== SHAPE ===")
    print(df.shape)

    print("\n=== COLUMNS ===")
    for column in df.columns:
        print(f"- {column}")

    print("\n=== DTYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("Nincs hiányzó érték.")
    else:
        print(missing)

    print("\n=== DRIVERS ===")
    print(df["driver_number"].unique())

    print("\n=== LAPS PER DRIVER ===")
    print(
        df.groupby("driver_number")["lap_number"]
        .agg(["min", "max", "count"])
    )

    print("\n=== COMPOUNDS ===")
    print(df["compound"].value_counts(dropna=False))

    print("\n=== PIT STOPS ===")
    print(df.groupby("driver_number")["pit_stop"].sum())

    print("\n=== DUPLICATE DRIVER/LAP ===")
    duplicates = df.duplicated(
        subset=["driver_number", "lap_number"]
    ).sum()

    print(f"Duplikált sorok: {duplicates}")

    print("\n=== SAMPLE ===")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()