from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/hungary_2026/ml_dataset.parquet")
OUTPUT_PATH = Path("data/processed/hungary_2026/ml_dataset_clean.parquet")

def main():
    print("Loading ML dataset...")

    df = pd.read_parquet(INPUT_PATH)

    print(f"Initial shape: {df.shape}")

    # 1. Compound
    # A gumi típusa egy adott stint alatt nem változik.
    # Ezért driverenként előre töltjük a hiányzó értékeket.
    df["compound"] = (
        df.groupby("driver_number")["compound"]
        .ffill()
    )

    # Ha még mindig marad hiányzó compound,
    # akkor az adott driver első ismert stintje előtt vagyunk.
    df["compound"] = df["compound"].fillna("UNKNOWN")

    # 2. Position
    # A pozíció folytonos versenyállapot,
    # ezért driverenként forward fill használható.

    df["position_missing"] = df["position"].isna().astype(int)

    df["position"] = (
        df.groupby("driver_number")["position"]
        .ffill()
    )

    # 3. Interval
    # Missing indicator:
    # megőrizzük azt az információt, hogy eredetileg hiányzott.
    df["interval_missing"] = df["interval"].isna().astype(int)

    interval_median = df["interval"].median()

    df["interval"] = df["interval"].fillna(interval_median)

    # 4. Interval change
    df["interval_change_missing"] = (
        df["interval_change"].isna().astype(int)
    )

    interval_change_median = df["interval_change"].median()

    df["interval_change"] = (
        df["interval_change"]
        .fillna(interval_change_median)
    )

    # 5. Previous lap / rolling features
    # Ezek a verseny elején természetesen hiányoznak,
    # mert még nincs elegendő előző kör.
    #
    # Nem töltjük őket mesterségesen.
    # A későbbi modell-előkészítésnél eldöntjük,
    # hogy szükséges-e ezeket a korai sorokat elhagyni.

    # 6. laps_since_pit
    # NaN itt nem azt jelenti, hogy "0 kör telt el".
    # Inkább azt, hogy még nem volt korábbi boxkiállás.
    #
    # Ezért külön flaget készítünk.
    df["has_previous_pit"] = (
        df["laps_since_pit"].notna().astype(int)
    )

    # A modell számára egy külön numerikus értéket használunk.
    # Az első stint esetében a tyre_age szolgálhat közelítő
    # információként arról, hány kör telt el a stint kezdete óta.
    df["laps_since_pit"] = df["laps_since_pit"].fillna(
        df["tyre_age"]
    )

    # 7. Gap features
    # gap_to_leader és gap_change esetében NEM végzünk
    # egyszerű imputálást.
    #
    # A NaN itt jelenthet például +1 LAP állapotot,
    # ezért az eredeti hiány információt külön flaggel őrizzük.
    df["gap_to_leader_missing"] = (
        df["gap_to_leader"].isna().astype(int)
    )

    df["gap_change_missing"] = (
        df["gap_change"].isna().astype(int)
    )

    # 8. Lap history features
    history_columns = [
        "previous_lap_time",
        "rolling_3_lap_avg",
        "rolling_5_lap_avg",
        "lap_time_delta",
        "position_change",
    ]

    for column in history_columns:
        df[f"{column}_missing"] = (
            df[column].isna().astype(int)
        )

    # 9. Missing value report
    print("\n=== MISSING VALUES AFTER CLEANING ===")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        print("\n=== LAPS DOWN ===")

        print(
            df["laps_down"]
            .value_counts(dropna=False)
            .sort_index()
        )

        print(
            f"Missing laps_down: "
            f"{df['laps_down'].isna().sum()}"
        )
    else:
        print(missing)

    # 10. Save
    df.to_parquet(OUTPUT_PATH, index=False)

    print("\n=== FINAL DATASET ===")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()