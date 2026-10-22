from pathlib import Path

import pandas as pd

INPUT_PATH = Path(
    "data/processed/hungary_2026/features.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/hungary_2026/ml_dataset.parquet"
)


def main():
    print("Loading features...")

    df = pd.read_parquet(INPUT_PATH)

    print(f"Features shape: {df.shape}")

    # 1. SELECT FEATURES
    #
    # Ezeket az adatokat ismeri a rendszer az aktuális
    # Race State időpontjában.
    #
    # next_lap_time NEM kerül az X-be, mert ez a target.
    #
    # A VSC feature-ök az aktuális, már befejezett kör
    # eseményeit írják le, ezért használhatók a következő
    # kör idejének előrejelzéséhez.
    #

    feature_columns = [
        "driver_number",
        "lap_number",
        "compound",
        "tyre_age",
        "position",
        "gap_to_leader",
        "laps_down",
        "interval",
        "previous_lap_time",
        "rolling_3_lap_avg",
        "rolling_5_lap_avg",
        "lap_time_delta",
        "position_change",
        "gap_change",
        "interval_change",
        "laps_since_pit",
        "track_air_temp_difference",
        "air_temperature",
        "track_temperature",
        "rainfall",
        "humidity",
        "wind_speed",

        # VSC features
        "vsc_duration_in_lap",
        "vsc_in_lap",
        "vsc_lap_fraction",
        "vsc_active_at_end",
    ]

    target_column = "next_lap_time"

    # 2. CHECK REQUIRED COLUMNS

    required_columns = feature_columns + [
        target_column
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # 3. CREATE ML DATASET

    ml_df = df[
        feature_columns + [target_column]
    ].copy()

    print(
        f"Initial ML dataset shape: "
        f"{ml_df.shape}"
    )

    # 4. REMOVE ROWS WITHOUT TARGET
    #
    # Az utolsó körnél nincs következő kör,
    # ezért next_lap_time = NaN.
    #
    # Ezekből nem lehet tanító példa.
    #

    missing_target = ml_df[
        target_column
    ].isna().sum()

    print(
        f"Rows without target: "
        f"{missing_target}"
    )

    ml_df = ml_df.dropna(
        subset=[target_column]
    ).copy()

    # 5. HANDLE COMPOUND
    #
    # A compound kategorikus változó:
    #
    # HARD
    # MEDIUM
    # SOFT
    #
    # Az ML modellhez később numerikus reprezentációra
    # lesz szükség.
    #
    # Egyelőre category típussá alakítjuk.
    # A tényleges encoding a train pipeline-ban történik.
    #

    ml_df["compound"] = (
        ml_df["compound"].astype("category")
    )

    # 6. DATA TYPES

    numeric_columns = [
        "driver_number",
        "lap_number",
        "tyre_age",
        "position",
        "gap_to_leader",
        "laps_down",
        "interval",
        "previous_lap_time",
        "rolling_3_lap_avg",
        "rolling_5_lap_avg",
        "lap_time_delta",
        "position_change",
        "gap_change",
        "interval_change",
        "laps_since_pit",
        "track_air_temp_difference",
        "air_temperature",
        "track_temperature",
        "rainfall",
        "humidity",
        "wind_speed",

        # VSC features
        "vsc_duration_in_lap",
        "vsc_in_lap",
        "vsc_lap_fraction",
        "vsc_active_at_end",

        # Target
        "next_lap_time",
    ]

    for column in numeric_columns:
        ml_df[column] = pd.to_numeric(
            ml_df[column],
            errors="coerce"
        )

    # 7. INSPECTION OF MISSING VALUES

    print("\n=== MISSING VALUES ===")

    missing_values = (
        ml_df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(
        missing_values[
            missing_values > 0
        ].to_string()
    )

    # 8. VSC FEATURE INSPECTION

    print("\n=== VSC FEATURES ===")

    vsc_columns = [
        "vsc_duration_in_lap",
        "vsc_in_lap",
        "vsc_lap_fraction",
        "vsc_active_at_end",
    ]

    print(
        ml_df[vsc_columns]
        .describe()
        .to_string()
    )

    print("\nVSC rows:")

    vsc_rows = ml_df[
        ml_df["vsc_in_lap"] == 1
    ][
        [
            "driver_number",
            "lap_number",
            "vsc_duration_in_lap",
            "vsc_lap_fraction",
            "vsc_active_at_end",
            target_column,
        ]
    ]

    print(
        vsc_rows
        .head(20)
        .to_string(index=False)
    )

    # 9. BASIC STATISTICS

    print("\n=== TARGET ===")

    print(
        f"Target: {target_column}"
    )

    print(
        f"Mean next lap time: "
        f"{ml_df[target_column].mean():.3f} s"
    )

    print(
        f"Minimum next lap time: "
        f"{ml_df[target_column].min():.3f} s"
    )

    print(
        f"Maximum next lap time: "
        f"{ml_df[target_column].max():.3f} s"
    )

    # 10. FINAL DATASET INFORMATION

    print("\n=== FINAL DATASET ===")

    print(
        f"Rows: {len(ml_df)}"
    )

    print(
        f"Columns: {len(ml_df.columns)}"
    )

    print("\nColumns:")

    for column in ml_df.columns:
        print(f"- {column}")

    # 11. SAVE

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ml_df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()