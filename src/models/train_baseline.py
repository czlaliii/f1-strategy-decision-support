import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/processed/hungary_2026/ml_dataset_clean.parquet"

# A verseny időbeli felosztása
TRAIN_LAST_LAP = 50


def main():
    print("Loading cleaned ML dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # Időrend biztosítása
    df = df.sort_values(
        ["lap_number", "driver_number"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # ---------------------------------------------------------

    train_df = df[
        df["lap_number"] <= TRAIN_LAST_LAP
    ].copy()

    validation_df = df[
        df["lap_number"] > TRAIN_LAST_LAP
    ].copy()

    print("\n=== DATA SPLIT ===")
    print(f"Training rows:    {len(train_df)}")
    print(f"Validation rows:  {len(validation_df)}")

    print(
        f"Training laps:    "
        f"{train_df['lap_number'].min()} - "
        f"{train_df['lap_number'].max()}"
    )

    print(
        f"Validation laps:  "
        f"{validation_df['lap_number'].min()} - "
        f"{validation_df['lap_number'].max()}"
    )

    # ---------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------
    #
    # A következő kör idejét egyszerűen az előző kör idejével
    # becsüljük.
    #
    # prediction = previous_lap_time
    # target     = next_lap_time
    #

    validation_df = validation_df.dropna(
        subset=["previous_lap_time", "next_lap_time"]
    ).copy()

    # ---------------------------------------------------------
    # BASELINE 1 - PREVIOUS LAP
    # ---------------------------------------------------------

    previous_lap_df = validation_df.dropna(
        subset=["previous_lap_time", "next_lap_time"]
    ).copy()

    y_true_previous = previous_lap_df["next_lap_time"]
    y_pred_previous = previous_lap_df["previous_lap_time"]

    mae_previous = mean_absolute_error(
        y_true_previous,
        y_pred_previous
    )

    rmse_previous = np.sqrt(
        mean_squared_error(
            y_true_previous,
            y_pred_previous
        )
    )

    r2_previous = r2_score(
        y_true_previous,
        y_pred_previous
    )

    # ---------------------------------------------------------
    # BASELINE 2 - ROLLING 3 LAP AVERAGE
    # ---------------------------------------------------------

    rolling_df = validation_df.dropna(
        subset=["rolling_3_lap_avg", "next_lap_time"]
    ).copy()

    y_true_rolling = rolling_df["next_lap_time"]
    y_pred_rolling = rolling_df["rolling_3_lap_avg"]

    mae_rolling = mean_absolute_error(
        y_true_rolling,
        y_pred_rolling
    )

    rmse_rolling = np.sqrt(
        mean_squared_error(
            y_true_rolling,
            y_pred_rolling
        )
    )

    r2_rolling = r2_score(
        y_true_rolling,
        y_pred_rolling
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print("\n=== BASELINE RESULTS ===")

    print("\nBaseline 1 - Previous Lap Time")
    print(f"MAE:  {mae_previous:.3f} s")
    print(f"RMSE: {rmse_previous:.3f} s")
    print(f"R²:   {r2_previous:.3f}")
    print(
        f"Samples: {len(previous_lap_df)}"
    )

    print("\nBaseline 2 - Rolling 3 Lap Average")
    print(f"MAE:  {mae_rolling:.3f} s")
    print(f"RMSE: {rmse_rolling:.3f} s")
    print(f"R²:   {r2_rolling:.3f}")
    print(
        f"Samples: {len(rolling_df)}"
    )

    print("\n=== INTERPRETATION ===")

    print(
        "Baseline 1 uses the previous lap time "
        "to predict the next lap time."
    )

    print(
        "Baseline 2 uses the average lap time of "
        "the previous three laps."
    )

    print(
        f"Previous Lap Time MAE: "
        f"{mae_previous:.3f} seconds."
    )

    print(
        f"Rolling 3 Lap Average MAE: "
        f"{mae_rolling:.3f} seconds."
    )

    print(
        f"Validation samples used: "
        f"{len(validation_df)}"
    )

if __name__ == "__main__":
    main()