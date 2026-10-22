import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "data/processed/hungary_2026/ml_dataset_clean.parquet"

TRAIN_LAST_LAP = 50


def main():
    print("Loading cleaned ML dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # --- SORT DATA ---

    df = df.sort_values(
        ["lap_number", "driver_number"]
    ).reset_index(drop=True)

    # --- TRAIN / VALIDATION SPLIT ---

    train_df = df[
        df["lap_number"] <= TRAIN_LAST_LAP
    ].copy()

    validation_df = df[
        df["lap_number"] > TRAIN_LAST_LAP
    ].copy()

    print("\n=== DATA SPLIT ===")

    print(f"Training rows:   {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")

    print(
        f"Training laps:   "
        f"{train_df['lap_number'].min()} - "
        f"{train_df['lap_number'].max()}"
    )

    print(
        f"Validation laps: "
        f"{validation_df['lap_number'].min()} - "
        f"{validation_df['lap_number'].max()}"
    )

    # --- FEATURES ---

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
        "position_missing",
        "interval_missing",
        "interval_change_missing",
        "has_previous_pit",
        "gap_to_leader_missing",
        "gap_change_missing",
        "previous_lap_time_missing",
        "rolling_3_lap_avg_missing",
        "rolling_5_lap_avg_missing",
        "lap_time_delta_missing",
        "position_change_missing",
    ]

    target_column = "next_lap_time"

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_validation = validation_df[feature_columns]
    y_validation = validation_df[target_column]

    # --- FEATURE TYPES ---

    categorical_features = [
        "compound"
    ]

    numerical_features = [
        column
        for column in feature_columns
        if column not in categorical_features
    ]

    # --- PREPROCESSING ---

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # --- MODEL ---

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "regressor",
                LinearRegression()
            )
        ]
    )

    # --- TRAIN ---

    print("\n=== TRAINING ===")

    model.fit(
        X_train,
        y_train
    )

    print("Linear Regression training completed.")

    # --- PREDICTION ---

    y_pred = model.predict(
        X_validation
    )

    # --- METRICS ---

    mae = mean_absolute_error(
        y_validation,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            y_pred
        )
    )

    r2 = r2_score(
        y_validation,
        y_pred
    )

    # --- RESULTS ---

    print("\n=== LINEAR REGRESSION RESULTS ===")

    print(f"MAE:  {mae:.3f} s")
    print(f"RMSE: {rmse:.3f} s")
    print(f"R²:   {r2:.3f}")

    print(
        f"\nValidation samples: "
        f"{len(y_validation)}"
    )

    # --- COMPARISON ---

    print("\n=== BASELINE COMPARISON ===")

    print(
        "Previous Lap Time MAE: "
        "3.682 s"
    )

    print(
        "Rolling 3 Lap Average MAE: "
        "3.627 s"
    )

    print(
        f"Linear Regression MAE: "
        f"{mae:.3f} s"
    )


if __name__ == "__main__":
    main()