import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "data/processed/hungary_2026/ml_dataset_clean.parquet"

TARGET_COLUMN = "next_lap_time"

FEATURE_COLUMNS = [
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
]


def main():

    print("Loading cleaned ML dataset...")

    df = pd.read_parquet(DATA_PATH)

    # ---------------------------------------------------------
    # 1. Train / validation split
    # ---------------------------------------------------------

    train_df = df[df["lap_number"] <= 50].copy()
    validation_df = df[df["lap_number"] > 50].copy()

    train_df = train_df.dropna(subset=[TARGET_COLUMN])
    validation_df = validation_df.dropna(subset=[TARGET_COLUMN])

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df[TARGET_COLUMN]

    print("\n=== DATA SPLIT ===")
    print(f"Training rows:   {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Training laps:   {train_df['lap_number'].min()} - {train_df['lap_number'].max()}")
    print(f"Validation laps: {validation_df['lap_number'].min()} - {validation_df['lap_number'].max()}")

    # ---------------------------------------------------------
    # 2. Preprocessing
    # ---------------------------------------------------------

    numerical_features = [
        column
        for column in FEATURE_COLUMNS
        if column != "compound"
    ]

    categorical_features = ["compound"]

    numerical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                numerical_features,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    # ---------------------------------------------------------
    # 3. Model
    # ---------------------------------------------------------

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------

    print("\n=== TRAINING ===")

    pipeline.fit(X_train, y_train)

    print("Gradient Boosting training completed.")

    # ---------------------------------------------------------
    # 5. Predictions
    # ---------------------------------------------------------

    predictions = pipeline.predict(X_validation)

    validation_results = validation_df[
        [
            "driver_number",
            "lap_number",
            "compound",
            "tyre_age",
            "position",
            "laps_since_pit",
            "previous_lap_time",
            "gap_to_leader",
            "interval",
            TARGET_COLUMN,
        ]
    ].copy()

    validation_results["predicted_lap_time"] = predictions

    validation_results["error"] = (
        validation_results[TARGET_COLUMN]
        - validation_results["predicted_lap_time"]
    )

    validation_results["absolute_error"] = (
        validation_results["error"].abs()
    )

    # ---------------------------------------------------------
    # 6. Overall metrics
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        y_validation,
        predictions,
    )

    rmse = mean_squared_error(
        y_validation,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_validation,
        predictions,
    )

    print("\n=== OVERALL RESULTS ===")
    print(f"MAE:  {mae:.3f} s")
    print(f"RMSE: {rmse:.3f} s")
    print(f"R²:   {r2:.3f}")

    # ---------------------------------------------------------
    # 7. Error statistics
    # ---------------------------------------------------------

    print("\n=== ERROR STATISTICS ===")

    print(
        validation_results["absolute_error"].describe().to_string()
    )

    # ---------------------------------------------------------
    # 8. Worst predictions
    # ---------------------------------------------------------

    print("\n=== 15 WORST PREDICTIONS ===")

    worst_predictions = validation_results.sort_values(
        "absolute_error",
        ascending=False,
    ).head(15)

    print(
        worst_predictions[
            [
                "driver_number",
                "lap_number",
                "compound",
                "tyre_age",
                "laps_since_pit",
                TARGET_COLUMN,
                "predicted_lap_time",
                "error",
                "absolute_error",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # 9. Error by compound
    # ---------------------------------------------------------

    print("\n=== ERROR BY COMPOUND ===")

    compound_error = (
        validation_results
        .groupby("compound")["absolute_error"]
        .agg(
            samples="count",
            mean_error="mean",
            median_error="median",
            max_error="max",
        )
        .sort_values("mean_error", ascending=False)
    )

    print(compound_error.to_string())

    # ---------------------------------------------------------
    # 10. Error by race lap
    # ---------------------------------------------------------

    print("\n=== ERROR BY LAP ===")

    lap_error = (
        validation_results
        .groupby("lap_number")["absolute_error"]
        .agg(
            samples="count",
            mean_error="mean",
            median_error="median",
            max_error="max",
        )
        .sort_values("mean_error", ascending=False)
    )

    print(lap_error.head(10).to_string())

    # ---------------------------------------------------------
    # 11. Error by tyre age
    # ---------------------------------------------------------

    print("\n=== ERROR BY TYRE AGE ===")

    tyre_age_error = (
        validation_results
        .groupby("tyre_age")["absolute_error"]
        .agg(
            samples="count",
            mean_error="mean",
            median_error="median",
            max_error="max",
        )
        .sort_index()
    )

    print(tyre_age_error.to_string())

    # ---------------------------------------------------------
    # 12. Error by laps since pit
    # ---------------------------------------------------------

    print("\n=== ERROR BY LAPS SINCE PIT ===")

    pit_history_error = (
        validation_results
        .groupby("laps_since_pit")["absolute_error"]
        .agg(
            samples="count",
            mean_error="mean",
            median_error="median",
            max_error="max",
        )
        .sort_index()
    )

    print(pit_history_error.head(20).to_string())

    # ---------------------------------------------------------
    # 13. Systematic over / under prediction
    # ---------------------------------------------------------

    print("\n=== SYSTEMATIC ERROR ===")

    mean_error = validation_results["error"].mean()

    print(
        f"Mean signed error: {mean_error:.3f} s"
    )

    if mean_error > 0:
        print("The model tends to UNDERPREDICT lap time.")
    elif mean_error < 0:
        print("The model tends to OVERPREDICT lap time.")
    else:
        print("The model has no significant overall bias.")

    # ---------------------------------------------------------
    # 14. Save detailed predictions
    # ---------------------------------------------------------

    output_path = (
        "data/processed/hungary_2026/"
        "gradient_boosting_predictions.parquet"
    )

    validation_results.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"\nDetailed predictions saved to: {output_path}"
    )


if __name__ == "__main__":
    main()