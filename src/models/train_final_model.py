from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "multi_race"

TRAIN_PATH = DATA_DIR / "train_context.parquet"
VALIDATION_PATH = DATA_DIR / "validation_context.parquet"
TEST_PATH = DATA_DIR / "test_context.parquet"

MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / "final_gradient_boosting_pipeline.joblib"

FEATURE_PATH = MODEL_DIR / "final_model_features.txt"

TARGET = "next_lap_time"


# --- BASELINE FEATURES ---

BASE_FEATURE_COLUMNS = [
    "driver_number",
    "lap_number",
    "lap_duration",
    "compound",
    "tyre_age",
    "pit_stop",
    "is_pit_out_lap",
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


# --- SELECTED PRACTICE FEATURES ---

FP1_FEATURES = [
    "fp1_available",
    "fp1_best_lap",
    "fp1_avg_top5_lap",
    "fp1_lap_count",
    "fp1_best_soft",
    "fp1_best_medium",
    "fp1_best_hard",
    "fp1_best_intermediate",
]


FP2_FEATURES = [
    "fp2_available",
    "fp2_best_lap",
    "fp2_avg_top5_lap",
    "fp2_lap_count",
    "fp2_best_soft",
    "fp2_best_medium",
    "fp2_best_hard",
    "fp2_best_intermediate",
]

FP3_FEATURES = [
    "fp3_available",
    "fp3_best_lap",
    "fp3_avg_top5_lap",
    "fp3_lap_count",
    "fp3_best_soft",
    "fp3_best_medium",
    "fp3_best_hard",
    "fp3_best_intermediate",
    "fp3_best_wet",
]


FINAL_FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    + FP1_FEATURES
    + FP2_FEATURES
    + FP3_FEATURES
)


# --- MISSING INDICATORS ---

MISSING_INDICATOR_COLUMNS = [
    "position",
    "interval",
    "interval_change",
    "gap_to_leader",
    "gap_change",
    "previous_lap_time",
    "rolling_3_lap_avg",
    "rolling_5_lap_avg",
    "lap_time_delta",
    "position_change",
]


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in MISSING_INDICATOR_COLUMNS:
        df[f"{column}_missing"] = (
            df[column].isna().astype(int)
        )

    return df


# --- MODEL ---

def build_model(feature_columns):
    categorical_features = [
        column
        for column in ["compound"]
        if column in feature_columns
    ]

    numerical_features = [
        column
        for column in feature_columns
        if column not in categorical_features
    ]

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
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
                "numeric",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# --- EVALUATION ---

def evaluate(
    model,
    X,
    y,
    name,
):
    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions,
    )

    rmse = mean_squared_error(
        y,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y,
        predictions,
    )

    print(f"\n{name}")
    print("-" * 60)
    print(f"MAE:   {mae:.3f} s")
    print(f"RMSE:  {rmse:.3f} s")
    print(f"R²:    {r2:.3f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# --- MAIN ---

def main():
    print("=" * 78)
    print("FINAL MULTI-RACE MODEL TRAINING")
    print("=" * 78)

    # --- Load data ---

    train_raw = pd.read_parquet(TRAIN_PATH)
    validation_raw = pd.read_parquet(VALIDATION_PATH)
    test_raw = pd.read_parquet(TEST_PATH)

    train = add_model_features(train_raw)
    validation = add_model_features(validation_raw)
    test = add_model_features(test_raw)

    print("\nDataset sizes:")
    print(f"Train:       {train.shape}")
    print(f"Validation:  {validation.shape}")
    print(f"Test:        {test.shape}")

    # --- Validate features ---

    feature_columns = (
        FINAL_FEATURE_COLUMNS
        + [
            "position_missing",
            "interval_missing",
            "interval_change_missing",
            "gap_to_leader_missing",
            "gap_change_missing",
            "previous_lap_time_missing",
            "rolling_3_lap_avg_missing",
            "rolling_5_lap_avg_missing",
            "lap_time_delta_missing",
            "position_change_missing",
        ]
    )

    missing_columns = [
        column
        for column in feature_columns
        if column not in train.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    print("\nFinal feature configuration:")
    print(f"Number of features: {len(feature_columns)}")

    # --- Split X / y ---

    X_train = train[feature_columns]
    y_train = train[TARGET]

    X_validation = validation[feature_columns]
    y_validation = validation[TARGET]

    X_test = test[feature_columns]
    y_test = test[TARGET]

    # --- Build and train ---

    model = build_model(feature_columns)

    print("\nTraining final model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training finished.")

    # --- Evaluate ---

    validation_metrics = evaluate(
        model,
        X_validation,
        y_validation,
        "VALIDATION",
    )

    test_metrics = evaluate(
        model,
        X_test,
        y_test,
        "TEST",
    )

    # --- Save model ---

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    FEATURE_PATH.write_text(
        "\n".join(feature_columns),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("FINAL MODEL SAVED")
    print("=" * 78)

    print(f"Model:    {MODEL_PATH}")
    print(f"Features: {FEATURE_PATH}")

    # --- Final summary ---

    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)

    print(
        f"\nValidation:"
        f" MAE={validation_metrics['mae']:.3f}s"
        f" RMSE={validation_metrics['rmse']:.3f}s"
        f" R²={validation_metrics['r2']:.3f}"
    )

    print(
        f"\nTest:"
        f" MAE={test_metrics['mae']:.3f}s"
        f" RMSE={test_metrics['rmse']:.3f}s"
        f" R²={test_metrics['r2']:.3f}"
    )


if __name__ == "__main__":
    main()