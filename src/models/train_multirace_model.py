from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path("data/processed/multi_race")

TRAIN_PATH = DATA_DIR / "train.parquet"
VALIDATION_PATH = DATA_DIR / "validation.parquet"
TEST_PATH = DATA_DIR / "test.parquet"

MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / "gradient_boosting_pipeline.joblib"


TARGET_COLUMN = "next_lap_time"


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

def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    indicator_columns = [
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

    for column in indicator_columns:
        result[f"{column}_missing"] = (
            result[column].isna().astype(int)
        )

    return result


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["pit_stop"] = (
        df["pit_stop"]
        .fillna(False)
        .astype(int)
    )

    df["is_pit_out_lap"] = (
        df["is_pit_out_lap"]
        .fillna(False)
        .astype(int)
    )

    df = add_missing_indicators(df)

    feature_columns = BASE_FEATURE_COLUMNS + [
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

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    return df[feature_columns + [TARGET_COLUMN]].copy()


def build_model():
    numerical_features = [
        column
        for column in BASE_FEATURE_COLUMNS
        if column not in ["compound"]
    ]

    categorical_features = [
        "compound"
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
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "cat",
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


def evaluate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
):
    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions,
        )
    )

    r2 = r2_score(
        y,
        predictions,
    )

    print(f"\n{name}")
    print("-" * 50)
    print(f"MAE:   {mae:.3f} s")
    print(f"RMSE:  {rmse:.3f} s")
    print(f"R²:    {r2:.3f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def evaluate_previous_lap(
    df: pd.DataFrame,
    name: str,
):
    valid = df[
        df["previous_lap_time"].notna()
    ].copy()

    mae = mean_absolute_error(
        valid[TARGET_COLUMN],
        valid["previous_lap_time"],
    )

    rmse = np.sqrt(
        mean_squared_error(
            valid[TARGET_COLUMN],
            valid["previous_lap_time"],
        )
    )

    r2 = r2_score(
        valid[TARGET_COLUMN],
        valid["previous_lap_time"],
    )

    print(f"\n{name} - Previous Lap")
    print("-" * 50)
    print(f"MAE:   {mae:.3f} s")
    print(f"RMSE:  {rmse:.3f} s")
    print(f"R²:    {r2:.3f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def main():
    print("=" * 70)
    print("MULTI-RACE GRADIENT BOOSTING TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    train_raw = pd.read_parquet(TRAIN_PATH)
    validation_raw = pd.read_parquet(VALIDATION_PATH)
    test_raw = pd.read_parquet(TEST_PATH)

    print("\nDataset sizes:")
    print(f"Train:       {train_raw.shape}")
    print(f"Validation:  {validation_raw.shape}")
    print(f"Test:        {test_raw.shape}")

    # ---------------------------------------------------------
    # Prepare datasets
    # ---------------------------------------------------------

    train = prepare_dataset(train_raw)
    validation = prepare_dataset(validation_raw)
    test = prepare_dataset(test_raw)

    feature_columns = [
        column
        for column in train.columns
        if column != TARGET_COLUMN
    ]

    X_train = train[feature_columns]
    y_train = train[TARGET_COLUMN]

    X_validation = validation[feature_columns]
    y_validation = validation[TARGET_COLUMN]

    X_test = test[feature_columns]
    y_test = test[TARGET_COLUMN]

    print("\nFeatures:")
    print(f"Number of features: {len(feature_columns)}")

    # ---------------------------------------------------------
    # Build model
    # ---------------------------------------------------------

    model = build_model()

    print("\nTraining Gradient Boosting model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training finished.")

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    validation_metrics = evaluate(
        model,
        X_validation,
        y_validation,
        "VALIDATION - Gradient Boosting",
    )

    test_metrics = evaluate(
        model,
        X_test,
        y_test,
        "TEST - Gradient Boosting",
    )

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    evaluate_previous_lap(
        validation_raw,
        "VALIDATION",
    )

    evaluate_previous_lap(
        test_raw,
        "TEST",
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)
    print(MODEL_PATH)

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

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