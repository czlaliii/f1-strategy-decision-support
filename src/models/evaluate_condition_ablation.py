from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "multi_race"
MODEL_DIR = DATA_DIR / "models"

TRAIN_PATH = DATA_DIR / "train_context.parquet"
VALIDATION_PATH = DATA_DIR / "validation_context.parquet"
TEST_PATH = DATA_DIR / "test_context.parquet"

OUTPUT_PATH = MODEL_DIR / "condition_ablation_results.csv"

TARGET = "next_lap_time"


# ============================================================
# BASELINE FEATURES
# ============================================================

BASE_FEATURES = [
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


# ============================================================
# PRACTICE FEATURES
# ============================================================

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


# ============================================================
# MISSING INDICATORS
# ============================================================

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

MISSING_FEATURES = [
    f"{column}_missing"
    for column in MISSING_INDICATOR_COLUMNS
]


# ============================================================
# RACE CONTROL FEATURES
# ============================================================

VSC_FEATURES = [
    "vsc_duration_in_lap",
    "vsc_in_lap",
    "vsc_lap_fraction",
    "vsc_active_at_end",
]

SC_FEATURES = [
    "sc_duration_in_lap",
    "sc_in_lap",
    "sc_lap_fraction",
    "sc_active_at_end",
]


PRACTICE_FEATURES = (
    FP1_FEATURES
    + FP2_FEATURES
    + FP3_FEATURES
)

FINAL_BASE_FEATURES = (
    BASE_FEATURES
    + MISSING_FEATURES
    + PRACTICE_FEATURES
)


# ============================================================
# DATA PREPARATION
# ============================================================

def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in MISSING_INDICATOR_COLUMNS:
        if column not in df.columns:
            raise ValueError(
                f"Missing source column for indicator: {column}"
            )

        df[f"{column}_missing"] = (
            df[column].isna().astype(int)
        )

    return df


def load_datasets():
    train = add_model_features(
        pd.read_parquet(TRAIN_PATH)
    )

    validation = add_model_features(
        pd.read_parquet(VALIDATION_PATH)
    )

    test = add_model_features(
        pd.read_parquet(TEST_PATH)
    )

    required_features = (
        FINAL_BASE_FEATURES
        + VSC_FEATURES
        + SC_FEATURES
    )

    for split_name, df in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:
        missing = [
            column
            for column in required_features
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{split_name} is missing columns: {missing}"
            )

        if df[TARGET].isna().any():
            raise ValueError(
                f"{split_name} contains null target values."
            )

    print("\nDataset sizes:")
    print(f"Train:      {train.shape}")
    print(f"Validation: {validation.shape}")
    print(f"Test:       {test.shape}")

    return train, validation, test


# ============================================================
# MODEL
# ============================================================

def build_model(feature_columns, train_df):
    # Remove features with no observed values in training.
    actual_features = [
        column
        for column in feature_columns
        if train_df[column].notna().any()
    ]

    categorical_features = [
        column
        for column in ["compound"]
        if column in actual_features
    ]

    numerical_features = [
        column
        for column in actual_features
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

    regressor = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ]
    )

    return pipeline, actual_features


# ============================================================
# EVALUATION
# ============================================================

def evaluate_experiment(
    name,
    requested_features,
    train,
    validation,
    test,
):
    model, actual_features = build_model(
        requested_features,
        train,
    )

    print(f"\nRunning: {name}")
    print(f"Features used: {len(actual_features)}")

    X_train = train[actual_features]
    y_train = train[TARGET]

    X_validation = validation[actual_features]
    y_validation = validation[TARGET]

    X_test = test[actual_features]
    y_test = test[TARGET]

    model.fit(X_train, y_train)

    validation_predictions = model.predict(
        X_validation
    )

    test_predictions = model.predict(
        X_test
    )

    result = {
        "experiment": name,
        "features": len(actual_features),
        "validation_mae": mean_absolute_error(
            y_validation,
            validation_predictions,
        ),
        "validation_rmse": mean_squared_error(
            y_validation,
            validation_predictions,
        ) ** 0.5,
        "validation_r2": r2_score(
            y_validation,
            validation_predictions,
        ),
        "test_mae": mean_absolute_error(
            y_test,
            test_predictions,
        ),
        "test_rmse": mean_squared_error(
            y_test,
            test_predictions,
        ) ** 0.5,
        "test_r2": r2_score(
            y_test,
            test_predictions,
        ),
    }

    print(
        f"Validation MAE: {result['validation_mae']:.3f} s"
    )
    print(
        f"Test MAE:       {result['test_mae']:.3f} s"
    )

    return result


def main():
    print("=" * 78)
    print("SAFETY CAR / VSC FEATURE ABLATION")
    print("=" * 78)

    train, validation, test = load_datasets()

    experiments = {
        "BASELINE + PRACTICE": FINAL_BASE_FEATURES,

        "BASELINE + PRACTICE + VSC": (
            FINAL_BASE_FEATURES
            + VSC_FEATURES
        ),

        "BASELINE + PRACTICE + SC": (
            FINAL_BASE_FEATURES
            + SC_FEATURES
        ),

        "BASELINE + PRACTICE + VSC + SC": (
            FINAL_BASE_FEATURES
            + VSC_FEATURES
            + SC_FEATURES
        ),
    }

    results = []

    for name, features in experiments.items():
        result = evaluate_experiment(
            name,
            features,
            train,
            validation,
            test,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 78)
    print("CONDITION ABLATION RESULTS")
    print("=" * 78)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")

    print(
        "\nModel selection should prioritize validation MAE. "
        "Test results are reported for final comparison, "
        "not for selecting the model."
    )


if __name__ == "__main__":
    main()