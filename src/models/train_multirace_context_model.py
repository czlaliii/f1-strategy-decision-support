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

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "multi_race"
)

TRAIN_PATH = DATA_DIR / "train_context.parquet"
VALIDATION_PATH = DATA_DIR / "validation_context.parquet"
TEST_PATH = DATA_DIR / "test_context.parquet"

MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / "gradient_boosting_with_context_pipeline.joblib"


TARGET = "next_lap_time"


# ============================================================================
# 1. EREDETI 35 BASE FEATURE
# ============================================================================

BASE_FEATURE_COLUMNS = [
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
    "pressure",
    "rainfall",
    "humidity",
    "wind_speed",
    "wind_direction",
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


# ============================================================================
# 2. CONTEXT FEATURE
# ============================================================================

CONTEXT_FEATURE_COLUMNS = [
    # Historical driver
    "driver_previous_race_pace_index",
    "driver_recent_3_race_pace_index",
    "driver_recent_5_race_pace_index",
    "driver_previous_finish_position",
    "driver_recent_3_finish_avg",
    "driver_recent_5_finish_avg",
    "driver_track_pace_index",
    "driver_track_race_count",
    "driver_track_finish_avg",

    # Historical team
    "team_recent_3_race_pace_index",
    "team_recent_5_race_pace_index",
    "team_recent_3_finish_avg",
    "team_recent_5_finish_avg",
    "team_track_pace_index",
    "team_track_race_count",

    # Pre-race qualifying
    "qualifying_position",
    "qualifying_best_lap",
    "q1",
    "q2",
    "q3",

    # FP1
    "fp1_available",
    "fp1_best_lap",
    "fp1_avg_top5_lap",
    "fp1_lap_count",
    "fp1_best_soft",
    "fp1_best_medium",
    "fp1_best_hard",
    "fp1_best_intermediate",

    # FP2
    "fp2_available",
    "fp2_best_lap",
    "fp2_avg_top5_lap",
    "fp2_lap_count",
    "fp2_best_soft",
    "fp2_best_medium",
    "fp2_best_hard",
    "fp2_best_intermediate",

    # FP3
    "fp3_available",
    "fp3_best_lap",
    "fp3_avg_top5_lap",
    "fp3_lap_count",
    "fp3_best_soft",
    "fp3_best_medium",
    "fp3_best_hard",
    "fp3_best_intermediate",
    "fp3_best_wet",

    # Sprint / sprint qualifying
    "sq_available",
    "sq_position",
    "sq_best_lap",
    "sq_avg_top5_lap",
    "sq_lap_count",
    "sprint_available",
    "sprint_position",
    "sprint_best_lap",
    "sprint_avg_top5_lap",
    "sprint_lap_count",
]


# A team új kategóriás feature.
CATEGORICAL_FEATURES = [
    "compound",
    "team",
]


NUMERICAL_FEATURES = [
    column
    for column in (
        BASE_FEATURE_COLUMNS
        + CONTEXT_FEATURE_COLUMNS
    )
    if column not in CATEGORICAL_FEATURES
]


FEATURE_COLUMNS = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


def rmse(y_true, y_pred) -> float:
    return mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5


def evaluate(
    name: str,
    y_true: pd.Series,
    y_pred,
) -> None:

    print(f"\n{name}")
    print("-" * 60)

    print(
        f"MAE:   {mean_absolute_error(y_true, y_pred):.3f} s"
    )

    print(
        f"RMSE:  {rmse(y_true, y_pred):.3f} s"
    )

    print(
        f"R²:    {r2_score(y_true, y_pred):.3f}"
    )

def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the same derived missingness indicators used by the
    baseline model.

    The source parquet does not physically contain these columns;
    they are created immediately before model training.
    """

    df = df.copy()

    # --------------------------------------------------------------
    # Missingness indicators
    # --------------------------------------------------------------

    missing_indicator_columns = [
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

    for column in missing_indicator_columns:
        df[f"{column}_missing"] = (
            df[column].isna().astype(int)
        )

    # --------------------------------------------------------------
    # Previous pit indicator
    # --------------------------------------------------------------

    df["has_previous_pit"] = (
        df["laps_since_pit"].notna().astype(int)
    )

    return df

def main() -> None:

    print("=" * 78)
    print("MULTI-RACE GRADIENT BOOSTING WITH HISTORICAL + PRE-RACE CONTEXT")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    train_df = pd.read_parquet(TRAIN_PATH)
    validation_df = pd.read_parquet(VALIDATION_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Create the same derived model features as in the baseline.
    train_df = add_model_features(train_df)
    validation_df = add_model_features(validation_df)
    test_df = add_model_features(test_df)

    print("\nDataset sizes:")

    print(
        f"Train:       {train_df.shape}"
    )

    print(
        f"Validation:  {validation_df.shape}"
    )

    print(
        f"Test:        {test_df.shape}"
    )

    # ------------------------------------------------------------------
    # Feature validation
    # ------------------------------------------------------------------

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in train_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing features:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_features
            )
        )

    print("\nFeature summary:")
    print(
        f"Base features:       {len(BASE_FEATURE_COLUMNS)}"
    )

    print(
        f"Context features:    {len(CONTEXT_FEATURE_COLUMNS) + 1}"
    )

    print(
        f"Total model inputs:  {len(FEATURE_COLUMNS)}"
    )

    print("\nCategorical:")
    for column in CATEGORICAL_FEATURES:
        print(f"- {column}")

    # ------------------------------------------------------------------
    # X / y
    # ------------------------------------------------------------------

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET]

    X_validation = validation_df[FEATURE_COLUMNS].copy()
    y_validation = validation_df[TARGET]

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    numeric_pipeline = Pipeline(
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
                numeric_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    print("\nTraining Gradient Boosting model...")
    pipeline.fit(
        X_train,
        y_train,
    )

    print("Training finished.")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    validation_pred = pipeline.predict(
        X_validation
    )

    evaluate(
        "VALIDATION - Gradient Boosting + Context",
        y_validation,
        validation_pred,
    )

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    test_pred = pipeline.predict(
        X_test
    )

    evaluate(
        "TEST - Gradient Boosting + Context",
        y_test,
        test_pred,
    )

    # ------------------------------------------------------------------
    # Previous-lap baseline
    # ------------------------------------------------------------------

    validation_baseline_mask = (
        validation_df["previous_lap_time"].notna()
    )

    test_baseline_mask = (
        test_df["previous_lap_time"].notna()
    )

    evaluate(
        "VALIDATION - Previous Lap Baseline",
        y_validation[validation_baseline_mask],
        validation_df.loc[
            validation_baseline_mask,
            "previous_lap_time",
        ],
    )

    evaluate(
        "TEST - Previous Lap Baseline",
        y_test[test_baseline_mask],
        test_df.loc[
            test_baseline_mask,
            "previous_lap_time",
        ],
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    print("\n" + "=" * 78)
    print("MODEL SAVED")
    print("=" * 78)

    print(MODEL_PATH)


if __name__ == "__main__":
    main()