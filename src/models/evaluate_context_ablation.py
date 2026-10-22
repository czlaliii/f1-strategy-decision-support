from pathlib import Path

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

TARGET = "next_lap_time"


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


HISTORICAL_FEATURES = [
    "driver_previous_race_pace_index",
    "driver_recent_3_race_pace_index",
    "driver_recent_5_race_pace_index",
    "driver_previous_finish_position",
    "driver_recent_3_finish_avg",
    "driver_recent_5_finish_avg",
    "driver_track_pace_index",
    "driver_track_race_count",
    "driver_track_finish_avg",
    "team_recent_3_race_pace_index",
    "team_recent_5_race_pace_index",
    "team_recent_3_finish_avg",
    "team_recent_5_finish_avg",
    "team_track_pace_index",
    "team_track_race_count",
]


QUALIFYING_FEATURES = [
    "qualifying_position",
    "qualifying_best_lap",
    "q1",
    "q2",
    "q3",
]


PRACTICE_FEATURES = [
    "fp1_available",
    "fp1_best_lap",
    "fp1_avg_top5_lap",
    "fp1_lap_count",
    "fp1_best_soft",
    "fp1_best_medium",
    "fp1_best_hard",
    "fp1_best_intermediate",
    "fp2_available",
    "fp2_best_lap",
    "fp2_avg_top5_lap",
    "fp2_lap_count",
    "fp2_best_soft",
    "fp2_best_medium",
    "fp2_best_hard",
    "fp2_best_intermediate",
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


SPRINT_FEATURES = [
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


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in [
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
    ]:
        df[f"{column}_missing"] = (
            df[column].isna().astype(int)
        )

    return df


def make_model(feature_columns, train_df):
    feature_columns = [
        column
        for column in feature_columns
        if column in train_df.columns
        and train_df[column].notna().any()
    ]

    categorical = [
        column
        for column in ["compound", "team"]
        if column in feature_columns
    ]

    numeric = [
        column
        for column in feature_columns
        if column not in categorical
    ]

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
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical,
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

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline, feature_columns


def evaluate_model(name, feature_columns, train_df, validation_df, test_df):
    pipeline, actual_features = make_model(
        feature_columns,
        train_df,
    )

    X_train = train_df[actual_features]
    y_train = train_df[TARGET]

    X_validation = validation_df[actual_features]
    y_validation = validation_df[TARGET]

    X_test = test_df[actual_features]
    y_test = test_df[TARGET]

    pipeline.fit(X_train, y_train)

    validation_pred = pipeline.predict(X_validation)
    test_pred = pipeline.predict(X_test)

    return {
        "experiment": name,
        "features": len(actual_features),
        "validation_mae": mean_absolute_error(
            y_validation,
            validation_pred,
        ),
        "validation_rmse": mean_squared_error(
            y_validation,
            validation_pred,
        ) ** 0.5,
        "validation_r2": r2_score(
            y_validation,
            validation_pred,
        ),
        "test_mae": mean_absolute_error(
            y_test,
            test_pred,
        ),
        "test_rmse": mean_squared_error(
            y_test,
            test_pred,
        ) ** 0.5,
        "test_r2": r2_score(
            y_test,
            test_pred,
        ),
    }


def main():
    print("=" * 78)
    print("CONTEXT FEATURE ABLATION STUDY")
    print("=" * 78)

    train_df = add_model_features(
        pd.read_parquet(TRAIN_PATH)
    )

    validation_df = add_model_features(
        pd.read_parquet(VALIDATION_PATH)
    )

    test_df = add_model_features(
        pd.read_parquet(TEST_PATH)
    )

    experiments = {
        "BASELINE": BASE_FEATURE_COLUMNS,

        "BASELINE + TEAM": (
            BASE_FEATURE_COLUMNS
            + ["team"]
        ),

        "BASELINE + HISTORICAL": (
            BASE_FEATURE_COLUMNS
            + HISTORICAL_FEATURES
            + ["team"]
        ),

        "BASELINE + QUALIFYING": (
            BASE_FEATURE_COLUMNS
            + QUALIFYING_FEATURES
        ),

        "BASELINE + PRACTICE": (
            BASE_FEATURE_COLUMNS
            + PRACTICE_FEATURES
        ),

        "BASELINE + QUALIFYING + PRACTICE": (
            BASE_FEATURE_COLUMNS
            + QUALIFYING_FEATURES
            + PRACTICE_FEATURES
        ),

        "BASELINE + SPRINT": (
            BASE_FEATURE_COLUMNS
            + SPRINT_FEATURES
        ),

        "BASELINE + ALL CONTEXT": (
            BASE_FEATURE_COLUMNS
            + ["team"]
            + HISTORICAL_FEATURES
            + QUALIFYING_FEATURES
            + PRACTICE_FEATURES
            + SPRINT_FEATURES
        ),
    }

    results = []

    for name, features in experiments.items():
        print(f"\nRunning: {name}")

        result = evaluate_model(
            name,
            features,
            train_df,
            validation_df,
            test_df,
        )

        results.append(result)

        print(
            f"Validation MAE: {result['validation_mae']:.3f}"
        )
        print(
            f"Test MAE:       {result['test_mae']:.3f}"
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 78)
    print("ABLATION RESULTS")
    print("=" * 78)

    print(
        results_df[
            [
                "experiment",
                "features",
                "validation_mae",
                "validation_rmse",
                "validation_r2",
                "test_mae",
                "test_rmse",
                "test_r2",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


if __name__ == "__main__":
    main()