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

    print(f"Dataset shape: {df.shape}")

    # --- 1. Train / validation split ---

    train_df = df[df["lap_number"] <= 50].copy()
    validation_df = df[df["lap_number"] > 50].copy()

    train_df = train_df.dropna(subset=[TARGET_COLUMN])
    validation_df = validation_df.dropna(subset=[TARGET_COLUMN])

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df[TARGET_COLUMN]

    # --- 2. Preprocessing ---

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

    # --- 3. Gradient Boosting model ---

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

    # --- 4. Training ---

    print("\n=== TRAINING ===")

    pipeline.fit(X_train, y_train)

    print("Gradient Boosting training completed.")

    # --- 5. Predictions ---

    predictions = pipeline.predict(X_validation)

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

    print("\n=== GRADIENT BOOSTING RESULTS ===")
    print(f"MAE:  {mae:.3f} s")
    print(f"RMSE: {rmse:.3f} s")
    print(f"R²:   {r2:.3f}")

    # --- 6. Feature importance ---

    # HistGradientBoostingRegressor has no feature_importances_,
    # so use permutation importance on the full pipeline.

    from sklearn.inspection import permutation_importance

    print("\n=== CALCULATING FEATURE IMPORTANCE ===")

    importance = permutation_importance(
        pipeline,
        X_validation,
        y_validation,
        n_repeats=10,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False,
    )

    print("\n=== FEATURE IMPORTANCE ===")

    print(
        importance_df.to_string(
            index=False,
            formatters={
                "importance_mean": "{:.6f}".format,
                "importance_std": "{:.6f}".format,
            },
        )
    )

    # --- 7. Top 10 features ---

    print("\n=== TOP 10 FEATURES ===")

    top_features = importance_df.head(10)

    for index, row in top_features.iterrows():
        print(
            f"{row['feature']:<30} "
            f"{row['importance_mean']:.6f}"
        )


if __name__ == "__main__":
    main()