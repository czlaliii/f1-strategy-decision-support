import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "data/processed/hungary_2026/ml_dataset_clean.parquet"

TRAIN_LAST_LAP = 50


def main():
    print("Loading cleaned ML dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # ---------------------------------------------------------
    # SORT DATA
    # ---------------------------------------------------------

    df = df.sort_values(
        ["lap_number", "driver_number"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # TRAIN DATA
    # ---------------------------------------------------------

    train_df = df[
        df["lap_number"] <= TRAIN_LAST_LAP
    ].copy()

    print("\n=== TRAIN DATA ===")
    print(f"Rows: {len(train_df)}")
    print(
        f"Laps: {train_df['lap_number'].min()} - "
        f"{train_df['lap_number'].max()}"
    )

    # ---------------------------------------------------------
    # FEATURES
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # FEATURE TYPES
    # ---------------------------------------------------------

    categorical_features = [
        "compound"
    ]

    numerical_features = [
        column
        for column in feature_columns
        if column not in categorical_features
    ]

    # ---------------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    print("\n=== TRAINING MODEL ===")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # ---------------------------------------------------------
    # GET FEATURE NAMES
    # ---------------------------------------------------------

    fitted_preprocessor = model.named_steps[
        "preprocessor"
    ]

    feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )

    # ---------------------------------------------------------
    # GET COEFFICIENTS
    # ---------------------------------------------------------

    regressor = model.named_steps[
        "regressor"
    ]

    coefficients = regressor.coef_

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )

    coefficient_df[
        "absolute_coefficient"
    ] = coefficient_df["coefficient"].abs()

    coefficient_df = coefficient_df.sort_values(
        "absolute_coefficient",
        ascending=False
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print("\n=== TOP 20 FEATURES BY ABSOLUTE COEFFICIENT ===")

    print(
        coefficient_df[
            [
                "feature",
                "coefficient",
                "absolute_coefficient",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # POSITIVE / NEGATIVE EFFECTS
    # ---------------------------------------------------------

    print("\n=== TOP 10 POSITIVE COEFFICIENTS ===")

    print(
        coefficient_df
        .sort_values(
            "coefficient",
            ascending=False
        )[
            [
                "feature",
                "coefficient",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n=== TOP 10 NEGATIVE COEFFICIENTS ===")

    print(
        coefficient_df
        .sort_values(
            "coefficient",
            ascending=True
        )[
            [
                "feature",
                "coefficient",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()