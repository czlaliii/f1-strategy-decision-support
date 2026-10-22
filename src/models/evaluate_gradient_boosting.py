"""Reproducible evaluation of Gradient Boosting with and without VSC inputs."""

import os
import tempfile
from pathlib import Path

# Keep Matplotlib headless and its cache inside the project so the script also
# works in non-interactive Windows environments.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "f1_strategy_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("data/processed/hungary_2026/ml_dataset_clean.parquet")
OUTPUT_DIR = Path("data/processed/hungary_2026/model_evaluation")
TARGET_COLUMN = "next_lap_time"
TRAIN_LAST_LAP = 50
RANDOM_STATE = 42
VSC_COLUMNS = [
    "vsc_duration_in_lap",
    "vsc_in_lap",
    "vsc_lap_fraction",
    "vsc_active_at_end",
]
BASE_FEATURE_COLUMNS = [
    "driver_number", "lap_number", "compound", "tyre_age", "position",
    "gap_to_leader", "laps_down", "interval", "previous_lap_time",
    "rolling_3_lap_avg", "rolling_5_lap_avg", "lap_time_delta",
    "position_change", "gap_change", "interval_change", "laps_since_pit",
    "track_air_temp_difference", "air_temperature", "track_temperature",
    "rainfall", "humidity", "wind_speed", "position_missing",
    "interval_missing", "interval_change_missing", "has_previous_pit",
    "gap_to_leader_missing", "gap_change_missing", "previous_lap_time_missing",
    "rolling_3_lap_avg_missing", "rolling_5_lap_avg_missing",
    "lap_time_delta_missing", "position_change_missing",
]

def build_pipeline(feature_columns):
    """Build the same preprocessing and model configuration as training."""
    categorical_features = ["compound"]
    numerical_features = [
        column for column in feature_columns if column not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numerical_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

def metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R^2": r2_score(y_true, predictions),
    }


def save_plots(results, importance_df):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        results["actual_next_lap_time"],
        results["predicted_next_lap_time"],
        alpha=0.7,
        edgecolors="none",
    )
    lower = min(results["actual_next_lap_time"].min(), results["predicted_next_lap_time"].min())
    upper = max(results["actual_next_lap_time"].max(), results["predicted_next_lap_time"].max())
    ax.plot([lower, upper], [lower, upper], "k--", linewidth=1, label="Perfect prediction")
    ax.set(xlabel="Actual next lap time (s)", ylabel="Predicted next lap time (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted_gradient_boosting.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(results["error"], bins=30, color="#4472C4", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Prediction error: actual - predicted (s)", ylabel="Validation samples")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_error_distribution.png", dpi=300)
    plt.close(fig)

    top_features = importance_df.head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
        color="#4472C4",
        capsize=3,
    )
    ax.set(xlabel="Increase in validation MAE after permutation", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance_top_15.png", dpi=300)
    plt.close(fig)

def main():
    print("Loading cleaned ML dataset...")
    df = pd.read_parquet(DATA_PATH).sort_values(["lap_number", "driver_number"])

    required_columns = BASE_FEATURE_COLUMNS + VSC_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    train_df = df[df["lap_number"] <= TRAIN_LAST_LAP].copy()
    validation_df = df[df["lap_number"] > TRAIN_LAST_LAP].copy()
    y_train = train_df[TARGET_COLUMN]
    y_validation = validation_df[TARGET_COLUMN]

    print(f"Training rows: {len(train_df)}; validation rows: {len(validation_df)}")
    print(f"Training laps: 1-{TRAIN_LAST_LAP}; validation laps: {validation_df['lap_number'].min()}-{validation_df['lap_number'].max()}")

    feature_sets = {
        "Without VSC": BASE_FEATURE_COLUMNS,
        "With VSC": BASE_FEATURE_COLUMNS + VSC_COLUMNS,
    }
    fitted_models = {}
    predictions = {}
    comparison = {}

    for name, feature_columns in feature_sets.items():
        model = build_pipeline(feature_columns)
        model.fit(train_df[feature_columns], y_train)
        prediction = model.predict(validation_df[feature_columns])
        fitted_models[name] = model
        predictions[name] = prediction
        comparison[name] = metrics(y_validation, prediction)

    comparison_df = pd.DataFrame(comparison)
    comparison_df["Change (With - Without)"] = (
        comparison_df["With VSC"] - comparison_df["Without VSC"]
    )
    print("\n=== MODEL COMPARISON ===")
    print(comparison_df.to_string(float_format="{:.6f}".format))

    # Permuting pipeline input columns measures the effect of each original feature,
    # while the fitted ColumnTransformer still applies its one-hot encoding internally.
    print("\nCalculating validation permutation importance for the VSC model...")
    importance = permutation_importance(
        fitted_models["With VSC"],
        validation_df[feature_sets["With VSC"]],
        y_validation,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": feature_sets["With VSC"],
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)

    print("\n=== TOP 15 FEATURE IMPORTANCE ===")
    print(importance_df.head(15).to_string(index=False, float_format="{:.6f}".format))

    results = validation_df[["lap_number", "driver_number"] + VSC_COLUMNS].copy()
    results["actual_next_lap_time"] = y_validation
    results["predicted_next_lap_time"] = predictions["With VSC"]
    results["error"] = results["actual_next_lap_time"] - results["predicted_next_lap_time"]
    results["absolute_error"] = results["error"].abs()
    results["vsc_in_lap"] = results["vsc_in_lap"].astype(int)

    worst = results.nlargest(15, "absolute_error")
    error_by_lap = results.groupby("lap_number")["absolute_error"].agg(["count", "mean", "median", "max"]).sort_values("mean", ascending=False)
    error_by_driver = results.groupby("driver_number")["absolute_error"].agg(["count", "mean", "median", "max"]).sort_values("mean", ascending=False)
    error_by_vsc = results.groupby("vsc_in_lap")["absolute_error"].agg(["count", "mean", "median", "max"])

    print("\n=== WORST 15 VALIDATION PREDICTIONS ===")
    print(worst.to_string(index=False, float_format="{:.3f}".format))
    print("\n=== HIGHEST MEAN ABSOLUTE ERROR BY LAP ===")
    print(error_by_lap.head(10).to_string(float_format="{:.3f}".format))
    print("\n=== HIGHEST MEAN ABSOLUTE ERROR BY DRIVER ===")
    print(error_by_driver.head(10).to_string(float_format="{:.3f}".format))
    print("\n=== ERROR BY VSC STATUS ===")
    print(error_by_vsc.to_string(float_format="{:.3f}".format))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUTPUT_DIR / "predictions_gradient_boosting.parquet", index=False)
    importance_df.to_parquet(OUTPUT_DIR / "feature_importance.parquet", index=False)
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv")
    save_plots(results, importance_df)
    print(f"\nSaved evaluation outputs to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
