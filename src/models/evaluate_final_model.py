from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "multi_race"
MODEL_DIR = DATA_DIR / "models"

MODEL_PATH = MODEL_DIR / "final_gradient_boosting_pipeline.joblib"
FEATURE_PATH = MODEL_DIR / "final_model_features.txt"

VALIDATION_PATH = DATA_DIR / "validation_context.parquet"
TEST_PATH = DATA_DIR / "test_context.parquet"

TARGET = "next_lap_time"


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


def evaluate_split(
    model,
    feature_columns,
    df,
    split_name,
):
    df = add_model_features(df)

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{split_name}: missing columns: {missing_columns}"
        )

    df = df[
        df[TARGET].notna()
    ].copy()

    X = df[feature_columns]
    y = df[TARGET]

    predictions = model.predict(X)

    df["prediction"] = predictions
    df["error"] = df["prediction"] - df[TARGET]
    df["absolute_error"] = df["error"].abs()

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

    print("\n" + "=" * 78)
    print(f"{split_name}")
    print("=" * 78)

    print(f"Rows: {len(df)}")
    print(f"MAE:  {mae:.3f} s")
    print(f"RMSE: {rmse:.3f} s")
    print(f"R²:   {r2:.3f}")

    return df


def print_group_analysis(
    df,
    column,
    title,
):
    if column not in df.columns:
        print(
            f"\n{title}: column '{column}' not available."
        )
        return

    grouped = (
        df.groupby(column, dropna=False)
        .agg(
            rows=("absolute_error", "size"),
            mae=("absolute_error", "mean"),
            rmse=("error", lambda x: np.sqrt(np.mean(x ** 2))),
        )
        .sort_values("mae")
    )

    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)

    print(
        grouped.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )


def print_lap_phase_analysis(df):
    df = df.copy()

    max_lap = df["lap_number"].max()

    if pd.isna(max_lap) or max_lap <= 0:
        return

    def phase(lap):
        ratio = lap / max_lap

        if ratio <= 0.33:
            return "EARLY"
        elif ratio <= 0.66:
            return "MIDDLE"
        return "LATE"

    df["lap_phase"] = df["lap_number"].apply(phase)

    print_group_analysis(
        df,
        "lap_phase",
        "ERROR BY RACE PHASE",
    )


def print_pit_analysis(df):
    if "pit_stop" not in df.columns:
        return

    df = df.copy()

    df["pit_category"] = np.where(
        df["pit_stop"].fillna(False),
        "PIT STOP",
        "NO PIT STOP",
    )

    print_group_analysis(
        df,
        "pit_category",
        "ERROR AROUND PIT STOPS",
    )


def print_vsc_analysis(df):
    vsc_columns = [
        "vsc_in_lap",
        "vsc_duration_in_lap",
        "vsc_lap_fraction",
        "vsc_active_at_end",
    ]

    available = [
        column
        for column in vsc_columns
        if column in df.columns
    ]

    if not available:
        print("\nNo VSC columns available.")
        return

    df = df.copy()

    df["vsc_category"] = (
        df[available]
        .fillna(0)
        .max(axis=1)
        .gt(0)
        .map({
            True: "VSC",
            False: "NORMAL",
        })
    )

    print_group_analysis(
        df,
        "vsc_category",
        "ERROR DURING VSC",
    )


def print_error_buckets(df):
    df = df.copy()

    bins = [
        -np.inf,
        1,
        2,
        3,
        5,
        10,
        20,
        np.inf,
    ]

    labels = [
        "<=1s",
        "1-2s",
        "2-3s",
        "3-5s",
        "5-10s",
        "10-20s",
        ">20s",
    ]

    df["error_bucket"] = pd.cut(
        df["absolute_error"],
        bins=bins,
        labels=labels,
    )

    counts = (
        df["error_bucket"]
        .value_counts()
        .reindex(labels)
        .fillna(0)
        .astype(int)
    )

    percentages = (
        counts / len(df) * 100
    )

    result = pd.DataFrame({
        "rows": counts,
        "percentage": percentages,
    })

    print("\n" + "-" * 78)
    print("ABSOLUTE ERROR DISTRIBUTION")
    print("-" * 78)

    print(
        result.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )


def print_worst_predictions(df, n=20):
    columns = [
        "race",
        "driver_number",
        "lap_number",
        "compound",
        "tyre_age",
        TARGET,
        "prediction",
        "error",
        "absolute_error",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    worst = (
        df.nlargest(
            n,
            "absolute_error",
        )[available]
    )

    print("\n" + "-" * 78)
    print(f"WORST {n} PREDICTIONS")
    print("-" * 78)

    print(
        worst.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


def main():
    print("=" * 78)
    print("FINAL MODEL ERROR ANALYSIS")
    print("=" * 78)

    # --------------------------------------------------------
    # Load model and feature list
    # --------------------------------------------------------

    model = joblib.load(MODEL_PATH)

    feature_columns = [
        line.strip()
        for line in FEATURE_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    print(
        f"\nLoaded model: {MODEL_PATH}"
    )

    print(
        f"Number of model features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Load validation and test
    # --------------------------------------------------------

    validation_raw = pd.read_parquet(
        VALIDATION_PATH
    )

    test_raw = pd.read_parquet(
        TEST_PATH
    )

    validation_df = evaluate_split(
        model,
        feature_columns,
        validation_raw,
        "VALIDATION - SILVERSTONE 2025",
    )

    test_df = evaluate_split(
        model,
        feature_columns,
        test_raw,
        "TEST - HUNGARY 2026",
    )

    # --------------------------------------------------------
    # Detailed analysis
    # --------------------------------------------------------

    print("\n\n")
    print("#" * 78)
    print("VALIDATION ERROR ANALYSIS")
    print("#" * 78)

    print_group_analysis(
        validation_df,
        "compound",
        "ERROR BY TYRE COMPOUND",
    )

    print_lap_phase_analysis(
        validation_df
    )

    print_pit_analysis(
        validation_df
    )

    print_vsc_analysis(
        validation_df
    )

    print_error_buckets(
        validation_df
    )

    print_worst_predictions(
        validation_df
    )

    print("\n\n")
    print("#" * 78)
    print("TEST ERROR ANALYSIS")
    print("#" * 78)

    print_group_analysis(
        test_df,
        "compound",
        "ERROR BY TYRE COMPOUND",
    )

    print_lap_phase_analysis(
        test_df
    )

    print_pit_analysis(
        test_df
    )

    print_vsc_analysis(
        test_df
    )

    print_error_buckets(
        test_df
    )

    print_worst_predictions(
        test_df
    )


if __name__ == "__main__":
    main()