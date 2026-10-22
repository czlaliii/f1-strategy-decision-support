"""Compare baseline vs. Safety Car (SC) features by lap on Silverstone 2025.

Run from the repository root:
    python -m src.models.analyze_sc_effect_by_lap

Uses the saved race-wise training and validation split only. The test split is
intentionally not loaded or used for model selection.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path("data/processed/multi_race")
TRAIN_PATH = DATA_DIR / "train_context.parquet"
VALIDATION_PATH = DATA_DIR / "validation_context.parquet"
OUTPUT_DIR = DATA_DIR / "models"

TARGET = "next_lap_time"
LAP_COLUMN = "lap_number"
DRIVER_COLUMN = "driver_number"
RACE_COLUMN = "race"
PRE_SC_LAPS = 3
POST_SC_LAPS = 3

# Keep these feature lists aligned with train_final_model.py and
# evaluate_condition_ablation.py.
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

MISSING_FEATURES = [
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

BASELINE_PRACTICE_FEATURES = (
    BASE_FEATURES + MISSING_FEATURES + FP1_FEATURES + FP2_FEATURES + FP3_FEATURES
)
SC_FEATURES = [
    "sc_duration_in_lap",
    "sc_in_lap",
    "sc_lap_fraction",
    "sc_active_at_end",
]
VSC_FEATURES = [
    "vsc_duration_in_lap",
    "vsc_in_lap",
    "vsc_lap_fraction",
    "vsc_active_at_end",
]

MODEL_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
    "random_state": 42,
}


def _require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} dataset is missing required columns: {missing}")


def _ensure_missing_indicators(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Rebuild *_missing flags from their source columns if the saved split omitted them."""
    source_by_indicator = {
        "position_missing": "position",
        "interval_missing": "interval",
        "interval_change_missing": "interval_change",
        "gap_to_leader_missing": "gap_to_leader",
        "gap_change_missing": "gap_change",
        "previous_lap_time_missing": "previous_lap_time",
        "rolling_3_lap_avg_missing": "rolling_3_lap_avg",
        "rolling_5_lap_avg_missing": "rolling_5_lap_avg",
        "lap_time_delta_missing": "lap_time_delta",
        "position_change_missing": "position_change",
    }
    df = df.copy()
    created = []
    for indicator, source in source_by_indicator.items():
        if indicator in df.columns:
            continue
        if source not in df.columns:
            raise ValueError(
                f"Cannot reconstruct {indicator!r} in {label}: "
                f"source column {source!r} is also missing."
            )
        df[indicator] = df[source].isna().astype("int8")
        created.append(indicator)

    if created:
        print(
            f"{label}: rebuilt {len(created)} missing-indicator columns from "
            "current null values: " + ", ".join(created)
        )
        print(
            "NOTE: if upstream preprocessing already filled any source values, "
            "their original missingness cannot be recovered from this split."
        )
    return df


def _usable_features(train_df: pd.DataFrame, requested: list[str]) -> list[str]:
    """Only remove features absent from train or entirely null in train."""
    missing = [column for column in requested if column not in train_df.columns]
    if missing:
        raise ValueError(
            "Expected model feature columns are missing from training data: "
            + ", ".join(missing)
        )
    all_null = [column for column in requested if train_df[column].isna().all()]
    usable = [column for column in requested if column not in all_null]
    if all_null:
        print("Features omitted because they are all null in train:", all_null)
    return usable


def _build_model(features: list[str]) -> Pipeline:
    categorical_features = [column for column in features if column == "compound"]
    numeric_features = [column for column in features if column not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor(**MODEL_PARAMS)),
        ]
    )


def _is_active(df: pd.DataFrame, prefixes: str) -> pd.Series:
    """Return a robust event-active flag from duration/fraction/state columns."""
    tests = []
    duration_col = f"{prefixes}_duration_in_lap"
    flag_col = f"{prefixes}_in_lap"
    fraction_col = f"{prefixes}_lap_fraction"
    end_col = f"{prefixes}_active_at_end"

    if duration_col in df:
        tests.append(pd.to_numeric(df[duration_col], errors="coerce").fillna(0) > 0.5)
    if flag_col in df:
        tests.append(pd.to_numeric(df[flag_col], errors="coerce").fillna(0) > 0.5)
    if fraction_col in df:
        tests.append(pd.to_numeric(df[fraction_col], errors="coerce").fillna(0) > 0.01)
    if end_col in df:
        tests.append(pd.to_numeric(df[end_col], errors="coerce").fillna(0) > 0.5)

    if not tests:
        raise ValueError(f"No usable {prefixes.upper()} status columns found.")
    result = tests[0].copy()
    for test in tests[1:]:
        result = result | test
    return result


def _find_lap_intervals(active_laps: list[int]) -> list[tuple[int, int]]:
    """Group consecutive SC-affected lap numbers into event intervals."""
    if not active_laps:
        return []
    values = sorted(set(int(lap) for lap in active_laps))
    intervals: list[tuple[int, int]] = []
    start = end = values[0]
    for lap in values[1:]:
        if lap <= end + 1:
            end = lap
        else:
            intervals.append((start, end))
            start = end = lap
    intervals.append((start, end))
    return intervals


def _classify_lap(
    lap: int,
    sc_intervals: list[tuple[int, int]],
    vsc_active: bool,
) -> str:
    # Current-lap conditions label the decision point; next_lap_time is the target.
    if any(start <= lap <= end for start, end in sc_intervals):
        return "SC_ACTIVE"
    # Keep VSC-only laps separate so they do not distort the SC comparison.
    if vsc_active:
        return "VSC_ACTIVE"
    if any(end < lap <= end + POST_SC_LAPS for _, end in sc_intervals):
        return "POST_SC"
    if any(start - PRE_SC_LAPS <= lap < start for start, _ in sc_intervals):
        return "PRE_SC"
    return "NORMAL"


def _format_interval(interval: tuple[int, int]) -> str:
    start, end = interval
    return str(start) if start == end else f"{start}-{end}"


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training split not found: {TRAIN_PATH}. Run from the repository root."
        )
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation split not found: {VALIDATION_PATH}. Run the context-split preparation first."
        )

    print("=" * 78)
    print("SAFETY CAR EFFECT BY LAP — SILVERSTONE 2025 VALIDATION")
    print("=" * 78)
    print("Loading training and validation splits (test split is not used)...")
    train_df = pd.read_parquet(TRAIN_PATH)
    validation_df = pd.read_parquet(VALIDATION_PATH)
    # Keep every validation lap for labeling; the last lap may lack a target
    # but still defines a target lap's state.
    validation_conditions_all = validation_df.copy()

    # The context split may omit these flags; recreate them before feature
    # selection and imputer fitting.
    train_df = _ensure_missing_indicators(train_df, "Training")
    validation_df = _ensure_missing_indicators(validation_df, "Validation")
    validation_conditions_all = _ensure_missing_indicators(
        validation_conditions_all, "Validation condition copy"
    )

    _require_columns(train_df, [TARGET], "Training")
    _require_columns(
        validation_df,
        [TARGET, LAP_COLUMN, DRIVER_COLUMN] + SC_FEATURES + VSC_FEATURES,
        "Validation",
    )

    if RACE_COLUMN in validation_df.columns:
        races = validation_df[RACE_COLUMN].dropna().astype(str).unique().tolist()
        print(f"Validation race(s): {races}")
        if len(races) > 1:
            raise ValueError(
                "This analysis expects one validation race. The validation dataset "
                f"contains: {races}"
            )

    train_df = train_df.loc[train_df[TARGET].notna()].copy()
    validation_df = validation_df.loc[validation_df[TARGET].notna()].copy()

    baseline_features = _usable_features(train_df, BASELINE_PRACTICE_FEATURES)
    sc_features = _usable_features(train_df, BASELINE_PRACTICE_FEATURES + SC_FEATURES)

    # Same rows, train/validation split, model settings, and baseline features.
    y_train = pd.to_numeric(train_df[TARGET], errors="raise")
    y_val = pd.to_numeric(validation_df[TARGET], errors="raise")

    print(f"Train rows:      {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Baseline features: {len(baseline_features)}")
    print(f"Baseline + SC features: {len(sc_features)}")

    print("\nFitting baseline + practice model...")
    baseline_model = _build_model(baseline_features)
    baseline_model.fit(train_df[baseline_features], y_train)
    baseline_predictions = baseline_model.predict(validation_df[baseline_features])

    print("Fitting baseline + practice + SC model...")
    sc_model = _build_model(sc_features)
    sc_model.fit(train_df[sc_features], y_train)
    sc_predictions = sc_model.predict(validation_df[sc_features])

    print("\nOverall validation metrics:")
    print(
        f"  Baseline + practice: MAE={mean_absolute_error(y_val, baseline_predictions):.3f}s, "
        f"RMSE={np.sqrt(mean_squared_error(y_val, baseline_predictions)):.3f}s, "
        f"R²={r2_score(y_val, baseline_predictions):.3f}"
    )
    print(
        f"  Baseline + practice + SC: MAE={mean_absolute_error(y_val, sc_predictions):.3f}s, "
        f"RMSE={np.sqrt(mean_squared_error(y_val, sc_predictions)):.3f}s, "
        f"R²={r2_score(y_val, sc_predictions):.3f}"
    )

    # A lap is SC-active if any driver row shows SC, so each event counts once.
    event_df = validation_conditions_all[[LAP_COLUMN]].copy()
    event_df["sc_active_row"] = _is_active(validation_conditions_all, "sc").to_numpy()
    event_df["vsc_active_row"] = _is_active(validation_conditions_all, "vsc").to_numpy()
    event_df["sc_duration"] = pd.to_numeric(
        validation_conditions_all["sc_duration_in_lap"], errors="coerce"
    ).fillna(0).to_numpy()
    event_df["sc_fraction"] = pd.to_numeric(
        validation_conditions_all["sc_lap_fraction"], errors="coerce"
    ).fillna(0).to_numpy()
    event_df["vsc_duration"] = pd.to_numeric(
        validation_conditions_all["vsc_duration_in_lap"], errors="coerce"
    ).fillna(0).to_numpy()
    event_df["vsc_fraction"] = pd.to_numeric(
        validation_conditions_all["vsc_lap_fraction"], errors="coerce"
    ).fillna(0).to_numpy()

    lap_conditions = (
        event_df.groupby(LAP_COLUMN, as_index=False)
        .agg(
            sc_active=("sc_active_row", "max"),
            vsc_active=("vsc_active_row", "max"),
            sc_duration_max=("sc_duration", "max"),
            sc_fraction_max=("sc_fraction", "max"),
            vsc_duration_max=("vsc_duration", "max"),
            vsc_fraction_max=("vsc_fraction", "max"),
        )
        .sort_values(LAP_COLUMN)
        .reset_index(drop=True)
    )

    active_sc_laps = lap_conditions.loc[
        lap_conditions["sc_active"].astype(bool), LAP_COLUMN
    ].astype(int).tolist()
    sc_intervals = _find_lap_intervals(active_sc_laps)
    print("\nSC-affected lap interval(s): " + (
        ", ".join(_format_interval(interval) for interval in sc_intervals)
        if sc_intervals
        else "none detected"
    ))
    if not sc_intervals:
        raise ValueError(
            "No SC-affected laps were detected in validation. Check the SC columns "
            "and race-condition preprocessing before interpreting this comparison."
        )

    lap_conditions["segment"] = [
        _classify_lap(
            int(row[LAP_COLUMN]), sc_intervals, bool(row["vsc_active"])
        )
        for _, row in lap_conditions.iterrows()
    ]
    lap_to_segment = dict(
        zip(lap_conditions[LAP_COLUMN].astype(int), lap_conditions["segment"])
    )

    row_results = validation_df[[DRIVER_COLUMN, LAP_COLUMN, TARGET]].copy()
    if RACE_COLUMN in validation_df.columns:
        row_results.insert(0, RACE_COLUMN, validation_df[RACE_COLUMN].to_numpy())
    row_results["segment"] = row_results[LAP_COLUMN].astype(int).map(lap_to_segment)
    # Label each prediction by its target lap's (lap_number + 1) SC/VSC segment.
    # Used for grouping only, not as a model input.
    row_results["target_lap_number"] = row_results[LAP_COLUMN].astype(int) + 1
    row_results["target_segment"] = row_results["target_lap_number"].map(lap_to_segment)
    row_results["baseline_prediction"] = baseline_predictions
    row_results["sc_prediction"] = sc_predictions
    row_results["baseline_abs_error"] = np.abs(y_val.to_numpy() - baseline_predictions)
    row_results["sc_abs_error"] = np.abs(y_val.to_numpy() - sc_predictions)
    # Signed error: positive = overprediction; negative = underprediction.
    row_results["baseline_signed_error"] = baseline_predictions - y_val.to_numpy()
    row_results["sc_signed_error"] = sc_predictions - y_val.to_numpy()
    # Positive improvement means the SC model's absolute error is lower.
    row_results["mae_improvement_per_row"] = (
        row_results["baseline_abs_error"] - row_results["sc_abs_error"]
    )

    lap_metrics = (
        row_results.groupby([LAP_COLUMN, "segment"], as_index=False)
        .agg(
            rows=(TARGET, "size"),
            drivers=(DRIVER_COLUMN, "nunique"),
            actual_next_lap_time_mean=(TARGET, "mean"),
            baseline_mae=("baseline_abs_error", "mean"),
            sc_mae=("sc_abs_error", "mean"),
            improvement=("mae_improvement_per_row", "mean"),
        )
        .sort_values(LAP_COLUMN)
        .reset_index(drop=True)
    )
    lap_metrics["improvement_pct"] = np.where(
        lap_metrics["baseline_mae"] > 0,
        100 * lap_metrics["improvement"] / lap_metrics["baseline_mae"],
        np.nan,
    )
    lap_metrics["sc_duration_max"] = lap_metrics[LAP_COLUMN].map(
        lap_conditions.set_index(LAP_COLUMN)["sc_duration_max"]
    )
    lap_metrics["sc_fraction_max"] = lap_metrics[LAP_COLUMN].map(
        lap_conditions.set_index(LAP_COLUMN)["sc_fraction_max"]
    )
    lap_metrics["vsc_duration_max"] = lap_metrics[LAP_COLUMN].map(
        lap_conditions.set_index(LAP_COLUMN)["vsc_duration_max"]
    )

    segment_order = ["SC_ACTIVE", "POST_SC", "PRE_SC", "VSC_ACTIVE", "NORMAL"]
    segment_summary = (
        row_results.groupby("segment", as_index=False)
        .agg(
            rows=(TARGET, "size"),
            laps=(LAP_COLUMN, "nunique"),
            baseline_mae=("baseline_abs_error", "mean"),
            sc_mae=("sc_abs_error", "mean"),
            improvement=("mae_improvement_per_row", "mean"),
        )
    )
    segment_summary["improvement_pct"] = np.where(
        segment_summary["baseline_mae"] > 0,
        100 * segment_summary["improvement"] / segment_summary["baseline_mae"],
        np.nan,
    )
    segment_summary["segment"] = pd.Categorical(
        segment_summary["segment"], categories=segment_order, ordered=True
    )
    segment_summary = segment_summary.sort_values("segment").reset_index(drop=True)
    segment_summary["segment"] = segment_summary["segment"].astype(str)

    # Main diagnostic: errors grouped by the target lap's condition (after prediction).
    target_segment_summary = (
        row_results.dropna(subset=["target_segment"])
        .groupby("target_segment", as_index=False)
        .agg(
            rows=(TARGET, "size"),
            target_laps=("target_lap_number", "nunique"),
            actual_mean=(TARGET, "mean"),
            baseline_bias=("baseline_signed_error", "mean"),
            sc_bias=("sc_signed_error", "mean"),
            baseline_mae=("baseline_abs_error", "mean"),
            sc_mae=("sc_abs_error", "mean"),
            improvement=("mae_improvement_per_row", "mean"),
        )
    )
    target_segment_summary["improvement_pct"] = np.where(
        target_segment_summary["baseline_mae"] > 0,
        100 * target_segment_summary["improvement"] / target_segment_summary["baseline_mae"],
        np.nan,
    )
    target_order = ["SC_ACTIVE", "POST_SC", "PRE_SC", "VSC_ACTIVE", "NORMAL"]
    target_segment_summary["target_segment"] = pd.Categorical(
        target_segment_summary["target_segment"], categories=target_order, ordered=True
    )
    target_segment_summary = target_segment_summary.sort_values("target_segment").reset_index(drop=True)
    target_segment_summary["target_segment"] = target_segment_summary["target_segment"].astype(str)

    target_lap_detail = (
        row_results.dropna(subset=["target_segment"])
        .groupby(["target_lap_number", "target_segment"], as_index=False)
        .agg(
            rows=(TARGET, "size"),
            drivers=(DRIVER_COLUMN, "nunique"),
            actual_mean=(TARGET, "mean"),
            baseline_prediction_mean=("baseline_prediction", "mean"),
            sc_prediction_mean=("sc_prediction", "mean"),
            baseline_bias=("baseline_signed_error", "mean"),
            sc_bias=("sc_signed_error", "mean"),
            baseline_mae=("baseline_abs_error", "mean"),
            sc_mae=("sc_abs_error", "mean"),
            improvement=("mae_improvement_per_row", "mean"),
        )
        .sort_values("target_lap_number")
        .reset_index(drop=True)
    )
    target_lap_detail["improvement_pct"] = np.where(
        target_lap_detail["baseline_mae"] > 0,
        100 * target_lap_detail["improvement"] / target_lap_detail["baseline_mae"],
        np.nan,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    segment_path = OUTPUT_DIR / "sc_effect_segment_summary.csv"
    lap_path = OUTPUT_DIR / "sc_effect_lap_detail.csv"
    row_path = OUTPUT_DIR / "sc_effect_row_predictions.csv"
    target_segment_path = OUTPUT_DIR / "sc_effect_target_segment_summary.csv"
    target_lap_path = OUTPUT_DIR / "sc_effect_target_lap_detail.csv"

    segment_summary.to_csv(segment_path, index=False, float_format="%.6f")
    lap_metrics.to_csv(lap_path, index=False, float_format="%.6f")
    row_results.to_csv(row_path, index=False, float_format="%.6f")
    target_segment_summary.to_csv(target_segment_path, index=False, float_format="%.6f")
    target_lap_detail.to_csv(target_lap_path, index=False, float_format="%.6f")

    print("\nSegment summary — MAE is lower-is-better:")
    display_segments = segment_summary.copy()
    for column in ["baseline_mae", "sc_mae", "improvement", "improvement_pct"]:
        display_segments[column] = display_segments[column].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else ""
        )
    print(display_segments.to_string(index=False))
    print("\nPer-lap detail (SC/VSC and nearby laps):")
    interesting_segments = {"SC_ACTIVE", "POST_SC", "PRE_SC", "VSC_ACTIVE"}
    display_laps = lap_metrics[lap_metrics["segment"].isin(interesting_segments)].copy()
    for column in [
        "actual_next_lap_time_mean",
        "baseline_mae",
        "sc_mae",
        "improvement",
        "improvement_pct",
        "sc_duration_max",
        "sc_fraction_max",
        "vsc_duration_max",
    ]:
        display_laps[column] = display_laps[column].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else ""
        )
    print(display_laps.to_string(index=False))

    print("\nTARGET-LAP ALIGNED SEGMENT SUMMARY")
    print("Each prediction is grouped by the observed conditions on the target lap (lap_number + 1).")
    print("Target-lap condition is used only for error analysis, not as a model input.")
    display_target = target_segment_summary.copy()
    for column in ["actual_mean", "baseline_bias", "sc_bias", "baseline_mae", "sc_mae", "improvement", "improvement_pct"]:
        display_target[column] = display_target[column].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else ""
        )
    print(display_target.to_string(index=False))

    print("\nTarget-lap details around SC (target laps 13-24):")
    display_target_laps = target_lap_detail[
        target_lap_detail["target_lap_number"].between(13, 24)
    ].copy()
    for column in ["actual_mean", "baseline_prediction_mean", "sc_prediction_mean", "baseline_bias", "sc_bias", "baseline_mae", "sc_mae", "improvement", "improvement_pct"]:
        display_target_laps[column] = display_target_laps[column].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else ""
        )
    print(display_target_laps.to_string(index=False))

    print("\nSaved analysis files:")
    print(f"  Input-lap segment summary: {segment_path}")
    print(f"  Input-lap details: {lap_path}")
    print(f"  Row-level predictions/errors: {row_path}")
    print(f"  Target-lap segment summary: {target_segment_path}")
    print(f"  Target-lap details: {target_lap_path}")
    print("\nInterpretation: positive 'improvement' means adding SC features lowered MAE;")
    print("negative 'improvement' means the SC model was less accurate.")
    print("The original 'segment' column is the input/current lap; 'target_segment' is the target lap.")


if __name__ == "__main__":
    main()
