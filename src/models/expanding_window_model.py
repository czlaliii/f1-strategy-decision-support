"""Train the next-lap model only on races run before a selected race.

The strategy model is a structured hybrid:
    next_lap_time = field_pace_ref + tyre effect + context residual
- Tyre effect: compound offset and per-compound degradation, estimated with a
  driver-race fixed-effects regression, so car, driver and track differences
  do not bias it.
- Context residual: gradient boosting on non-tyre features only, so it cannot
  absorb the tyre effect.
- Fuel: the same regression estimates the per-lap fuel effect; the simulator
  lowers field_pace_ref by it each lap.
- An XGBoost next-lap model is trained on the same races; the strategy
  worker uses it to anchor the simulated pace (lap_time_models.AnchoredLapTimeModel).
It does not use the previous lap time, so a pit stop changes the prediction
through the compound and tyre age instead of being masked by old-tyre pace.

Run from the project root to pretrain models:
    python -m src.models.expanding_window_model --race hungarian_2025
    python -m src.models.expanding_window_model --all
"""
import argparse
import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.race_dataset import (
    DATASET_VERSION,
    DRY_COMPOUNDS,
    PROCESSED_ROOT,
    RaceDataError,
    list_local_races,
    load_race_dataset,
)
from src.simulator.race_simulator import MODEL_FEATURES

MODEL_DIR = PROCESSED_ROOT / "expanding_window_models"
MODEL_VERSION = 5
TARGET = "next_lap_time"
REFERENCE = "field_pace_ref"
# Laps over 200 s likely include stoppages or restarts.
MAX_VALID_LAP_TIME = 200.0
# Target laps slower than this share of the race's median racing lap are
# treated as SC/VSC or incident laps.
MAX_RELATIVE_LAP_TIME = 1.07
MIN_TRAINING_RACES = 3
# Percentile of observed stint lengths used as a compound's typical tyre life.
TYRE_LIFE_PERCENTILE = 0.9
CATEGORICAL_FEATURES = ["compound"]

TYRE_FEATURES = ["compound", "tyre_age"]
CONTEXT_FEATURES = [
    "field_pace_ref",
    "driver_pace_offset",
    "position",
    "interval",
    "track_temperature",
    "air_temperature",
    "humidity",
    "wind_speed",
    "rainfall",
]
STRATEGY_FEATURES = TYRE_FEATURES + CONTEXT_FEATURES


class InsufficientHistoryError(ValueError):
    """Raised when too few earlier races exist to train a model."""


@dataclass(frozen=True)
class TrainedModel:
    pipeline: BaseEstimator
    training_races: tuple[str, ...]
    training_rows: int
    # Typical tyre life in laps per dry compound, from the training races.
    tyre_life_laps: dict
    # Next-lap XGBoost model on the same races; anchors the simulation pace.
    xgboost: object = None


def estimate_tyre_and_fuel_effects(rows: pd.DataFrame) -> dict[str, float]:
    """Driver-race fixed-effects least squares on racing target laps.

    Demeaning within each driver-race removes car, driver, track and day
    effects; compounds are compared via drivers who used several of them.
    MEDIUM at tyre age 0 is the reference level.
    """
    age = rows["tyre_age"].to_numpy(dtype=float) + 1
    compound = rows["compound"].astype(str)
    X = pd.DataFrame({
        "offset_SOFT": (compound == "SOFT").astype(float),
        "offset_HARD": (compound == "HARD").astype(float),
        **{f"degradation_{c}": age * (compound == c) for c in DRY_COMPOUNDS},
        "fuel_per_lap": rows["laps_remaining"].astype(float),
    }, index=rows.index)
    y = rows[TARGET].astype(float)
    group = rows["race"].astype(str) + "_" + rows["driver_number"].astype(str)
    X_within = X - X.groupby(group).transform("mean")
    y_within = y - y.groupby(group).transform("mean")
    coefficients, *_ = np.linalg.lstsq(X_within.to_numpy(), y_within.to_numpy(), rcond=None)
    effects = dict(zip(X.columns, coefficients.astype(float)))
    # Physical floor: tyres do not get faster with age.
    for c in DRY_COMPOUNDS:
        effects[f"degradation_{c}"] = max(effects[f"degradation_{c}"], 0.0)
    return effects


class StructuredLapTimeModel(RegressorMixin, BaseEstimator):
    """field_pace_ref + fixed-effects tyre model + boosted context residual."""

    def fit(self, X, y, rows):
        self.effects_ = estimate_tyre_and_fuel_effects(rows)
        self.fuel_effect_per_lap = self.effects_["fuel_per_lap"]
        residual = (
            np.asarray(y, dtype=float)
            - X["field_pace_ref"].to_numpy(dtype=float)
            - self.tyre_effect(X)
        )
        self.pipeline = build_pipeline(CONTEXT_FEATURES, categorical=[]).fit(
            X[CONTEXT_FEATURES], residual
        )
        self.feature_names_in_ = np.asarray(STRATEGY_FEATURES, dtype=object)
        return self

    def tyre_effect(self, X) -> np.ndarray:
        compound = X["compound"].astype(str).to_numpy()
        age = X["tyre_age"].to_numpy(dtype=float) + 1
        effect = np.zeros(len(X))
        for c in DRY_COMPOUNDS:
            mask = compound == c
            effect[mask] = (
                self.effects_.get(f"offset_{c}", 0.0)
                + self.effects_[f"degradation_{c}"] * age[mask]
            )
        return effect

    def predict(self, X):
        return (
            X["field_pace_ref"].to_numpy(dtype=float)
            + self.tyre_effect(X)
            + self.pipeline.predict(X[CONTEXT_FEATURES])
        )


def build_pipeline(features=MODEL_FEATURES, categorical=CATEGORICAL_FEATURES) -> Pipeline:
    """Same preprocessing and hyperparameters as train_gradient_boosting.py."""
    numerical_features = [f for f in features if f not in categorical]
    transformers = [
        ("numerical", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numerical_features),
    ]
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])


def training_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    """Baseline filter: valid target and current/next lap times under 200 s."""
    lap = pd.to_numeric(dataset["lap_duration"], errors="coerce")
    target = pd.to_numeric(dataset[TARGET], errors="coerce")
    valid = lap.between(0, MAX_VALID_LAP_TIME, inclusive="neither") & target.between(
        0, MAX_VALID_LAP_TIME, inclusive="neither"
    )
    return dataset.loc[valid]


def racing_target_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    """Rows whose target is a green-flag racing lap on a dry compound.

    Uses the whole race only to select labelled examples, never as a feature.
    """
    target = pd.to_numeric(dataset[TARGET], errors="coerce")
    racing_times = pd.to_numeric(dataset["lap_duration"], errors="coerce").where(
        dataset["is_racing_lap"].astype(bool)
    )
    race_median = racing_times.groupby(dataset["race"]).transform("median")
    valid = (
        dataset["target_is_racing_lap"].astype(bool)
        & dataset["compound"].isin(DRY_COMPOUNDS)
        & dataset["tyre_age"].notna()
        & dataset[REFERENCE].notna()
        & target.gt(0)
        & target.le(MAX_RELATIVE_LAP_TIME * race_median)
    )
    return dataset.loc[valid]


def _model_frame(rows: pd.DataFrame, features) -> pd.DataFrame:
    columns = list(features) + [TARGET]
    frame = rows[columns].astype({
        column: float for column in columns if column not in CATEGORICAL_FEATURES
    })
    return frame.astype({"compound": str})


def fit_strategy_model(train: pd.DataFrame) -> StructuredLapTimeModel:
    rows = racing_target_rows(train)
    frame = _model_frame(rows, STRATEGY_FEATURES)
    return StructuredLapTimeModel().fit(frame[STRATEGY_FEATURES], frame[TARGET], rows)


def fit_baseline_model(train: pd.DataFrame) -> Pipeline:
    """The previous approach: absolute target, all laps under 200 s."""
    frame = _model_frame(training_rows(train), MODEL_FEATURES)
    return build_pipeline(MODEL_FEATURES).fit(frame[MODEL_FEATURES], frame[TARGET])


def races_before(race_folder: str) -> pd.DataFrame:
    """Local races whose race date is strictly earlier than the selected race."""
    races = list_local_races()
    selected = races.loc[races["race"] == race_folder]
    if selected.empty:
        raise RaceDataError(f"Unknown local race folder: {race_folder}")
    return races.loc[races["race_date"] < selected["race_date"].iloc[0]]


def load_training_data(race_folder: str) -> tuple[pd.DataFrame, list[str]]:
    """Concatenate datasets of all usable races before the selected race."""
    frames, used = [], []
    for race in races_before(race_folder).itertuples():
        try:
            frames.append(load_race_dataset(race.race, race.session_key))
            used.append(race.race)
        except RaceDataError:
            continue
    if len(used) < MIN_TRAINING_RACES:
        raise InsufficientHistoryError(
            f"Only {len(used)} earlier race(s) available before {race_folder}; "
            f"at least {MIN_TRAINING_RACES} are needed."
        )
    columns = sorted(set(STRATEGY_FEATURES) | set(MODEL_FEATURES) | {
        TARGET, "lap_duration", "is_racing_lap", "target_is_racing_lap", "race",
        "laps_remaining", "stint_number",
    })
    frames = [frame[columns] for frame in frames]
    return pd.concat(frames, ignore_index=True), used


def typical_tyre_life(train: pd.DataFrame) -> dict[str, float]:
    """TYRE_LIFE_PERCENTILE of the oldest tyre age reached per stint, by compound.

    Final stints end at the flag rather than at the tyre's limit, so a high
    percentile is used instead of the mean.
    """
    stints = (
        train.dropna(subset=["stint_number", "tyre_age"])
        .groupby(["race", "driver_number", "stint_number"])
        .agg(compound=("compound", "first"), life=("tyre_age", "max"))
    )
    stints = stints[stints["compound"].isin(DRY_COMPOUNDS)]
    life = stints.groupby("compound")["life"].quantile(TYRE_LIFE_PERCENTILE) + 1
    return {compound: float(laps) for compound, laps in life.items()}


def train_model(race_folder: str) -> TrainedModel:
    """Fit the strategy model and the XGBoost next-lap model on races before the selected race."""
    # Imported here: lap_time_models imports this module.
    from src.models.lap_time_models import fit_xgboost, load_history

    train, used = load_training_data(race_folder)
    model = fit_strategy_model(train)
    history, history_races = load_history(race_folder)
    return TrainedModel(
        model, tuple(used), int(len(racing_target_rows(train))), typical_tyre_life(train),
        fit_xgboost(history, history_races),
    )


def load_or_train_model(race_folder: str, retrain: bool = False) -> TrainedModel:
    """Load the cached models for this race, training them on first use."""
    import xgboost

    model_path = MODEL_DIR / f"{race_folder}.joblib"
    xgboost_path = MODEL_DIR / f"{race_folder}.xgb.joblib"
    meta_path = MODEL_DIR / f"{race_folder}.json"
    if model_path.exists() and xgboost_path.exists() and meta_path.exists() and not retrain:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("model_version") == MODEL_VERSION
            and meta.get("dataset_version") == DATASET_VERSION
            and meta.get("sklearn_version") == sklearn.__version__
            and meta.get("xgboost_version") == xgboost.__version__
        ):
            return TrainedModel(
                joblib.load(model_path),
                tuple(meta["training_races"]),
                int(meta["training_rows"]),
                meta["tyre_life_laps"],
                joblib.load(xgboost_path),
            )

    model = train_model(race_folder)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipeline, model_path)
    joblib.dump(model.xgboost, xgboost_path)
    meta_path.write_text(json.dumps({
        "race": race_folder,
        "training_races": list(model.training_races),
        "training_rows": model.training_rows,
        "tyre_life_laps": model.tyre_life_laps,
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "features": STRATEGY_FEATURES,
    }, indent=2), encoding="utf-8")
    return model


def fill_with_training_medians(model, state: pd.Series) -> pd.Series:
    """Fill missing numeric features with the medians learned in training.

    The simulator would otherwise turn NaN into 0, which differs from how
    the model saw missing values during training.
    """
    pipeline = model.pipeline if isinstance(model, StructuredLapTimeModel) else model
    preprocessor = pipeline.named_steps["preprocessor"]
    numerical = preprocessor.named_transformers_["numerical"]
    numerical_features = preprocessor.transformers_[0][2]
    state = state.copy()
    for feature, median in zip(numerical_features, numerical.named_steps["imputer"].statistics_):
        if pd.isna(state.get(feature)):
            state[feature] = median
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--race", help="Local race folder name, e.g. hungarian_2025")
    group.add_argument("--all", action="store_true", help="Train a model for every local race")
    parser.add_argument("--retrain", action="store_true", help="Ignore cached models")
    args = parser.parse_args()

    races = [args.race] if args.race else list_local_races()["race"].tolist()
    for race in races:
        try:
            model = load_or_train_model(race, retrain=args.retrain)
            print(f"{race}: {len(model.training_races)} races, {model.training_rows} rows")
        except (InsufficientHistoryError, RaceDataError) as error:
            print(f"{race}: skipped ({error})")


if __name__ == "__main__":
    main()
