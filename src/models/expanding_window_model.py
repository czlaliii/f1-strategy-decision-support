"""Train the next-lap model only on races run before a selected race.

Run from the project root to pretrain models:
    python -m src.models.expanding_window_model --race hungarian_2025
    python -m src.models.expanding_window_model --all
"""
import argparse
import json
from dataclasses import dataclass

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.race_dataset import (
    DATASET_VERSION,
    PROCESSED_ROOT,
    RaceDataError,
    list_local_races,
    load_race_dataset,
)
from src.simulator.race_simulator import MODEL_FEATURES

MODEL_DIR = PROCESSED_ROOT / "expanding_window_models"
TARGET = "next_lap_time"
# Laps over 200 s likely include stoppages or restarts.
MAX_VALID_LAP_TIME = 200.0
MIN_TRAINING_RACES = 3
CATEGORICAL_FEATURES = ["compound"]


class InsufficientHistoryError(ValueError):
    """Raised when too few earlier races exist to train a model."""


@dataclass(frozen=True)
class TrainedModel:
    pipeline: Pipeline
    training_races: tuple[str, ...]
    training_rows: int


def build_pipeline() -> Pipeline:
    """Same preprocessing and hyperparameters as train_gradient_boosting.py."""
    numerical_features = [f for f in MODEL_FEATURES if f not in CATEGORICAL_FEATURES]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numerical_features),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL_FEATURES),
        ]
    )
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
    """Keep rows with a valid target and normal current/next lap times."""
    lap = pd.to_numeric(dataset["lap_duration"], errors="coerce")
    target = pd.to_numeric(dataset[TARGET], errors="coerce")
    valid = lap.between(0, MAX_VALID_LAP_TIME, inclusive="neither") & target.between(
        0, MAX_VALID_LAP_TIME, inclusive="neither"
    )
    return dataset.loc[valid]


def races_before(race_folder: str) -> pd.DataFrame:
    """Local races whose race date is strictly earlier than the selected race."""
    races = list_local_races()
    selected = races.loc[races["race"] == race_folder]
    if selected.empty:
        raise RaceDataError(f"Unknown local race folder: {race_folder}")
    return races.loc[races["race_date"] < selected["race_date"].iloc[0]]


def train_model(race_folder: str) -> TrainedModel:
    """Fit a model on all usable races before the selected race."""
    frames, used = [], []
    for race in races_before(race_folder).itertuples():
        try:
            rows = training_rows(load_race_dataset(race.race, race.session_key))
        except RaceDataError:
            continue
        if not rows.empty:
            rows = rows[MODEL_FEATURES + [TARGET]].astype({
                column: float
                for column in MODEL_FEATURES + [TARGET]
                if column not in CATEGORICAL_FEATURES
            })
            frames.append(rows.astype({"compound": str}))
            used.append(race.race)

    if len(used) < MIN_TRAINING_RACES:
        raise InsufficientHistoryError(
            f"Only {len(used)} earlier race(s) available before {race_folder}; "
            f"at least {MIN_TRAINING_RACES} are needed."
        )

    train = pd.concat(frames, ignore_index=True)
    pipeline = build_pipeline().fit(train[MODEL_FEATURES], train[TARGET])
    return TrainedModel(pipeline, tuple(used), len(train))


def load_or_train_model(race_folder: str, retrain: bool = False) -> TrainedModel:
    """Load the cached model for this race, training it on first use."""
    model_path = MODEL_DIR / f"{race_folder}.joblib"
    meta_path = MODEL_DIR / f"{race_folder}.json"
    if model_path.exists() and meta_path.exists() and not retrain:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("dataset_version") == DATASET_VERSION
            and meta.get("sklearn_version") == sklearn.__version__
            and meta.get("features") == MODEL_FEATURES
        ):
            return TrainedModel(
                joblib.load(model_path),
                tuple(meta["training_races"]),
                int(meta["training_rows"]),
            )

    model = train_model(race_folder)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipeline, model_path)
    meta_path.write_text(json.dumps({
        "race": race_folder,
        "training_races": list(model.training_races),
        "training_rows": model.training_rows,
        "dataset_version": DATASET_VERSION,
        "sklearn_version": sklearn.__version__,
        "features": MODEL_FEATURES,
    }, indent=2), encoding="utf-8")
    return model


def fill_with_training_medians(pipeline: Pipeline, state: pd.Series) -> pd.Series:
    """Fill missing numeric features with the medians learned in training.

    The simulator would otherwise turn NaN into 0, which differs from how
    the model saw missing values during training.
    """
    numerical = pipeline.named_steps["preprocessor"].named_transformers_["numerical"]
    medians = numerical.named_steps["imputer"].statistics_
    numerical_features = [f for f in MODEL_FEATURES if f not in CATEGORICAL_FEATURES]
    state = state.copy()
    for feature, median in zip(numerical_features, medians):
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
