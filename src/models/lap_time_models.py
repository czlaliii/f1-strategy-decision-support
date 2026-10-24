"""Next-lap time models for comparison: XGBoost (tabular) and LSTM (lap sequences).

Both predict next_lap_time - field_pace_ref (the field's recent green-lap
pace), which removes track-to-track scale; the reference is added back.
Features describe the state at the end of the current lap only, so nothing
from later laps is used. Models are trained on earlier races only (see
compare_lap_time_models.py); early stopping uses the chronologically last
races of the training window.
"""
import time

import numpy as np
import pandas as pd

from src.features.race_dataset import RaceDataError, load_race_dataset
from src.models.expanding_window_model import REFERENCE, TARGET, racing_target_rows, races_before

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
# Numeric state features; *_rel columns are relative to field_pace_ref.
NUMERIC_FEATURES = [
    "lap_rel",
    "previous_lap_rel",
    "rolling_3_rel",
    "rolling_5_rel",
    "lap_time_delta",
    "tyre_age",
    "laps_since_pit",
    "stint_number",
    "laps_remaining",
    "lap_number",
    "field_pace_ref",
    "driver_pace_offset",
    "position",
    "interval",
    "gap_to_leader",
    "track_temperature",
    "air_temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "is_racing_lap",
    "pit_stop",
    "vsc_in_lap",
]
FEATURES = NUMERIC_FEATURES + [f"compound_{c}" for c in COMPOUNDS]
# Share of training races (latest first) held out for early stopping.
VALIDATION_SHARE = 0.15
SEED = 42


def add_model_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Relative lap times and compound one-hot columns; all known at the end of the lap."""
    df = dataset.copy()
    reference = df[REFERENCE]
    df["lap_rel"] = df["lap_duration"] - reference
    df["previous_lap_rel"] = df["previous_lap_time"] - reference
    df["rolling_3_rel"] = df["rolling_3_lap_avg"] - reference
    df["rolling_5_rel"] = df["rolling_5_lap_avg"] - reference
    for column in ["is_racing_lap", "pit_stop"]:
        df[column] = df[column].astype(float)
    for compound in COMPOUNDS:
        df[f"compound_{compound}"] = (df["compound"].astype(str).str.upper() == compound).astype(float)
    df["target_rel"] = df[TARGET] - reference
    return df


def load_history(race_folder: str) -> tuple[pd.DataFrame, list[str]]:
    """All lap rows of races before race_folder, oldest first, with model features."""
    frames, used = [], []
    for race in races_before(race_folder).itertuples():
        try:
            frames.append(add_model_features(load_race_dataset(race.race, race.session_key).assign(race=race.race)))
            used.append(race.race)
        except RaceDataError:
            continue
    columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    # Columns that are empty in one race would otherwise decide the combined dtype.
    frames = [frame.dropna(axis=1, how="all") for frame in frames]
    return pd.concat(frames, ignore_index=True).reindex(columns=columns), used


def split_train_validation(history: pd.DataFrame, races: list[str]):
    """Chronological split: the latest VALIDATION_SHARE of races validate."""
    count = max(1, int(round(len(races) * VALIDATION_SHARE)))
    validation_races = set(races[-count:])
    is_validation = history["race"].isin(validation_races)
    return history[~is_validation], history[is_validation]


# --- XGBoost -------------------------------------------------------------

def fit_xgboost(history: pd.DataFrame, races: list[str]):
    from xgboost import XGBRegressor

    train, validation = split_train_validation(history, races)
    train, validation = racing_target_rows(train), racing_target_rows(validation)
    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        early_stopping_rounds=100,
        random_state=SEED,
        n_jobs=-1,
    )
    started = time.perf_counter()
    model.fit(
        train[FEATURES], train["target_rel"],
        eval_set=[(validation[FEATURES], validation["target_rel"])],
        verbose=False,
    )
    model.training_seconds_ = time.perf_counter() - started
    return model


def predict_xgboost(model, rows: pd.DataFrame) -> np.ndarray:
    return rows[REFERENCE].to_numpy(dtype=float) + model.predict(rows[FEATURES])


# --- LSTM ----------------------------------------------------------------

SEQUENCE_LENGTH = 10


def build_sequences(dataset: pd.DataFrame, target_index, medians, means, stds):
    """For each target row, the driver's last SEQUENCE_LENGTH laps up to and including it.

    Shorter histories are left-padded with zeros; the last channel flags real laps.
    Features are filled with training medians and standardised with training statistics.
    """
    values = dataset[FEATURES].astype(float).fillna(medians)
    values = ((values - means) / stds).to_numpy(dtype=np.float32)
    position_in_df = {index: i for i, index in enumerate(dataset.index)}
    order = dataset.sort_values(["race", "driver_number", "lap_number"]).index
    previous_rows = {}
    run = []
    last_key = None
    for index in order:
        key = (dataset.at[index, "race"], dataset.at[index, "driver_number"])
        if key != last_key:
            run, last_key = [], key
        run.append(position_in_df[index])
        previous_rows[index] = run[-SEQUENCE_LENGTH:]
    sequences = np.zeros((len(target_index), SEQUENCE_LENGTH, len(FEATURES) + 1), dtype=np.float32)
    for row, index in enumerate(target_index):
        history = previous_rows[index]
        sequences[row, SEQUENCE_LENGTH - len(history):, :-1] = values[history]
        sequences[row, SEQUENCE_LENGTH - len(history):, -1] = 1.0
    return sequences


class LapSequenceModel:
    """LSTM over the last laps, then a small dense head predicting next lap vs reference."""

    def __init__(self, hidden_size=64, epochs=40, patience=5, batch_size=256, learning_rate=1e-3):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def _network(self, inputs):
        import torch
        from torch import nn

        class Network(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.lstm = nn.LSTM(inputs, hidden, batch_first=True)
                self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

            def forward(self, x):
                output, _ = self.lstm(x)
                return self.head(output[:, -1, :]).squeeze(-1)

        torch.manual_seed(SEED)
        return Network(self.hidden_size)

    def fit(self, history: pd.DataFrame, races: list[str]):
        import torch

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        train, validation = split_train_validation(history, races)
        train_rows, validation_rows = racing_target_rows(train), racing_target_rows(validation)
        numeric = train[FEATURES].astype(float)
        self.medians_ = numeric.median()
        filled = numeric.fillna(self.medians_)
        self.means_ = filled.mean()
        self.stds_ = filled.std().replace(0, 1.0)

        def tensors(frame, rows):
            x = build_sequences(frame, rows.index, self.medians_, self.means_, self.stds_)
            return torch.from_numpy(x), torch.from_numpy(rows["target_rel"].to_numpy(dtype=np.float32))

        x_train, y_train = tensors(train, train_rows)
        x_validation, y_validation = tensors(validation, validation_rows)
        self.network_ = self._network(x_train.shape[-1])
        optimiser = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate)
        loss_function = torch.nn.HuberLoss(delta=1.0)
        best, best_state, waited = np.inf, None, 0
        started = time.perf_counter()
        generator = torch.Generator().manual_seed(SEED)
        for _ in range(self.epochs):
            self.network_.train()
            for batch in torch.randperm(len(x_train), generator=generator).split(self.batch_size):
                optimiser.zero_grad()
                loss = loss_function(self.network_(x_train[batch]), y_train[batch])
                loss.backward()
                optimiser.step()
            self.network_.eval()
            with torch.no_grad():
                validation_mae = float((self.network_(x_validation) - y_validation).abs().mean())
            if validation_mae < best - 1e-4:
                best, waited = validation_mae, 0
                best_state = {k: v.clone() for k, v in self.network_.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self.network_.load_state_dict(best_state)
        self.training_seconds_ = time.perf_counter() - started
        return self

    def predict(self, dataset: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
        """Predict lap times for `rows`, using the full race dataset for their lap histories."""
        import torch

        x = torch.from_numpy(build_sequences(dataset, rows.index, self.medians_, self.means_, self.stds_))
        self.network_.eval()
        with torch.no_grad():
            relative = self.network_(x).numpy()
        return rows[REFERENCE].to_numpy(dtype=float) + relative


# --- XGBoost-anchored strategy model --------------------------------------

# Half-life (laps) of the XGBoost correction in simulations. Chosen on real
# stint rollouts (see evaluate_anchoring.py): 19/23 test races improved over
# the structured model alone (Wilcoxon p = 0.016).
ANCHOR_HALF_LIFE_LAPS = 10


class AnchoredLapTimeModel:
    """Structured model shifted by the XGBoost-vs-structured gap at the start state.

    XGBoost is the more accurate next-lap estimator, but it barely reacts to
    compound and tyre age, so it cannot rank tyre strategies. The structured
    model provides the tyre and fuel effects; the XGBoost correction sets the
    current pace level and fades with a half-life of half_life laps. The
    correction is the same for every strategy on a given lap, so it improves
    lap-time accuracy without changing the strategy ranking.
    """

    def __init__(self, structured, offset, start_lap, half_life=ANCHOR_HALF_LIFE_LAPS):
        self.structured = structured
        self.offset = float(offset)
        self.start_lap = int(start_lap)
        self.half_life = half_life
        self.feature_names_in_ = np.asarray(list(structured.feature_names_in_) + ["lap_number"], dtype=object)
        self.fuel_effect_per_lap = structured.fuel_effect_per_lap
        self.effects_ = structured.effects_
        self.pipeline = structured.pipeline

    def correction(self, lap_numbers):
        steps = np.asarray(lap_numbers, dtype=float) - self.start_lap
        decay = 1.0 if self.half_life is None else 0.5 ** (steps / self.half_life)
        return self.offset * decay

    def predict(self, X):
        base = self.structured.predict(X[list(self.structured.feature_names_in_)])
        return base + self.correction(X["lap_number"])


def anchored_model(structured, xgboost_model, feature_row: pd.Series, state: pd.Series):
    """AnchoredLapTimeModel for one state.

    feature_row: the lap's row from add_model_features (XGBoost input);
    state: the structured model's state at the same lap.
    """
    xgboost_frame = pd.DataFrame([feature_row[FEATURES]]).astype(float)
    columns = list(structured.feature_names_in_)
    structured_frame = pd.DataFrame([state[columns]])
    structured_frame = structured_frame.astype({c: float for c in columns if c != "compound"}).astype({"compound": str})
    offset = predict_xgboost(xgboost_model, xgboost_frame)[0] - structured.predict(structured_frame)[0]
    if not np.isfinite(offset):
        offset = 0.0
    return AnchoredLapTimeModel(structured, offset, feature_row["lap_number"])
