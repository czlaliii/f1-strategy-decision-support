import numpy as np
import pandas as pd

from src.models.lap_time_models import (
    FEATURES,
    SEQUENCE_LENGTH,
    AnchoredLapTimeModel,
    build_sequences,
    split_train_validation,
)
from src.simulator.race_simulator import RaceSimulator, Strategy


def laps_frame(laps=12):
    rows = []
    for driver in (1, 2):
        for lap in range(1, laps + 1):
            row = {feature: 0.0 for feature in FEATURES}
            # lap_rel encodes the lap number so the sequence content can be checked.
            row.update({"race": "r", "driver_number": driver, "lap_number": lap, "lap_rel": float(lap)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_sequences_contain_only_current_and_earlier_laps():
    df = laps_frame()
    numeric = df[FEATURES].astype(float)
    medians, means, stds = numeric.median(), pd.Series(0.0, index=FEATURES), pd.Series(1.0, index=FEATURES)
    target = df[(df["driver_number"] == 1) & (df["lap_number"].isin([3, 12]))].index
    sequences = build_sequences(df, target, medians, means, stds)
    lap_channel = FEATURES.index("lap_rel")

    early, late = sequences
    assert early.shape == (SEQUENCE_LENGTH, len(FEATURES) + 1)
    # Lap 3: laps 1-3 at the end, the rest left-padded and flagged as padding.
    assert early[-3:, lap_channel].tolist() == [1.0, 2.0, 3.0]
    assert early[:, -1].tolist() == [0.0] * (SEQUENCE_LENGTH - 3) + [1.0] * 3
    # Lap 12: the last SEQUENCE_LENGTH laps, never lap 13+ or the other driver's laps.
    assert late[:, lap_channel].tolist() == [float(lap) for lap in range(3, 13)]


def test_validation_split_is_chronological():
    history = pd.DataFrame({"race": ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3, "x": np.arange(12)})
    train, validation = split_train_validation(history, ["a", "b", "c", "d"])
    assert set(validation["race"]) == {"d"} and set(train["race"]) == {"a", "b", "c"}


class FlatModel:
    """Structured-model stand-in: 90 s plus 0.1 s per tyre lap."""

    feature_names_in_ = ["compound", "tyre_age", "field_pace_ref"]
    fuel_effect_per_lap = 0.0
    effects_ = {}
    pipeline = None

    def predict(self, X):
        return 90.0 + 0.1 * X["tyre_age"].to_numpy(dtype=float)


def test_anchor_correction_halves_every_half_life():
    model = AnchoredLapTimeModel(FlatModel(), offset=-0.8, start_lap=20, half_life=10)
    X = pd.DataFrame({"compound": "MEDIUM", "tyre_age": 0.0, "field_pace_ref": 90.0, "lap_number": [20, 30, 40]})
    assert np.allclose(model.predict(X), [89.2, 89.6, 89.8])


def test_anchor_does_not_change_strategy_ranking():
    state = pd.Series({"lap_number": 20, "compound": "MEDIUM", "tyre_age": 10.0, "field_pace_ref": 90.0})
    strategies = [Strategy((25,), ("MEDIUM", "HARD")), Strategy((35,), ("MEDIUM", "HARD"))]

    def totals(predictor):
        return RaceSimulator(predictor, state, race_last_lap=50, pit_loss_seconds=20).total_times(strategies)

    plain = totals(FlatModel())
    anchored = totals(AnchoredLapTimeModel(FlatModel(), offset=-0.8, start_lap=20))
    # Same correction on every lap for every strategy: totals shift equally.
    assert np.allclose(anchored - plain, (anchored - plain)[0])
    assert (anchored - plain)[0] < 0
