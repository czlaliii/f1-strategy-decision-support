import numpy as np
import pandas as pd

from src.models.expanding_window_model import STRATEGY_FEATURES, estimate_tyre_and_fuel_effects
from src.simulator.race_simulator import RaceSimulator, Strategy

TRUE_EFFECTS = {
    "offset_SOFT": -0.4,
    "offset_HARD": 0.3,
    "degradation_SOFT": 0.08,
    "degradation_MEDIUM": 0.04,
    "degradation_HARD": 0.02,
    "fuel_per_lap": 0.05,
}


def synthetic_rows(seed=0):
    """Two-stint races where faster drivers prefer SOFT, which biases raw means."""
    rng = np.random.default_rng(seed)
    rows = []
    for race in range(6):
        race_pace = 75 + 10 * race
        for driver in range(20):
            driver_pace = rng.normal(0, 1.0)
            first, second = ("SOFT", "HARD") if driver_pace < 0 else ("MEDIUM", "HARD")
            if driver % 3 == 0:
                first, second = second, "MEDIUM"
            for lap in range(2, 60):
                compound, age = (first, lap - 2) if lap < 30 else (second, lap - 30)
                age_term = TRUE_EFFECTS[f"degradation_{compound}"] * (age + 1)
                offset = TRUE_EFFECTS.get(f"offset_{compound}", 0.0)
                laps_remaining = 60 - lap
                rows.append({
                    "race": f"r{race}",
                    "driver_number": driver,
                    "compound": compound,
                    "tyre_age": age,
                    "laps_remaining": laps_remaining,
                    "next_lap_time": race_pace + driver_pace + offset + age_term
                    + TRUE_EFFECTS["fuel_per_lap"] * laps_remaining + rng.normal(0, 0.1),
                })
    return pd.DataFrame(rows)


def test_fixed_effects_recover_known_tyre_and_fuel_effects():
    effects = estimate_tyre_and_fuel_effects(synthetic_rows())
    for name, expected in TRUE_EFFECTS.items():
        assert abs(effects[name] - expected) < 0.02, name


class ReferencePredictor:
    """Predicts the field reference plus 1 s; mimics a model with a fuel effect."""

    feature_names_in_ = np.asarray(STRATEGY_FEATURES, dtype=object)
    fuel_effect_per_lap = 0.05

    def predict(self, features):
        return features["field_pace_ref"].to_numpy() + 1.0


def test_simulator_lowers_field_reference_by_fuel_effect():
    state = {feature: 0.0 for feature in STRATEGY_FEATURES}
    state.update({"compound": "MEDIUM", "tyre_age": 5, "field_pace_ref": 80.0, "lap_number": 10})
    result = RaceSimulator(ReferencePredictor(), state, race_last_lap=14).simulate(
        Strategy((), ("MEDIUM",))
    )
    assert np.allclose(result["predicted_lap_time"], [81.0, 80.95, 80.9, 80.85])
