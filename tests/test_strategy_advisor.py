import numpy as np
import pandas as pd

from src.simulator import strategy_advisor as advisor
from src.simulator.race_simulator import RaceSimulator, Strategy

FEATURES = np.asarray(["compound", "tyre_age", "field_pace_ref"], dtype=object)


class TyreModel:
    """Lap time = reference + compound offset + degradation per lap of tyre age."""

    feature_names_in_ = FEATURES
    fuel_effect_per_lap = 0.0
    effects_ = {"degradation_SOFT": 0.1, "degradation_MEDIUM": 0.05, "degradation_HARD": 0.03}
    offsets = {"SOFT": -0.5, "MEDIUM": 0.0, "HARD": 0.3}

    def predict(self, X):
        rate = X["compound"].map(lambda c: self.effects_[f"degradation_{c}"]).to_numpy()
        offset = X["compound"].map(self.offsets).to_numpy()
        return X["field_pace_ref"].to_numpy() + offset + rate * (X["tyre_age"].to_numpy() + 1)


def state(lap=10, compound="MEDIUM", age=9):
    return pd.Series({"compound": compound, "tyre_age": age, "field_pace_ref": 80.0, "lap_number": lap})


def test_two_compound_rule_filters_candidates():
    candidates = advisor.candidate_strategies(10, 30, "MEDIUM", {"MEDIUM"})
    assert all(len(set(s.compounds)) >= 2 for s in candidates)
    assert Strategy((), ("MEDIUM",)) not in candidates
    # Once two compounds were used, finishing without a stop is allowed.
    assert Strategy((), ("HARD",)) in advisor.candidate_strategies(20, 30, "HARD", {"MEDIUM", "HARD"})


def test_rule_waived_when_wet_tyres_were_used():
    assert advisor.satisfies_compound_rule({"INTERMEDIATE", "MEDIUM"})
    assert not advisor.satisfies_compound_rule({"MEDIUM"})


def test_recommendation_and_pit_window():
    model = TyreModel()
    candidates = advisor.candidate_strategies(10, 30, "MEDIUM", {"MEDIUM"})
    results = advisor.evaluate_strategies(model, state(), candidates, 30, pit_loss_seconds=20.0)
    recommendation = advisor.recommend(results)

    assert recommendation.best["remaining_time"] == results["remaining_time"].min()
    assert recommendation.plans["delta"].iloc[0] == 0
    assert (recommendation.plans["delta"] >= 0).all()
    start, end = recommendation.pit_window
    assert start <= recommendation.best["pit_laps"][0] <= end


def test_total_times_match_per_strategy_simulation():
    model = TyreModel()
    candidates = advisor.candidate_strategies(10, 30, "MEDIUM", {"MEDIUM"})[:50]
    simulator = RaceSimulator(model, state(), race_last_lap=30, pit_loss_seconds=20.0)
    totals = simulator.total_times(candidates)
    separate = [simulator.simulate(s)["cumulative_time"].iloc[-1] for s in candidates]
    assert np.allclose(totals, separate)


def test_tyre_wear_and_compound_pace():
    assert advisor.tyre_wear_percent(14, "MEDIUM", {"MEDIUM": 30.0}) == 50.0
    assert advisor.tyre_wear_percent(80, "MEDIUM", {"MEDIUM": 30.0}) == 100.0
    assert advisor.tyre_wear_percent(5, "SOFT", {}) is None

    pace = advisor.compound_pace(TyreModel(), state(age=19))
    current = pace.iloc[0]
    assert current["set"] == "Jelenlegi szett" and current["delta_to_current"] == 0
    new_soft = pace[pace["set"] == "Új SOFT"].iloc[0]
    assert np.isclose(new_soft["lap_time"], 80.0 - 0.5 + 0.1)
