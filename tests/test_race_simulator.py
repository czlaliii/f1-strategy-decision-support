import pandas as pd
import pytest

from src.simulator.race_simulator import MODEL_FEATURES, RaceSimulator, Strategy


class ConstantPredictor:
    def predict(self, features):
        assert list(features.columns) == MODEL_FEATURES
        return [80.0]


def current_state():
    state = {feature: 0.0 for feature in MODEL_FEATURES}
    state.update(
        {
            "driver_number": 44,
            "lap_number": 50,
            "compound": "MEDIUM",
            "tyre_age": 10,
            "position": 2,
            "previous_lap_time": 81.0,
            "rolling_3_lap_avg": 81.0,
            "rolling_5_lap_avg": 81.0,
            "laps_since_pit": 10,
        }
    )
    return pd.Series(state)


def simulator():
    return RaceSimulator(
        ConstantPredictor(),
        current_state(),
        race_last_lap=57,
        pit_loss_seconds=20.0,
        starting_cumulative_time=100.0,
    )


def test_pit_stop_changes_compound_and_resets_tyre_age():
    result = simulator().simulate(Strategy((55,), ("MEDIUM", "HARD")))

    pit_lap = result.loc[result["lap_number"] == 55].iloc[0]
    first_lap_after_pit = result.loc[result["lap_number"] == 56].iloc[0]
    assert pit_lap["pit_stop"]
    assert pit_lap["compound_after_lap"] == "HARD"
    assert pit_lap["tyre_age_after_lap"] == 0
    assert first_lap_after_pit["compound"] == "HARD"
    assert first_lap_after_pit["tyre_age"] == 0


def test_no_pit_keeps_compound_and_increments_tyre_age():
    result = simulator().simulate(Strategy((), ("MEDIUM",)))

    assert result["compound"].eq("MEDIUM").all()
    assert result["pit_stop"].eq(False).all()
    assert result["tyre_age"].tolist()[:3] == [10, 11, 12]


def test_cumulative_time_includes_lap_times_and_pit_loss():
    result = simulator().simulate(Strategy((55,), ("MEDIUM", "HARD")))

    assert result.iloc[0]["cumulative_time"] == 180.0
    assert result.loc[result["lap_number"] == 55, "cumulative_time"].iloc[0] == 520.0
    assert result.iloc[-1]["cumulative_time"] == 680.0


def test_simulation_stops_at_actual_final_lap():
    result = simulator().simulate(Strategy((), ("MEDIUM",)))

    assert result["lap_number"].tolist() == list(range(51, 58))


def test_invalid_strategy_raises_clear_error():
    with pytest.raises(ValueError, match="first strategy compound"):
        simulator().simulate(Strategy((55,), ("HARD", "MEDIUM")))

    with pytest.raises(ValueError, match="strictly increasing"):
        Strategy((55, 55), ("MEDIUM", "HARD", "SOFT"))
