import pandas as pd
import pytest

from src.simulator.race_simulator import MODEL_FEATURES, Strategy
from src.simulator.strategy_comparator import StrategyComparator


class RecordingPredictor:
    def __init__(self):
        self.first_feature_rows = []

    def predict(self, features):
        for _, row in features.iterrows():
            if row["lap_number"] == 50:
                self.first_feature_rows.append(row.copy())
        return [80.0] * len(features)


def initial_state():
    state = {feature: 0.0 for feature in MODEL_FEATURES}
    state.update(
        {
            "driver_number": 14,
            "lap_number": 50,
            "compound": "MEDIUM",
            "tyre_age": 15,
            "position": 14,
            "previous_lap_time": 85.0,
            "rolling_3_lap_avg": 85.0,
            "rolling_5_lap_avg": 85.0,
            "laps_since_pit": 15,
        }
    )
    return pd.Series(state)


def comparator(predictor=None):
    return StrategyComparator(predictor or RecordingPredictor(), race_last_lap=70, pit_loss_seconds=20)


def strategies():
    return [
        Strategy((52,), ("MEDIUM", "HARD")),
        Strategy((55,), ("MEDIUM", "SOFT")),
    ]


def test_multiple_strategies_run_from_same_state_independently():
    predictor = RecordingPredictor()
    comparator(predictor).compare(initial_state(), strategies(), starting_cumulative_time=100)

    assert len(predictor.first_feature_rows) == 2
    assert predictor.first_feature_rows[0].equals(predictor.first_feature_rows[1])


def test_comparator_returns_one_result_per_strategy_with_time_columns():
    result = comparator().compare(initial_state(), strategies(), 100)

    assert len(result) == 2
    assert {"predicted_remaining_race_time", "predicted_total_race_time"}.issubset(result.columns)
    assert result["strategy_id"].tolist() == ["S1", "S2"]


def test_two_stop_strategy_structure_is_supported():
    strategy = Strategy((50, 62), ("MEDIUM", "HARD", "MEDIUM"))

    assert strategy.pit_laps == (50, 62)
    assert strategy.compounds == ("MEDIUM", "HARD", "MEDIUM")


def test_invalid_strategy_or_empty_comparison_raises_error():
    with pytest.raises(ValueError, match="one compound per stint"):
        Strategy((55,), ("MEDIUM",))
    with pytest.raises(ValueError, match="At least one"):
        comparator().compare(initial_state(), [])


def test_invalid_pit_lap_for_current_state_raises_error():
    with pytest.raises(ValueError, match="after the current lap"):
        comparator().compare(
            initial_state(),
            [Strategy((50, 62), ("MEDIUM", "HARD", "MEDIUM"))],
        )
