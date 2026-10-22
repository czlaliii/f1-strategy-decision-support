"""Deterministic, lap-by-lap simulation from a known race-state snapshot."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

ALLOWED_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
MODEL_FEATURES = [
    "driver_number", "lap_number", "compound", "tyre_age", "position",
    "gap_to_leader", "laps_down", "interval", "previous_lap_time",
    "rolling_3_lap_avg", "rolling_5_lap_avg", "lap_time_delta",
    "position_change", "gap_change", "interval_change", "laps_since_pit",
    "track_air_temp_difference", "air_temperature", "track_temperature",
    "rainfall", "humidity", "wind_speed", "vsc_duration_in_lap",
    "vsc_in_lap", "vsc_lap_fraction", "vsc_active_at_end", "position_missing",
    "interval_missing", "interval_change_missing", "has_previous_pit",
    "gap_to_leader_missing", "gap_change_missing", "previous_lap_time_missing",
    "rolling_3_lap_avg_missing", "rolling_5_lap_avg_missing",
    "lap_time_delta_missing", "position_change_missing",
]

class Predictor(Protocol):
    def predict(self, features: pd.DataFrame): ...

@dataclass(frozen=True)
class Strategy:
    """Compounds describe consecutive stints, including the current stint."""

    pit_laps: tuple[int, ...]
    compounds: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "pit_laps", tuple(self.pit_laps))
        object.__setattr__(
            self, "compounds", tuple(compound.upper() for compound in self.compounds)
        )
        if len(self.compounds) != len(self.pit_laps) + 1:
            raise ValueError("Strategy compounds must contain one compound per stint.")
        if not set(self.compounds).issubset(ALLOWED_COMPOUNDS):
            raise ValueError(f"Allowed compounds: {sorted(ALLOWED_COMPOUNDS)}.")
        if tuple(sorted(set(self.pit_laps))) != self.pit_laps:
            raise ValueError("Pit laps must be unique and strictly increasing.")

class RaceSimulator:
    """Simulate future laps without accessing future historical race-state rows."""

    def __init__(
        self,
        predictor: Predictor,
        current_state: pd.Series | dict,
        race_last_lap: int,
        pit_loss_seconds: float = 22.0,
        starting_cumulative_time: float = 0.0,
    ):
        self.predictor = predictor
        self.current_state = dict(current_state)
        self.race_last_lap = int(race_last_lap)
        self.pit_loss_seconds = float(pit_loss_seconds)
        self.starting_cumulative_time = float(starting_cumulative_time)
        self._validate_state()

    def _validate_state(self):
        missing = [feature for feature in MODEL_FEATURES if feature not in self.current_state]
        if missing:
            raise ValueError(f"Current race state is missing model features: {missing}")
        if int(self.current_state["lap_number"]) >= self.race_last_lap:
            raise ValueError("Current lap must be before the final race lap.")
        compound = str(self.current_state["compound"]).upper()
        if compound not in ALLOWED_COMPOUNDS:
            raise ValueError(f"Current compound must be one of {sorted(ALLOWED_COMPOUNDS)}.")
        if float(self.current_state["tyre_age"]) < 0:
            raise ValueError("Tyre age cannot be negative.")

    def _validate_strategy(self, strategy: Strategy):
        current_lap = int(self.current_state["lap_number"])
        if strategy.compounds[0] != str(self.current_state["compound"]).upper():
            raise ValueError("The first strategy compound must match the current compound.")
        if any(lap <= current_lap or lap >= self.race_last_lap for lap in strategy.pit_laps):
            raise ValueError("Pit laps must be after the current lap and before the final lap.")

    @staticmethod
    def _numeric(value, default=0.0):
        return default if pd.isna(value) else value

    def simulate(self, strategy: Strategy) -> pd.DataFrame:
        return self.simulate_many([strategy])[0]

    def simulate_many(self, strategies: Sequence[Strategy]) -> list[pd.DataFrame]:
        """Simulate strategies in lockstep with one model call per lap.

        Every strategy starts from its own copy of the current state, so the
        results match separate simulate() calls.
        """
        for strategy in strategies:
            self._validate_strategy(strategy)
        initial = {key: self._numeric(value) for key, value in self.current_state.items()}
        initial["compound"] = str(initial["compound"]).upper()
        states = [dict(initial) for _ in strategies]
        cumulative_times = [self.starting_cumulative_time] * len(strategies)
        records = [[] for _ in strategies]
        pit_plans = [
            dict(zip(strategy.pit_laps, strategy.compounds[1:]))
            for strategy in strategies
        ]

        # Each state is the information available at the end of its lap. It predicts
        # the next lap, then is updated only from the prediction and strategy.
        while states and int(states[0]["lap_number"]) < self.race_last_lap:
            simulated_lap = int(states[0]["lap_number"]) + 1
            features = pd.DataFrame(
                [{feature: state[feature] for feature in MODEL_FEATURES} for state in states]
            )
            predictions = np.asarray(self.predictor.predict(features), dtype=float)
            if len(predictions) != len(states):
                raise ValueError("The predictor must return one value per feature row.")

            for index, (state, pit_to_compound) in enumerate(zip(states, pit_plans)):
                predicted_lap_time = float(predictions[index])
                pit_stop = simulated_lap in pit_to_compound
                pit_loss = self.pit_loss_seconds if pit_stop else 0.0
                cumulative_times[index] += predicted_lap_time + pit_loss

                compound_after_lap = state["compound"]
                tyre_age_after_lap = float(state["tyre_age"]) + 1
                if pit_stop:
                    compound_after_lap = pit_to_compound[simulated_lap]
                    tyre_age_after_lap = 0.0

                records[index].append(
                    {
                        "lap_number": simulated_lap,
                        "driver_number": state["driver_number"],
                        "position": state["position"],
                        "compound": state["compound"],
                        "tyre_age": state["tyre_age"],
                        "predicted_lap_time": predicted_lap_time,
                        "pit_stop": pit_stop,
                        "pit_loss_seconds": pit_loss,
                        "compound_after_lap": compound_after_lap,
                        "tyre_age_after_lap": tyre_age_after_lap,
                        "cumulative_time": cumulative_times[index],
                    }
                )

                previous_lap_time = state["previous_lap_time"]
                state["lap_number"] = simulated_lap
                state["compound"] = compound_after_lap
                state["tyre_age"] = tyre_age_after_lap
                state["previous_lap_time"] = predicted_lap_time
                state["lap_time_delta"] = predicted_lap_time - previous_lap_time
                state["rolling_3_lap_avg"] = (2 * state["rolling_3_lap_avg"] + predicted_lap_time) / 3
                state["rolling_5_lap_avg"] = (4 * state["rolling_5_lap_avg"] + predicted_lap_time) / 5
                state["position_change"] = 0.0
                state["gap_change"] = 0.0
                state["interval_change"] = 0.0
                state["laps_since_pit"] = 0.0 if pit_stop else state["laps_since_pit"] + 1
                state["has_previous_pit"] = 1 if pit_stop else state["has_previous_pit"]

        return [pd.DataFrame(strategy_records) for strategy_records in records]
