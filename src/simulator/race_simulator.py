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
        # Fitted sklearn models expose their input columns; older ones use MODEL_FEATURES.
        self.features = list(getattr(predictor, "feature_names_in_", MODEL_FEATURES))
        # Models with a field pace reference expose the field's per-lap gain from fuel burn.
        self.fuel_effect_per_lap = float(getattr(predictor, "fuel_effect_per_lap", 0.0))
        self.current_state = dict(current_state)
        self.race_last_lap = int(race_last_lap)
        self.pit_loss_seconds = float(pit_loss_seconds)
        self.starting_cumulative_time = float(starting_cumulative_time)
        self._validate_state()

    def _validate_state(self):
        required = set(self.features) | {"lap_number", "compound", "tyre_age"}
        missing = [feature for feature in sorted(required) if feature not in self.current_state]
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
        """Simulate strategies in lockstep and return one lap table per strategy.

        Every strategy starts from its own copy of the current state, so the
        results match separate simulate() calls.
        """
        laps = self._run(strategies, keep_laps=True)
        columns = list(laps[0].keys()) if laps else []
        return [
            pd.DataFrame({column: [lap[column][index] for lap in laps] for column in columns})
            for index in range(len(strategies))
        ]

    def total_times(self, strategies: Sequence[Strategy]) -> np.ndarray:
        """Cumulative race time at the final lap for every strategy."""
        return self._run(strategies, keep_laps=False)

    def _run(self, strategies, keep_laps):
        """Vectorised core: one row per strategy, one model call per lap."""
        for strategy in strategies:
            self._validate_strategy(strategy)
        initial = {key: self._numeric(value) for key, value in self.current_state.items()}
        initial["compound"] = str(initial["compound"]).upper()
        count = len(strategies)
        states = pd.DataFrame({key: [value] * count for key, value in initial.items()})
        cumulative = np.full(count, self.starting_cumulative_time, dtype=float)

        # lap -> (strategy indices pitting on that lap, compound fitted)
        pits = {}
        for index, strategy in enumerate(strategies):
            for lap, compound in zip(strategy.pit_laps, strategy.compounds[1:]):
                pits.setdefault(lap, ([], []))
                pits[lap][0].append(index)
                pits[lap][1].append(compound)

        laps = []
        # Each state is the information available at the end of its lap. It predicts
        # the next lap, then is updated only from the prediction and strategy.
        lap = int(initial["lap_number"])
        while count and lap < self.race_last_lap:
            simulated_lap = lap + 1
            predictions = np.asarray(self.predictor.predict(states[self.features]), dtype=float)
            if len(predictions) != count:
                raise ValueError("The predictor must return one value per feature row.")

            pit_stop = np.zeros(count, dtype=bool)
            compound_after = states["compound"].to_numpy(dtype=object).copy()
            if simulated_lap in pits:
                indices, compounds = pits[simulated_lap]
                pit_stop[indices] = True
                compound_after[indices] = compounds
            pit_loss = np.where(pit_stop, self.pit_loss_seconds, 0.0)
            cumulative = cumulative + predictions + pit_loss
            tyre_age_after = np.where(pit_stop, 0.0, states["tyre_age"].to_numpy(dtype=float) + 1)

            if keep_laps:
                laps.append({
                    "lap_number": [simulated_lap] * count,
                    "driver_number": states["driver_number"].tolist() if "driver_number" in states else [None] * count,
                    "position": states["position"].tolist() if "position" in states else [None] * count,
                    "compound": states["compound"].tolist(),
                    "tyre_age": states["tyre_age"].tolist(),
                    "predicted_lap_time": predictions.tolist(),
                    "pit_stop": pit_stop.tolist(),
                    "pit_loss_seconds": pit_loss.tolist(),
                    "compound_after_lap": compound_after.tolist(),
                    "tyre_age_after_lap": tyre_age_after.tolist(),
                    "cumulative_time": cumulative.tolist(),
                })

            self._advance_states(states, simulated_lap, predictions, pit_stop, compound_after, tyre_age_after)
            lap = simulated_lap

        return laps if keep_laps else cumulative

    def _advance_states(self, states, lap, lap_times, pit_stop, compounds, tyre_ages):
        """Move all states to the end of `lap`; history features are updated only if present."""
        states["lap_number"] = lap
        states["compound"] = compounds
        states["tyre_age"] = tyre_ages
        if "field_pace_ref" in states:
            states["field_pace_ref"] = states["field_pace_ref"] - self.fuel_effect_per_lap
        if "laps_remaining" in states:
            states["laps_remaining"] = states["laps_remaining"] - 1
        if "previous_lap_time" in states:
            previous_lap_time = states["previous_lap_time"].to_numpy(dtype=float)
            states["previous_lap_time"] = lap_times
            if "lap_time_delta" in states:
                states["lap_time_delta"] = lap_times - previous_lap_time
        if "rolling_3_lap_avg" in states:
            states["rolling_3_lap_avg"] = (2 * states["rolling_3_lap_avg"] + lap_times) / 3
        if "rolling_5_lap_avg" in states:
            states["rolling_5_lap_avg"] = (4 * states["rolling_5_lap_avg"] + lap_times) / 5
        for change in ("position_change", "gap_change", "interval_change"):
            if change in states:
                states[change] = 0.0
        if "laps_since_pit" in states:
            states["laps_since_pit"] = np.where(pit_stop, 0.0, states["laps_since_pit"] + 1)
        if "has_previous_pit" in states:
            states["has_previous_pit"] = np.where(pit_stop, 1, states["has_previous_pit"])
