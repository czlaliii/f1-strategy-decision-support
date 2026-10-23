"""Turn race simulations into a pit-stop recommendation and tyre insights.

All quantities are model estimates for a historical replay, not team data.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.simulator.race_simulator import RaceSimulator, Strategy

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
COMPOUND_SHORT = {"SOFT": "S", "MEDIUM": "M", "HARD": "H"}
# Pit laps whose race time is within this margin of the best form the window.
PIT_WINDOW_TOLERANCE_SECONDS = 0.5
# Two-stop candidates use every TWO_STOP_STEP-th lap to keep the search fast.
TWO_STOP_STEP = 2
MIN_STINT_LAPS = 5


@dataclass(frozen=True)
class Recommendation:
    best: pd.Series
    pit_window: tuple[int, int] | None
    plans: pd.DataFrame


def satisfies_compound_rule(compounds) -> bool:
    """Dry races require at least two different dry compounds."""
    used = set(compounds)
    if used - set(DRY_COMPOUNDS):
        return True
    return len(used) >= 2


def candidate_strategies(current_lap, last_lap, current_compound, used_compounds):
    """0-, 1- and 2-stop strategies that respect the two-compound rule."""
    used = set(used_compounds) | {current_compound}
    strategies = []
    if satisfies_compound_rule(used):
        strategies.append(Strategy((), (current_compound,)))
    for pit in range(current_lap + 1, last_lap):
        for compound in DRY_COMPOUNDS:
            if satisfies_compound_rule(used | {compound}):
                strategies.append(Strategy((pit,), (current_compound, compound)))
    for first in range(current_lap + 1, last_lap - MIN_STINT_LAPS, TWO_STOP_STEP):
        for second in range(first + MIN_STINT_LAPS, last_lap, TWO_STOP_STEP):
            for middle in DRY_COMPOUNDS:
                for final in DRY_COMPOUNDS:
                    if satisfies_compound_rule(used | {middle, final}):
                        strategies.append(
                            Strategy((first, second), (current_compound, middle, final))
                        )
    return strategies


def plan_label(strategy: Strategy) -> str:
    stops = len(strategy.pit_laps)
    route = "→".join(COMPOUND_SHORT.get(c, c[:1]) for c in strategy.compounds)
    return f"{stops} kiállás · {route}"


def evaluate_strategies(predictor, state, strategies, last_lap, pit_loss_seconds):
    """Simulate every strategy and return one row per strategy with its race time."""
    totals = RaceSimulator(
        predictor, state, race_last_lap=last_lap, pit_loss_seconds=pit_loss_seconds,
    ).total_times(strategies)
    return pd.DataFrame({
        "strategy": strategies,
        "plan": [plan_label(s) for s in strategies],
        "stops": [len(s.pit_laps) for s in strategies],
        "pit_laps": [s.pit_laps for s in strategies],
        "remaining_time": totals,
    })


def recommend(results: pd.DataFrame) -> Recommendation:
    """Best strategy, its pit window and the best variant of every plan."""
    results = results.sort_values("remaining_time").reset_index(drop=True)
    best = results.iloc[0]

    pit_window = None
    if best["stops"] > 0:
        same_plan = results[results["plan"] == best["plan"]]
        close = same_plan[same_plan["remaining_time"] <= best["remaining_time"] + PIT_WINDOW_TOLERANCE_SECONDS]
        first_stops = [laps[0] for laps in close["pit_laps"]]
        pit_window = (min(first_stops), max(first_stops))

    plans = results.drop_duplicates("plan").copy()
    plans["delta"] = plans["remaining_time"] - best["remaining_time"]
    return Recommendation(best, pit_window, plans.reset_index(drop=True))


def tyre_wear_percent(tyre_age, compound, tyre_life_laps) -> float | None:
    """Laps on the set as a share of the compound's typical life (an estimate)."""
    life = tyre_life_laps.get(compound)
    if not life or pd.isna(tyre_age):
        return None
    return float(min(100.0, 100.0 * (float(tyre_age) + 1) / life))


def degradation_curve(model, compound, ages) -> np.ndarray:
    """Model slowdown versus a fresh set of the same compound, in seconds."""
    rate = model.effects_.get(f"degradation_{compound}", 0.0)
    return rate * np.asarray(ages, dtype=float)


def observed_dropoff(stint_rows: pd.DataFrame, model, compound) -> pd.DataFrame:
    """Driver's green laps in the current stint, relative to field and own pace.

    The values are shifted so their median matches the model curve, because
    only the change over the stint is comparable, not the absolute level.
    """
    rows = stint_rows[stint_rows["is_green_lap"].astype(bool)].dropna(
        subset=["lap_duration", "field_pace_ref", "tyre_age"]
    )
    if rows.empty:
        return pd.DataFrame(columns=["tyre_age", "dropoff"])
    relative = (
        rows["lap_duration"] - rows["field_pace_ref"] - rows["driver_pace_offset"].fillna(0)
    ).to_numpy(dtype=float)
    ages = rows["tyre_age"].to_numpy(dtype=float)
    model_values = degradation_curve(model, compound, ages)
    shift = np.median(relative - model_values)
    return pd.DataFrame({"tyre_age": ages, "dropoff": relative - shift})


def compound_pace(model, state: pd.Series) -> pd.DataFrame:
    """Next-lap estimate on the current set and on a new set of each compound."""
    rows = [("Jelenlegi szett", state["compound"], float(state["tyre_age"]))]
    rows += [(f"Új {c}", c, 0.0) for c in DRY_COMPOUNDS]
    features = list(model.feature_names_in_)
    frame = pd.DataFrame([
        {**{f: state[f] for f in features}, "compound": compound, "tyre_age": age}
        for _, compound, age in rows
    ])
    predictions = model.predict(frame)
    current = predictions[0]
    return pd.DataFrame({
        "set": [name for name, _, _ in rows],
        "compound": [compound for _, compound, _ in rows],
        "tyre_age": [int(age) for _, _, age in rows],
        "lap_time": predictions,
        "delta_to_current": predictions - current,
    })
