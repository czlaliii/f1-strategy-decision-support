"""Strategy computations that run in a separate worker process.

The functions only take and return plain data, so they can cross the process
boundary. Datasets, models and finished "what if" analyses stay in the
worker's memory between laps.

Simulations use the structured model anchored to the XGBoost next-lap
estimate at the decision lap (see lap_time_models.AnchoredLapTimeModel).
"""
import pandas as pd

from src.dashboard.config import PIT_LOSS_SECONDS
from src.features.race_dataset import RaceDataError, load_race_dataset
from src.models.expanding_window_model import (
    InsufficientHistoryError,
    fill_with_training_medians,
    load_or_train_model,
)
from src.models.lap_time_models import add_model_features, anchored_model
from src.simulator import strategy_advisor as advisor

# Per-race dataset, trained models and XGBoost feature rows, kept in the worker's memory.
_RACE_RESOURCES: dict[tuple[str, int], tuple] = {}
# (race, session, driver, pit lap) -> what-if result; a stop is analysed once.
_WHAT_IF_CACHE: dict[tuple, dict] = {}


def _race_resources(race_folder, session_key):
    key = (race_folder, int(session_key))
    if key not in _RACE_RESOURCES:
        dataset = load_race_dataset(race_folder, int(session_key))
        trained = load_or_train_model(race_folder)
        feature_rows = add_model_features(dataset)
        feature_rows.index = pd.MultiIndex.from_arrays([
            pd.to_numeric(feature_rows["driver_number"], errors="coerce"),
            pd.to_numeric(feature_rows["lap_number"], errors="coerce"),
        ])
        _RACE_RESOURCES[key] = (dataset, trained, feature_rows)
    return _RACE_RESOURCES[key]


def _message(text, **extra):
    return {"status": "message", "message": text, **extra}


def warm_up(race_folder, session_key):
    """Load (or train) the race's dataset and model before the first lap needs them."""
    try:
        _race_resources(race_folder, session_key)
        return {"status": "ready"}
    except (InsufficientHistoryError, RaceDataError) as error:
        return _message(f"Ehhez a versenyhez nem készíthető modell: {error}")


def _state_at(model, driver_data, lap):
    """(state row, model state) at the end of `lap`, or (None, reason) if unusable."""
    lap_numbers = pd.to_numeric(driver_data["lap_number"], errors="coerce")
    state_rows = driver_data.loc[lap_numbers == lap]
    if state_rows.empty:
        return None, (
            f"Ehhez a versenyzőhöz nincs modellállapot a {lap}. körre. "
            "Az ajánlás ezért nem számítható ki biztonságosan."
        )
    state_row = state_rows.iloc[0]
    # The pace reference and tyre age cannot be imputed meaningfully.
    if pd.isna(state_row.get("field_pace_ref")) or pd.isna(state_row.get("tyre_age")):
        return None, f"A {lap}. körre még nincs elég tiszta köridő a tempó becsléséhez."
    features = list(dict.fromkeys(list(model.feature_names_in_) + ["lap_number", "tyre_age", "compound"]))
    state = fill_with_training_medians(model, state_row[features])
    compound = str(state.get("compound", "UNKNOWN")).upper()
    if compound not in advisor.DRY_COMPOUNDS:
        return None, f"A {lap}. kör gumikeveréke ({compound}) nem támogatott a szimulátorban."
    return state_row, state


def _predictor(trained, feature_rows, state_row, state):
    """Structured model anchored to XGBoost at this state; plain structured model if not possible."""
    key = (pd.to_numeric(state_row.get("driver_number"), errors="coerce"), int(state["lap_number"]))
    if trained.xgboost is None or key not in feature_rows.index:
        return trained.pipeline
    row = feature_rows.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return anchored_model(trained.pipeline, trained.xgboost, row, state)


def _evaluate_from(trained, feature_rows, driver_data, lap, race_last_lap):
    """(state row, state, predictor, simulated strategies) from the end of `lap`; results may be a reason string."""
    state_row, state = _state_at(trained.pipeline, driver_data, lap)
    if state_row is None:
        return None, None, None, state
    lap_numbers = pd.to_numeric(driver_data["lap_number"], errors="coerce")
    used = set(driver_data.loc[lap_numbers <= lap, "compound"].dropna())
    strategies = advisor.candidate_strategies(lap, race_last_lap, state["compound"], used)
    if not strategies:
        return None, None, None, "Nincs a szabályoknak megfelelő stratégia a verseny hátralévő részére."
    predictor = _predictor(trained, feature_rows, state_row, state)
    results = advisor.evaluate_strategies(predictor, state, strategies, race_last_lap, PIT_LOSS_SECONDS)
    return state_row, state, predictor, results


def _actual_segments(driver_data, up_to_lap):
    """Stints actually driven up to a lap, as lap ranges."""
    rows = driver_data.loc[pd.to_numeric(driver_data["lap_number"], errors="coerce") <= up_to_lap]
    rows = rows.dropna(subset=["stint_number"]).sort_values("lap_number")
    return [
        {"start": int(group["lap_number"].min()), "end": int(group["lap_number"].max()),
         "compound": str(group["compound"].iloc[0]).upper()}
        for _, group in rows.groupby("stint_number", sort=True)
    ]


def _what_if(cache_key, trained, feature_rows, driver_data, stop, previous_stop_lap, race_last_lap):
    """Compare an actual pit stop with the model's best plan from a few laps before it."""
    if cache_key in _WHAT_IF_CACHE:
        return _WHAT_IF_CACHE[cache_key]
    base = {"stop_lap": stop["lap"], "from": stop["from"], "to": stop["to"], "stint": stop["stint"]}
    if stop["from"] not in advisor.DRY_COMPOUNDS or stop["to"] not in advisor.DRY_COMPOUNDS:
        result = {**base, "status": "skipped", "reason": "Nem száraz keverék; a modell csak száraz versenyt szimulál."}
    else:
        decision = advisor.decision_lap(stop["lap"], previous_stop_lap)
        *_, results = _evaluate_from(trained, feature_rows, driver_data, decision, race_last_lap)
        if isinstance(results, str):
            result = {**base, "status": "skipped", "reason": results}
        else:
            recommendation = advisor.recommend(results)
            actual = advisor.best_matching_first_stop(results, stop["lap"], stop["to"])
            best = recommendation.best["strategy"]
            history = _actual_segments(driver_data, decision)
            result = {
                **base,
                "status": "ok",
                "decision_lap": decision,
                "model_plan": recommendation.best["plan"],
                "model_pit_laps": list(best.pit_laps),
                "model_compounds": list(best.compounds),
                "model_window": list(recommendation.pit_window) if recommendation.pit_window else None,
                "model_segments": history[:-1] + advisor.stint_segments(
                    history[-1]["start"] if history else 1, race_last_lap, best.compounds, best.pit_laps
                ),
                # None: the actual stop is outside the simulated options (e.g. too early).
                "delta": None if actual is None else float(actual["remaining_time"] - recommendation.best["remaining_time"]),
                "actual_plan": None if actual is None else actual["plan"],
            }
    _WHAT_IF_CACHE[cache_key] = result
    return result


def _what_ifs(race_key, driver_number, trained, feature_rows, driver_data, current_lap, race_last_lap):
    """What-if analyses for every stop completed before the current lap."""
    stops = [s for s in advisor.actual_pit_stops(driver_data) if s["lap"] < current_lap]
    analyses, previous = [], None
    for stop in stops:
        key = (*race_key, driver_number, stop["lap"])
        analyses.append(_what_if(key, trained, feature_rows, driver_data, stop, previous, race_last_lap))
        previous = stop["lap"]
    return {"analyses": analyses, "actual_segments": _actual_segments(driver_data, current_lap)}


def compute_strategy(race_folder, session_key, driver_number, current_lap):
    """Recommendation, tyre insights and what-if analyses for one driver at the end of current_lap."""
    try:
        dataset, trained, feature_rows = _race_resources(race_folder, session_key)
    except (InsufficientHistoryError, RaceDataError) as error:
        return _message(f"Ehhez a versenyhez nem készíthető modell: {error}")
    model = trained.pipeline
    # Final lap from the data; equals the scheduled distance unless the race was shortened.
    race_last_lap = int(pd.to_numeric(dataset["lap_number"], errors="coerce").max())
    driver_data = dataset.loc[pd.to_numeric(dataset["driver_number"], errors="coerce") == driver_number]
    try:
        what_if = _what_ifs((race_folder, int(session_key)), driver_number, trained, feature_rows,
                            driver_data, current_lap, race_last_lap)
    except (ValueError, KeyError, TypeError) as error:
        what_if = {"analyses": [], "actual_segments": [], "error": str(error)}
    what_if["race_last_lap"] = race_last_lap

    if current_lap >= race_last_lap - 1:
        return _message("A verseny végéhez közel már nincs értelmezhető jövőbeli bokszkör.", what_if=what_if)
    try:
        state_row, state, predictor, results = _evaluate_from(trained, feature_rows, driver_data, current_lap, race_last_lap)
        if isinstance(results, str):
            return _message(results, what_if=what_if)
        recommendation = advisor.recommend(results)
        compound = state["compound"]
        tyre_age = float(state["tyre_age"])
        stint = state_row.get("stint_number")
        lap_numbers = pd.to_numeric(driver_data["lap_number"], errors="coerce")
        stint_rows = driver_data.loc[(driver_data["stint_number"] == stint) & (lap_numbers <= current_lap)]
        observed = advisor.observed_dropoff(stint_rows, model, compound)
        pace = advisor.compound_pace(predictor, state)
    except (ValueError, KeyError, TypeError) as error:
        return _message(f"A stratégia nem számítható ki: {error}", what_if=what_if)

    best = recommendation.best["strategy"]
    return {
        "status": "ok",
        "current_lap": int(current_lap),
        "race_last_lap": race_last_lap,
        "training_races": len(trained.training_races),
        "last_training_race": trained.training_races[-1],
        # XGBoost minus structured estimate for the next lap (s); None: not anchored.
        "anchor_offset": float(predictor.offset) if hasattr(predictor, "offset") else None,
        "best": {
            "plan": recommendation.best["plan"],
            "pit_laps": list(best.pit_laps),
            "compounds": list(best.compounds),
        },
        "pit_window": list(recommendation.pit_window) if recommendation.pit_window else None,
        "current_stint": None if pd.isna(stint) else int(stint),
        "plans": [
            {"plan": row["plan"], "pit_laps": list(row["pit_laps"]), "delta": float(row["delta"])}
            for _, row in recommendation.plans.iterrows()
        ],
        "tyre": {
            "compound": compound,
            "age": tyre_age,
            "wear": advisor.tyre_wear_percent(tyre_age, compound, trained.tyre_life_laps),
            "life": trained.tyre_life_laps,
        },
        "degradation": {c: float(model.effects_.get(f"degradation_{c}", 0.0)) for c in advisor.DRY_COMPOUNDS},
        "observed": {
            "tyre_age": observed["tyre_age"].astype(float).tolist(),
            "dropoff": observed["dropoff"].astype(float).tolist(),
        },
        "pace": pace.astype({"tyre_age": int}).to_dict("records"),
        "what_if": what_if,
    }
