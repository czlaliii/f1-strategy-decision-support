"""Strategy computations that run in a separate worker process.

The functions only take and return plain data, so they can cross the process
boundary. Datasets and models stay loaded in the worker between laps.
"""
import pandas as pd

from src.dashboard.config import PIT_LOSS_SECONDS
from src.features.race_dataset import RaceDataError, load_race_dataset
from src.models.expanding_window_model import (
    InsufficientHistoryError,
    fill_with_training_medians,
    load_or_train_model,
)
from src.simulator import strategy_advisor as advisor

# Per-race dataset and model, kept in the worker's memory.
_RACE_RESOURCES: dict[tuple[str, int], tuple] = {}


def _race_resources(race_folder, session_key):
    key = (race_folder, int(session_key))
    if key not in _RACE_RESOURCES:
        dataset = load_race_dataset(race_folder, int(session_key))
        model = load_or_train_model(race_folder)
        _RACE_RESOURCES[key] = (dataset, model)
    return _RACE_RESOURCES[key]


def _message(text):
    return {"status": "message", "message": text}


def warm_up(race_folder, session_key):
    """Load (or train) the race's dataset and model before the first lap needs them."""
    try:
        _race_resources(race_folder, session_key)
        return {"status": "ready"}
    except (InsufficientHistoryError, RaceDataError) as error:
        return _message(f"Ehhez a versenyhez nem készíthető modell: {error}")


def compute_strategy(race_folder, session_key, driver_number, current_lap):
    """Recommendation and tyre insights for one driver at the end of current_lap."""
    try:
        dataset, trained = _race_resources(race_folder, session_key)
    except (InsufficientHistoryError, RaceDataError) as error:
        return _message(f"Ehhez a versenyhez nem készíthető modell: {error}")
    model = trained.pipeline
    # Final lap from the data; equals the scheduled distance unless the race was shortened.
    race_last_lap = int(pd.to_numeric(dataset["lap_number"], errors="coerce").max())

    driver_data = dataset.loc[pd.to_numeric(dataset["driver_number"], errors="coerce") == driver_number]
    lap_numbers = pd.to_numeric(driver_data["lap_number"], errors="coerce")
    state_rows = driver_data.loc[lap_numbers == current_lap]
    if state_rows.empty:
        return _message(
            f"Ehhez a versenyzőhöz nincs modellállapot a {current_lap}. körre. "
            "Az ajánlás ezért nem számítható ki biztonságosan."
        )
    state_row = state_rows.iloc[0]
    # The pace reference and tyre age cannot be imputed meaningfully.
    if pd.isna(state_row.get("field_pace_ref")) or pd.isna(state_row.get("tyre_age")):
        return _message(f"A {current_lap}. körre még nincs elég tiszta köridő a tempó becsléséhez.")
    model_features = list(dict.fromkeys(list(model.feature_names_in_) + ["lap_number", "tyre_age", "compound"]))
    initial_state = fill_with_training_medians(model, state_row[model_features])
    compound = str(initial_state.get("compound", "UNKNOWN")).upper()
    if compound not in advisor.DRY_COMPOUNDS:
        return _message(f"A {current_lap}. kör gumikeveréke ({compound}) nem támogatott a szimulátorban.")
    if current_lap >= race_last_lap - 1:
        return _message("A verseny végéhez közel már nincs értelmezhető jövőbeli bokszkör.")

    used_compounds = set(driver_data.loc[lap_numbers <= current_lap, "compound"].dropna())
    strategies = advisor.candidate_strategies(current_lap, race_last_lap, compound, used_compounds)
    if not strategies:
        return _message("Nincs a szabályoknak megfelelő stratégia a verseny hátralévő részére.")
    try:
        results = advisor.evaluate_strategies(model, initial_state, strategies, race_last_lap, PIT_LOSS_SECONDS)
        recommendation = advisor.recommend(results)
        tyre_age = float(initial_state["tyre_age"])
        stint = state_row.get("stint_number")
        stint_rows = driver_data.loc[(driver_data["stint_number"] == stint) & (lap_numbers <= current_lap)]
        observed = advisor.observed_dropoff(stint_rows, model, compound)
        pace = advisor.compound_pace(model, initial_state)
    except (ValueError, KeyError, TypeError) as error:
        return _message(f"A stratégia nem számítható ki: {error}")

    best = recommendation.best["strategy"]
    return {
        "status": "ok",
        "current_lap": int(current_lap),
        "race_last_lap": race_last_lap,
        "training_races": len(trained.training_races),
        "last_training_race": trained.training_races[-1],
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
    }
