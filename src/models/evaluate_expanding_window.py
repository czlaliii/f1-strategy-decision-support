"""Compare the baseline and strategy next-lap models on unseen races.

Each test race gets models trained only on earlier races (expanding window).
Metrics, all on green-flag racing laps of the test race:
  - mae_next_lap: next-lap MAE; persistence baseline = current lap time.
  - soft_minus_hard: mean predicted SOFT - HARD difference at tyre age 2.
  - degradation_per_lap: mean predicted slowdown per lap of MEDIUM tyre age 5 -> 25.
  - stint_mae_per_lap / stint_bias_per_lap: error of multi-lap simulations
    over real stints of at least MIN_STINT_LAPS racing laps without a pit stop.

Run from the project root:
    python -m src.models.evaluate_expanding_window
    python -m src.models.evaluate_expanding_window --races hungarian_2025 italian_2024
"""
import argparse

import numpy as np
import pandas as pd

from src.features.race_dataset import list_local_races, load_race_dataset
from src.models.expanding_window_model import (
    MAX_RELATIVE_LAP_TIME,
    MIN_TRAINING_RACES,
    MODEL_DIR,
    STRATEGY_FEATURES,
    TARGET,
    fill_with_training_medians,
    fit_baseline_model,
    fit_strategy_model,
    load_training_data,
    racing_target_rows,
)
from src.simulator.race_simulator import MODEL_FEATURES, RaceSimulator, Strategy

MIN_STINT_LAPS = 8
# Default test set: every 3rd race with at least this many earlier races.
MIN_PRIOR_RACES = 20


def default_test_races() -> list[str]:
    races = list_local_races()
    eligible = races.iloc[MIN_PRIOR_RACES:]
    return eligible["race"].iloc[::3].tolist()


def _features(rows: pd.DataFrame, features) -> pd.DataFrame:
    return rows[list(features)].astype({"compound": str})


def tyre_sensitivity(model, rows: pd.DataFrame, features) -> tuple[float, float]:
    X = _features(rows, features)
    soft = model.predict(X.assign(compound="SOFT", tyre_age=2.0))
    hard = model.predict(X.assign(compound="HARD", tyre_age=2.0))
    young = model.predict(X.assign(compound="MEDIUM", tyre_age=5.0))
    old = model.predict(X.assign(compound="MEDIUM", tyre_age=25.0))
    return float(np.mean(soft - hard)), float(np.mean(old - young) / 20)


def stint_segments(dataset: pd.DataFrame) -> list[tuple[pd.Series, pd.DataFrame]]:
    """(start state, following laps) for clean same-stint runs of racing laps."""
    racing_median = dataset.loc[dataset["is_racing_lap"], "lap_duration"].median()
    clean = (
        dataset["is_racing_lap"]
        & dataset["lap_duration"].le(MAX_RELATIVE_LAP_TIME * racing_median)
        & dataset["compound"].isin(["SOFT", "MEDIUM", "HARD"])
    )
    segments = []
    for _, laps in dataset.assign(clean=clean).groupby(["driver_number", "stint_number"]):
        laps = laps.sort_values("lap_number").reset_index(drop=True)
        run_id = (~laps["clean"] | laps["lap_number"].diff().ne(1)).cumsum()
        for _, run in laps[laps["clean"]].groupby(run_id[laps["clean"]]):
            if len(run) > MIN_STINT_LAPS:
                segments.append((run.iloc[0], run.iloc[1:]))
    return segments


def stint_errors(model, dataset: pd.DataFrame, features) -> tuple[float, float]:
    errors = []
    for start, following in stint_segments(dataset):
        columns = list(dict.fromkeys(list(features) + ["lap_number", "tyre_age"]))
        state = fill_with_training_medians(model, start[columns])
        if pd.isna(state["lap_number"]):
            continue
        simulation = RaceSimulator(
            model, state, race_last_lap=int(following["lap_number"].iloc[-1]),
            pit_loss_seconds=0.0,
        ).simulate(Strategy((), (str(start["compound"]),)))
        predicted = simulation["predicted_lap_time"].sum()
        errors.append((predicted - following["lap_duration"].sum()) / len(following))
    if not errors:
        return float("nan"), float("nan")
    return float(np.mean(np.abs(errors))), float(np.mean(errors))


def evaluate_race(race: pd.Series) -> list[dict]:
    train, used = load_training_data(race.race)
    test = load_race_dataset(race.race, race.session_key)
    test = test.assign(race=race.race)
    rows = racing_target_rows(test)
    y = rows[TARGET].to_numpy(dtype=float)
    persistence = rows["lap_duration"].to_numpy(dtype=float)
    has_last_lap = ~np.isnan(persistence)

    results = []
    for name, fit, features in [
        ("baseline", fit_baseline_model, MODEL_FEATURES),
        ("strategy", fit_strategy_model, STRATEGY_FEATURES),
    ]:
        model = fit(train)
        prediction = model.predict(_features(rows, features))
        soft_minus_hard, degradation = tyre_sensitivity(model, rows, features)
        stint_mae, stint_bias = stint_errors(model, test, features)
        results.append({
            "race": race.race,
            "model": name,
            "training_races": len(used),
            "eval_rows": len(rows),
            "mae_next_lap": float(np.mean(np.abs(prediction - y))),
            "mae_persistence": float(np.mean(np.abs(persistence - y)[has_last_lap])),
            "soft_minus_hard": soft_minus_hard,
            "degradation_per_lap": degradation,
            "stint_mae_per_lap": stint_mae,
            "stint_bias_per_lap": stint_bias,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--races", nargs="+", help="Test race folders (default: every 3rd late race)")
    args = parser.parse_args()

    races = list_local_races()
    selected = args.races or default_test_races()
    results = []
    for race in races[races["race"].isin(selected)].itertuples():
        if (races["race_date"] < race.race_date).sum() < MIN_TRAINING_RACES:
            print(f"{race.race}: skipped (not enough earlier races)")
            continue
        race_results = evaluate_race(race)
        results.extend(race_results)
        for row in race_results:
            print(
                f"{row['race']:<22} {row['model']:<9} MAE {row['mae_next_lap']:.3f} "
                f"(persistence {row['mae_persistence']:.3f})  S-H {row['soft_minus_hard']:+.3f}  "
                f"deg {row['degradation_per_lap']:+.3f}  stint MAE {row['stint_mae_per_lap']:.3f} "
                f"bias {row['stint_bias_per_lap']:+.3f}",
                flush=True,
            )

    table = pd.DataFrame(results)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output = MODEL_DIR / "evaluation.csv"
    table.to_csv(output, index=False)
    summary = table.groupby("model")[[
        "mae_next_lap", "mae_persistence", "soft_minus_hard",
        "degradation_per_lap", "stint_mae_per_lap", "stint_bias_per_lap",
    ]].mean()
    print("\nMean over test races:")
    print(summary.round(3).to_string())
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
