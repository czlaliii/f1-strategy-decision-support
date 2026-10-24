"""Multi-lap rollout accuracy of the structured model with and without XGBoost anchoring.

For each test race the structured strategy model and XGBoost are trained on
earlier races only. Every clean stint run (same stint, green racing laps) is
simulated from its first lap with the actual compound and no pit stop, and
the simulated lap times are compared with the real ones. Variants: the
structured model alone, and anchored to XGBoost with a constant correction or
with half-lives of 10 and 5 laps (lap_time_models.AnchoredLapTimeModel).

Outputs (data/processed/model_comparison/): anchoring_stints.csv and anchoring.md.

Run from the project root:
    python -m src.models.evaluate_anchoring
    python -m src.models.evaluate_anchoring --races hungarian_2025 italian_2025
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from src.features.race_dataset import list_local_races, load_race_dataset
from src.models.compare_lap_time_models import OUTPUT_DIR
from src.models.evaluate_expanding_window import default_test_races, stint_segments
from src.models.expanding_window_model import STRATEGY_FEATURES, fill_with_training_medians, fit_strategy_model
from src.models.lap_time_models import (
    AnchoredLapTimeModel,
    add_model_features,
    anchored_model,
    fit_xgboost,
    load_history,
)
from src.simulator.race_simulator import RaceSimulator, Strategy

BASELINE = "strukturált"
# Variant -> half-life in laps (None: constant correction).
ANCHORED_VARIANTS = {"horgony állandó": None, "horgony fél.idő 10": 10, "horgony fél.idő 5": 5}


def evaluate_race(race, session_key):
    history, used = load_history(race)
    raw = load_race_dataset(race, session_key).assign(race=race)
    # stint_segments re-indexes rows, so feature rows are looked up by driver and lap.
    feature_rows = add_model_features(raw).set_index(["driver_number", "lap_number"], drop=False)
    structured, xgboost_model = fit_strategy_model(history), fit_xgboost(history, used)
    columns = list(dict.fromkeys(STRATEGY_FEATURES + ["lap_number", "tyre_age"]))
    rows = []
    for start, following in stint_segments(raw):
        if pd.isna(start.get("field_pace_ref")):
            continue
        state = fill_with_training_medians(structured, start[columns])
        lap = int(start["lap_number"])
        anchored = anchored_model(structured, xgboost_model, feature_rows.loc[(start["driver_number"], lap)], state)
        variants = {BASELINE: structured}
        variants.update({
            name: AnchoredLapTimeModel(structured, anchored.offset, lap, half_life)
            for name, half_life in ANCHORED_VARIANTS.items()
        })
        actual = following["lap_duration"].to_numpy(dtype=float)
        for name, model in variants.items():
            simulation = RaceSimulator(
                model, state, race_last_lap=int(following["lap_number"].iloc[-1]), pit_loss_seconds=0.0,
            ).simulate(Strategy((), (str(start["compound"]),)))
            predicted = simulation["predicted_lap_time"].to_numpy(dtype=float)
            rows.append({
                "race": race, "variant": name, "start_lap": lap, "laps": len(actual),
                "lap_mae": float(np.mean(np.abs(predicted - actual))),
                "first_lap_ae": float(abs(predicted[0] - actual[0])),
                "total_error_per_lap": float((predicted.sum() - actual.sum()) / len(actual)),
            })
    return rows


def write_summary(table):
    variants = [BASELINE, *ANCHORED_VARIANTS]
    summary = table.groupby("variant").agg(
        stints=("lap_mae", "size"),
        lap_mae=("lap_mae", "mean"),
        first_lap_ae=("first_lap_ae", "mean"),
        abs_total_per_lap=("total_error_per_lap", lambda s: s.abs().mean()),
        bias_per_lap=("total_error_per_lap", "mean"),
    ).loc[variants]
    per_race = table.groupby(["race", "variant"])["lap_mae"].mean().unstack()
    tests = len(ANCHORED_VARIANTS)
    lines = [
        "# Szimulált stintek pontossága – XGBoost-horgonyzás",
        "",
        f"Tesztversenyek: {table['race'].nunique()}, stintek: {int(summary['stints'].iloc[0])} "
        "(mindegyik versenynél csak korábbi versenyeken tanítva).",
        "",
        "| Változat | Kör-MAE (s) | Első kör AE (s) | Összidő abs. hibája/kör (s) | Torzítás/kör (s) | "
        "Jobb versenyek | Wilcoxon p | Bonferroni p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in summary.iterrows():
        if name == BASELINE:
            comparison = "– | – | –"
        else:
            p = stats.wilcoxon(per_race[name], per_race[BASELINE]).pvalue
            wins = int((per_race[name] < per_race[BASELINE]).sum())
            comparison = f"{wins}/{len(per_race)} | {p:.4f} | {min(1.0, p * tests):.4f}"
        lines.append(f"| {name} | {row.lap_mae:.3f} | {row.first_lap_ae:.3f} | {row.abs_total_per_lap:.3f} | "
                     f"{row.bias_per_lap:+.3f} | {comparison} |")
    (OUTPUT_DIR / "anchoring.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--races", nargs="+", help="Test race folders (default: every 3rd late race)")
    args = parser.parse_args()

    races = list_local_races().set_index("race")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for race in args.races or default_test_races():
        rows += evaluate_race(race, races.loc[race, "session_key"])
        print(race, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "anchoring_stints.csv", index=False)
    print()
    print(write_summary(table))
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
