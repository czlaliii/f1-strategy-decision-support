"""Compare next-lap time models on unseen races (expanding window).

Models: persistence baseline (next lap = current lap), the structured hybrid
strategy model, XGBoost and LSTM. For each test race every model is trained
on earlier races only and scored on the same green-flag racing laps.

Outputs (data/processed/model_comparison/): per-race metrics CSV, a Markdown
summary with a paired Wilcoxon test and bootstrap CI for XGBoost vs LSTM
(all laps, and separately the opening laps and the rest of the race), and figures.

Run from the project root:
    python -m src.models.compare_lap_time_models
    python -m src.models.compare_lap_time_models --models baseline strategy xgboost
    python -m src.models.compare_lap_time_models --races hungarian_2025 italian_2025
    python -m src.models.compare_lap_time_models --summary-only   # from saved results
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

from src.features.race_dataset import PROCESSED_ROOT, list_local_races, load_race_dataset
from src.models.evaluate_expanding_window import default_test_races
from src.models.expanding_window_model import (
    MIN_TRAINING_RACES,
    STRATEGY_FEATURES,
    TARGET,
    fit_strategy_model,
    racing_target_rows,
)
from src.models.lap_time_models import (
    LapSequenceModel,
    add_model_features,
    fit_xgboost,
    load_history,
    predict_xgboost,
)

OUTPUT_DIR = PROCESSED_ROOT / "model_comparison"
ALL_MODELS = ["baseline", "strategy", "xgboost", "lstm"]
MODEL_LABELS = {"baseline": "Persistence", "strategy": "Strukturált hibrid", "xgboost": "XGBoost", "lstm": "LSTM"}
BOOTSTRAP_SAMPLES = 10000
# Opening laps: the field pace reference rests on few green laps and restarts are common.
OPENING_LAPS = 10


def metrics(actual, predicted):
    error = predicted - actual
    residual = ((actual - actual.mean()) ** 2).sum()
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "median_ae": float(np.median(np.abs(error))),
        "r2": float(1 - (error ** 2).sum() / residual) if residual > 0 else np.nan,
        "bias": float(np.mean(error)),
    }


def evaluate_race(race, models):
    history, used = load_history(race.race)
    test = add_model_features(load_race_dataset(race.race, race.session_key).assign(race=race.race))
    rows = racing_target_rows(test)
    actual = rows[TARGET].to_numpy(dtype=float)
    predictions, seconds = {}, {}

    if "baseline" in models:
        has_lap = rows["lap_duration"].notna()
        predictions["baseline"] = rows["lap_duration"].where(has_lap, rows["field_pace_ref"]).to_numpy(dtype=float)
        seconds["baseline"] = 0.0
    if "strategy" in models:
        started = time.perf_counter()
        model = fit_strategy_model(history)
        seconds["strategy"] = time.perf_counter() - started
        predictions["strategy"] = model.predict(rows[STRATEGY_FEATURES].astype({"compound": str}))
    if "xgboost" in models:
        model = fit_xgboost(history, used)
        seconds["xgboost"] = model.training_seconds_
        predictions["xgboost"] = predict_xgboost(model, rows)
    if "lstm" in models:
        model = LapSequenceModel().fit(history, used)
        seconds["lstm"] = model.training_seconds_
        predictions["lstm"] = model.predict(test, rows)

    results, errors = [], []
    for name, predicted in predictions.items():
        results.append({"race": race.race, "model": name, "training_races": len(used), "n": len(rows),
                        "train_seconds": seconds[name], **metrics(actual, predicted)})
        errors.append(pd.DataFrame({"race": race.race, "model": name, "error": predicted - actual,
                                    "tyre_age": rows["tyre_age"].to_numpy(), "lap_number": rows["lap_number"].to_numpy()}))
    return results, pd.concat(errors, ignore_index=True)


def paired_comparison(table, first, second):
    """Paired Wilcoxon signed-rank test and bootstrap CI of the per-race MAE difference."""
    pivot = table.pivot(index="race", columns="model", values="mae").dropna(subset=[first, second])
    return _paired(pivot, first, second)


def _paired(pivot, first, second):
    difference = (pivot[first] - pivot[second]).to_numpy()
    test = stats.wilcoxon(pivot[first], pivot[second])
    rng = np.random.default_rng(42)
    means = rng.choice(difference, size=(BOOTSTRAP_SAMPLES, len(difference)), replace=True).mean(axis=1)
    return {
        "races": len(difference),
        "mean_difference": float(difference.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "wins_first": int((difference < 0).sum()),
        "wilcoxon_p": float(test.pvalue),
    }


def write_figures(table, errors, models):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    labels = [MODEL_LABELS[m] for m in models]
    pivot = table.pivot(index="race", columns="model", values="mae")[models]
    figure, axis = plt.subplots(figsize=(14, 5))
    pivot.rename(columns=MODEL_LABELS).plot.bar(ax=axis, width=0.8)
    axis.set_xlabel("Tesztverseny")
    axis.set_ylabel("MAE (s)")
    axis.set_title("Következő kör becslése – MAE versenyenként (nem látott versenyek)")
    axis.legend(title=None)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "mae_by_race.png", dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    data = [errors.loc[errors["model"] == m, "error"].abs().clip(upper=5) for m in models]
    axis.boxplot(data, tick_labels=labels, showfliers=False)
    axis.set_ylabel("|hiba| (s), 5 s-nál levágva")
    axis.set_title("Abszolút hibák eloszlása")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "error_distribution.png", dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    bins = pd.cut(errors["tyre_age"], [-1, 2, 5, 10, 20, 40, 80], labels=["0–2", "3–5", "6–10", "11–20", "21–40", "40+"])
    by_age = errors.assign(abs_error=errors["error"].abs(), age=bins).groupby(["age", "model"], observed=True)["abs_error"].mean().unstack()
    by_age[models].rename(columns=MODEL_LABELS).plot(ax=axis, marker="o")
    axis.set_xlabel("Gumikor (kör)")
    axis.set_ylabel("MAE (s)")
    axis.set_title("Hiba a gumikor szerint")
    axis.legend(title=None)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "mae_by_tyre_age.png", dpi=120)
    plt.close(figure)


def phase_comparison(errors, first, second):
    """Per-race MAE and the paired test, separately for opening laps and the rest."""
    errors = errors.assign(abs_error=errors["error"].abs())
    phases = {
        "minden kör": errors,
        f"első {OPENING_LAPS} kör": errors[errors["lap_number"] <= OPENING_LAPS],
        f"{OPENING_LAPS}. kör után": errors[errors["lap_number"] > OPENING_LAPS],
    }
    results = {}
    for name, rows in phases.items():
        pivot = rows.groupby(["race", "model"])["abs_error"].mean().unstack()
        results[name] = {
            "share": len(rows) / len(errors),
            "mae": pivot.mean(),
            **_paired(pivot.dropna(subset=[first, second]), first, second),
        }
    return results


def write_summary(table, models, errors=None):
    summary = table.groupby("model")[["mae", "rmse", "median_ae", "r2", "bias", "train_seconds"]].mean().loc[models]
    lines = [
        "# Következő kör becslése – modell-összehasonlítás",
        "",
        f"Tesztversenyek: {table['race'].nunique()} (mindegyiknél csak korábbi versenyeken tanítva).",
        "",
        "| Modell | MAE (s) | RMSE (s) | Medián AE (s) | R² | Torzítás (s) | Tanítás (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, row in summary.iterrows():
        lines.append(f"| {MODEL_LABELS[model]} | {row.mae:.3f} | {row.rmse:.3f} | {row.median_ae:.3f} | "
                     f"{row.r2:.3f} | {row.bias:+.3f} | {row.train_seconds:.1f} |")
    if {"xgboost", "lstm"} <= set(models):
        c = paired_comparison(table, "xgboost", "lstm")
        lines += [
            "",
            "## XGBoost vs LSTM (versenyenkénti MAE, páros összehasonlítás)",
            "",
            f"- Átlagos különbség (XGBoost − LSTM): {c['mean_difference']:+.3f} s, "
            f"95% bootstrap CI [{c['ci_low']:+.3f}; {c['ci_high']:+.3f}] s",
            f"- Az XGBoost {c['wins_first']} / {c['races']} versenyen volt pontosabb",
            f"- Wilcoxon előjeles rangpróba: p = {c['wilcoxon_p']:.4f}",
        ]
        if errors is not None:
            lines += [
                "",
                f"## Érzékenységvizsgálat: a verseny eleje ({OPENING_LAPS} kör) külön",
                "",
                "| Szakasz | Körök aránya | " + " | ".join(f"{MODEL_LABELS[m]} MAE" for m in models)
                + " | XGBoost − LSTM (95% CI) | XGBoost jobb | Wilcoxon p |",
                "|---|---|" + "---|" * len(models) + "---|---|---|",
            ]
            for name, r in phase_comparison(errors, "xgboost", "lstm").items():
                lines.append(
                    f"| {name} | {r['share']:.0%} | " + " | ".join(f"{r['mae'][m]:.3f}" for m in models)
                    + f" | {r['mean_difference']:+.3f} [{r['ci_low']:+.3f}; {r['ci_high']:+.3f}] | "
                    f"{r['wins_first']}/{r['races']} | {r['wilcoxon_p']:.4f} |"
                )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--races", nargs="+", help="Test race folders (default: every 3rd late race)")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument("--summary-only", action="store_true", help="Rebuild summary and figures from saved results")
    args = parser.parse_args()

    if args.summary_only:
        table = pd.read_csv(OUTPUT_DIR / "per_race.csv")
        errors = pd.read_parquet(OUTPUT_DIR / "errors.parquet")
        models = [m for m in ALL_MODELS if m in set(table["model"])]
        write_figures(table, errors, models)
        print(write_summary(table, models, errors))
        return

    races = list_local_races()
    selected = races[races["race"].isin(args.races or default_test_races())]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    for race in selected.itertuples():
        if (races["race_date"] < race.race_date).sum() < MIN_TRAINING_RACES:
            print(f"{race.race}: skipped (not enough earlier races)")
            continue
        race_results, race_errors = evaluate_race(race, args.models)
        results += race_results
        errors.append(race_errors)
        print(f"{race.race:<20} " + "  ".join(f"{MODEL_LABELS[r['model']]} {r['mae']:.3f}" for r in race_results), flush=True)

    table = pd.DataFrame(results)
    errors = pd.concat(errors, ignore_index=True)
    table.to_csv(OUTPUT_DIR / "per_race.csv", index=False)
    errors.to_parquet(OUTPUT_DIR / "errors.parquet", index=False)
    models = [m for m in ALL_MODELS if m in args.models]
    write_figures(table, errors, models)
    print()
    print(write_summary(table, models, errors))
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
