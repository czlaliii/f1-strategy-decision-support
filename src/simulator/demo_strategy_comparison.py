"""Compare selected Hungary 2026 lap-50 strategy candidates for driver 14."""

from pathlib import Path

import pandas as pd

from src.models.gradient_boosting_predictor import load_gradient_boosting_model
from src.simulator.race_simulator import Strategy
from src.simulator.strategy_comparator import StrategyComparator

DATA_DIR = Path("data/processed/hungary_2026")
OUTPUT_PATH = DATA_DIR / "strategy_evaluation/strategy_comparison_lap50.csv"
CURRENT_LAP = 50
DRIVER_NUMBER = 14


def main():
    dataset = pd.read_parquet(DATA_DIR / "ml_dataset_clean.parquet")
    race_state = pd.read_parquet(DATA_DIR / "race_state_with_conditions.parquet")
    initial_state = dataset.loc[
        (dataset["driver_number"] == DRIVER_NUMBER)
        & (dataset["lap_number"] == CURRENT_LAP)
    ].iloc[0]
    starting_cumulative_time = float(
        race_state.loc[
            (race_state["driver_number"] == DRIVER_NUMBER)
            & (race_state["lap_number"] <= CURRENT_LAP),
            "lap_duration",
        ].sum()
    )
    strategies = [
        Strategy((52,), ("MEDIUM", "HARD")),
        Strategy((55,), ("MEDIUM", "HARD")),
        Strategy((58,), ("MEDIUM", "HARD")),
        Strategy((55,), ("MEDIUM", "SOFT")),
    ]

    comparison = StrategyComparator(
        predictor=load_gradient_boosting_model(),
        race_last_lap=int(race_state["lap_number"].max()),
        pit_loss_seconds=22.0,
    ).compare(initial_state, strategies, starting_cumulative_time)

    print("=== STRATEGY COMPARISON ===")
    print(f"Starting lap: {CURRENT_LAP}")
    print(f"Driver: {DRIVER_NUMBER}")
    print(f"Current compound: {initial_state['compound']}")
    print(f"Current tyre age: {initial_state['tyre_age']}")
    print()
    print(comparison.to_string(index=False, float_format="{:.3f}".format))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
