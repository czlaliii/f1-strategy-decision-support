"""Run one deterministic Hungary 2026 race-simulation example."""

from pathlib import Path

import pandas as pd

from src.models.gradient_boosting_predictor import load_gradient_boosting_model
from src.simulator.race_simulator import RaceSimulator, Strategy

DATA_DIR = Path("data/processed/hungary_2026")
CURRENT_LAP = 50
DRIVER_NUMBER = 14


def main():
    dataset = pd.read_parquet(DATA_DIR / "ml_dataset_clean.parquet")
    race_state = pd.read_parquet(DATA_DIR / "race_state_with_conditions.parquet")
    current_state = dataset.loc[
        (dataset["driver_number"] == DRIVER_NUMBER)
        & (dataset["lap_number"] == CURRENT_LAP)
    ].iloc[0]
    historical_laps = race_state.loc[
        (race_state["driver_number"] == DRIVER_NUMBER)
        & (race_state["lap_number"] <= CURRENT_LAP),
        "lap_duration",
    ]

    simulator = RaceSimulator(
        predictor=load_gradient_boosting_model(),
        current_state=current_state,
        race_last_lap=int(race_state["lap_number"].max()),
        pit_loss_seconds=22.0,
        starting_cumulative_time=float(historical_laps.sum()),
    )
    strategy = Strategy(pit_laps=(55,), compounds=(current_state["compound"], "HARD"))
    simulation = simulator.simulate(strategy)

    print("=== RACE SIMULATION ===")
    print(f"Driver: {DRIVER_NUMBER}")
    print(f"Starting lap: {CURRENT_LAP}")
    print(f"Starting compound: {current_state['compound']}")
    print(f"Starting tyre age: {current_state['tyre_age']}")
    print("Strategy: PIT lap 55, new tyre HARD")
    print(f"Predicted remaining race time: {simulation['cumulative_time'].iloc[-1] - historical_laps.sum():.3f} s")
    print(f"Predicted total race time: {simulation['cumulative_time'].iloc[-1]:.3f} s")
    print("\nFirst simulation rows:")
    print(simulation.head(6).to_string(index=False))
    print("\nLast simulation rows:")
    print(simulation.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
