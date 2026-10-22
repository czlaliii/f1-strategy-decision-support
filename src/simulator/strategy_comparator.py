"""Compare explicitly supplied race strategies from one race-state snapshot."""

from collections.abc import Sequence

import pandas as pd

from src.simulator.race_simulator import Predictor, RaceSimulator, Strategy


class StrategyComparator:
    """Run independent deterministic simulations for supplied strategies."""

    def __init__(
        self,
        predictor: Predictor,
        race_last_lap: int,
        pit_loss_seconds: float = 22.0,
    ):
        self.predictor = predictor
        self.race_last_lap = int(race_last_lap)
        self.pit_loss_seconds = float(pit_loss_seconds)

    def compare(
        self,
        initial_state: pd.Series | dict,
        strategies: Sequence[Strategy],
        starting_cumulative_time: float = 0.0,
    ) -> pd.DataFrame:
        """Return race-time summaries for strategies in their input order.

        The simulator copies the supplied state for each candidate.  No result of
        an earlier strategy is reused as input for a later one.
        """
        if not strategies:
            raise ValueError("At least one strategy is required for comparison.")

        records = []
        for index, strategy in enumerate(strategies, start=1):
            simulation = RaceSimulator(
                predictor=self.predictor,
                current_state=initial_state,
                race_last_lap=self.race_last_lap,
                pit_loss_seconds=self.pit_loss_seconds,
                starting_cumulative_time=starting_cumulative_time,
            ).simulate(strategy)
            total_time = float(simulation["cumulative_time"].iloc[-1])
            records.append(
                {
                    "strategy_id": f"S{index}",
                    "pit_laps": ",".join(map(str, strategy.pit_laps)) or "none",
                    "compounds": ",".join(strategy.compounds),
                    "predicted_remaining_race_time": total_time - starting_cumulative_time,
                    "predicted_total_race_time": total_time,
                }
            )

        return pd.DataFrame(records)
