"""Callbacks for comparing race strategies and plotting their projections."""
import pandas as pd
import plotly.express as px
from dash import State
from dash.dependencies import Input, Output

from src.dashboard.config import PIT_LOSS_SECONDS
from src.simulator.race_simulator import RaceSimulator, Strategy
from src.simulator.strategy_comparator import StrategyComparator


def format_race_time(seconds):
    """Format a total race time, e.g. 1:40:55.739."""
    if seconds is None or pd.isna(seconds):
        return "–"
    total_seconds = int(seconds)
    milliseconds = round((seconds - total_seconds) * 1000)
    if milliseconds == 1000:
        total_seconds += 1
        milliseconds = 0
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{remaining_seconds:02d}.{milliseconds:03d}"

def register_strategy_callbacks(app, dataset, race_state, predictor, race_last_lap):

    @app.callback(
        Output("results-table", "data"),
        Output("strategy-chart", "figure"),
        Output("simulation-message", "children"),
        Input("run-simulation", "n_clicks"),
        State("driver", "value"),
        State("current-lap", "value"),
        State("pit-lap", "value"),
        State("new-compound", "value"),
        prevent_initial_call=True,
    )
    def run_simulation(
        n_clicks,
        driver_number,
        current_lap,
        pit_lap,
        new_compound,
    ):
        if None in (
            driver_number,
            current_lap,
            pit_lap,
            new_compound,
        ):
            return [], {}, "Töltsd ki az összes mezőt!"
        current_lap = int(current_lap)
        pit_lap = int(pit_lap)
        if current_lap < 1 or current_lap >= race_last_lap:
            return (
                [],
                {},
                f"A kezdőkör 1 és {race_last_lap - 1} között legyen.",
            )
        if not current_lap < pit_lap < race_last_lap:
            return (
                [],
                {},
                "A bokszkiállásnak a kezdőkör után, a cél előtt kell lennie.",
            )
        selected = dataset.loc[
            (dataset["driver_number"] == driver_number)
            & (dataset["lap_number"] == current_lap)
        ]
        if selected.empty:
            return (
                [],
                {},
                "Ehhez a versenyzőhöz nincs adat a kiválasztott körben.",
            )
        initial_state = selected.iloc[0]
        if initial_state["compound"] not in {"SOFT", "MEDIUM", "HARD"}:
            return (
                [],
                {},
                "A jelenlegi gumikeverék nem támogatott a szimulátorban.",
            )
        cumulative_rows = race_state.loc[
            (race_state["driver_number"] == driver_number)
            & (race_state["lap_number"] <= current_lap),
            "lap_duration",
        ]
        starting_cumulative_time = float(cumulative_rows.sum())
        strategies = [
            Strategy(
                (pit_lap,),
                (initial_state["compound"], new_compound),
            ),
        ]
        # Alternatívák: három körrel korábban vagy később.
        for alternative_lap in (pit_lap - 3, pit_lap + 3):
            if current_lap < alternative_lap < race_last_lap:
                strategies.append(
                    Strategy(
                        (alternative_lap,),
                        (initial_state["compound"], new_compound),
                    )
                )
        comparator = StrategyComparator(
            predictor=predictor,
            race_last_lap=race_last_lap,
            pit_loss_seconds=PIT_LOSS_SECONDS,
        )
        try:
            comparison = comparator.compare(
                initial_state,
                strategies,
                starting_cumulative_time,
            )
            results = comparison.to_dict("records")
            for row in results:
                row["predicted_remaining_race_time"] = format_race_time(
                    row["predicted_remaining_race_time"]
                )
                row["predicted_total_race_time"] = format_race_time(
                    row["predicted_total_race_time"]
                )
            simulator = RaceSimulator(
                predictor=predictor,
                current_state=initial_state,
                race_last_lap=race_last_lap,
                pit_loss_seconds=PIT_LOSS_SECONDS,
                starting_cumulative_time=starting_cumulative_time,
            )
            curves = []
            for row in comparison.to_dict("records"):
                strategy_id = row["strategy_id"]
                strategy = next(
                    strategy
                    for strategy in strategies
                    if ",".join(map(str, strategy.pit_laps))
                    == row["pit_laps"]
                    and ",".join(strategy.compounds)
                    == row["compounds"]
                )
                simulated = simulator.simulate(strategy)
                simulated["strategy_id"] = strategy_id
                curves.append(simulated)
            chart_data = pd.concat(
                curves,
                ignore_index=True,
            )
            figure = px.line(
                chart_data,
                x="lap_number",
                y="cumulative_time",
                color="strategy_id",
                markers=True,
                title="Stratégiák becsült kumulált versenyideje",
                labels={
                    "lap_number": "Kör",
                    "cumulative_time": "Kumulált idő (s)",
                    "strategy_id": "Stratégia",
                },
            )
            return (
                results,
                figure,
                "A szimuláció sikeresen lefutott.",
            )
        except (ValueError, KeyError) as error:
            return (
                [],
                {},
                f"A szimuláció nem futtatható: {error}",
            )
