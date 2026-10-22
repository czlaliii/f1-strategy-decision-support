"""Connect replay lap progression to the historical strategy simulator."""
import pandas as pd
from dash import no_update
from dash.dependencies import Input, Output, State

from src.dashboard.config import PIT_LOSS_SECONDS
from src.dashboard.replay_service import REPLAY_CACHE, time_to_lap
from src.simulator.race_simulator import Strategy
from src.simulator.strategy_comparator import StrategyComparator

SUPPORTED_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def _format_race_time(seconds):
    if seconds is None or pd.isna(seconds):
        return "–"
    total_ms = round(float(seconds) * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _empty_outputs(message, key=None):
    return message, "Nincs még kiszámítható ajánlás.", [], {"key": key, "status": message}


def register_replay_strategy_callbacks(app, dataset, race_state, predictor, race_last_lap):
    """Show the best future pit lap/compound and refresh once per replay lap."""

    @app.callback(
        Output("live-strategy-message", "children"),
        Output("live-strategy-summary", "children"),
        Output("live-strategy-table", "data"),
        Output("live-strategy-state", "data"),
        Input("replay-data-store", "data"),
        Input("replay-clock-store", "data"),
        Input("replay-driver", "value"),
        State("live-strategy-state", "data"),
    )
    def update_live_strategy(replay_data, current_time, replay_driver, previous_state):
        if not replay_data or not replay_data.get("replay_id"):
            return _empty_outputs(
                "Tölts be egy verseny-sessiont a stratégiai ajánláshoz.",
                None,
            )
        replay_id = replay_data["replay_id"]
        replay = REPLAY_CACHE.get(replay_id)
        if replay is None:
            return _empty_outputs("A replay adatai nem találhatók. Töltsd be újra a sessiont.", replay_id)

        # The configured ML dataset currently contains Hungary 2026 race data only.
        # Do not silently apply that model state to a different Grand Prix/session.
        event_name = str(replay.get("event_name", ""))
        session_name = str(replay.get("session_name", ""))
        is_supported_race = (
            int(replay.get("year", 0)) == 2026
            and "hungar" in event_name.lower()
            and session_name.upper() in {"R", "RACE"}
        )
        if not is_supported_race:
            key = f"{replay_id}|unsupported"
            if (previous_state or {}).get("key") == key:
                return no_update, no_update, no_update, no_update
            return _empty_outputs(
                "A körönkénti ajánlás jelenleg a 2026-os Magyar Nagydíj Race sessionhöz érhető el, mert ehhez áll rendelkezésre illesztett tanító- és versenyállapot-adat.",
                key,
            )

        if not replay_driver:
            return _empty_outputs("Válassz versenyzőt a stratégiai elemzéshez.", replay_id)
        current_lap = int(time_to_lap(replay_id, float(current_time or 0.0)))
        cache_key = f"{replay_id}|{replay_driver}|lap-{current_lap}"
        if (previous_state or {}).get("key") == cache_key:
            return no_update, no_update, no_update, no_update

        laps = replay["laps"]
        if "Driver" not in laps.columns or "DriverNumber" not in laps.columns:
            return _empty_outputs("A session köradataiból nem olvasható ki a versenyző rajtszáma.", cache_key)
        selected_laps = laps.loc[laps["Driver"].astype(str) == str(replay_driver)]
        driver_numbers = pd.to_numeric(selected_laps["DriverNumber"], errors="coerce").dropna()
        if driver_numbers.empty:
            return _empty_outputs(f"Nem található rajtszám ehhez a versenyzőhöz: {replay_driver}.", cache_key)
        driver_number = int(driver_numbers.iloc[0])

        driver_data = dataset.loc[
            pd.to_numeric(dataset["driver_number"], errors="coerce") == driver_number
        ]
        state_rows = driver_data.loc[
            pd.to_numeric(driver_data["lap_number"], errors="coerce") == current_lap
        ]
        if state_rows.empty:
            return _empty_outputs(
                f"A {replay_driver} versenyzőhöz nincs modellállapot a {current_lap}. körre. Az ajánlás ezért nem számítható ki biztonságosan.",
                cache_key,
            )
        initial_state = state_rows.iloc[0]
        current_compound = str(initial_state.get("compound", "UNKNOWN")).upper()
        if current_compound not in SUPPORTED_COMPOUNDS:
            return _empty_outputs(
                f"A {current_lap}. kör gumikeveréke ({current_compound}) nem támogatott a szimulátorban.",
                cache_key,
            )
        if current_lap >= int(race_last_lap) - 1:
            return _empty_outputs("A verseny végéhez közel már nincs értelmezhető jövőbeli bokszkör.", cache_key)

        cumulative_rows = race_state.loc[
            (pd.to_numeric(race_state["driver_number"], errors="coerce") == driver_number)
            & (pd.to_numeric(race_state["lap_number"], errors="coerce") <= current_lap),
            "lap_duration",
        ]
        starting_cumulative_time = float(pd.to_numeric(cumulative_rows, errors="coerce").fillna(0).sum())
        strategies = [
            Strategy((pit_lap,), (current_compound, new_compound))
            for pit_lap in range(current_lap + 1, int(race_last_lap))
            for new_compound in SUPPORTED_COMPOUNDS
        ]
        if not strategies:
            return _empty_outputs("Nincs elérhető bokszkiállási lehetőség a verseny hátralévő részében.", cache_key)

        comparator = StrategyComparator(
            predictor=predictor,
            race_last_lap=int(race_last_lap),
            pit_loss_seconds=PIT_LOSS_SECONDS,
        )
        try:
            comparison = comparator.compare(
                initial_state,
                strategies,
                starting_cumulative_time,
            )
            if comparison.empty:
                return _empty_outputs("A szimulátor nem adott értékelhető stratégiát erre a körre.", cache_key)
            comparison = comparison.copy()
            comparison["predicted_total_race_time"] = pd.to_numeric(
                comparison["predicted_total_race_time"], errors="coerce"
            )
            comparison["predicted_remaining_race_time"] = pd.to_numeric(
                comparison["predicted_remaining_race_time"], errors="coerce"
            )
            comparison = comparison.dropna(subset=["predicted_total_race_time"]).sort_values(
                "predicted_total_race_time", ascending=True
            )
            if comparison.empty:
                return _empty_outputs("A szimuláció nem adott érvényes időbecslést.", cache_key)
            best = comparison.iloc[0]
            best_pit_laps = str(best["pit_laps"])
            compounds = str(best["compounds"])
            compound_parts = compounds.split(",")
            next_compound = compound_parts[-1].strip() if compound_parts else "?"
            summary = (
                f"Ajánlott kiállás: {best_pit_laps}. kör · "
                f"következő gumi: {next_compound} · "
                f"becsült teljes versenyidő: {_format_race_time(best['predicted_total_race_time'])}"
            )
            table = []
            for _, row in comparison.head(5).iterrows():
                table.append({
                    "pit_laps": str(row["pit_laps"]),
                    "compounds": str(row["compounds"]),
                    "predicted_remaining_race_time": _format_race_time(row["predicted_remaining_race_time"]),
                    "predicted_total_race_time": _format_race_time(row["predicted_total_race_time"]),
                })
            return (
                f"{replay_driver} · aktuális kör: {current_lap} · az ajánlás a kör állapotából újraszámolva.",
                summary,
                table,
                {"key": cache_key, "status": "ok"},
            )
        except (ValueError, KeyError, TypeError) as error:
            return _empty_outputs(f"A stratégia nem számítható ki: {error}", cache_key)
