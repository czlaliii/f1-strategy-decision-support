"""Race progression graph, redrawn once per completed race lap."""
from dash import no_update
from dash.dependencies import Input, Output, State

from src.analysis.grid_finish import race_grid_and_result
from src.dashboard.race_progression import build_progression_figure, progression_data
from src.dashboard.replay_service import REPLAY_CACHE, time_to_lap


def _progression(replay_id, replay):
    """Built once per loaded session and kept with the replay."""
    if "progression" not in replay:
        grid = None
        if replay.get("race_folder"):
            table = race_grid_and_result(replay["race_folder"], int(replay["session_key"]))
            grid = dict(zip(table["Driver"], table["grid"]))
        replay["progression"] = progression_data(
            replay["laps"],
            grid=grid,
            stints=replay.get("stints"),
            retirements=replay.get("retirements"),
            track_status=replay.get("track_status"),
            lap_of=lambda seconds: time_to_lap(replay_id, seconds),
        )
    return replay["progression"]


def register_progression_callbacks(app):
    @app.callback(
        Output("race-progression", "figure"),
        Output("progression-key", "data"),
        Input("replay-data-store", "data"),
        Input("replay-clock-store", "data"),
        Input("replay-driver", "value"),
        State("progression-key", "data"),
    )
    def show_progression(data, current_time, driver, shown_key):
        replay_id = (data or {}).get("replay_id")
        replay = REPLAY_CACHE.get(replay_id)
        if replay is None or current_time is None:
            return no_update, no_update
        progression = _progression(replay_id, replay)
        positions = progression["positions"]
        finished = positions[positions["TimeSeconds"] <= float(current_time)]
        leader_lap = int(finished["Lap"].max()) if len(finished) else -1
        key = f"{replay_id}|{leader_lap}|{driver}"
        if key == shown_key:
            return no_update, no_update
        colors = replay.get("driver_colors", {})
        # Teams share a colour, so the colour doubles as the team key.
        figure = build_progression_figure(progression, float(current_time), colors, team_of=colors, highlight=driver)
        return figure, key
