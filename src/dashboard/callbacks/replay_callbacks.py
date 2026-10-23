"""Callbacks for loading, controlling and rendering a historical replay."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import ClientsideFunction, State, ctx, no_update
from dash.dependencies import Input, Output

from src.dashboard.config import PROJECT_ROOT
from src.dashboard.replay_player import (
    build_current_snapshot,
    build_leaderboard,
    build_position_window,
    build_track_figure,
    format_elapsed_time,
    format_lap_time,
    load_local_session_data,
    load_session_data,
    track_status_at,
)
from src.dashboard.replay_service import (
    REPLAY_CACHE,
    get_lap_start_times,
    lap_to_time,
    time_to_lap,
)


# Track status banner: label and CSS modifier per race-control state.
TRACK_STATUS_DISPLAY = {
    "SC": ("SAFETY CAR", "sc"),
    "VSC": ("VIRTUAL SAFETY CAR", "vsc"),
    "RED": ("PIROS ZÁSZLÓ", "red"),
}

# Seconds of car positions sent ahead per unit of playback speed, and the
# remaining lead time that triggers the next window.
POSITION_WINDOW_SECONDS = 15.0
POSITION_WINDOW_MARGIN_SECONDS = 5.0


def records_from_dataframe(df):
    """Convert a DataFrame to JSON-compatible records for Dash components."""
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))

_LOCAL_RACE_FOLDER_CACHE: dict[int, Path] | None = None


def _normalize_event_name(value: object) -> str:
    """Normalize event names so FastF1 and OpenF1 names can be matched."""
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _local_race_folder_for_selection(year: int, event_name: str):
    """Return (folder name, session_key) when this Race session is stored locally.

    The OpenF1 race catalog maps year/event to session_key. We then locate the
    downloaded folder by checking each local drivers.json. This replaces the
    previous one-off branches for Australia 2025 and Hungary 2026.
    """
    catalog_path = PROJECT_ROOT / "data" / "metadata" / "race_catalog.csv"
    raw_root = PROJECT_ROOT / "data" / "raw"
    if not catalog_path.exists() or not raw_root.exists():
        return None

    try:
        catalog = pd.read_csv(catalog_path)
        required = {"year", "meeting_name", "session_key"}
        if not required.issubset(catalog.columns):
            return None
        years = pd.to_numeric(catalog["year"], errors="coerce")
        matches = catalog.loc[
            (years == int(year))
            & catalog["meeting_name"].map(_normalize_event_name).eq(
                _normalize_event_name(event_name)
            )
        ]
        if matches.empty:
            return None
        session_keys = pd.to_numeric(matches["session_key"], errors="coerce").dropna().astype(int).unique()
        if len(session_keys) != 1:
            return None
        session_key = int(session_keys[0])
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return None

    # Look only at the small drivers.json file in each downloaded race folder.
    for race_dir in raw_root.iterdir():
        if not race_dir.is_dir():
            continue
        drivers_path = race_dir / "drivers.json"
        laps_path = race_dir / "laps.json"
        if not drivers_path.exists() or not laps_path.exists():
            continue
        try:
            drivers = pd.read_json(drivers_path)
            if "session_key" not in drivers.columns:
                continue
            keys = pd.to_numeric(drivers["session_key"], errors="coerce")
            if keys.eq(session_key).any():
                return race_dir.name, session_key
        except (ValueError, OSError, pd.errors.ParserError):
            continue
    return None


def register_replay_callbacks(app):

    @app.callback(
        Output("replay-data-store", "data"),
        Output("replay-clock-store", "data", allow_duplicate=True),
        Output("replay-driver", "options"),
        Output("replay-driver", "value"),
        Output("replay-time-slider", "min"),
        Output("replay-time-slider", "max"),
        Output("replay-time-slider", "value"),
        Output("replay-message", "children"),
        Output("replay-session-info", "children"),
        Input("load-replay", "n_clicks"),
        State("season-year", "value"),
        State("event-name", "value"),
        State("session-name", "value"),
        prevent_initial_call=True,
    )
    def load_replay(n_clicks, year, event_name, session_name):
        if None in (year, event_name, session_name):
            return (
                None, 0.0, [], None, 0, 1, 0,
                "Válassz évet, nagydíjat és sessiont.",
                "",
            )
        try:
            local_race = None
            race_folder = local_session_key = None
            if session_name == "R":
                local_race = _local_race_folder_for_selection(
                    int(year),
                    str(event_name),
                )

            if local_race is not None:
                race_folder, local_session_key = local_race
                replay = load_local_session_data(
                    race_name=race_folder,
                    session_key=local_session_key,
                )
                replay_source = f"helyi OpenF1-adatok: {race_folder}"
            else:
                # Races not downloaded yet and non-Race sessions still use the FastF1 loader.
                replay = load_session_data(
                    year,
                    event_name,
                    session_name,
                )
                replay_source = "FastF1"
            laps = replay["laps"]
            frames = replay["frames"]
            track = replay["track"]
            if frames.empty:
                return (
                    None, 0.0, [], None, 0, 1, 0,
                    "Nem található használható telemetria.",
                    "",
                )
            driver_names = sorted(
                frames["Driver"].dropna().unique().tolist()
            )
            replay_id = f"{year}:{event_name}:{session_name}"
            frames_by_driver = {
                driver: group.sort_values("TimeSeconds").reset_index(drop=True)
                for driver, group in frames.groupby("Driver", sort=False)
            }
            REPLAY_CACHE[replay_id] = {
                "laps": laps,
                "frames": frames,
                "frames_by_driver": frames_by_driver,
                "track": track,
                "year": int(year),
                "event_name": str(event_name),
                "session_name": str(session_name),
                "driver_colors": replay.get("driver_colors", {}),
                "stints": replay.get("stints"),
                "track_status": replay.get("track_status"),
                "retirements": replay.get("retirements", {}),
                # Set only for locally stored races; used by the strategy advisor.
                "race_folder": race_folder,
                "session_key": local_session_key,
            }
            data = {"replay_id": replay_id}
            max_lap = max(
                1,
                int(pd.to_numeric(
                    frames["LapNumber"], errors="coerce"
                ).max()),
            )
            return (
                data,
                0.0,
                [
                    {"label": name, "value": name}
                    for name in driver_names
                ],
                driver_names[0] if driver_names else None,
                1,
                max_lap,
                1,
                (
                    f"{event_name} — {session_name}: "
                    f"{len(frames)} telemetria-minta betöltve ({replay_source})."
                ),
                f"{year} · {event_name} · {session_name}",
            )
        except Exception as error:
            return (
                None, 0.0, [], None, 0, 1, 0,
                f"A session nem tölthető be: {error}",
                "",
            )

    @app.callback(
        Output("replay-time-slider", "value", allow_duplicate=True),
        Output("replay-interval", "disabled"),
        Output("replay-play", "children"),
        Output("replay-clock-store", "data"),
        Input("replay-play", "n_clicks"),
        Input("replay-back-5", "n_clicks"),
        Input("replay-forward-5", "n_clicks"),
        Input("replay-time-slider", "value"),
        Input("replay-interval", "n_intervals"),
        State("replay-interval", "disabled"),
        State("replay-speed", "value"),
        State("replay-clock-store", "data"),
        State("replay-data-store", "data"),
        prevent_initial_call=True,
    )
    def control_replay(
        play_clicks,
        back_clicks,
        forward_clicks,
        selected_lap,
        n_intervals,
        interval_disabled,
        speed,
        current_time,
        data,
    ):
        if not data or "replay_id" not in data:
            return no_update, True, "Lejátszás", no_update
        replay_id = data["replay_id"]
        replay = REPLAY_CACHE.get(replay_id)
        if replay is None:
            return no_update, True, "Lejátszás", no_update
        frames = replay["frames"]
        if frames.empty:
            return no_update, True, "Lejátszás", no_update
        max_time = float(frames["TimeSeconds"].max())
        max_lap = int(
            pd.to_numeric(
                frames["LapNumber"], errors="coerce"
            ).max()
        )
        current_time = float(current_time or 0.0)
        speed = float(speed or 1)
        triggered = ctx.triggered_id
        if triggered == "replay-play":
            if interval_disabled and current_time >= max_time:
                return 1, False, "Szünet", lap_to_time(replay_id, 1)
            new_disabled = not interval_disabled
            return (
                time_to_lap(replay_id, current_time),
                new_disabled,
                "Lejátszás" if new_disabled else "Szünet",
                current_time,
            )
        if triggered == "replay-back-5":
            current_lap = time_to_lap(replay_id, current_time)
            new_lap = max(1, current_lap - 1)
            new_time = lap_to_time(replay_id, new_lap)
            return new_lap, True, "Lejátszás", new_time
        if triggered == "replay-forward-5":
            current_lap = time_to_lap(replay_id, current_time)
            new_lap = min(max_lap, current_lap + 1)
            new_time = lap_to_time(replay_id, new_lap)
            return new_lap, True, "Lejátszás", new_time
        if triggered == "replay-time-slider":
            new_lap = int(selected_lap or 1)
            new_time = lap_to_time(replay_id, new_lap)
            return new_lap, True, "Lejátszás", new_time
        if triggered == "replay-interval":
            next_time = min(
                max_time,
                current_time + 0.25 * speed,
            )
            next_lap = time_to_lap(replay_id, next_time)
            if next_time >= max_time:
                return next_lap, True, "Lejátszás", next_time
            return next_lap, False, "Szünet", next_time
        return no_update, no_update, no_update, no_update

    @app.callback(
        Output("replay-current-time", "children"),
        Output("replay-total-time", "children"),
        Output("leaderboard-table", "data"),
        Output("replay-current-info", "children"),
        Output("replay-table", "columns"),
        Output("replay-table", "data"),
        Output("replay-chart", "figure"),
        Output("replay-view-state", "data"),
        Input("replay-data-store", "data"),
        Input("replay-clock-store", "data"),
        Input("replay-driver", "value"),
        State("replay-view-state", "data"),
    )
    def update_replay_view(data, current_time, driver, view_state):
        if not data or current_time is None:
            return (
                "00:00:00", "00:00:00", [],
                "Nincs betöltött session.", [], [], {}, None,
            )
        replay_id = data.get("replay_id")
        replay = REPLAY_CACHE.get(replay_id)
        if replay is None:
            return (
                "00:00:00", "00:00:00", [],
                "A replay adatai nem találhatók. Töltsd be újra a sessiont.",
                [], [], {}, None,
            )
        frames = replay["frames"]
        laps = replay["laps"]
        current_time = float(current_time)
        total_time = float(frames["TimeSeconds"].max())
        current_frames = build_current_snapshot(
            replay["frames_by_driver"],
            current_time,
        )
        leaderboard = build_leaderboard(
            current_frames,
            laps,
            current_time,
            replay.get("stints"),
            replay.get("retirements"),
        )
        current_time_text = format_elapsed_time(current_time)
        total_time_text = format_elapsed_time(total_time)
        if driver is None:
            return (
                current_time_text,
                total_time_text,
                records_from_dataframe(leaderboard),
                "Válassz versenyzőt.",
                [],
                [],
                {},
                None,
            )
        driver_laps = laps.loc[
            (laps["Driver"] == driver)
            & (laps["TimeSeconds"] <= current_time)
        ].copy()
        if driver_laps.empty:
            info = f"{driver}: még nincs befejezett köradat."
        else:
            row = driver_laps.sort_values("TimeSeconds").iloc[-1]
            tyres = leaderboard.loc[leaderboard["Driver"] == driver]
            tyre = tyres.iloc[0] if not tyres.empty else row

            def shown(value):
                return "–" if pd.isna(value) else value

            info = (
                f"{driver} | {int(row['LapNumber'])}. kör | "
                f"Utolsó köridő: {format_lap_time(row['LapTime'])} | "
                f"Gumi: {shown(tyre.get('Compound'))} | "
                f"Gumikor: {shown(tyre.get('TyreLife'))} kör | "
                f"Stint: {shown(tyre.get('Stint'))}"
            )
        preferred_columns = [
            "LapNumber",
            "LapTime",
            "Sector1Time",
            "Sector2Time",
            "Sector3Time",
            "Compound",
            "TyreLife",
            "Stint",
            "Position",
            "IsPersonalBest",
        ]
        available_columns = [
            column for column in preferred_columns
            if column in driver_laps.columns
        ]
        # The lap table and chart change only when a lap is completed.
        view_key = f"{replay_id}|{driver}|{len(driver_laps)}"
        if view_key == view_state:
            return (
                current_time_text,
                total_time_text,
                records_from_dataframe(leaderboard),
                info,
                no_update,
                no_update,
                no_update,
                no_update,
            )
        # Show the latest completed lap first in the table.
        lap_table = (
            driver_laps[available_columns]
            .sort_values("LapNumber", ascending=False)
            .copy()
        )
        for column in [
            "LapTime",
            "Sector1Time",
            "Sector2Time",
            "Sector3Time",
        ]:
            if column in lap_table.columns:
                lap_table[column] = lap_table[column].map(
                    format_lap_time
                )
        table_columns = [
            {"name": column, "id": column}
            for column in available_columns
        ]
        # Chart runs chronologically; the table is sorted by descending lap.
        chart_data = (
            driver_laps
            .dropna(subset=["LapTime"])
            .sort_values("LapNumber", ascending=True)
        )
        if chart_data.empty:
            chart = {}
        else:
            lap_times = pd.to_numeric(chart_data["LapTime"], errors="coerce").dropna()
            if lap_times.empty:
                chart = {}
            else:
                y_min = float(lap_times.min())
                y_max = float(lap_times.max())
                spread = y_max - y_min
                # Tight, data-fitted Y axis so lap-time differences stay visible.
                padding = max(spread * 0.12, abs(y_min) * 0.005, 0.1)
                chart = px.line(
                    chart_data,
                    x="LapNumber",
                    y="LapTime",
                    markers=True,
                    title=f"{driver} köridői",
                    labels={
                        "LapNumber": "Befejezett kör",
                        "LapTime": "Köridő (másodperc)",
                    },
                    hover_data={
                        "LapNumber": True,
                        "LapTime": ":.3f",
                        "Compound": True if "Compound" in chart_data.columns else False,
                        "TyreLife": True if "TyreLife" in chart_data.columns else False,
                    },
                )
                chart.update_traces(
                    line={"width": 3},
                    marker={"size": 7},
                    hovertemplate=(
                        "Kör: %{x}<br>"
                        "Köridő: %{y:.3f} s"
                        "<extra></extra>"
                    ),
                )
                chart.update_layout(
                    template="plotly_white",
                    height=360,
                    margin={"l": 18, "r": 18, "t": 52, "b": 35},
                    hovermode="x unified",
                    showlegend=False,
                    xaxis={
                        "title": "Befejezett kör",
                        "dtick": 1 if chart_data["LapNumber"].nunique() <= 25 else None,
                        "showgrid": False,
                        "zeroline": False,
                    },
                    yaxis={
                        "title": "Köridő (másodperc)",
                        "range": [y_min - padding, y_max + padding],
                        "tickformat": ".1f",
                        "gridcolor": "#e7ebf0",
                        "zeroline": False,
                    },
                )
        return (
            current_time_text,
            total_time_text,
            records_from_dataframe(leaderboard),
            info,
            table_columns,
            records_from_dataframe(lap_table),
            chart,
            view_key,
        )

    @app.callback(
        Output("replay-track", "figure"),
        Input("replay-data-store", "data"),
    )
    def draw_track(data):
        """The track line is drawn once per session; cars are animated in the browser."""
        replay_id = (data or {}).get("replay_id")
        replay = REPLAY_CACHE.get(replay_id)
        # A new revision per session drops the previous session's zoom.
        return build_track_figure(replay["track"] if replay else None, revision=replay_id)

    @app.callback(
        Output("replay-anim-data", "data"),
        Input("replay-clock-store", "data"),
        Input("replay-data-store", "data"),
        Input("replay-speed", "value"),
        State("replay-anim-data", "data"),
    )
    def send_position_window(current_time, data, speed, window):
        """Send car positions ahead of the clock; only when the current window runs out."""
        replay_id = (data or {}).get("replay_id")
        replay = REPLAY_CACHE.get(replay_id)
        if replay is None or current_time is None:
            return None
        current_time = float(current_time)
        speed = float(speed or 1)
        if (
            window
            and window.get("replay_id") == replay_id
            and window.get("speed") == speed
            and window["start"] <= current_time
            <= window["end"] - POSITION_WINDOW_MARGIN_SECONDS * speed
        ):
            return no_update
        window = build_position_window(
            replay["frames_by_driver"],
            replay.get("driver_colors"),
            current_time - 1.0,
            current_time + POSITION_WINDOW_SECONDS * speed,
            replay.get("retirements"),
        )
        window.update({"replay_id": replay_id, "speed": speed})
        return window

    @app.callback(
        Output("track-status-banner", "children"),
        Output("track-status-banner", "className"),
        Output("track-status-key", "data"),
        Input("replay-clock-store", "data"),
        Input("replay-data-store", "data"),
        State("track-status-key", "data"),
    )
    def show_track_status(current_time, data, shown_key):
        """Race-control state at the replay time; only sent when it changes."""
        replay = REPLAY_CACHE.get((data or {}).get("replay_id"))
        if replay is None or current_time is None:
            key = "none"
            label, modifier = "Nincs betöltött session", "none"
        elif replay.get("track_status") is None:
            key = "unknown"
            label, modifier = "Pályastátusz: nincs versenyirányítási adat (FastF1)", "none"
        else:
            status = track_status_at(replay["track_status"], float(current_time))
            if status is None:
                key, label, modifier = "green", "ZÖLD ZÁSZLÓ · szabad verseny", "green"
            else:
                label, modifier = TRACK_STATUS_DISPLAY[status["status"]]
                if status["ending"]:
                    label += " · befejeződik"
                key = f"{status['status']}|{status['ending']}"
        if key == shown_key:
            return no_update, no_update, no_update
        return label, f"track-status track-status-{modifier}", key

    app.clientside_callback(
        ClientsideFunction(namespace="replay", function_name="sync"),
        Output("replay-anim-anchor", "data"),
        Input("replay-clock-store", "data"),
        Input("replay-interval", "disabled"),
        Input("replay-speed", "value"),
        Input("replay-anim-data", "data"),
    )
