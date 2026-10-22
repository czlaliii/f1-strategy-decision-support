"""Callbacks for the event calendar and session selection."""
import fastf1
import pandas as pd
from dash.dependencies import Input, Output

from src.dashboard.config import SESSION_CODES


def register_session_callbacks(app):

    @app.callback(
        Output("race-view", "style"),
        Output("replay-view", "style"),
        Input("session-name", "value"),
    )
    def switch_session_view(session_name):
        return (
            {"display": "block"}
            if session_name == "R"
            else {"display": "none"},
            {"display": "block"},
        )

    @app.callback(
        Output("event-name", "options"),
        Output("event-name", "value"),
        Output("calendar-message", "children"),
        Input("season-year", "value"),
    )
    def update_events(year):
        if year is None:
            return [], None, "Válassz évet."
        try:
            schedule = fastf1.get_event_schedule(
                int(year),
                include_testing=False,
            )
            events = schedule["EventName"].dropna().tolist()
            options = [
                {"label": event, "value": event}
                for event in events
            ]
            default_event = events[0] if events else None
            return (
                options,
                default_event,
                f"{year}: {len(events)} versenyhétvége betöltve.",
            )
        except Exception as error:
            return (
                [],
                None,
                f"A versenynaptár nem tölthető be: {error}",
            )

    @app.callback(
        Output("session-name", "options"),
        Output("session-name", "value"),
        Input("season-year", "value"),
        Input("event-name", "value"),
    )
    def update_sessions(year, event_name):
        if year is None or event_name is None:
            return [], None
        try:
            schedule = fastf1.get_event_schedule(
                int(year),
                include_testing=False,
            )
            event_rows = schedule.loc[
                schedule["EventName"] == event_name
            ]
            if event_rows.empty:
                return [], None
            event = event_rows.iloc[0]
            sessions = []
            for number in range(1, 6):
                column = f"Session{number}"
                if column not in schedule.columns:
                    continue
                session_name = event[column]
                if pd.isna(session_name):
                    continue
                session_name = str(session_name).strip()
                code = SESSION_CODES.get(session_name)
                if code is not None:
                    sessions.append({
                        "label": f"{code} — {session_name}",
                        "value": code,
                    })
            default_session = (
                "R" if any(s["value"] == "R" for s in sessions)
                else sessions[-1]["value"] if sessions
                else None
            )
            return sessions, default_session
        except Exception:
            return [], None
