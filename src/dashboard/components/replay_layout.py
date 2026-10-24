"""Replay workspace: tab navigation, playback bar and the tab panels.

Panels: track map + leaderboard (default), strategy, lap times, race progression.
Switching tabs is handled in the browser (assets/navigation.js).
"""
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from src.dashboard.components.progression_layout import progression_layout
from src.dashboard.theme import (
    ACCENT,
    FASTEST_LAP,
    PERSONAL_BEST,
    TABLE_CELL,
    TABLE_HEADER,
    TABLE_STYLE,
    TABLE_ZEBRA,
    TEXT_MUTED,
)

TABS = [
    ("track", "Pálya"),
    ("strategy", "Stratégia"),
    ("laps", "Köridők"),
    ("progress", "Versenyfolyamat"),
]


def _panel(title, body, class_name=""):
    return dbc.Card(
        className=f"replay-panel h-100 {class_name}",
        children=[dbc.CardHeader(title, className="mini-panel-header"), dbc.CardBody(body, className="p-2 p-lg-3")],
    )


def _stores():
    return [
        dcc.Store(id="replay-data-store"),
        dcc.Store(id="replay-clock-store", data=0.0),
        # Car positions ahead of the clock and the clock anchor for the browser animation.
        dcc.Store(id="replay-anim-data"),
        dcc.Store(id="replay-anim-anchor"),
        # Last rendered lap table/chart, so they redraw only after a completed lap.
        dcc.Store(id="replay-view-state"),
        # Last shown track status, so the banner is sent only when it changes.
        dcc.Store(id="track-status-key"),
        dcc.Store(id="live-strategy-state", data={}),
        # Strategy job queued for the worker process, and a poll for its result.
        dcc.Store(id="strategy-request"),
        dcc.Interval(id="strategy-poll", interval=400, n_intervals=0),
        dcc.Interval(id="replay-interval", interval=250, n_intervals=0, disabled=True),
    ]


def _navigation():
    return html.Nav(
        [
            html.Button(label, id=f"nav-{key}", n_clicks=0, className="nav-tab active" if key == "track" else "nav-tab")
            for key, label in TABS
        ],
        className="nav-tabs-bar",
    )


def _playback_bar():
    return html.Div(
        className="control-bar",
        children=[
            html.Div(
                [
                    dbc.Button("−1 kör", id="replay-back-5", n_clicks=0, color="secondary", outline=True, className="replay-control-button"),
                    dbc.Button("Lejátszás", id="replay-play", n_clicks=0, color="danger", className="replay-control-button replay-play"),
                    dbc.Button("+1 kör", id="replay-forward-5", n_clicks=0, color="secondary", outline=True, className="replay-control-button"),
                ],
                className="control-buttons",
            ),
            html.Div(
                [
                    dcc.Slider(
                        id="replay-time-slider",
                        min=1,
                        max=1,
                        value=1,
                        step=1,
                        marks=None,
                        tooltip={"placement": "top", "always_visible": False},
                        className="replay-slider",
                    ),
                    html.Div(
                        [
                            html.Span(id="replay-current-time", children="00:00:00", className="time-current"),
                            html.Span(" / ", className="time-separator"),
                            html.Span(id="replay-total-time", children="00:00:00", className="time-total"),
                        ],
                        className="replay-time-display",
                    ),
                ],
                className="control-timeline",
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id="replay-speed",
                        options=[{"label": "×1 valós idő", "value": 1}, {"label": "×2 gyors", "value": 2}, {"label": "×5 nagyon gyors", "value": 5}],
                        value=1,
                        clearable=False,
                        className="dashboard-dropdown control-speed",
                    ),
                    dcc.Dropdown(
                        id="replay-driver",
                        options=[],
                        value=None,
                        clearable=False,
                        placeholder="Versenyző",
                        className="dashboard-dropdown control-driver",
                    ),
                ],
                className="control-selects",
            ),
        ],
    )


def _track_panel():
    leaderboard = dash_table.DataTable(
        id="leaderboard-table",
        columns=[
            {"name": "Poz.", "id": "Position"},
            {"name": "Versenyző", "id": "Driver"},
            {"name": "Kör", "id": "Lap"},
            {"name": "Köridő", "id": "LastLapTime"},
            {"name": "Gumi", "id": "Compound"},
            {"name": "Kor", "id": "TyreLife"},
            {"name": "Stint", "id": "Stint"},
            {"name": "Állapot", "id": "Status"},
        ],
        data=[],
        page_size=22,
        sort_action="native",
        style_table=TABLE_STYLE,
        style_cell={**TABLE_CELL, "textAlign": "center", "fontSize": "12px", "padding": "7px 6px"},
        style_header=TABLE_HEADER,
        style_data_conditional=[
            TABLE_ZEBRA,
            {"if": {"filter_query": "{Position} = 1"}, "fontWeight": "800", "color": ACCENT},
            # Retired drivers: greyed out at the bottom of the table.
            {"if": {"filter_query": "{Status} contains 'Kiesett'"}, "color": TEXT_MUTED, "fontStyle": "italic"},
            # Holder of the race's fastest lap so far (marked with a stopwatch).
            {"if": {"filter_query": '{Driver} contains "⏱"', "column_id": "Driver"},
             "color": FASTEST_LAP, "fontWeight": "800"},
        ],
    )
    track = dcc.Graph(
        id="replay-track",
        # Only zoom in/out and reset; drag pans, double-click resets.
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": False,
            "doubleClick": "reset",
            "modeBarButtons": [["zoomIn2d", "zoomOut2d", "resetScale2d"]],
        },
        className="dashboard-graph track-map graph-track",
    )
    return dbc.Row(
        className="g-3",
        children=[
            dbc.Col(
                _panel("PÁLYA FELÜLNÉZET", [
                    html.Div("Nincs betöltött session", id="track-status-banner", className="track-status track-status-none"),
                    track,
                ]),
                xs=12, xl=7,
            ),
            dbc.Col(_panel("LEADERBOARD", leaderboard), xs=12, xl=5),
        ],
    )


def _strategy_panel():
    plan_table = dash_table.DataTable(
        id="live-strategy-table",
        columns=[
            {"name": "Terv", "id": "plan"},
            {"name": "Kiállás köre", "id": "pit_laps"},
            {"name": "Delta", "id": "delta"},
        ],
        data=[],
        page_size=8,
        style_table=TABLE_STYLE,
        style_cell=TABLE_CELL,
        style_header=TABLE_HEADER,
        style_data_conditional=[{"if": {"row_index": 0}, "fontWeight": "800", "color": ACCENT}],
    )
    compound_table = dash_table.DataTable(
        id="live-compound-table",
        columns=[
            {"name": "Szett", "id": "set"},
            {"name": "Keverék", "id": "compound"},
            {"name": "Kor", "id": "tyre_age"},
            {"name": "Köridő", "id": "lap_time"},
            {"name": "Δ", "id": "delta"},
        ],
        data=[],
        style_table=TABLE_STYLE,
        style_cell=TABLE_CELL,
        style_header=TABLE_HEADER,
    )
    return _panel(
        "STRATÉGIAI AJÁNLÁS · KÖRÖNKÉNT FRISSÍTVE",
        [
            dbc.Alert(
                id="live-strategy-message",
                color="secondary",
                className="mb-3 strategy-message",
                children="Tölts be egy támogatott verseny-sessiont a körönkénti stratégiai ajánláshoz.",
            ),
            html.Div(
                id="live-strategy-summary",
                className="mb-3",
                children="A javasolt kiállási ablak és gumikeverék itt fog megjelenni.",
            ),
            html.P(
                "A rendszer a replay aktuális körének állapotából újraszámolja a 0, 1 és 2 kiállásos terveket a kétkeverékes szabály figyelembevételével. Ez történeti adatokra épülő modellbecslés, nem valós idejű csapatutasítás.",
                className="text-muted small mb-3",
            ),
            # What-if: each completed pit stop versus the model's plan from a few laps before.
            html.Div(id="live-what-if", className="mb-3"),
            dbc.Row(
                className="g-3",
                children=[
                    dbc.Col([html.Div("ALTERNATÍV TERVEK · DELTA A JAVASOLTHOZ A CÉLIG", className="kpi-label mb-2"), plan_table], xs=12, lg=6),
                    dbc.Col(
                        [
                            html.Div("GUMI KORA ÉS ÁLLAPOTA", className="kpi-label mb-2"),
                            html.Div(id="live-tyre-status", className="tyre-status-box mb-3"),
                            html.Div("KEVERÉKEK TEMPÓJA · BECSÜLT KÖVETKEZŐ KÖR", className="kpi-label mb-2"),
                            compound_table,
                            html.Div(
                                "Az OpenF1 nem közli a még meglévő gumiszetteket, ezért az új szettek becsült ideje látható.",
                                className="text-muted small mt-2",
                            ),
                        ],
                        xs=12, lg=6,
                    ),
                ],
            ),
            dcc.Graph(id="live-pace-dropoff", config={"displaylogo": False, "responsive": True}, className="dashboard-graph graph-medium mt-3"),
        ],
    )


def _laps_panel():
    lap_table = dash_table.DataTable(
        id="replay-table",
        columns=[],
        data=[],
        page_size=10,
        sort_action="native",
        sort_by=[{"column_id": "LapNumber", "direction": "desc"}],
        style_table=TABLE_STYLE,
        style_cell=TABLE_CELL,
        style_header=TABLE_HEADER,
        style_data_conditional=[
            TABLE_ZEBRA,
            # Personal best in green, the race's fastest lap in purple, as on F1 timing screens.
            {"if": {"filter_query": '{IsPersonalBest} = "✓"', "column_id": ["LapTime", "IsPersonalBest"]},
             "color": PERSONAL_BEST, "fontWeight": "800"},
            {"if": {"filter_query": '{LapTime} contains "⏱"', "column_id": "LapTime"},
             "color": FASTEST_LAP, "fontWeight": "800"},
        ],
    )
    return html.Div([
        dbc.Row(
            className="g-3",
            children=[
                dbc.Col(_panel("VERSENYZŐI RÉSZLETEK", html.Div(id="replay-current-info", className="driver-info")), xs=12, lg=4),
                dbc.Col(
                    _panel("KÖRIDŐK ALAKULÁSA", dcc.Graph(id="replay-chart", config={"displaylogo": False, "responsive": True}, className="dashboard-graph graph-medium")),
                    xs=12, lg=8,
                ),
            ],
        ),
        html.Div(_panel("BEFEJEZETT KÖRÖK", lap_table), className="mt-3"),
    ])


def replay_layout():
    panels = {
        "track": _track_panel(),
        "strategy": _strategy_panel(),
        "laps": _laps_panel(),
        "progress": progression_layout(),
    }
    return html.Div(
        id="replay-view",
        # Hidden until a session is loaded (see show_replay_view).
        style={"display": "none"},
        children=[
            *_stores(),
            html.Div(id="replay-session-info", className="session-info-line", children="Nincs betöltött session."),
            _navigation(),
            _playback_bar(),
            *[
                html.Div(
                    panel,
                    id=f"panel-{key}",
                    className="tab-panel",
                    # Only the first tab is visible initially (see assets/navigation.js).
                    style=None if key == "track" else {"display": "none"},
                )
                for key, panel in panels.items()
            ],
        ],
    )
