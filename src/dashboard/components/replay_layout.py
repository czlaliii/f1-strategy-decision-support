"""Bootstrap-styled historical session replay section."""
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html


def replay_layout():
    return dbc.Card(
        id="replay-view",
        className="dashboard-card section-card mb-4",
        style={"display": "none"},
        children=[
            dcc.Store(id="replay-data-store"),
            dcc.Store(id="replay-clock-store", data=0.0),
            # Car positions ahead of the clock and the clock anchor for the browser animation.
            dcc.Store(id="replay-anim-data"),
            dcc.Store(id="replay-anim-anchor"),
            # Last rendered lap table/chart, so they redraw only after a completed lap.
            dcc.Store(id="replay-view-state"),
            dbc.CardHeader(
                className="section-heading",
                children=dbc.Row(
                    align="center",
                    children=[
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div("03", className="section-index"),
                                    html.Div(
                                        [
                                            html.H2("Session-visszajátszás", className="section-title"),
                                            html.P(
                                                "Kövesd a pályán zajló eseményeket, a sorrendet és az egyes versenyzők köridejét.",
                                                className="section-description mb-0",
                                            ),
                                        ]
                                    ),
                                ],
                                className="section-title-wrap",
                            ),
                            width=True,
                        ),
                        dbc.Col(
                            dbc.Button(
                                [html.Span("↓", className="button-symbol"), " Session betöltése"],
                                id="load-replay",
                                n_clicks=0,
                                color="dark",
                                outline=True,
                                className="secondary-action",
                            ),
                            width="auto",
                        ),
                    ],
                    className="g-3",
                ),
            ),
            dbc.CardBody(
                children=[
                    html.Div(id="replay-message", className="inline-status mb-3"),
                    dbc.Alert(id="replay-session-info", color="light", className="session-info-banner mb-3", children="Nincs betöltött session."),
                    dcc.Store(id="live-strategy-state", data={}),
                    # Strategy job queued for the worker process, and a poll for its result.
                    dcc.Store(id="strategy-request"),
                    dcc.Interval(id="strategy-poll", interval=400, n_intervals=0),
                    dbc.Card(
                        className="replay-panel mb-3 live-strategy-panel",
                        children=[
                            dbc.CardHeader(
                                "STRATÉGIAI AJÁNLÁS · KÖRÖNKÉNT FRISSÍTVE",
                                className="mini-panel-header",
                            ),
                            dbc.CardBody(
                                [
                                    dbc.Alert(
                                        id="live-strategy-message",
                                        color="info",
                                        className="mb-3",
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
                                    dbc.Row(
                                        className="g-3",
                                        children=[
                                            dbc.Col(
                                                [
                                                    html.Div("ALTERNATÍV TERVEK · DELTA A JAVASOLTHOZ A CÉLIG", className="kpi-label mb-2"),
                                                    dash_table.DataTable(
                                                        id="live-strategy-table",
                                                        columns=[
                                                            {"name": "Terv", "id": "plan"},
                                                            {"name": "Kiállás köre", "id": "pit_laps"},
                                                            {"name": "Delta", "id": "delta"},
                                                        ],
                                                        data=[],
                                                        page_size=8,
                                                        style_table={"overflowX": "auto"},
                                                style_cell={"padding": "8px 10px", "textAlign": "left", "fontFamily": "inherit", "fontSize": "13px", "border": "none", "borderBottom": "1px solid #e9edf2", "whiteSpace": "normal"},
                                                style_header={"fontWeight": "700", "backgroundColor": "#f4f6f9", "color": "#4a5568", "border": "none", "fontSize": "11px"},
                                                        style_data_conditional=[{"if": {"row_index": 0}, "fontWeight": "700", "color": "#a91f2d"}],
                                                    ),
                                                ],
                                                xs=12, lg=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Div("GUMI KORA ÉS ÁLLAPOTA", className="kpi-label mb-2"),
                                                    html.Div(id="live-tyre-status", className="tyre-status-box mb-3"),
                                                    html.Div("KEVERÉKEK TEMPÓJA · BECSÜLT KÖVETKEZŐ KÖR", className="kpi-label mb-2"),
                                                    dash_table.DataTable(
                                                        id="live-compound-table",
                                                        columns=[
                                                            {"name": "Szett", "id": "set"},
                                                            {"name": "Keverék", "id": "compound"},
                                                            {"name": "Kor", "id": "tyre_age"},
                                                            {"name": "Köridő", "id": "lap_time"},
                                                            {"name": "Δ", "id": "delta"},
                                                        ],
                                                        data=[],
                                                        style_table={"overflowX": "auto"},
                                                style_cell={"padding": "8px 10px", "textAlign": "left", "fontFamily": "inherit", "fontSize": "13px", "border": "none", "borderBottom": "1px solid #e9edf2", "whiteSpace": "normal"},
                                                style_header={"fontWeight": "700", "backgroundColor": "#f4f6f9", "color": "#4a5568", "border": "none", "fontSize": "11px"},
                                                    ),
                                                    html.Div(
                                                        "Az OpenF1 nem közli a még meglévő gumiszetteket, ezért az új szettek becsült ideje látható.",
                                                        className="text-muted small mt-2",
                                                    ),
                                                ],
                                                xs=12, lg=6,
                                            ),
                                        ],
                                    ),
                                    dcc.Graph(
                                        id="live-pace-dropoff",
                                        config={"displaylogo": False, "responsive": True},
                                        className="dashboard-graph mt-3",
                                    ),
                                ],
                                className="p-3 p-lg-4",
                            ),
                        ],
                    ),
                    dbc.Row(
                        className="g-3",
                        children=[
                            dbc.Col(
                                dbc.Card(
                                    className="replay-panel h-100",
                                    children=[
                                        dbc.CardHeader("POZÍCIÓ A PÁLYÁN", className="mini-panel-header"),
                                        dbc.CardBody(dcc.Graph(id="replay-track", config={"displaylogo": False, "responsive": True}, className="dashboard-graph"), className="p-2"),
                                    ],
                                ),
                                xs=12, xl=8,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    className="replay-panel h-100",
                                    children=[
                                        dbc.CardHeader("LEADERBOARD", className="mini-panel-header"),
                                        dbc.CardBody(
                                            dash_table.DataTable(
                                                id="leaderboard-table",
                                                columns=[
                                                    {"name": "Poz.", "id": "Position"},
                                                    {"name": "Versenyző", "id": "Driver"},
                                                    {"name": "Kör", "id": "Lap"},
                                                    {"name": "Köridő", "id": "LastLapTime"},
                                                    {"name": "Gumi", "id": "Compound"},
                                                    {"name": "Gumikor (kör)", "id": "TyreLife"},
                                                    {"name": "Stint", "id": "Stint"},
                                                ],
                                                data=[],
                                                page_size=22,
                                                sort_action="native",
                                                style_table={"overflowX": "auto"},
                                                style_cell={"padding": "8px", "textAlign": "center", "fontFamily": "inherit", "fontSize": "12px", "border": "none", "borderBottom": "1px solid #e9edf2"},
                                                style_header={"fontWeight": "700", "backgroundColor": "#f4f6f9", "color": "#4a5568", "border": "none", "fontSize": "11px"},
                                                style_data_conditional=[{"if": {"filter_query": "{Position} = 1"}, "fontWeight": "bold", "color": "#bd2638"}],
                                            ),
                                            className="p-2",
                                        ),
                                    ],
                                ),
                                xs=12, xl=4,
                            ),
                        ],
                    ),
                    dbc.Card(
                        className="replay-controls-card mt-3",
                        children=dbc.CardBody(
                            [
                                dbc.Row(
                                    className="g-3 align-items-center",
                                    children=[
                                        dbc.Col(
                                            [
                                                html.Div("LEJÁTSZÁS VEZÉRLÉSE", className="mini-panel-header mb-2"),
                                                html.Div(
                                                    [
                                                        dbc.Button("−1 kör", id="replay-back-5", n_clicks=0, color="secondary", outline=True, className="replay-control-button"),
                                                        dbc.Button("Lejátszás", id="replay-play", n_clicks=0, color="danger", className="replay-control-button"),
                                                        dbc.Button("+1 kör", id="replay-forward-5", n_clicks=0, color="secondary", outline=True, className="replay-control-button"),
                                                    ],
                                                    className="replay-button-row",
                                                ),
                                            ],
                                            xs=12, lg=7,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Label("Lejátszási sebesség", html_for="replay-speed", class_name="field-label"),
                                                dcc.Dropdown(
                                                    id="replay-speed",
                                                    options=[{"label": "×1 · Valós idő", "value": 1}, {"label": "×2 · Gyors", "value": 2}, {"label": "×5 · Nagyon gyors", "value": 5}],
                                                    value=1,
                                                    clearable=False,
                                                    className="dashboard-dropdown",
                                                ),
                                            ],
                                            xs=12, lg=5,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Lejátszás pozíciója", html_for="replay-time-slider", class_name="field-label mt-3"),
                                        dcc.Slider(
                                            id="replay-time-slider",
                                            min=1,
                                            max=1,
                                            value=1,
                                            step=1,
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            className="replay-slider",
                                        ),
                                    ],
                                    className="replay-timeline",
                                ),
                                html.Div(
                                    [
                                        html.Span(id="replay-current-time", children="00:00:00", className="time-current"),
                                        html.Span(" / ", className="time-separator"),
                                        html.Span(id="replay-total-time", children="00:00:00", className="time-total"),
                                    ],
                                    className="replay-time-display mt-2",
                                ),
                            ],
                            className="p-3 p-lg-4",
                        ),
                    ),
                    dcc.Interval(id="replay-interval", interval=250, n_intervals=0, disabled=True),
                    dbc.Row(
                        className="g-3 mt-1",
                        children=[
                            dbc.Col(
                                dbc.Card(
                                    className="replay-panel h-100",
                                    children=[
                                        dbc.CardHeader("VERSENYZŐI RÉSZLETEK", className="mini-panel-header"),
                                        dbc.CardBody(
                                            [
                                                dbc.Label("Versenyző kiválasztása", html_for="replay-driver", class_name="field-label"),
                                                dcc.Dropdown(id="replay-driver", options=[], value=None, clearable=False, placeholder="Válassz versenyzőt", className="dashboard-dropdown"),
                                                html.Div(id="replay-current-info", className="driver-info mt-3"),
                                            ],
                                            className="p-3",
                                        ),
                                    ],
                                ),
                                xs=12, lg=4,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    className="replay-panel h-100",
                                    children=[
                                        dbc.CardHeader("KÖRIDŐK ALAKULÁSA", className="mini-panel-header"),
                                        dbc.CardBody(dcc.Graph(id="replay-chart", config={"displaylogo": False, "responsive": True}, className="dashboard-graph"), className="p-2"),
                                    ],
                                ),
                                xs=12, lg=8,
                            ),
                        ],
                    ),
                    dbc.Card(
                        className="replay-panel mt-3",
                        children=[
                            dbc.CardHeader("BEFEJEZETT KÖRÖK", className="mini-panel-header"),
                            dbc.CardBody(
                                dash_table.DataTable(
                                    id="replay-table",
                                    columns=[],
                                    data=[],
                                    page_size=10,
                                    sort_action="native",
                                    sort_by=[{"column_id": "LapNumber", "direction": "desc"}],
                                    style_table={"overflowX": "auto"},
                                    style_cell={"padding": "10px 12px", "textAlign": "left", "fontFamily": "inherit", "fontSize": "13px", "border": "none", "borderBottom": "1px solid #e9edf2"},
                                    style_header={"fontWeight": "700", "backgroundColor": "#f4f6f9", "color": "#4a5568", "border": "none", "fontSize": "11px"},
                                    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafbfc"}],
                                ),
                                className="p-2",
                            ),
                        ],
                    ),
                ]
            ),
        ],
    )
