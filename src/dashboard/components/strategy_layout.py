"""Bootstrap-styled race-strategy simulation section."""
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html


def strategy_layout(drivers, race_last_lap):
    return dbc.Card(
        id="race-view",
        className="dashboard-card section-card mb-4",
        children=[
            dbc.CardHeader(
                className="section-heading",
                children=dbc.Row(
                    align="center",
                    children=[
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div("02", className="section-index"),
                                    html.Div(
                                        [
                                            html.H2("Stratégia-szimuláció", className="section-title"),
                                            html.P(
                                                "Hasonlítsd össze a tervezett és az alternatív bokszkiállási időpontokat.",
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
                            dbc.Badge("STRATÉGIAI ELEMZÉS", className="section-badge", color="light"),
                            width="auto",
                        ),
                    ],
                ),
            ),
            dbc.CardBody(
                children=[
                    dbc.Row(
                        className="g-3 align-items-end simulation-controls",
                        children=[
                            dbc.Col(
                                [
                                    dbc.Label("Versenyző", html_for="driver", class_name="field-label"),
                                    dcc.Dropdown(
                                        id="driver",
                                        options=[{"label": f"Versenyző #{driver}", "value": int(driver)} for driver in drivers],
                                        value=drivers[0] if drivers else None,
                                        clearable=False,
                                        className="dashboard-dropdown",
                                    ),
                                ], xs=12, md=6, lg=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Szimuláció kezdőköre", html_for="current-lap", class_name="field-label"),
                                    dbc.Input(id="current-lap", type="number", min=1, max=race_last_lap - 1, value=50, step=1, class_name="dashboard-input"),
                                ], xs=12, md=6, lg=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Tervezett bokszkör", html_for="pit-lap", class_name="field-label"),
                                    dbc.Input(id="pit-lap", type="number", min=2, max=race_last_lap - 1, value=55, step=1, class_name="dashboard-input"),
                                ], xs=12, md=6, lg=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Új gumikeverék", html_for="new-compound", class_name="field-label"),
                                    dcc.Dropdown(
                                        id="new-compound",
                                        options=[{"label": compound, "value": compound} for compound in ["SOFT", "MEDIUM", "HARD"]],
                                        value="HARD",
                                        clearable=False,
                                        className="dashboard-dropdown",
                                    ),
                                ], xs=12, md=6, lg=3,
                            ),
                        ],
                    ),
                    dbc.Row(
                        className="mt-3 align-items-center g-3",
                        children=[
                            dbc.Col(
                                dbc.Button(
                                    [html.Span("▶", className="button-symbol"), " Stratégiák összehasonlítása"],
                                    id="run-simulation",
                                    n_clicks=0,
                                    color="danger",
                                    className="primary-action",
                                ),
                                xs=12, md="auto",
                            ),
                            dbc.Col(html.Div(id="simulation-message", className="inline-status"), xs=12, md=True),
                        ],
                    ),
                    html.Hr(className="section-divider"),
                    html.Div(
                        [
                            html.H3("Eredmények", className="subsection-title"),
                            html.P("A becsült hátralévő és teljes versenyidő alapján hasonlíthatók össze a stratégiák.", className="subsection-description"),
                        ],
                        className="subsection-heading",
                    ),
                    dash_table.DataTable(
                        id="results-table",
                        columns=[
                            {"name": "Stratégia", "id": "strategy_id"},
                            {"name": "Bokszkiállás köre", "id": "pit_laps"},
                            {"name": "Gumikeverékek", "id": "compounds"},
                            {"name": "Hátralévő idő", "id": "predicted_remaining_race_time"},
                            {"name": "Becsült teljes idő", "id": "predicted_total_race_time"},
                        ],
                        data=[],
                        sort_action="native",
                        page_size=10,
                        style_table={"overflowX": "auto", "minWidth": "100%"},
                        style_cell={"padding": "12px 14px", "textAlign": "left", "fontFamily": "inherit", "fontSize": "14px", "border": "none", "borderBottom": "1px solid #e9edf2", "minWidth": "110px", "maxWidth": "260px", "whiteSpace": "normal", "height": "auto"},
                        style_header={"fontWeight": "700", "backgroundColor": "#f4f6f9", "color": "#4a5568", "border": "none", "borderBottom": "1px solid #dfe4ea", "textTransform": "uppercase", "fontSize": "11px", "letterSpacing": "0.05em"},
                        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafbfc"}],
                    ),
                    html.Div(
                        [
                            html.H3("Becsült kumulált versenyidő", className="subsection-title mt-4"),
                            html.P("A görbék a stratégiák várható időalakulását mutatják körönként.", className="subsection-description"),
                        ],
                        className="subsection-heading",
                    ),
                    dcc.Graph(id="strategy-chart", config={"displaylogo": False, "responsive": True}, className="dashboard-graph"),
                ]
            ),
        ],
    )
