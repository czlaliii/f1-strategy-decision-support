"""Section comparing starting grid and final classification."""
import dash_bootstrap_components as dbc
from dash import dcc, html


def grid_layout():
    return dbc.Card(
        id="grid-view",
        className="dashboard-card section-card mb-4",
        # Hidden until a session is loaded (see show_replay_view).
        style={"display": "none"},
        children=[
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
                                            html.H2("Rajt–cél összefüggés", className="section-title"),
                                            html.P(
                                                "Mennyire határozza meg a rajtpozíció a végeredményt ezen a versenyen és az összes helyi versenyen.",
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
                            dbc.Badge("STATISZTIKA", className="section-badge", color="light"),
                            width="auto",
                        ),
                    ],
                ),
            ),
            dbc.CardBody(
                dcc.Loading(
                    dbc.Row(
                        className="g-3",
                        children=[
                            dbc.Col(
                                dbc.Card(
                                    className="replay-panel h-100",
                                    children=[
                                        dbc.CardHeader("RAJTPOZÍCIÓ ÉS VÉGEREDMÉNY", className="mini-panel-header"),
                                        dbc.CardBody(
                                            dcc.Graph(
                                                id="grid-scatter",
                                                config={"displaylogo": False, "responsive": True, "displayModeBar": False},
                                                className="dashboard-graph",
                                            ),
                                            className="p-2",
                                        ),
                                    ],
                                ),
                                xs=12, xl=7,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    className="replay-panel h-100",
                                    children=[
                                        dbc.CardHeader("RANGKORRELÁCIÓ", className="mini-panel-header"),
                                        dbc.CardBody(
                                            [
                                                html.Div(id="grid-stats"),
                                                dcc.Graph(
                                                    id="grid-trend",
                                                    config={"displaylogo": False, "responsive": True, "displayModeBar": False},
                                                    className="dashboard-graph mt-2",
                                                ),
                                            ],
                                            className="p-3",
                                        ),
                                    ],
                                ),
                                xs=12, xl=5,
                            ),
                        ],
                    ),
                    type="circle",
                    color="#d62839",
                ),
                className="p-3 p-lg-4",
            ),
        ],
    )
