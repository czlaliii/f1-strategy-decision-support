"""Tab panel comparing the starting grid and the final classification."""
import dash_bootstrap_components as dbc
from dash import dcc, html


def grid_layout():
    return dcc.Loading(
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
                                    className="dashboard-graph graph-tall",
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
                                        className="dashboard-graph graph-short mt-2",
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
        color="#e10600",
    )
