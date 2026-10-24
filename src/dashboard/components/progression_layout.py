"""Tab panel with the lap-by-lap race progression graph."""
import dash_bootstrap_components as dbc
from dash import dcc, html


def progression_layout():
    return dbc.Card(
        className="replay-panel",
        children=[
            dbc.CardHeader("VERSENYFOLYAMAT · POZÍCIÓK KÖRRŐL KÖRRE", className="mini-panel-header"),
            dbc.CardBody(
                [
                    dcc.Store(id="progression-key"),
                    dcc.Graph(
                        id="race-progression",
                        config={"displaylogo": False, "responsive": True, "displayModeBar": False},
                        className="dashboard-graph graph-progression",
                    ),
                    html.Div(
                        [
                            html.Span([html.Span(className="legend-line"), "a csapat második versenyzője szaggatott"]),
                            html.Span([html.Span(className="legend-pit"), "kiállás (in-lap)"]),
                            html.Span(["✕ kiesés"]),
                            html.Span([html.Span(className="legend-band legend-band-sc"), "SC"]),
                            html.Span([html.Span(className="legend-band legend-band-vsc"), "VSC"]),
                            html.Span([html.Span(className="legend-band legend-band-red"), "piros zászló"]),
                        ],
                        className="progression-legend",
                    ),
                ],
                className="p-2 p-lg-3",
            ),
        ],
    )
