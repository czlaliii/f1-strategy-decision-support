"""Bootstrap-styled controls for choosing a season, event and session."""
import dash_bootstrap_components as dbc
from dash import dcc, html


def session_selector_layout(years):
    return dbc.Card(
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
                                    html.Div("01", className="section-index"),
                                    html.Div(
                                        [
                                            html.H2("Versenyhétvége", className="section-title"),
                                            html.P(
                                                "Válaszd ki, melyik esemény adataival dolgozzunk.",
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
                            dbc.Badge("ADATVÁLASZTÁS", className="section-badge", color="light"),
                            width="auto",
                        ),
                    ],
                ),
            ),
            dbc.CardBody(
                children=[
                    dbc.Row(
                        className="g-3",
                        children=[
                            dbc.Col(
                                [
                                    dbc.Label("Szezon", html_for="season-year", class_name="field-label"),
                                    dcc.Dropdown(
                                        id="season-year",
                                        options=[{"label": str(year), "value": year} for year in years],
                                        value=2026,
                                        clearable=False,
                                        className="dashboard-dropdown",
                                    ),
                                ],
                                xs=12, md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Nagydíj", html_for="event-name", class_name="field-label"),
                                    dcc.Dropdown(
                                        id="event-name",
                                        options=[],
                                        value=None,
                                        clearable=False,
                                        placeholder="Válassz nagydíjat",
                                        className="dashboard-dropdown",
                                    ),
                                ],
                                xs=12, md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Session", html_for="session-name", class_name="field-label"),
                                    dcc.Dropdown(
                                        id="session-name",
                                        options=[],
                                        value=None,
                                        clearable=False,
                                        placeholder="Válassz sessiont",
                                        className="dashboard-dropdown",
                                    ),
                                ],
                                xs=12, md=4,
                            ),
                        ],
                    ),
                    html.Div(id="calendar-message", className="inline-status mt-3"),
                    dbc.Row(
                        className="g-3 mt-1 align-items-center",
                        children=[
                            dbc.Col(
                                dbc.Button(
                                    [html.Span("↓", className="button-symbol"), " Session betöltése"],
                                    id="load-replay",
                                    n_clicks=0,
                                    color="danger",
                                    className="primary-action",
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                # Loading can take a while; the spinner shows until the message updates.
                                dcc.Loading(
                                    html.Div(id="replay-message", className="inline-status"),
                                    type="circle",
                                    color="#d62839",
                                ),
                                width=True,
                            ),
                        ],
                    ),
                ]
            ),
        ],
    )
