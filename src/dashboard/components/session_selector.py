"""Race-weekend selector: season, event, session and the load button."""
import dash_bootstrap_components as dbc
from dash import dcc, html


def session_selector_layout(years):
    return dbc.Card(
        className="dashboard-card mb-3",
        children=dbc.CardBody(
            [
                html.Div(
                    [
                        html.H2("Versenyhétvége", className="section-title"),
                        html.Span("Válaszd ki az eseményt, majd töltsd be a sessiont.", className="section-description"),
                    ],
                    className="section-heading-inline mb-3",
                ),
                dbc.Row(
                    className="g-3 align-items-end",
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
                            xs=12, sm=4, lg=2,
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
                            xs=12, sm=8, lg=4,
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
                            xs=12, sm=6, lg=3,
                        ),
                        dbc.Col(
                            dbc.Button(
                                [html.Span("↓", className="button-symbol"), " Session betöltése"],
                                id="load-replay",
                                n_clicks=0,
                                color="danger",
                                className="primary-action w-100",
                            ),
                            xs=12, sm=6, lg=3,
                        ),
                    ],
                ),
                html.Div(
                    [
                        html.Div(id="calendar-message", className="inline-status"),
                        # Loading can take a while; the spinner shows until the message updates.
                        dcc.Loading(
                            html.Div(id="replay-message", className="inline-status"),
                            type="circle",
                            color="#e10600",
                        ),
                    ],
                    className="selector-status mt-3",
                ),
            ],
            className="p-3 p-lg-4",
        ),
    )
