"""Compose the dashboard: top bar, race-weekend selector and the tabbed replay."""
import dash_bootstrap_components as dbc
from dash import html

from src.dashboard.components.replay_layout import replay_layout
from src.dashboard.components.session_selector import session_selector_layout


def build_layout(years):
    """Dark page shell; data-bs-theme switches Bootstrap components to dark mode."""
    return html.Div(
        className="app-shell",
        **{"data-bs-theme": "dark"},
        children=[
            html.Header(
                className="app-topbar",
                children=dbc.Container(
                    fluid=True,
                    className="dashboard-shell d-flex align-items-center gap-3",
                    children=[
                        html.Span(className="brand-mark"),
                        html.Div(
                            [
                                html.Div(["F1 ", html.Span("STRATEGY", className="brand-accent"), " DASHBOARD"], className="brand-title"),
                                html.Div("Történeti versenyadatok · visszajátszás · stratégiai döntéstámogatás", className="brand-subtitle"),
                            ]
                        ),
                    ],
                ),
            ),
            dbc.Container(
                fluid=True,
                className="dashboard-shell py-3 py-lg-4",
                children=[
                    html.Main([session_selector_layout(years), replay_layout()]),
                    html.Footer(
                        "F1 Strategy Decision Support · Historical data mode",
                        className="dashboard-footer mt-4 pt-3",
                    ),
                ],
            ),
        ],
    )
