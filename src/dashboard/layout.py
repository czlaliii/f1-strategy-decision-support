"""Compose the dashboard from clearly separated functional sections."""
import dash_bootstrap_components as dbc
from dash import html

from src.dashboard.components.grid_layout import grid_layout
from src.dashboard.components.replay_layout import replay_layout
from src.dashboard.components.session_selector import session_selector_layout


def build_layout(years):
    """Build the top-level page without changing any callback component IDs."""
    return dbc.Container(
        fluid=True,
        className="dashboard-shell py-4 py-lg-5",
        children=[
            html.Header(
                className="dashboard-hero mb-4 mb-lg-5",
                children=[
                    html.Div(
                        className="hero-kicker",
                        children=[
                            html.Span(className="hero-kicker-dot"),
                            "RACE ENGINEERING · DECISION SUPPORT",
                        ],
                    ),
                    html.H1("F1 Strategy Dashboard", className="dashboard-title"),
                    html.P(
                        "Történelmi versenyadatok elemzése, körönkénti visszajátszás "
                        "és bokszstratégiák összehasonlítása egy helyen.",
                        className="dashboard-subtitle mb-0",
                    ),
                ],
            ),
            html.Main(
                className="dashboard-sections",
                children=[
                    session_selector_layout(years),
                    replay_layout(),
                    grid_layout(),
                ],
            ),
            html.Footer(
                "F1 Strategy Decision Support · Historical data mode",
                className="dashboard-footer mt-4 pt-3",
            ),
        ],
    )
