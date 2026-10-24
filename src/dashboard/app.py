"""Application entry point for the F1 strategy decision-support dashboard."""
import dash_bootstrap_components as dbc
from dash import Dash

from src.dashboard.callbacks.models_callbacks import register_models_callbacks
from src.dashboard.callbacks.navigation_callbacks import register_navigation_callbacks
from src.dashboard.callbacks.progression_callbacks import register_progression_callbacks
from src.dashboard.callbacks.replay_callbacks import register_replay_callbacks
from src.dashboard.callbacks.replay_strategy_callbacks import (
    register_replay_strategy_callbacks,
)
from src.dashboard.callbacks.session_callbacks import register_session_callbacks
from src.dashboard.config import YEARS
from src.dashboard.layout import build_layout
from src.dashboard.theme import register_plotly_template


def create_app():
    """Build the Dash app and register callbacks."""
    register_plotly_template()
    # update_title=None: the browser tab title stays fixed while callbacks run.
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], update_title=None)
    app.title = "F1 Strategy Simulator"
    app.layout = build_layout(YEARS)
    register_session_callbacks(app)
    register_replay_callbacks(app)
    register_replay_strategy_callbacks(app)
    register_progression_callbacks(app)
    register_navigation_callbacks(app)
    register_models_callbacks(app)
    return app
app = create_app()
if __name__ == "__main__":
    app.run(debug=True)
