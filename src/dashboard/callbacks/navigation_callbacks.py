"""Tab navigation of the replay workspace (runs in the browser)."""
from dash import ClientsideFunction
from dash.dependencies import Input, Output

from src.dashboard.components.replay_layout import TABS


def register_navigation_callbacks(app):
    keys = [key for key, _ in TABS]
    app.clientside_callback(
        ClientsideFunction(namespace="nav", function_name="switchTab"),
        *[Output(f"panel-{key}", "style") for key in keys],
        *[Output(f"nav-{key}", "className") for key in keys],
        *[Input(f"nav-{key}", "n_clicks") for key in keys],
        prevent_initial_call=True,
    )
