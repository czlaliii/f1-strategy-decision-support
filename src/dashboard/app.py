"""Application entry point for the F1 strategy decision-support dashboard."""
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash

from src.dashboard.callbacks.replay_callbacks import register_replay_callbacks
from src.dashboard.callbacks.replay_strategy_callbacks import (
    register_replay_strategy_callbacks,
)
from src.dashboard.callbacks.session_callbacks import register_session_callbacks
from src.dashboard.callbacks.strategy_callbacks import register_strategy_callbacks
from src.dashboard.config import DATASET_PATH, RACE_STATE_PATH, YEARS
from src.dashboard.layout import build_layout
from src.models.gradient_boosting_predictor import load_gradient_boosting_model


def create_app():
    """Build the Dash app, load thesis data and register callbacks."""
    dataset = pd.read_parquet(DATASET_PATH)
    race_state = pd.read_parquet(RACE_STATE_PATH)
    predictor = load_gradient_boosting_model()
    race_last_lap = int(race_state["lap_number"].max())
    drivers = (
        dataset[["driver_number"]]
        .drop_duplicates()
        .sort_values("driver_number")["driver_number"]
        .tolist()
    )
    # update_title=None: the browser tab title stays fixed while callbacks run.
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], update_title=None)
    app.title = "F1 Strategy Simulator"
    app.layout = build_layout(YEARS, drivers, race_last_lap)
    register_session_callbacks(app)
    register_replay_callbacks(app)
    register_replay_strategy_callbacks(app)
    register_strategy_callbacks(
        app,
        dataset=dataset,
        race_state=race_state,
        predictor=predictor,
        race_last_lap=race_last_lap,
    )
    return app
app = create_app()
if __name__ == "__main__":
    app.run(debug=True)
