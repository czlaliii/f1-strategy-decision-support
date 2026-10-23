"""Connect replay lap progression to the historical strategy advisor."""
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, no_update
from dash.dependencies import Input, Output, State

from src.dashboard.config import PIT_LOSS_SECONDS
from src.dashboard.replay_player import format_lap_time
from src.dashboard.replay_service import REPLAY_CACHE, time_to_lap
from src.features.race_dataset import RaceDataError, load_race_dataset
from src.models.expanding_window_model import (
    InsufficientHistoryError,
    fill_with_training_medians,
    load_or_train_model,
)
from src.simulator import strategy_advisor as advisor

SUPPORTED_COMPOUNDS = advisor.DRY_COMPOUNDS
COMPOUND_COLORS = {"SOFT": "#d62839", "MEDIUM": "#e0b100", "HARD": "#8a94a6"}
COMPOUND_NAMES = {"SOFT": "Soft", "MEDIUM": "Medium", "HARD": "Hard"}


def _empty_figure(message=""):
    figure = go.Figure()
    figure.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{"text": message, "showarrow": False, "font": {"color": "#697586"}}] if message else [],
        plot_bgcolor="#ffffff",
    )
    return figure


def _empty_outputs(message, key=None):
    return (
        message,
        html.Div("Nincs még kiszámítható ajánlás.", className="text-muted"),
        [],
        "",
        _empty_figure(),
        [],
        {"key": key, "status": message},
    )


def _no_change():
    return (no_update,) * 7


def _compound_pill(compound):
    return html.Span(
        COMPOUND_NAMES.get(compound, compound),
        className="compound-pill",
        style={"backgroundColor": COMPOUND_COLORS.get(compound, "#697586")},
    )


def _kpi(label, value, sub=None):
    return html.Div(
        [
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(sub, className="kpi-sub") if sub else None,
        ],
        className="kpi-tile",
    )


def _signed(seconds, digits=1):
    return f"{seconds:+.{digits}f} s".replace("-", "−")


def _summary_tiles(recommendation, current_stint):
    best = recommendation.best
    strategy = best["strategy"]
    if recommendation.pit_window:
        start, end = recommendation.pit_window
        window = f"{start}–{end}. kör" if start != end else f"{start}. kör"
        optimum = ", ".join(f"{lap}." for lap in strategy.pit_laps)
        window_sub = f"optimum: {optimum} kör"
        next_compound = _compound_pill(strategy.compounds[1])
    else:
        window, window_sub = "Nincs kiállás", "a jelenlegi szettel a célig"
        next_compound = html.Span("Marad a jelenlegi", className="kpi-muted")

    stops = len(strategy.pit_laps)
    total_stops = (int(current_stint) - 1 if pd.notna(current_stint) else 0) + stops
    alternatives = recommendation.plans.iloc[1:]
    if alternatives.empty:
        delta_value, delta_sub = "–", "nincs alternatíva"
    else:
        runner_up = alternatives.iloc[0]
        delta_value = f"{runner_up['delta']:.1f} s előny"
        delta_sub = f"a célig vs. {runner_up['plan']}"

    return html.Div(
        [
            _kpi("Javasolt kiállási ablak", window, window_sub),
            _kpi("Következő gumikeverék", next_compound, best["plan"]),
            _kpi("Tervezett kiállások", f"{stops} hátralévő", f"a futamon összesen {total_stops}"),
            _kpi("Delta a következő legjobb tervhez", delta_value, delta_sub),
        ],
        className="kpi-grid",
    )


def _plan_rows(recommendation, limit=8):
    rows = []
    for _, plan in recommendation.plans.head(limit).iterrows():
        rows.append({
            "plan": plan["plan"],
            "pit_laps": ", ".join(f"{lap}." for lap in plan["pit_laps"]) or "–",
            "delta": "javasolt" if plan["delta"] == 0 else _signed(plan["delta"]),
        })
    return rows


def _tyre_status(compound, tyre_age, wear, tyre_life):
    life = tyre_life.get(compound)
    wear_bar = (
        dbc.Progress(
            value=wear,
            label=f"{wear:.0f}%",
            color="danger" if wear >= 85 else "warning" if wear >= 60 else "success",
            className="tyre-wear-bar",
        )
        if wear is not None
        else html.Div("Nincs becslés.", className="text-muted small")
    )
    return html.Div(
        [
            html.Div([_compound_pill(compound), html.Span(f"{int(tyre_age)} kör a szetten", className="ms-2 fw-semibold")]),
            html.Div("Becsült elhasználódás", className="kpi-label mt-3"),
            wear_bar,
            html.Div(
                f"Becslés: a {COMPOUND_NAMES.get(compound, compound)} keverék tipikus élettartama "
                f"~{life:.0f} kör a korábbi versenyek alapján." if life else "",
                className="text-muted small mt-2",
            ),
        ]
    )


def _dropoff_figure(model, compound, tyre_age, tyre_life, observed):
    figure = go.Figure()
    for name in SUPPORTED_COMPOUNDS:
        max_age = max(tyre_life.get(name, 30), tyre_age + 5)
        ages = np.arange(0, int(max_age) + 1)
        current = name == compound
        figure.add_trace(go.Scatter(
            x=ages,
            y=advisor.degradation_curve(model, name, ages),
            mode="lines",
            name=f"{COMPOUND_NAMES[name]} (modell)",
            line={"color": COMPOUND_COLORS[name], "width": 3 if current else 1.5, "dash": "solid" if current else "dot"},
            hovertemplate="%{x}. kör a szetten: +%{y:.2f} s<extra></extra>",
        ))
    if not observed.empty:
        figure.add_trace(go.Scatter(
            x=observed["tyre_age"],
            y=observed["dropoff"],
            mode="markers",
            name="Megfigyelt körök (igazítva)",
            marker={"color": COMPOUND_COLORS.get(compound, "#17212f"), "size": 7, "line": {"width": 1, "color": "#17212f"}},
            hovertemplate="%{x}. kör a szetten: %{y:+.2f} s<extra></extra>",
        ))
    figure.add_vline(x=tyre_age, line={"color": "#17212f", "dash": "dash", "width": 1})
    figure.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title={"text": "Köridő-visszaesés a friss szetthez képest", "font": {"size": 13}},
        xaxis={"title": "Gumikor (kör)", "gridcolor": "#eef1f5"},
        yaxis={"title": "Lassulás (s)", "gridcolor": "#eef1f5", "zeroline": True},
        legend={"orientation": "h", "y": -0.25, "font": {"size": 11}},
        plot_bgcolor="#ffffff",
    )
    return figure


def _compound_rows(pace):
    rows = []
    for _, row in pace.iterrows():
        rows.append({
            "set": row["set"],
            "compound": COMPOUND_NAMES.get(row["compound"], row["compound"]),
            "tyre_age": int(row["tyre_age"]),
            "lap_time": format_lap_time(row["lap_time"]),
            "delta": "–" if row["set"] == "Jelenlegi szett" else _signed(row["delta_to_current"], 2),
        })
    return rows


# Per-race dataset and model, kept in memory after the first lap.
_RACE_RESOURCES: dict[tuple[str, int], tuple] = {}


def _race_resources(race_folder, session_key):
    key = (race_folder, int(session_key))
    if key not in _RACE_RESOURCES:
        dataset = load_race_dataset(race_folder, int(session_key))
        model = load_or_train_model(race_folder)
        _RACE_RESOURCES[key] = (dataset, model)
    return _RACE_RESOURCES[key]


def register_replay_strategy_callbacks(app):
    """Show the pit recommendation and tyre insights, refreshed once per replay lap."""

    @app.callback(
        Output("live-strategy-message", "children"),
        Output("live-strategy-summary", "children"),
        Output("live-strategy-table", "data"),
        Output("live-tyre-status", "children"),
        Output("live-pace-dropoff", "figure"),
        Output("live-compound-table", "data"),
        Output("live-strategy-state", "data"),
        Input("replay-data-store", "data"),
        Input("replay-clock-store", "data"),
        Input("replay-driver", "value"),
        State("live-strategy-state", "data"),
    )
    def update_live_strategy(replay_data, current_time, replay_driver, previous_state):
        if not replay_data or not replay_data.get("replay_id"):
            return _empty_outputs("Tölts be egy verseny-sessiont a stratégiai ajánláshoz.", None)
        replay_id = replay_data["replay_id"]
        replay = REPLAY_CACHE.get(replay_id)
        if replay is None:
            return _empty_outputs("A replay adatai nem találhatók. Töltsd be újra a sessiont.", replay_id)

        # Only locally stored races have the raw data needed for the model state.
        race_folder = replay.get("race_folder")
        session_key = replay.get("session_key")
        if not race_folder or session_key is None:
            key = f"{replay_id}|unsupported"
            if (previous_state or {}).get("key") == key:
                return _no_change()
            return _empty_outputs(
                "A stratégiai ajánlás csak helyben letöltött OpenF1 Race sessionhöz érhető el.",
                key,
            )
        try:
            dataset, trained = _race_resources(race_folder, session_key)
        except (InsufficientHistoryError, RaceDataError) as error:
            key = f"{replay_id}|no-model"
            if (previous_state or {}).get("key") == key:
                return _no_change()
            return _empty_outputs(f"Ehhez a versenyhez nem készíthető modell: {error}", key)
        model = trained.pipeline
        # Final lap from the data; equals the scheduled distance unless the race was shortened.
        race_last_lap = int(pd.to_numeric(dataset["lap_number"], errors="coerce").max())

        if not replay_driver:
            return _empty_outputs("Válassz versenyzőt a stratégiai elemzéshez.", replay_id)
        current_lap = int(time_to_lap(replay_id, float(current_time or 0.0)))
        cache_key = f"{replay_id}|{replay_driver}|lap-{current_lap}"
        if (previous_state or {}).get("key") == cache_key:
            return _no_change()

        laps = replay["laps"]
        if "Driver" not in laps.columns or "DriverNumber" not in laps.columns:
            return _empty_outputs("A session köradataiból nem olvasható ki a versenyző rajtszáma.", cache_key)
        selected_laps = laps.loc[laps["Driver"].astype(str) == str(replay_driver)]
        driver_numbers = pd.to_numeric(selected_laps["DriverNumber"], errors="coerce").dropna()
        if driver_numbers.empty:
            return _empty_outputs(f"Nem található rajtszám ehhez a versenyzőhöz: {replay_driver}.", cache_key)
        driver_number = int(driver_numbers.iloc[0])

        driver_data = dataset.loc[
            pd.to_numeric(dataset["driver_number"], errors="coerce") == driver_number
        ]
        lap_numbers = pd.to_numeric(driver_data["lap_number"], errors="coerce")
        state_rows = driver_data.loc[lap_numbers == current_lap]
        if state_rows.empty:
            return _empty_outputs(
                f"A {replay_driver} versenyzőhöz nincs modellállapot a {current_lap}. körre. Az ajánlás ezért nem számítható ki biztonságosan.",
                cache_key,
            )
        state_row = state_rows.iloc[0]
        # The pace reference and tyre age cannot be imputed meaningfully.
        if pd.isna(state_row.get("field_pace_ref")) or pd.isna(state_row.get("tyre_age")):
            return _empty_outputs(
                f"A {current_lap}. körre még nincs elég tiszta köridő a tempó becsléséhez.",
                cache_key,
            )
        model_features = list(dict.fromkeys(list(model.feature_names_in_) + ["lap_number", "tyre_age", "compound"]))
        initial_state = fill_with_training_medians(model, state_row[model_features])
        current_compound = str(initial_state.get("compound", "UNKNOWN")).upper()
        if current_compound not in SUPPORTED_COMPOUNDS:
            return _empty_outputs(
                f"A {current_lap}. kör gumikeveréke ({current_compound}) nem támogatott a szimulátorban.",
                cache_key,
            )
        if current_lap >= race_last_lap - 1:
            return _empty_outputs("A verseny végéhez közel már nincs értelmezhető jövőbeli bokszkör.", cache_key)

        used_compounds = set(driver_data.loc[lap_numbers <= current_lap, "compound"].dropna())
        strategies = advisor.candidate_strategies(current_lap, race_last_lap, current_compound, used_compounds)
        if not strategies:
            return _empty_outputs("Nincs a szabályoknak megfelelő stratégia a verseny hátralévő részére.", cache_key)
        try:
            results = advisor.evaluate_strategies(model, initial_state, strategies, race_last_lap, PIT_LOSS_SECONDS)
            recommendation = advisor.recommend(results)

            tyre_age = float(initial_state["tyre_age"])
            stint = state_row.get("stint_number")
            wear = advisor.tyre_wear_percent(tyre_age, current_compound, trained.tyre_life_laps)
            stint_rows = driver_data.loc[
                (driver_data["stint_number"] == stint) & (lap_numbers <= current_lap)
            ]
            observed = advisor.observed_dropoff(stint_rows, model, current_compound)
            pace = advisor.compound_pace(model, initial_state)
        except (ValueError, KeyError, TypeError) as error:
            return _empty_outputs(f"A stratégia nem számítható ki: {error}", cache_key)

        history = (
            f"a modell {len(trained.training_races)} korábbi versenyen tanult "
            f"(utolsó: {trained.training_races[-1]})"
        )
        return (
            f"{replay_driver} · aktuális kör: {current_lap}/{race_last_lap} · {history}.",
            _summary_tiles(recommendation, stint),
            _plan_rows(recommendation),
            _tyre_status(current_compound, tyre_age, wear, trained.tyre_life_laps),
            _dropoff_figure(model, current_compound, tyre_age, trained.tyre_life_laps, observed),
            _compound_rows(pace),
            {"key": cache_key, "status": "ok"},
        )
