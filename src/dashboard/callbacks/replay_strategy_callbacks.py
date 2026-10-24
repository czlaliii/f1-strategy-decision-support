"""Connect replay lap progression to the historical strategy advisor.

The advisor runs in a separate worker process (see strategy_service), so
replay playback never waits for it: one callback queues a job when the lap
or driver changes, another polls for the result and renders it.
"""
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, no_update
from dash.dependencies import Input, Output, State

from src.dashboard import strategy_service, strategy_worker
from src.dashboard.replay_player import format_lap_time
from src.dashboard.replay_service import REPLAY_CACHE, time_to_lap
from src.dashboard.theme import GRID, TEXT_MUTED
from src.simulator import strategy_advisor as advisor

SUPPORTED_COMPOUNDS = advisor.DRY_COMPOUNDS
# Official tyre colours; light compounds need dark text on top of them.
COMPOUND_COLORS = {"SOFT": "#e10600", "MEDIUM": "#ffd12e", "HARD": "#f2f2f2"}
COMPOUND_TEXT = {"SOFT": "#ffffff", "MEDIUM": "#111111", "HARD": "#111111"}
COMPOUND_NAMES = {"SOFT": "Soft", "MEDIUM": "Medium", "HARD": "Hard"}


def _empty_figure(message=""):
    figure = go.Figure()
    figure.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{"text": message, "showarrow": False, "font": {"color": TEXT_MUTED}}] if message else [],
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _empty_outputs(message, what_if=None):
    return (
        message,
        html.Div("Nincs még kiszámítható ajánlás.", className="text-muted"),
        [],
        "",
        _empty_figure(),
        [],
        _what_if_section(what_if),
    )


def _compound_pill(compound):
    return html.Span(
        COMPOUND_NAMES.get(compound, compound),
        className="compound-pill",
        style={"backgroundColor": COMPOUND_COLORS.get(compound, "#697586"), "color": COMPOUND_TEXT.get(compound, "#ffffff")},
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


def _laps_text(laps):
    return ", ".join(f"{lap}." for lap in laps)


def _summary_tiles(result):
    best = result["best"]
    if result["pit_window"]:
        start, end = result["pit_window"]
        window = f"{start}–{end}. kör" if start != end else f"{start}. kör"
        window_sub = f"optimum: {_laps_text(best['pit_laps'])} kör"
        next_compound = _compound_pill(best["compounds"][1])
    else:
        window, window_sub = "Nincs kiállás", "a jelenlegi szettel a célig"
        next_compound = html.Span("Marad a jelenlegi", className="kpi-muted")

    stops = len(best["pit_laps"])
    stint = result["current_stint"]
    total_stops = (stint - 1 if stint else 0) + stops
    alternatives = result["plans"][1:]
    if not alternatives:
        delta_value, delta_sub = "–", "nincs alternatíva"
    else:
        runner_up = alternatives[0]
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


def _plan_rows(result, limit=8):
    return [
        {
            "plan": plan["plan"],
            "pit_laps": _laps_text(plan["pit_laps"]) or "–",
            "delta": "javasolt" if plan["delta"] == 0 else _signed(plan["delta"]),
        }
        for plan in result["plans"][:limit]
    ]


def _tyre_status(tyre):
    compound, wear, life = tyre["compound"], tyre["wear"], tyre["life"].get(tyre["compound"])
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
            html.Div([_compound_pill(compound), html.Span(f"{int(tyre['age'])} kör a szetten", className="ms-2 fw-semibold")]),
            html.Div("Becsült elhasználódás", className="kpi-label mt-3"),
            wear_bar,
            html.Div(
                f"Becslés: a {COMPOUND_NAMES.get(compound, compound)} keverék tipikus élettartama "
                f"~{life:.0f} kör a korábbi versenyek alapján." if life else "",
                className="text-muted small mt-2",
            ),
        ]
    )


def _dropoff_figure(result):
    tyre, observed = result["tyre"], result["observed"]
    figure = go.Figure()
    for name in SUPPORTED_COMPOUNDS:
        max_age = max(tyre["life"].get(name, 30), tyre["age"] + 5)
        ages = np.arange(0, int(max_age) + 1)
        current = name == tyre["compound"]
        figure.add_trace(go.Scatter(
            x=ages,
            y=result["degradation"][name] * ages,
            mode="lines",
            name=f"{COMPOUND_NAMES[name]} (modell)",
            line={"color": COMPOUND_COLORS[name], "width": 3 if current else 1.5, "dash": "solid" if current else "dot"},
            hovertemplate="%{x}. kör a szetten: +%{y:.2f} s<extra></extra>",
        ))
    if observed["tyre_age"]:
        figure.add_trace(go.Scatter(
            x=observed["tyre_age"],
            y=observed["dropoff"],
            mode="markers",
            name="Megfigyelt körök (igazítva)",
            marker={"color": COMPOUND_COLORS.get(tyre["compound"], "#ffffff"), "size": 7, "line": {"width": 1, "color": "#0b0c10"}},
            hovertemplate="%{x}. kör a szetten: %{y:+.2f} s<extra></extra>",
        ))
    figure.add_vline(x=tyre["age"], line={"color": TEXT_MUTED, "dash": "dash", "width": 1})
    figure.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title={"text": "Köridő-visszaesés a friss szetthez képest", "font": {"size": 13}},
        xaxis={"title": "Gumikor (kör)", "gridcolor": GRID},
        yaxis={"title": "Lassulás (s)", "gridcolor": GRID, "zeroline": True},
        legend={"orientation": "h", "y": -0.25, "font": {"size": 11}},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _compound_rows(result):
    return [
        {
            "set": row["set"],
            "compound": COMPOUND_NAMES.get(row["compound"], row["compound"]),
            "tyre_age": int(row["tyre_age"]),
            "lap_time": format_lap_time(row["lap_time"]),
            "delta": "–" if row["set"] == "Jelenlegi szett" else _signed(row["delta_to_current"], 2),
        }
        for row in result["pace"]
    ]


# Delta (s) up to which an actual stop counts as matching / close to the model's plan.
WHAT_IF_MATCH_SECONDS = 0.5
WHAT_IF_CLOSE_SECONDS = 3.0


def _stint_bar(segments, race_last_lap, decision_lap=None, until_lap=None):
    """Horizontal stint bar: one coloured block per stint, scaled to race distance."""
    blocks = []
    for segment in segments:
        laps = segment["end"] - segment["start"] + 1
        width = 100 * laps / race_last_lap
        compound = segment["compound"]
        blocks.append(html.Div(
            COMPOUND_NAMES.get(compound, compound)[:1] if width >= 4 else "",
            className="stint-seg",
            title=f"{COMPOUND_NAMES.get(compound, compound)} · {segment['start']}–{segment['end']}. kör",
            style={"width": f"{width}%", "backgroundColor": COMPOUND_COLORS.get(compound, "#697586"),
                   "color": COMPOUND_TEXT.get(compound, "#ffffff")},
        ))
    if until_lap is not None and until_lap < race_last_lap:
        # Laps not yet driven in the replay.
        blocks.append(html.Div(className="stint-seg stint-seg-future",
                               style={"width": f"{100 * (race_last_lap - until_lap) / race_last_lap}%"}))
    markers = []
    if decision_lap is not None:
        markers.append(html.Div(className="stint-decision", title=f"Döntési pont: {decision_lap}. kör",
                                style={"left": f"{100 * decision_lap / race_last_lap}%"}))
    return html.Div([html.Div(blocks, className="stint-bar"), *markers], className="stint-bar-wrap")


def _what_if_card(index, analysis, actual_segments, race_last_lap, current_lap):
    stop = f"{analysis['stop_lap']}. kör · {COMPOUND_NAMES.get(analysis['from'], analysis['from'])} → {COMPOUND_NAMES.get(analysis['to'], analysis['to'])}"
    head = [html.Span(f"{index}. KIÁLLÁS", className="kpi-label"), html.Span(stop, className="what-if-stop")]
    if analysis["status"] != "ok":
        return html.Div([html.Div(head, className="what-if-head"),
                         html.Div(analysis["reason"], className="text-muted small")], className="what-if-card")

    delta = analysis["delta"]
    if delta is None:
        chip, tone = "A valós kiállás kívül esett a szimulált lehetőségeken", "neutral"
    elif delta <= WHAT_IF_MATCH_SECONDS:
        chip, tone = f"Közel optimális döntés · {delta:.1f} s", "good"
    else:
        chip = f"A modell terve {delta:.1f} s-mal gyorsabb lett volna a célig"
        tone = "warn" if delta <= WHAT_IF_CLOSE_SECONDS else "bad"
    head.append(html.Span(chip, className=f"what-if-chip what-if-chip-{tone}"))

    model_stops = " → ".join(COMPOUND_NAMES.get(c, c) for c in analysis["model_compounds"])
    window = analysis["model_window"]
    model_text = (
        f"{_laps_text(analysis['model_pit_laps'])} kör · {model_stops}"
        + (f" · ablak {window[0]}–{window[1]}." if window else "")
        if analysis["model_pit_laps"] else f"Nincs kiállás · {model_stops}"
    )
    return html.Div(
        [
            html.Div(head, className="what-if-head"),
            html.Div([html.Div("Valós", className="what-if-row-label"),
                      _stint_bar(actual_segments, race_last_lap, until_lap=current_lap)], className="what-if-row"),
            html.Div([html.Div("Modell", className="what-if-row-label"),
                      _stint_bar(analysis["model_segments"], race_last_lap, decision_lap=analysis["decision_lap"])],
                     className="what-if-row"),
            html.Div(
                [
                    html.Div([html.Div("Valós döntés", className="kpi-label"), html.Div(stop, className="what-if-value")]),
                    html.Div([html.Div(f"Modell javaslata · a {analysis['decision_lap']}. kör állapotából", className="kpi-label"),
                              html.Div(model_text, className="what-if-value")]),
                ],
                className="what-if-compare",
            ),
        ],
        className="what-if-card",
    )


def _what_if_section(what_if):
    if not what_if:
        return ""
    analyses = what_if.get("analyses", [])
    title = html.Div("MI LETT VOLNA, HA… · VALÓS KIÁLLÁS KONTRA MODELLJAVASLAT", className="kpi-label mb-2")
    if not analyses:
        return html.Div([title, html.Div(
            "A versenyző még nem állt ki; az összevetés az első kiállás után jelenik meg.",
            className="text-muted small",
        )])
    race_last_lap = what_if["race_last_lap"]
    current_lap = max((s["end"] for s in what_if["actual_segments"]), default=0)
    cards = [
        _what_if_card(index, analysis, what_if["actual_segments"], race_last_lap, current_lap)
        for index, analysis in enumerate(analyses, start=1)
    ]
    legend = html.Div(
        [html.Span([html.Span(className="legend-dot", style={"backgroundColor": COMPOUND_COLORS[c]}), COMPOUND_NAMES[c]],
                   className="me-3") for c in SUPPORTED_COMPOUNDS]
        + [html.Span([html.Span(className="legend-decision"), "döntési pont"])],
        className="what-if-legend",
    )
    # Latest stop first: it is the one the replay has just shown.
    return html.Div([title, *reversed(cards), legend])


def _render(result, driver):
    if result.get("status") != "ok":
        return _empty_outputs(result.get("message", "Nincs még kiszámítható ajánlás."), result.get("what_if"))
    message = (
        f"{driver} · aktuális kör: {result['current_lap']}/{result['race_last_lap']} · "
        f"a modell {result['training_races']} korábbi versenyen tanult "
        f"(utolsó: {result['last_training_race']})."
    )
    return (
        message,
        _summary_tiles(result),
        _plan_rows(result),
        _tyre_status(result["tyre"]),
        _dropoff_figure(result),
        _compound_rows(result),
        _what_if_section(result.get("what_if")),
    )


def _message_request(text, key=None):
    return {"key": key or f"msg|{text}", "message": text}


def register_replay_strategy_callbacks(app):
    """Queue strategy jobs on lap/driver changes and render results when ready."""

    @app.callback(
        Output("strategy-request", "data"),
        Input("replay-data-store", "data"),
        Input("replay-clock-store", "data"),
        Input("replay-driver", "value"),
        State("strategy-request", "data"),
    )
    def request_strategy(replay_data, current_time, replay_driver, previous_request):
        """Cheap: runs on every clock tick but only queues work when the lap or driver changes."""
        def changed(request):
            return no_update if (previous_request or {}).get("key") == request["key"] else request

        if not replay_data or not replay_data.get("replay_id"):
            return changed(_message_request("Tölts be egy verseny-sessiont a stratégiai ajánláshoz."))
        replay_id = replay_data["replay_id"]
        replay = REPLAY_CACHE.get(replay_id)
        if replay is None:
            return changed(_message_request("A replay adatai nem találhatók. Töltsd be újra a sessiont."))
        race_folder, session_key = replay.get("race_folder"), replay.get("session_key")
        if not race_folder or session_key is None:
            return changed(_message_request(
                "A stratégiai ajánlás csak helyben letöltött OpenF1 Race sessionhöz érhető el."
            ))
        # Load or train the model in the worker as soon as the race is loaded.
        strategy_service.submit(
            f"warm|{race_folder}|{session_key}", strategy_worker.warm_up, race_folder, int(session_key)
        )
        if not replay_driver:
            return changed(_message_request("Válassz versenyzőt a stratégiai elemzéshez."))

        laps = replay["laps"]
        selected = laps.loc[laps["Driver"].astype(str) == str(replay_driver)]
        numbers = pd.to_numeric(selected.get("DriverNumber"), errors="coerce").dropna()
        if numbers.empty:
            return changed(_message_request(f"Nem található rajtszám ehhez a versenyzőhöz: {replay_driver}."))
        driver_number = int(numbers.iloc[0])
        current_lap = int(time_to_lap(replay_id, float(current_time or 0.0)))

        key = f"{race_folder}|{session_key}|{driver_number}|{current_lap}"
        if (previous_request or {}).get("key") == key:
            return no_update
        strategy_service.submit(
            key, strategy_worker.compute_strategy, race_folder, int(session_key), driver_number, current_lap
        )
        return {"key": key, "driver": str(replay_driver), "lap": current_lap}

    @app.callback(
        Output("live-strategy-message", "children"),
        Output("live-strategy-summary", "children"),
        Output("live-strategy-table", "data"),
        Output("live-tyre-status", "children"),
        Output("live-pace-dropoff", "figure"),
        Output("live-compound-table", "data"),
        Output("live-what-if", "children"),
        Output("live-strategy-state", "data"),
        Input("strategy-poll", "n_intervals"),
        Input("strategy-request", "data"),
        State("live-strategy-state", "data"),
    )
    def render_strategy(_, request, view_state):
        """Cheap: shows a pending note while the worker runs, then the finished result."""
        unchanged = (no_update,) * 8
        if not request:
            return unchanged
        view_state = view_state or {}
        key = request["key"]
        if view_state.get("rendered") == key:
            return unchanged
        if "message" in request:
            return (*_empty_outputs(request["message"]), {"rendered": key})

        done, result = strategy_service.result(key)
        if not done:
            if view_state.get("pending") == key:
                return unchanged
            note = f"{request['driver']} · {request['lap']}. kör · számítás folyamatban…"
            # Keep the previous recommendation visible until the new one is ready.
            return (note, *(no_update,) * 6, {**view_state, "pending": key})
        return (*_render(result, request["driver"]), {"rendered": key})
