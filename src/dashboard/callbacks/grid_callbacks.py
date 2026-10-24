"""Starting grid versus final classification: scatter, rank correlation and race history."""
import numpy as np
import plotly.graph_objects as go
from dash import html
from dash.dependencies import Input, Output

from src.analysis.grid_finish import all_races_correlation, race_grid_and_result, rank_correlation
from src.dashboard.replay_service import REPLAY_CACHE

# Interpretation bands for Spearman's rho (absolute value).
RHO_BANDS = [(0.8, "erős"), (0.6, "közepesen erős"), (0.4, "mérsékelt"), (0.0, "gyenge")]


def _message_figure(text):
    figure = go.Figure()
    figure.update_layout(
        height=420, plot_bgcolor="#ffffff", margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[{"text": text, "showarrow": False, "font": {"color": "#697586"}}],
    )
    return figure


def _scatter(table):
    finishers = table[table["status"] == "Célba ért"].dropna(subset=["grid", "finish"])
    out = table[table["status"] != "Célba ért"].dropna(subset=["grid"])
    size = int(max(table["grid"].max(skipna=True) or 20, finishers["finish"].max() if len(finishers) else 20))
    out_row = size + 1.5
    figure = go.Figure()
    # Diagonal: finished where they started. Above it = places gained.
    figure.add_trace(go.Scatter(
        x=[0.5, size + 0.5], y=[0.5, size + 0.5], mode="lines", hoverinfo="skip",
        line={"color": "#c3c9d3", "dash": "dash", "width": 1.5}, name="nem változott",
    ))
    change = finishers["grid"] - finishers["finish"]
    figure.add_trace(go.Scatter(
        x=finishers["grid"], y=finishers["finish"], mode="markers+text",
        text=finishers["Driver"], textposition="middle right", textfont={"size": 10, "color": "#354052"},
        marker={"size": 13, "color": finishers["team_colour"], "line": {"width": 1.5, "color": "#ffffff"}},
        customdata=np.column_stack([change]),
        hovertemplate="%{text}<br>Rajt: %{x} → Cél: %{y}<br>Változás: %{customdata[0]:+d} hely<extra></extra>",
        name="célba ért",
    ))
    if len(out):
        figure.add_trace(go.Scatter(
            x=out["grid"], y=[out_row] * len(out), mode="markers+text",
            text=out["Driver"], textposition="top center", textfont={"size": 9, "color": "#9aa3af"},
            marker={"symbol": "x", "size": 10, "color": out["team_colour"]},
            customdata=out["status"],
            hovertemplate="%{text}<br>Rajt: %{x} · %{customdata}<extra></extra>",
            name="kiesett",
        ))
    tick_values = list(range(1, size + 1)) + ([out_row] if len(out) else [])
    tick_text = [str(v) for v in range(1, size + 1)] + (["KI"] if len(out) else [])
    figure.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor="#ffffff",
        showlegend=False,
        xaxis={"title": "Rajtpozíció", "range": [0.4, size + 0.9], "dtick": 1, "gridcolor": "#eef1f5", "zeroline": False},
        # P1 at the top, so points above the diagonal gained places.
        yaxis={"title": "Végeredmény", "range": [out_row + 0.8, 0.4], "tickvals": tick_values,
               "ticktext": tick_text, "gridcolor": "#eef1f5", "zeroline": False},
    )
    return figure


def _p_text(p):
    return "p < 0,001" if p < 0.001 else f"p = {p:.3f}".replace(".", ",")


def _stat(label, value, sub=None):
    return html.Div([html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value"),
                     html.Div(sub, className="kpi-sub") if sub else None], className="kpi-tile")


def _stats(table, result, history, race_folder):
    finishers = table[table["status"] == "Célba ért"].dropna(subset=["grid", "finish"])
    change = (finishers["grid"] - finishers["finish"]).astype(int)
    best = finishers.loc[change.idxmax()] if len(finishers) else None
    worst = finishers.loc[change.idxmin()] if len(finishers) else None
    rho = result["spearman"]
    strength = next(label for limit, label in RHO_BANDS if abs(rho) >= limit)
    rank = (history["spearman"] < rho).mean() * 100 if len(history) else np.nan
    retired = int((table["status"] != "Célba ért").sum())
    return html.Div([
        html.Div([
            _stat("Spearman ρ", f"{rho:.2f}".replace(".", ","), f"{_p_text(result['spearman_p'])} · n = {result['n']}"),
            _stat("Kendall τ", f"{result['kendall']:.2f}".replace(".", ","), _p_text(result["kendall_p"])),
            _stat("Átlagos helyezésváltozás", f"{result['mean_abs_change']:.1f} hely".replace(".", ","),
                  f"legtöbbet javított: {best['Driver']} ({change.max():+d})" if best is not None else None),
            _stat("Legtöbbet vesztett", f"{worst['Driver']} ({change.min():+d})" if worst is not None else "–",
                  f"{retired} versenyző nem ért célba" if retired else "mindenki célba ért"),
        ], className="kpi-grid"),
        html.P(
            f"{strength.capitalize()} pozitív kapcsolat: a rajtsorrend {'nagyrészt megmaradt' if rho >= 0.6 else 'jelentősen átrendeződött'}. "
            f"Ez a verseny a helyi versenyek {rank:.0f}%-ánál erősebb összefüggést mutat. "
            "A korreláció csak a célba érőkre számolt, és nem okozati kapcsolat: a rajtpozíció és a végeredmény is az autó tempójától függ.",
            className="text-muted small mt-3 mb-0",
        ),
        html.Div(f"Adatforrás: {table.attrs.get('source', '–')}.", className="text-muted small mt-1"),
    ])


def _trend(history, race_folder, pooled_rho):
    figure = go.Figure()
    current = history["race"] == race_folder
    figure.add_trace(go.Scatter(
        x=history["race_date"], y=history["spearman"], mode="markers",
        marker={"size": 7, "color": "#b8c0cc"}, text=history["race"],
        hovertemplate="%{text}<br>ρ = %{y:.2f}<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=history.loc[current, "race_date"], y=history.loc[current, "spearman"], mode="markers",
        marker={"size": 13, "color": "#d62839", "line": {"width": 2, "color": "#ffffff"}},
        text=history.loc[current, "race"], hovertemplate="%{text} (kiválasztott)<br>ρ = %{y:.2f}<extra></extra>",
    ))
    median = history["spearman"].median()
    figure.add_hline(y=median, line={"color": "#697586", "dash": "dot", "width": 1},
                     annotation_text=f"medián {median:.2f} · összesített ρ {pooled_rho:.2f}",
                     annotation_position="bottom right", annotation_font={"size": 10, "color": "#697586"})
    figure.update_layout(
        height=230, margin={"l": 10, "r": 10, "t": 30, "b": 10}, plot_bgcolor="#ffffff", showlegend=False,
        title={"text": "Spearman ρ versenyenként (helyi versenyek)", "font": {"size": 12}},
        yaxis={"range": [min(0, history["spearman"].min() - 0.05), 1.02], "gridcolor": "#eef1f5", "zeroline": False},
        xaxis={"gridcolor": "#eef1f5"},
    )
    return figure


def register_grid_callbacks(app):
    @app.callback(
        Output("grid-scatter", "figure"),
        Output("grid-stats", "children"),
        Output("grid-trend", "figure"),
        Input("replay-data-store", "data"),
    )
    def show_grid_analysis(data):
        """Rendered once per loaded session; the all-race history is computed once per process."""
        replay = REPLAY_CACHE.get((data or {}).get("replay_id"))
        if replay is None or not replay.get("race_folder"):
            text = "A rajt–cél elemzés helyben letöltött OpenF1 Race sessionhöz érhető el."
            return _message_figure(text), "", _message_figure("")
        race_folder, session_key = replay["race_folder"], int(replay["session_key"])
        table = race_grid_and_result(race_folder, session_key)
        result = rank_correlation(table)
        if np.isnan(result["spearman"]):
            return _message_figure("Ehhez a versenyhez nincs elég rajt- és eredményadat."), "", _message_figure("")
        history, pooled = all_races_correlation()
        pooled_rho = float(pooled[["grid", "finish"]].corr(method="spearman").iloc[0, 1]) if len(pooled) else np.nan
        return _scatter(table), _stats(table, result, history, race_folder), _trend(history, race_folder, pooled_rho)
