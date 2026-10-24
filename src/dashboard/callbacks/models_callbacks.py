"""Models tab: rendered from saved evaluation results when the tab is first opened."""
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State

from src.dashboard import model_results as results
from src.dashboard.components.models_layout import GRAPH_CONFIG
from src.models.compare_lap_time_models import MODEL_LABELS, OPENING_LAPS


def _kpi(label, value, sub=None, highlight=False):
    return html.Div(
        [html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value"),
         html.Div(sub, className="kpi-sub") if sub else None],
        className="kpi-tile" + (" kpi-tile-best" if highlight else ""),
    )


def _graph(figure, title):
    return html.Div([html.Div(title, className="kpi-label mb-1"), dcc.Graph(figure=figure, config=GRAPH_CONFIG)],
                    className="mt-3")


def _p(value):
    return "< 0,001" if value < 0.001 else f"{value:.3f}".replace(".", ",")


def _s(value, digits=3, sign=False):
    return (f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}").replace(".", ",")


def lap_section():
    try:
        table, errors = results.load_lap_model_results()
    except results.MissingResults as error:
        return dbc.Alert(str(error), color="secondary")
    summary, tests = results.lap_model_summary(table, errors)
    best = summary["mae"].idxmin()
    tiles = [
        _kpi(MODEL_LABELS[model], f"{_s(row.mae)} s",
             f"RMSE {_s(row.rmse)} s · torzítás {_s(row.bias, sign=True)} s · tanítás {row.train_seconds:.1f} s",
             highlight=model == best)
        for model, row in summary.iterrows()
    ]
    children = [
        html.P(
            f"Következő kör idejének becslése {table['race'].nunique()} nem látott versenyen: minden versenynél "
            "csak az annál korábbi versenyeken tanítva (bővülő ablak). A csempék a versenyenkénti MAE átlagát mutatják.",
            className="text-muted small mb-2",
        ),
        html.Div(tiles, className="kpi-grid"),
    ]
    if tests:
        overall = tests["all"]
        later = tests["phases"][f"{OPENING_LAPS}. kör után"]
        children.append(html.Div(
            [
                html.Div("XGBOOST − LSTM · PÁROS ÖSSZEHASONLÍTÁS VERSENYENKÉNT", className="kpi-label mb-1"),
                html.Div([
                    f"Minden kör: {_s(overall['mean_difference'], sign=True)} s "
                    f"(95% CI {_s(overall['ci_low'], sign=True)}; {_s(overall['ci_high'], sign=True)}), "
                    f"az XGBoost {overall['wins_first']}/{overall['races']} versenyen pontosabb, "
                    f"Wilcoxon p = {_p(overall['wilcoxon_p'])}.",
                    html.Br(),
                    f"A {OPENING_LAPS}. kör után ({later['share']:.0%} a körökből): "
                    f"{_s(later['mean_difference'], sign=True)} s, {later['wins_first']}/{later['races']} verseny, "
                    f"p = {_p(later['wilcoxon_p'])}.",
                ], className="models-test-text"),
            ],
            className="models-test mt-3",
        ))
    children += [
        _graph(results.mae_by_race_figure(table), "MAE VERSENYENKÉNT · a jelmagyarázatra kattintva további modellek"),
        _graph(results.mae_by_tyre_age_figure(errors), "HIBA A GUMIKOR SZERINT"),
    ]
    children.append(anchoring_block())
    return children


def anchoring_block():
    try:
        anchoring, stints = results.load_anchoring_summary()
    except results.MissingResults as error:
        return dbc.Alert(str(error), color="secondary", className="mt-3")
    rows = [
        html.Tr([
            html.Td(row.variant), html.Td(f"{_s(row.lap_mae)} s"),
            html.Td(row.wins or "–"), html.Td("–" if row.p is None else _p(row.p)),
        ], className="is-used" if row.variant == "horgony fél.idő 10" else None)
        for row in anchoring.itertuples()
    ]
    return html.Div(
        [
            html.Div("A PROGRAMBAN HASZNÁLT MODELL · XGBOOST-HORGONY", className="kpi-label mb-1"),
            html.P(
                "Az XGBoost a pontosabb következőkör-becslő, de alig reagál a gumikeverékre és a gumikorra, így "
                "önmagában nem tud stratégiát rangsorolni. A stratégiai szimuláció ezért a strukturált modellt "
                "használja, a köridőszintet pedig az XGBoost becsléséhez igazítja; a korrekció 10 körönként "
                f"feleződik. Mérés: {stints} valós stint szimulációja az első körétől.",
                className="text-muted small mb-2",
            ),
            html.Table(
                [html.Thead(html.Tr([html.Th("Változat"), html.Th("Kör-MAE"), html.Th("Jobb verseny"), html.Th("Wilcoxon p")])),
                 html.Tbody(rows)],
                className="models-table",
            ),
        ],
        className="mt-3",
    )


def style_section():
    try:
        clusters, centres, teams, meta = results.load_style_results()
    except results.MissingResults as error:
        return dbc.Alert(str(error), color="secondary"), [], None
    reliability = meta["reliability"]
    silhouette = meta["silhouette"][str(meta["k"])]
    tiles = [
        _kpi("Versenyek", str(len(meta["races"])), "száraz versenyek telemetriával"),
        _kpi("Profilok", str(len(clusters)), "versenyző × szezon, min. 3 verseny"),
        _kpi("Klaszterek (k)", str(meta["k"]), f"silhouette {_s(silhouette)} · stabilitás (ARI) {_s(meta['stability_ari'], 2)}"),
        _kpi("Mérési megbízhatóság", f"r = {_s(min(reliability.values()), 2)}–{_s(max(reliability.values()), 2)}",
             "páros vs páratlan körök"),
    ]
    children = [
        html.P(
            "Minden mutató a csapattárs átlagához mérve (az autó hatása így kiesik): későbbi fékezés, "
            "kanyarsebesség, korábbi teljes gáz, kigyorsítás, rajt, gumikímélés, konzisztencia, nettó előzés. "
            "A K-Means standardizált profilokon fut; k a silhouette-érték alapján.",
            className="text-muted small mb-2",
        ),
        html.Div(tiles, className="kpi-grid"),
        _graph(results.cluster_scatter_figure(clusters, meta), "VERSENYZŐK A KÉT FŐ KOMPONENS SÍKJÁBAN"),
        _graph(results.cluster_profile_figure(centres, meta), "KLASZTERKÖZÉPPONTOK · + = a csapattársnál több"),
    ]
    if silhouette < 0.25:
        children.append(dbc.Alert(
            "A silhouette-érték alacsony: a vezetési stílusok inkább folytonos skálát alkotnak, mint éles "
            "csoportokat. A klaszterek a skála két végét jelölik.",
            color="secondary", className="mt-3 mb-0 small",
        ))
    seasons = sorted(teams["season"].unique(), reverse=True)
    # Default: the season with the most races, i.e. the most reliable profile.
    default = int(teams.groupby("season")["races"].max().idxmax())
    options = [{"label": f"{s} ({int(teams.loc[teams['season'] == s, 'races'].max())} verseny)", "value": int(s)}
               for s in seasons]
    return children, options, default


def register_models_callbacks(app):
    @app.callback(
        Output("models-lap-section", "children"),
        Output("models-style-section", "children"),
        Output("models-season", "options"),
        Output("models-season", "value"),
        Input("nav-models", "n_clicks"),
        State("models-season", "options"),
        prevent_initial_call=True,
    )
    def render_models(_, rendered):
        if rendered:
            return no_update, no_update, no_update, no_update
        style, options, season = style_section()
        return lap_section(), style, options, season

    @app.callback(
        Output("models-car-profile", "figure"),
        Input("models-season", "value"),
    )
    def car_profile(season):
        if season is None:
            return go.Figure().update_layout(height=60, xaxis={"visible": False}, yaxis={"visible": False})
        try:
            _, _, teams, _ = results.load_style_results()
        except results.MissingResults:
            return no_update
        return results.car_profile_figure(teams, season)
