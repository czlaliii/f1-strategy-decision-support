"""Models tab: evaluation of the lap-time models, driving-style clusters, car profiles and planned models.

Sections are filled when the tab is first opened (see models_callbacks.py).
"""
import dash_bootstrap_components as dbc
from dash import dcc, html

GRAPH_CONFIG = {"displaylogo": False, "responsive": True, "displayModeBar": False}

# Models still to be built; they get their place on the tab already.
PLANNED_MODELS = [
    ("PIT STOP IDŐZÍTŐ", "XGBoost · osztályozás",
     "Körönként becsli, hogy a versenyző a következő körökben kiáll-e (igen/nem). Bemenet: gumikor, keverék, "
     "köridő-romlás, rés az előtte és mögötte haladóhoz, SC/VSC. Értékelés: F1-pontszám és PR-görbe korábbi "
     "versenyeken tanítva, versenyenkénti validációval."),
    ("GUMISTRATÉGIA-VÁLASZTÓ", "Random Forest vs XGBoost · többosztályos osztályozás",
     "A verseny elején javasolt stratégiatípust (pl. M→H, S→H, M→H→H) választja ki a pálya, az időjárás és "
     "a rajthely alapján. A két modell összevetése: makro-F1, keveredési mátrix, páros teszt versenyenként."),
]


def section(title, body, class_name=""):
    return dbc.Card(
        className=f"replay-panel mb-3 {class_name}",
        children=[dbc.CardHeader(title, className="mini-panel-header"), dbc.CardBody(body, className="p-2 p-lg-3")],
    )


def planned_card(title, method, description):
    return dbc.Card(
        className="replay-panel h-100 planned-model",
        children=[
            dbc.CardHeader([title, html.Span("TERVEZETT", className="planned-badge")], className="mini-panel-header"),
            dbc.CardBody([html.Div(method, className="planned-method"), html.P(description, className="planned-text")],
                         className="p-3"),
        ],
    )


def models_layout():
    return html.Div([
        section("KÖRTELJESÍTMÉNY-BECSLŐ · XGBOOST VS LSTM", html.Div(id="models-lap-section", children="Betöltés…")),
        section("PILÓTASTÍLUS-KLASZTEREZÉS · K-MEANS", html.Div(id="models-style-section", children="Betöltés…")),
        section("AUTÓPROFIL · CSAPATOK A MEZŐNYHÖZ KÉPEST", [
            html.Div([
                html.Span("Szezon", className="field-label mb-0"),
                dcc.Dropdown(id="models-season", options=[], value=None, clearable=False,
                             className="dashboard-dropdown models-season"),
            ], className="d-flex align-items-center gap-2 mb-2"),
            html.P(
                "A csapat két versenyzőjének átlaga a mezőny mediánjához képest, versenyenként, majd a szezonra "
                "átlagolva. Zöld: erősebb, piros: gyengébb (a szezonon belüli szórás egységében); a cellák a "
                "mért értéket mutatják. Csak száraz, szabad levegős körök.",
                className="text-muted small mb-2",
            ),
            dcc.Graph(id="models-car-profile", config=GRAPH_CONFIG, className="dashboard-graph"),
        ]),
        dbc.Row([dbc.Col(planned_card(*model), xs=12, lg=6, className="mb-3") for model in PLANNED_MODELS], className="g-3"),
    ])
