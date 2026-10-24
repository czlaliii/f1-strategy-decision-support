"""Saved model-evaluation results for the Models tab: loaders and Plotly figures.

Nothing is trained here; the numbers come from the offline scripts:
    python -m src.models.compare_lap_time_models   (lap-time models)
    python -m src.models.evaluate_anchoring        (anchored strategy model)
    python -m src.analysis.style_report            (car profiles, K-Means)
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.analysis.style_report import CAR_METRICS, DRIVER_METRICS, REPORT_DIR
from src.dashboard.theme import ACCENT, GRID, TEXT_MUTED
from src.models.compare_lap_time_models import MODEL_LABELS, OUTPUT_DIR as LAP_MODEL_DIR, paired_comparison, phase_comparison

MODEL_ORDER = ["baseline", "strategy", "xgboost", "lstm"]
MODEL_COLORS = {"baseline": "#6b7180", "strategy": "#f5c400", "xgboost": ACCENT, "lstm": "#3fa7ff"}
# Shown by default; the others can be switched on in the legend.
HEADLINE_MODELS = {"xgboost", "lstm"}
CLUSTER_COLORS = [ACCENT, "#3fa7ff", "#f5c400", "#2ecc71", "#b38cff", "#ff8c42"]
TYRE_AGE_BINS = [-1, 2, 5, 10, 20, 40, 80]
TYRE_AGE_LABELS = ["0–2", "3–5", "6–10", "11–20", "21–40", "40+"]


class MissingResults(FileNotFoundError):
    """A results file has not been generated yet; the message says which script makes it."""


def _require(path, command):
    if not path.exists():
        raise MissingResults(f"Hiányzik: {path.name}. Futtasd: {command}")
    return path


# --- Lap-time models -----------------------------------------------------

def load_lap_model_results():
    table = pd.read_csv(_require(LAP_MODEL_DIR / "per_race.csv", "python -m src.models.compare_lap_time_models"))
    errors = pd.read_parquet(_require(LAP_MODEL_DIR / "errors.parquet", "python -m src.models.compare_lap_time_models"))
    return table, errors


def lap_model_summary(table, errors):
    """Mean metrics per model and the XGBoost vs LSTM paired tests (all laps, after the opening laps)."""
    models = [m for m in MODEL_ORDER if m in set(table["model"])]
    summary = table.groupby("model")[["mae", "rmse", "median_ae", "bias", "train_seconds"]].mean().loc[models]
    tests = None
    if {"xgboost", "lstm"} <= set(models):
        tests = {"all": paired_comparison(table, "xgboost", "lstm"),
                 "phases": phase_comparison(errors, "xgboost", "lstm")}
    return summary, tests


def _style(figure, height, y_title, x_title=None):
    figure.update_layout(
        height=height, margin={"l": 50, "r": 16, "t": 16, "b": 40},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
        yaxis={"title": y_title, "gridcolor": GRID, "zeroline": False},
        xaxis={"title": x_title, "showgrid": False},
        hoverlabel={"namelength": -1},
    )
    return figure


def mae_by_race_figure(table):
    pivot = table.pivot(index="race", columns="model", values="mae")
    pivot = pivot.loc[pivot.get("xgboost", pivot.iloc[:, 0]).sort_values().index]
    figure = go.Figure()
    for model in [m for m in MODEL_ORDER if m in pivot.columns]:
        figure.add_bar(
            x=pivot.index, y=pivot[model], name=MODEL_LABELS[model], marker_color=MODEL_COLORS[model],
            visible=True if model in HEADLINE_MODELS else "legendonly",
            hovertemplate="%{x}<br>" + MODEL_LABELS[model] + ": %{y:.3f} s<extra></extra>",
        )
    figure.update_layout(barmode="group", xaxis_tickangle=-40)
    return _style(figure, 380, "MAE (s)")


def mae_by_tyre_age_figure(errors):
    data = errors.assign(abs_error=errors["error"].abs(),
                         age=pd.cut(errors["tyre_age"], TYRE_AGE_BINS, labels=TYRE_AGE_LABELS))
    by_age = data.groupby(["age", "model"], observed=True)["abs_error"].mean().unstack()
    figure = go.Figure()
    for model in [m for m in MODEL_ORDER if m in by_age.columns]:
        figure.add_scatter(
            x=by_age.index.astype(str), y=by_age[model], name=MODEL_LABELS[model], mode="lines+markers",
            line={"color": MODEL_COLORS[model], "width": 3}, marker={"size": 8},
            visible=True if model in HEADLINE_MODELS else "legendonly",
            hovertemplate="Gumikor %{x} kör<br>" + MODEL_LABELS[model] + ": %{y:.3f} s<extra></extra>",
        )
    return _style(figure, 320, "MAE (s)", "Gumikor (kör)")


def load_anchoring_summary():
    stints = pd.read_csv(_require(LAP_MODEL_DIR / "anchoring_stints.csv", "python -m src.models.evaluate_anchoring"))
    from scipy import stats

    from src.models.evaluate_anchoring import ANCHORED_VARIANTS, BASELINE

    per_race = stints.groupby(["race", "variant"])["lap_mae"].mean().unstack()
    rows = []
    for variant in [BASELINE, *ANCHORED_VARIANTS]:
        row = {"variant": variant, "lap_mae": stints.loc[stints["variant"] == variant, "lap_mae"].mean(),
               "wins": None, "p": None}
        if variant != BASELINE:
            row["wins"] = f"{int((per_race[variant] < per_race[BASELINE]).sum())}/{len(per_race)}"
            row["p"] = float(stats.wilcoxon(per_race[variant], per_race[BASELINE]).pvalue)
        rows.append(row)
    return pd.DataFrame(rows), int(stints.loc[stints["variant"] == BASELINE].shape[0])


# --- Driving style and car profiles --------------------------------------

def load_style_results():
    command = "python -m src.analysis.style_report"
    clusters = pd.read_csv(_require(REPORT_DIR / "driver_clusters.csv", command))
    centres = pd.read_csv(_require(REPORT_DIR / "cluster_centres.csv", command), index_col="cluster")
    teams = pd.read_csv(_require(REPORT_DIR / "team_season.csv", command))
    meta = json.loads(_require(REPORT_DIR / "report_meta.json", command).read_text(encoding="utf-8"))
    return clusters, centres, teams, meta


def cluster_scatter_figure(clusters, meta):
    """Driver-seasons on the first two principal components of the clustered features."""
    features = clusters[list(DRIVER_METRICS)]
    x = np.clip(StandardScaler().fit_transform(features.fillna(features.mean())), -meta["clip_z"], meta["clip_z"])
    pca = PCA(n_components=2, random_state=42).fit(x)
    points = pca.transform(x)
    figure = go.Figure()
    for cluster, group in clusters.groupby("cluster"):
        index = group.index.to_numpy()
        figure.add_scatter(
            x=points[index, 0], y=points[index, 1], mode="markers+text",
            text=[f"{d} {str(s)[2:]}" for d, s in zip(group["driver"], group["season"])],
            textposition="top center", textfont={"size": 10, "color": TEXT_MUTED},
            marker={"size": 11, "color": CLUSTER_COLORS[int(cluster) % len(CLUSTER_COLORS)],
                    "line": {"width": 1, "color": "#ffffff"}},
            name=f"{int(cluster)}. klaszter: {meta['cluster_names'].get(str(cluster), '')}",
            customdata=group["races"], hovertemplate="%{text} · %{customdata} verseny<extra></extra>",
        )
    explained = pca.explained_variance_ratio_ * 100
    return _style(figure, 460, f"2. főkomponens ({explained[1]:.0f}% variancia)",
                  f"1. főkomponens ({explained[0]:.0f}% variancia)")


def _heatmap(matrix, x_labels, y_labels, text, colorbar_title, height):
    figure = go.Figure(go.Heatmap(
        z=matrix, x=x_labels, y=y_labels, text=text, texttemplate="%{text}",
        colorscale=[[0, "#b30500"], [0.5, "#1c1e26"], [1, "#2ecc71"]], zmid=0, zmin=-2, zmax=2,
        colorbar={"title": colorbar_title, "thickness": 12},
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
    ))
    figure.update_layout(height=height, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                         xaxis={"side": "top", "tickangle": -30}, yaxis={"autorange": "reversed"})
    return figure


def cluster_profile_figure(centres, meta):
    labels = [f"{DRIVER_METRICS[m][0]}" for m in centres.columns]
    names = [f"{c}. klaszter" for c in centres.index]
    text = [[f"{v:+.2f}" for v in row] for row in centres.to_numpy()]
    return _heatmap(centres.to_numpy(), labels, names, text, "szórás", 120 + 60 * len(centres))


def car_profile_figure(teams, season):
    data = teams[teams["season"] == season].set_index("team_name")[list(CAR_METRICS)]
    data = data.loc[data["gain_slow"].sort_values(ascending=False).index]
    # Colour by standard score within the season, text in physical units.
    scaled = (data - data.mean()) / data.std().replace(0, 1)
    labels = [f"{label} ({unit})" for label, unit in CAR_METRICS.values()]
    text = [["–" if pd.isna(v) else f"{v:+.2f}" for v in row] for row in data.to_numpy()]
    return _heatmap(scaled.to_numpy(), labels, list(data.index), text, "szórás", 160 + 34 * len(data))
