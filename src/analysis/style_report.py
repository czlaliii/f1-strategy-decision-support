"""Car profiles per team and driving-style profiles with K-Means clusters per driver.

Builds on the per-race tables of driving_style.py. Every metric is first made
relative to the field median of the same race (and the same corner), so
track differences cancel out.
- Car profile (team-season): mean of the team's drivers, e.g. top speed with
  DRS closed, apex speed in slow/medium/fast corners, time per lap in each
  zone type.
- Driver profile: relative to the field (what the viewer sees), and car
  adjusted: minus the team average of the same race, i.e. half the gap to
  the teammate. Only the car-adjusted values are clustered, so the clusters
  describe drivers, not cars.

Reliability: for every telemetry metric the driver-race values computed
from odd and even laps are correlated (split-half). Clusters: K-Means on
standardised driver-season profiles; k chosen by silhouette, stability by
bootstrap adjusted Rand index.

Outputs (data/processed/driving_style/report/): CSV tables, summary.md and figures.

Run from the project root:
    python -m src.analysis.style_report
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.analysis.driving_style import OUTPUT_DIR, load_or_extract, races_with_telemetry
from src.features.race_dataset import RaceDataError

REPORT_DIR = OUTPUT_DIR / "report"
SEED = 42
MIN_SEASON_RACES = 3
# Races with fewer clean dry laps (e.g. wet races) are left out.
MIN_RACE_LAPS = 300
# Free-air laps a driver needs in a race for telemetry metrics.
MIN_DRIVER_LAPS = 8
# Standardised features are clipped here so one extreme profile cannot form its own cluster.
CLIP_Z = 2.5
K_RANGE = range(2, 7)
BOOTSTRAP_SAMPLES = 200

# Metric -> (label, unit); sign chosen so that a larger value means "more" of the label.
DRIVER_METRICS = {
    "late_braking_m": ("Későbbi fékezés", "m"),
    "apex_speed_slow": ("Kanyarsebesség – lassú", "km/h"),
    "apex_speed_medium": ("Kanyarsebesség – közepes", "km/h"),
    "apex_speed_fast": ("Kanyarsebesség – gyors", "km/h"),
    "early_throttle_m": ("Korábbi teljes gáz", "m"),
    "exit_gain": ("Kigyorsítás (+150 m)", "km/h"),
    "launch_distance_m": ("Rajt: megtett út 5 s alatt", "m"),
    "places_gained_lap1": ("Rajt: nyert hely az 1. körben", "hely"),
    "tyre_saving": ("Gumikímélés (kisebb kopás)", "s/kör/10 kör"),
    "consistency": ("Konzisztencia (kisebb szórás)", "s"),
    "net_passes": ("Nettó előzés", "db"),
}
CAR_METRICS = {
    "top_speed_no_drs": ("Végsebesség DRS nélkül", "km/h"),
    "apex_speed_slow": ("Kanyarsebesség – lassú", "km/h"),
    "apex_speed_medium": ("Kanyarsebesség – közepes", "km/h"),
    "apex_speed_fast": ("Kanyarsebesség – gyors", "km/h"),
    "gain_full": ("Időnyereség – teljes gázos szakasz", "s/kör"),
    "gain_slow": ("Időnyereség – lassú kanyarzóna", "s/kör"),
    "gain_medium": ("Időnyereség – közepes kanyarzóna", "s/kör"),
    "gain_fast": ("Időnyereség – gyors kanyarzóna", "s/kör"),
}
TELEMETRY_METRICS = ["late_braking_m", "apex_speed_slow", "apex_speed_medium", "apex_speed_fast",
                     "early_throttle_m", "exit_gain"]


def _relative(frame, column, by):
    return frame[column] - frame.groupby(by)[column].transform("median")


def corner_profile(corners, laps_filter=None) -> pd.DataFrame:
    """Driver-race corner metrics relative to the field (free-air laps, median per corner first)."""
    data = corners[corners["free_air"]]
    if laps_filter is not None:
        data = data[laps_filter(data)]
    laps = data.groupby(["race", "driver_number"])["lap_number"].transform("nunique")
    data = data[laps >= (MIN_DRIVER_LAPS if laps_filter is None else MIN_DRIVER_LAPS // 2)]
    per_corner = data.groupby(["race", "corner", "class", "driver_number"])[
        ["brake_onset_m", "apex_speed", "full_throttle_m", "exit_gain"]].median().reset_index()
    for column in ["brake_onset_m", "apex_speed", "full_throttle_m", "exit_gain"]:
        per_corner[column] = _relative(per_corner, column, ["race", "corner"])
    overall = per_corner.groupby(["race", "driver_number"]).agg(
        late_braking_m=("brake_onset_m", lambda s: -s.mean()),
        early_throttle_m=("full_throttle_m", lambda s: -s.mean()),
        exit_gain=("exit_gain", "mean"),
    )
    by_class = per_corner.pivot_table(index=["race", "driver_number"], columns="class", values="apex_speed", aggfunc="mean")
    by_class = by_class.reindex(columns=["slow", "medium", "fast"]).add_prefix("apex_speed_")
    return overall.join(by_class).reset_index()


def lap_profile(laps) -> pd.DataFrame:
    data = laps[laps["free_air"]]
    data = data[data.groupby(["race", "driver_number"])["lap_number"].transform("count") >= MIN_DRIVER_LAPS]
    per_driver = data.groupby(["race", "driver_number"])[
        ["top_speed_no_drs", "time_full", "time_slow", "time_medium", "time_fast"]].median().reset_index()
    per_driver["top_speed_no_drs"] = _relative(per_driver, "top_speed_no_drs", "race")
    for zone in ["full", "slow", "medium", "fast"]:
        # Time gained per lap versus the field median: positive = faster.
        per_driver[f"gain_{zone}"] = -_relative(per_driver, f"time_{zone}", "race")
    return per_driver.drop(columns=[f"time_{z}" for z in ["full", "slow", "medium", "fast"]])


def start_profile(starts) -> pd.DataFrame:
    data = starts.copy()
    data["launch_distance_m"] = _relative(data, "launch_distance_m", "race")
    return data[["race", "driver_number", "launch_distance_m", "places_gained_lap1"]]


def stint_profile(stints) -> pd.DataFrame:
    data = stints.copy()
    data["degradation"] = _relative(data, "degradation", ["race", "compound"])
    data["scatter"] = _relative(data, "scatter", "race")
    weighted = data.assign(w_deg=data["degradation"] * data["laps"], w_sc=data["scatter"] * data["laps"])
    sums = weighted.groupby(["race", "driver_number"])[["w_deg", "w_sc", "laps"]].sum()
    return pd.DataFrame({
        # Per 10 laps of tyre age, so values are readable.
        "tyre_saving": -sums["w_deg"] / sums["laps"] * 10,
        "consistency": -sums["w_sc"] / sums["laps"],
    }).reset_index()


def driver_race_table(tables) -> pd.DataFrame:
    info = pd.concat([t["starts"][["race", "driver_number", "driver", "team_name", "team_colour"]] for t in tables])
    corners = pd.concat([t["corners"] for t in tables], ignore_index=True)
    laps = pd.concat([t["laps"] for t in tables], ignore_index=True)
    starts = pd.concat([t["starts"] for t in tables], ignore_index=True)
    stints = pd.concat([t["stints"] for t in tables], ignore_index=True)
    passes = pd.concat([t["passes"] for t in tables], ignore_index=True)
    passes["net_passes"] = passes["passes_made"] - passes["passes_lost"]
    table = info.drop_duplicates(["race", "driver_number"])
    for part in [corner_profile(corners), lap_profile(laps), start_profile(starts),
                 stint_profile(stints) if not stints.empty else None,
                 passes[["race", "driver_number", "net_passes"]]]:
        if part is not None:
            table = table.merge(part, on=["race", "driver_number"], how="left")
    return table


def split_half_reliability(tables) -> pd.Series:
    """Correlation of driver-race telemetry metrics from odd vs even laps."""
    corners = pd.concat([t["corners"] for t in tables], ignore_index=True)
    halves = [corner_profile(corners, lambda d, p=p: d["lap_number"] % 2 == p).set_index(["race", "driver_number"])
              for p in (0, 1)]
    joined = halves[0].join(halves[1], lsuffix="_even", rsuffix="_odd", how="inner")
    return pd.Series({m: joined[f"{m}_even"].corr(joined[f"{m}_odd"]) for m in TELEMETRY_METRICS})


def car_adjusted(table, metrics) -> pd.DataFrame:
    """Driver value minus the team mean of the same race (NaN without a teammate in the data)."""
    adjusted = table.copy()
    teams = table.groupby(["race", "team_name"])
    count = teams["driver_number"].transform("count")
    for metric in metrics:
        adjusted[metric] = (table[metric] - teams[metric].transform("mean")).where(count >= 2)
    return adjusted


def season_means(table, keys, metrics, seasons) -> pd.DataFrame:
    data = table.assign(season=table["race"].map(seasons))
    grouped = data.groupby(["season", *keys])
    result = grouped[metrics].mean()
    result["races"] = grouped["race"].nunique()
    return result.reset_index()


def choose_clusters(features: pd.DataFrame):
    """K-Means on standardised features: k by silhouette, bootstrap ARI for stability."""
    x = np.clip(StandardScaler().fit_transform(features.fillna(features.mean())), -CLIP_Z, CLIP_Z)
    scores = {}
    for k in K_RANGE:
        if k >= len(x):
            break
        labels = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit_predict(x)
        scores[k] = silhouette_score(x, labels)
    k = max(scores, key=scores.get)
    model = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(x)
    rng = np.random.default_rng(SEED)
    stability = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = rng.choice(len(x), size=len(x), replace=True)
        boot = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x[sample])
        stability.append(adjusted_rand_score(model.labels_, boot.predict(x)))
    return model, x, scores, float(np.mean(stability))


def cluster_names(centres: pd.DataFrame) -> dict[int, str]:
    """Readable name from the two strongest standardised traits of each centre."""
    names = {}
    for cluster, row in centres.iterrows():
        top = row.abs().sort_values(ascending=False).index[:2]
        parts = [("+" if row[m] > 0 else "−") + DRIVER_METRICS[m][0] for m in top]
        names[cluster] = ", ".join(parts)
    return names


def write_figures(team_season, driver_season, clusters, x):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    for season, data in team_season.groupby("season"):
        matrix = data.set_index("team_name")[list(CAR_METRICS)]
        scaled = (matrix - matrix.mean()) / matrix.std().replace(0, 1)
        figure, axis = plt.subplots(figsize=(11, 0.45 * len(matrix) + 2))
        image = axis.imshow(scaled.to_numpy(), cmap="RdYlGn", vmin=-2, vmax=2, aspect="auto")
        axis.set_xticks(range(len(CAR_METRICS)), [CAR_METRICS[m][0] for m in CAR_METRICS], rotation=35, ha="right")
        axis.set_yticks(range(len(matrix)), matrix.index)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iat[i, j]
                if pd.notna(value):
                    axis.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7)
        axis.set_title(f"Autóprofil a mezőnyhöz képest – {season} (zöld = erősebb)")
        figure.colorbar(image, ax=axis, label="szórás egységben")
        figure.tight_layout()
        figure.savefig(REPORT_DIR / f"car_profile_{season}.png", dpi=120)
        plt.close(figure)

    points = PCA(n_components=2, random_state=SEED).fit_transform(x)
    figure, axis = plt.subplots(figsize=(10, 7))
    for cluster, group in clusters.groupby("cluster"):
        idx = group.index.to_numpy()
        axis.scatter(points[idx, 0], points[idx, 1], label=group["cluster_name"].iloc[0], s=40)
        for i in idx:
            axis.annotate(f"{clusters.at[i, 'driver']} {str(clusters.at[i, 'season'])[2:]}", points[i], fontsize=7)
    axis.set_xlabel("1. főkomponens")
    axis.set_ylabel("2. főkomponens")
    axis.set_title("Pilótastílus-klaszterek (autóhatás nélkül, versenyző × szezon)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "driver_clusters.png", dpi=120)
    plt.close(figure)


def write_summary(races, reliability, team_season, scores, k, stability, centres, clusters):
    lines = [
        "# Autó- és pilótaprofilok telemetriából",
        "",
        f"Versenyek telemetriával: {len(races)}; versenyző × szezon profilok: {len(clusters)} "
        f"(legalább {MIN_SEASON_RACES} verseny).",
        "",
        "## Megbízhatóság (páros vs páratlan körök, versenyző × verseny)",
        "",
        "| Mutató | r |",
        "|---|---|",
        *[f"| {DRIVER_METRICS[m][0]} | {r:.2f} |" for m, r in reliability.items()],
        "",
        "## Autóprofil szezononként (a mezőny mediánjához képest)",
        "",
    ]
    for season, data in team_season.groupby("season"):
        lines += [f"### {season}", "", "| Csapat | Verseny | " + " | ".join(f"{v[0]} ({v[1]})" for v in CAR_METRICS.values()) + " |",
                  "|---|---|" + "---|" * len(CAR_METRICS)]
        for _, row in data.sort_values("gain_slow", ascending=False).iterrows():
            lines.append(f"| {row.team_name} | {row.races} | " + " | ".join(
                "–" if pd.isna(row[m]) else f"{row[m]:+.2f}" for m in CAR_METRICS) + " |")
        lines.append("")
    lines += [
        "## Pilótastílus-klaszterek (K-Means, autóhatás nélkül)",
        "",
        "Silhouette k szerint: " + ", ".join(f"k={key}: {value:.3f}" for key, value in scores.items()),
        f"; választott k = {k}; bootstrap stabilitás (átlagos ARI, {BOOTSTRAP_SAMPLES} minta): {stability:.2f}.",
        "",
        "Klaszterközéppontok (standardizált, + = a csapattárs átlagánál több):",
        "",
        "| Klaszter | " + " | ".join(DRIVER_METRICS[m][0] for m in centres.columns) + " | Tagok |",
        "|---|" + "---|" * (len(centres.columns) + 1),
    ]
    for cluster, row in centres.iterrows():
        members = clusters[clusters["cluster"] == cluster]
        names = ", ".join(f"{d} {str(s)[2:]}" for d, s in zip(members["driver"], members["season"]))
        lines.append(f"| {cluster} | " + " | ".join(f"{v:+.2f}" for v in row) + f" | {names} |")
    (REPORT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    races = races_with_telemetry()
    tables = []
    for race in races.itertuples():
        try:
            tables.append(load_or_extract(race.race, int(race.session_key)))
        except (RaceDataError, ValueError, KeyError) as error:
            print(f"{race.race}: skipped ({error})")
    tables = [t for t in tables if len(t["laps"]) >= MIN_RACE_LAPS]
    used = sorted({t["laps"]["race"].iloc[0] for t in tables if not t["laps"].empty})
    seasons = races.set_index("race")["race_date"].dt.year.to_dict()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    driver_race = driver_race_table(tables)
    driver_race.to_csv(REPORT_DIR / "driver_race.csv", index=False)
    team_season = season_means(driver_race.groupby(["race", "team_name"])[list(CAR_METRICS)].mean().reset_index(),
                               ["team_name"], list(CAR_METRICS), seasons)
    team_season.to_csv(REPORT_DIR / "team_season.csv", index=False)
    field_season = season_means(driver_race, ["driver", "team_name"], list(DRIVER_METRICS), seasons)
    field_season.to_csv(REPORT_DIR / "driver_season_vs_field.csv", index=False)

    adjusted = car_adjusted(driver_race, list(DRIVER_METRICS))
    driver_season = season_means(adjusted, ["driver"], list(DRIVER_METRICS), seasons)
    driver_season = driver_season[driver_season["races"] >= MIN_SEASON_RACES].reset_index(drop=True)
    model, x, scores, stability = choose_clusters(driver_season[list(DRIVER_METRICS)])
    centres = pd.DataFrame(model.cluster_centers_, columns=list(DRIVER_METRICS))
    names = cluster_names(centres)
    clusters = driver_season.assign(cluster=model.labels_, cluster_name=[names[c] for c in model.labels_])
    clusters.to_csv(REPORT_DIR / "driver_clusters.csv", index=False)

    reliability = split_half_reliability(tables)
    centres.to_csv(REPORT_DIR / "cluster_centres.csv", index_label="cluster")
    (REPORT_DIR / "report_meta.json").write_text(json.dumps({
        "races": used,
        "reliability": {m: float(r) for m, r in reliability.items()},
        "silhouette": {str(k): float(v) for k, v in scores.items()},
        "k": int(model.n_clusters),
        "stability_ari": stability,
        "cluster_names": {str(c): n for c, n in names.items()},
        "clip_z": CLIP_Z,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    write_figures(team_season, driver_season, clusters, x)
    print(write_summary(used, reliability, team_season, scores, model.n_clusters, stability, centres, clusters))
    print(f"\nSaved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
