"""Starting grid versus final classification: per race and across all local races.

Sources: OpenF1 starting_grid.json and session_result.json (downloaded with
src.ingestion.download_results). If they are missing, the grid and result
fall back to the first and last records of position.json.

Correlation uses finishers only: DNF, DNS and DSQ positions do not reflect
race pace, so including them would mix two different effects.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from src.features.race_dataset import RAW_ROOT, list_local_races

COLUMNS = ["driver_number", "Driver", "team_colour", "grid", "finish", "status"]


def _read(race_folder, filename):
    path = RAW_ROOT / race_folder / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path)


def _position_feed(race_folder, session_key):
    """(pre-start positions, last positions) from position.json, by driver number."""
    position = _read(race_folder, "position.json")
    laps = _read(race_folder, "laps.json")
    if position.empty or laps.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    position = position[position["session_key"] == session_key].copy()
    position["date"] = pd.to_datetime(position["date"], utc=True, format="mixed")
    position = position.sort_values("date")
    laps = laps[laps["session_key"] == session_key]
    start = pd.to_datetime(laps["date_start"], utc=True, format="mixed").min()
    pre_start = position[position["date"] <= start].groupby("driver_number")["position"].last()
    final = position.groupby("driver_number")["position"].last()
    return pre_start, final


def _official_grid(race_folder, pre_start):
    """Race grid; on sprint weekends the grid matching the pre-start order is the race's."""
    grid = _read(race_folder, "starting_grid.json")
    if grid.empty or not {"session_key", "driver_number", "position"}.issubset(grid.columns):
        return pd.Series(dtype=float)
    best, best_matches = None, -1
    for _, candidate in grid.groupby("session_key"):
        order = candidate.set_index("driver_number")["position"]
        matches = int((order.reindex(pre_start.index) == pre_start).sum())
        if matches > best_matches:
            best, best_matches = order, matches
    return best


def race_grid_and_result(race_folder, session_key):
    """One row per driver: grid slot, classified position and status."""
    drivers = _read(race_folder, "drivers.json")
    drivers = drivers[drivers["session_key"] == session_key].drop_duplicates("driver_number")
    pre_start, final_feed = _position_feed(race_folder, session_key)
    grid = _official_grid(race_folder, pre_start)
    result = _read(race_folder, "session_result.json")
    official = not result.empty and "position" in result.columns

    rows = []
    for _, driver in drivers.iterrows():
        number = int(driver["driver_number"])
        start = grid.get(number, np.nan) if len(grid) else pre_start.get(number, np.nan)
        if pd.isna(start):
            # Pit-lane starters can be missing from the official grid.
            start = pre_start.get(number, np.nan)
        if official:
            row = result[result["driver_number"] == number]
            if row.empty:
                continue
            row = row.iloc[0]
            status = "DSQ" if row.get("dsq") else "DNS" if row.get("dns") else "DNF" if row.get("dnf") else "Célba ért"
            finish = row["position"] if status == "Célba ért" else np.nan
        else:
            status, finish = "Célba ért", final_feed.get(number, np.nan)
        rows.append({
            "driver_number": number,
            "Driver": str(driver.get("name_acronym", number)),
            "team_colour": f"#{str(driver.get('team_colour') or '808080').lstrip('#')}",
            "grid": start,
            "finish": finish,
            "status": status,
        })
    table = pd.DataFrame(rows, columns=COLUMNS)
    table.attrs["source"] = "hivatalos (OpenF1 session_result / starting_grid)" if official and len(grid) else "pozícióadatból becsült"
    return table


def rank_correlation(table):
    """Spearman and Kendall rank correlation between grid and finish for finishers."""
    finishers = table[(table["status"] == "Célba ért")].dropna(subset=["grid", "finish"])
    n = len(finishers)
    if n < 3:
        return {"n": n, "spearman": np.nan, "spearman_p": np.nan, "kendall": np.nan, "kendall_p": np.nan}
    spearman = stats.spearmanr(finishers["grid"], finishers["finish"])
    kendall = stats.kendalltau(finishers["grid"], finishers["finish"])
    gain = finishers["grid"] - finishers["finish"]
    return {
        "n": n,
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "kendall": float(kendall.statistic),
        "kendall_p": float(kendall.pvalue),
        "mean_abs_change": float(gain.abs().mean()),
    }


@lru_cache(maxsize=1)
def all_races_correlation():
    """Per-race Spearman rho over every local race, oldest first, plus pooled pairs."""
    rows, pooled = [], []
    for race in list_local_races().itertuples():
        table = race_grid_and_result(race.race, race.session_key)
        stats_row = rank_correlation(table)
        if np.isnan(stats_row["spearman"]):
            continue
        rows.append({"race": race.race, "race_date": race.race_date, **stats_row})
        pooled.append(table[table["status"] == "Célba ért"].dropna(subset=["grid", "finish"])[["grid", "finish"]])
    per_race = pd.DataFrame(rows)
    pooled = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame(columns=["grid", "finish"])
    return per_race, pooled
