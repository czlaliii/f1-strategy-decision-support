"""Per-race driving and car metrics from OpenF1 car_data telemetry and lap data.

Telemetry (~4 Hz: speed, throttle, binary brake, DRS) is mapped onto lap
distance: speed is integrated over each lap and scaled to the race's median
lap length, then every channel is sampled on a fixed STEP_M grid. Corners are
the local minima of the field's median speed trace, classed by apex speed.

Only clean dry laps are used (green flag, no pit in/out lap, not lap 1, dry
compound, no rainfall). Per lap and corner the metrics are:
- brake_onset_m: distance before the apex where braking starts (smaller = later braking)
- apex_speed: minimum speed within APEX_WINDOW_M of the apex
- full_throttle_m: distance after the apex until throttle >= FULL_THROTTLE
- exit_gain: speed EXIT_DISTANCE_M after the apex minus apex speed
Per lap: time in full-throttle and slow/medium/fast corner zones and the top
speed with DRS closed. Per driver: launch distance in the first seconds and
places gained on lap 1. Per stint: tyre degradation slope and lap-time
scatter. Per driver: on-track passes made and lost (pit cycles excluded).

Results are cached as parquet files in data/processed/driving_style/.

Run from the project root:
    python -m src.analysis.driving_style --race bahrain_2023
    python -m src.analysis.driving_style --all
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from src.analysis.grid_finish import race_grid_and_result
from src.features.race_dataset import (
    DRY_COMPOUNDS,
    PROCESSED_ROOT,
    RAW_ROOT,
    RaceDataError,
    list_local_races,
    load_race_dataset,
)

OUTPUT_DIR = PROCESSED_ROOT / "driving_style"
VERSION = 1
TABLES = ["corners", "laps", "starts", "stints", "passes", "reference"]

STEP_M = 5.0
RESAMPLE_S = 0.05
# Laps whose integrated length differs more than this from the median are dropped.
MAX_LENGTH_DEVIATION = 0.03
# Minimum dip of the field's speed trace (km/h) that counts as a corner.
APEX_PROMINENCE_KMH = 10
MIN_CORNER_GAP_M = 60
# Apex speed classes (km/h): slow < 120 <= medium < 200 <= fast.
SLOW_CORNER_KMH = 120
FAST_CORNER_KMH = 200
APEX_WINDOW_M = 40
BRAKE_SEARCH_M = 350
THROTTLE_SEARCH_M = 250
EXIT_DISTANCE_M = 150
FULL_THROTTLE = 95
BRAKE_ON = 50
# A corner is a braking corner if most clean laps brake before it.
BRAKING_CORNER_SHARE = 0.5
# OpenF1 DRS codes for an open flap.
DRS_OPEN = (10, 12, 14)
# Gap to the car ahead (s) above which a lap counts as driven in free air.
FREE_AIR_SECONDS = 1.0
LAUNCH_SECONDS = 5.0
MIN_STINT_LAPS = 8


def corner_class(apex_speed: float) -> str:
    if apex_speed < SLOW_CORNER_KMH:
        return "slow"
    return "medium" if apex_speed < FAST_CORNER_KMH else "fast"


def _seconds(values) -> np.ndarray:
    return pd.to_datetime(values, utc=True, format="ISO8601").astype("int64").to_numpy() / 1e9


def load_car_data(race_folder: str, driver_number: int) -> pd.DataFrame:
    path = RAW_ROOT / race_folder / f"car_data_driver_{driver_number}.json"
    if not path.exists():
        return pd.DataFrame()
    car = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    if car.empty:
        return car
    car["t"] = _seconds(car["date"])
    return car.sort_values("t").reset_index(drop=True)


def clean_laps(dataset: pd.DataFrame) -> pd.DataFrame:
    """Green, dry, non-pit laps after lap 1 with a valid lap time."""
    rainfall = pd.to_numeric(dataset["rainfall"], errors="coerce").fillna(0)
    mask = (
        dataset["is_green_lap"].astype(bool)
        & ~dataset["pit_stop"].astype(bool)
        & ~dataset["is_pit_out_lap"].fillna(False).astype(bool)
        & (dataset["lap_number"] > 1)
        & dataset["compound"].isin(DRY_COMPOUNDS)
        & rainfall.eq(0)
        & dataset["lap_duration"].notna()
        & dataset["date_start"].notna()
    )
    laps = dataset.loc[mask].copy()
    interval = pd.to_numeric(laps["interval"], errors="coerce")
    laps["free_air"] = interval.isna() | interval.ge(FREE_AIR_SECONDS)
    return laps


def _lap_trace(car: pd.DataFrame, start: float, duration: float):
    """(time grid, distance, channels at the time grid) for one lap, or None if data is missing."""
    window = car[(car["t"] >= start - 1) & (car["t"] <= start + duration + 1)]
    if len(window) < 20 or window["t"].diff().max() > 2.0:
        return None
    times = np.arange(start, start + duration, RESAMPLE_S)
    speed = np.interp(times, window["t"], window["speed"]) / 3.6
    distance = np.concatenate([[0.0], np.cumsum((speed[1:] + speed[:-1]) / 2 * RESAMPLE_S)])
    channels = {c: np.interp(times, window["t"], window[c].astype(float)) for c in ["speed", "throttle", "brake"]}
    # DRS is a code, not a level: take the nearest sample.
    nearest = np.clip(np.searchsorted(window["t"].to_numpy(), times), 0, len(window) - 1)
    channels["drs"] = window["drs"].to_numpy(dtype=float)[nearest]
    return distance, channels


def lap_traces(race_folder: str, laps: pd.DataFrame):
    """Channels of every clean lap on a common distance grid."""
    raw = []
    for driver, driver_laps in laps.groupby("driver_number"):
        car = load_car_data(race_folder, int(driver))
        if car.empty:
            continue
        starts = _seconds(driver_laps["date_start"])
        for (index, lap), start in zip(driver_laps.iterrows(), starts):
            trace = _lap_trace(car, start, float(lap["lap_duration"]))
            if trace is not None:
                raw.append((index, *trace))
    if not raw:
        raise RaceDataError(f"No car_data telemetry for {race_folder}")
    lengths = np.array([distance[-1] for _, distance, _ in raw])
    length = float(np.median(lengths))
    grid = np.arange(0.0, length, STEP_M)
    keep = np.abs(lengths / length - 1) <= MAX_LENGTH_DEVIATION
    index, channels = [], {c: [] for c in ["speed", "throttle", "brake", "drs"]}
    for (row, distance, values), ok in zip(raw, keep):
        if not ok:
            continue
        scaled = distance / distance[-1] * length
        index.append(row)
        for name, value in values.items():
            if name == "drs":
                nearest = np.clip(np.searchsorted(scaled, grid), 0, len(scaled) - 1)
                channels[name].append(value[nearest])
            else:
                channels[name].append(np.interp(grid, scaled, value))
    return laps.loc[index], grid, {name: np.array(values) for name, values in channels.items()}


def reference_profile(grid, channels) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Field median trace and detected corners (apex position, speed, class, braking)."""
    speed = uniform_filter1d(np.median(channels["speed"], axis=0), 5, mode="wrap")
    throttle = np.median(channels["throttle"], axis=0)
    apexes, _ = find_peaks(-speed, prominence=APEX_PROMINENCE_KMH, distance=int(MIN_CORNER_GAP_M / STEP_M))
    braking = []
    for apex in apexes:
        window = channels["brake"][:, max(0, apex - int(BRAKE_SEARCH_M / STEP_M)):apex + 1]
        braking.append(float((window >= BRAKE_ON).any(axis=1).mean()) >= BRAKING_CORNER_SHARE)
    corners = pd.DataFrame({
        "corner": np.arange(1, len(apexes) + 1),
        "apex_index": apexes,
        "apex_m": grid[apexes],
        "apex_speed": speed[apexes],
        "class": [corner_class(s) for s in speed[apexes]],
        "braking": braking,
    })
    reference = pd.DataFrame({"distance": grid, "speed": speed, "throttle": throttle})
    reference["zone"] = zone_labels(grid, throttle, corners)
    return reference, corners


def zone_labels(grid, throttle, corners) -> np.ndarray:
    """Full throttle where the field is flat out; otherwise the class of the nearest corner."""
    labels = np.full(len(grid), "full", dtype=object)
    if corners.empty:
        return labels
    partial = throttle < FULL_THROTTLE
    nearest = np.abs(grid[:, None] - corners["apex_m"].to_numpy()[None, :]).argmin(axis=1)
    labels[partial] = corners["class"].to_numpy()[nearest[partial]]
    return labels


def _brake_onset(brake_row, apex, lo):
    """Distance before the apex where the last braking run before it starts (sub-grid interpolated)."""
    on = brake_row[lo:apex + 1] >= BRAKE_ON
    if not on.any():
        return np.nan
    starts = np.where(on & ~np.concatenate([[False], on[:-1]]))[0]
    start = starts[-1]
    if start == 0:
        return np.nan
    before, after = brake_row[lo + start - 1], brake_row[lo + start]
    fraction = (BRAKE_ON - before) / (after - before) if after != before else 1.0
    position = (start - 1 + fraction) * STEP_M
    return (apex - lo) * STEP_M - position


def corner_metrics(laps, channels, corners) -> pd.DataFrame:
    speed, throttle, brake = channels["speed"], channels["throttle"], channels["brake"]
    n = speed.shape[1]
    frames = []
    for corner in corners.to_dict("records"):
        apex = corner["apex_index"]
        around = slice(max(0, apex - int(APEX_WINDOW_M / STEP_M)), min(n, apex + int(APEX_WINDOW_M / STEP_M) + 1))
        apex_speed = speed[:, around].min(axis=1)
        lo = max(0, apex - int(BRAKE_SEARCH_M / STEP_M))
        entry_speed = speed[:, lo:apex + 1].max(axis=1)
        if corner["braking"]:
            onset = np.array([_brake_onset(row, apex, lo) for row in brake])
        else:
            onset = np.full(len(laps), np.nan)
        hi = min(n, apex + int(THROTTLE_SEARCH_M / STEP_M) + 1)
        flat = throttle[:, apex:hi] >= FULL_THROTTLE
        full_throttle = np.where(flat.any(axis=1), flat.argmax(axis=1) * STEP_M, np.nan)
        exit_speed = speed[:, min(n - 1, apex + int(EXIT_DISTANCE_M / STEP_M))]
        frames.append(pd.DataFrame({
            "driver_number": laps["driver_number"].to_numpy(),
            "lap_number": laps["lap_number"].to_numpy(),
            "free_air": laps["free_air"].to_numpy(),
            "corner": corner["corner"],
            "class": corner["class"],
            "entry_speed": entry_speed,
            "brake_onset_m": onset,
            "apex_speed": apex_speed,
            "full_throttle_m": full_throttle,
            "exit_gain": exit_speed - apex_speed,
        }))
    return pd.concat(frames, ignore_index=True)


def lap_metrics(laps, channels, reference) -> pd.DataFrame:
    speed = channels["speed"]
    seconds = STEP_M / np.maximum(speed / 3.6, 1.0)
    frame = pd.DataFrame({
        "driver_number": laps["driver_number"].to_numpy(),
        "lap_number": laps["lap_number"].to_numpy(),
        "compound": laps["compound"].to_numpy(),
        "tyre_age": laps["tyre_age"].to_numpy(),
        "free_air": laps["free_air"].to_numpy(),
        "lap_duration": laps["lap_duration"].to_numpy(),
    })
    for zone in ["full", "slow", "medium", "fast"]:
        mask = (reference["zone"] == zone).to_numpy()
        frame[f"time_{zone}"] = seconds[:, mask].sum(axis=1) if mask.any() else np.nan
    closed = ~np.isin(channels["drs"], DRS_OPEN)
    frame["top_speed_no_drs"] = np.where(closed, speed, np.nan).max(axis=1)
    frame["top_speed"] = speed.max(axis=1)
    return frame


def start_metrics(race_folder, session_key, dataset) -> pd.DataFrame:
    """Launch distance in the first LAUNCH_SECONDS and places gained on lap 1."""
    first = dataset[dataset["lap_number"] == 1]
    lights_out = np.nanmin(_seconds(first["date_start"].dropna())) if first["date_start"].notna().any() else np.nan
    grid = race_grid_and_result(race_folder, session_key).set_index("driver_number")["grid"]
    rows = []
    for row in first.itertuples():
        driver = int(row.driver_number)
        launch = np.nan
        car = load_car_data(race_folder, driver) if np.isfinite(lights_out) else pd.DataFrame()
        if not car.empty:
            window = car[(car["t"] >= lights_out - 1) & (car["t"] <= lights_out + LAUNCH_SECONDS + 1)]
            if len(window) >= 5 and window["speed"].iloc[0] < 5:
                times = np.arange(lights_out, lights_out + LAUNCH_SECONDS, RESAMPLE_S)
                speed = np.interp(times, window["t"], window["speed"]) / 3.6
                launch = float(np.sum(speed) * RESAMPLE_S)
        rows.append({
            "driver_number": driver,
            "grid": grid.get(driver, np.nan),
            "lap1_position": row.position,
            "launch_distance_m": launch,
        })
    starts = pd.DataFrame(rows)
    starts["places_gained_lap1"] = starts["grid"] - pd.to_numeric(starts["lap1_position"], errors="coerce")
    return starts


def stint_metrics(dataset) -> pd.DataFrame:
    """Degradation slope (s/lap of tyre age) and lap scatter per long stint, relative to the field pace."""
    laps = clean_laps(dataset)
    laps = laps.assign(relative=laps["lap_duration"] - laps["field_pace_ref"]).dropna(subset=["relative", "tyre_age"])
    rows = []
    for (driver, stint), group in laps.groupby(["driver_number", "stint_number"]):
        if len(group) < MIN_STINT_LAPS:
            continue
        age = group["tyre_age"].to_numpy(dtype=float)
        relative = group["relative"].to_numpy(dtype=float)
        if np.ptp(age) == 0:
            continue
        slope, intercept = np.polyfit(age, relative, 1)
        residual = relative - (slope * age + intercept)
        rows.append({
            "driver_number": int(driver), "stint_number": int(stint),
            "compound": group["compound"].iloc[0], "laps": len(group),
            "degradation": float(slope), "scatter": float(np.std(residual, ddof=2)),
        })
    return pd.DataFrame(rows)


def pass_counts(dataset) -> pd.DataFrame:
    """On-track passes: pairs that swap order between consecutive green laps, neither car pitting."""
    data = dataset[["driver_number", "lap_number", "position", "is_green_lap", "pit_stop", "is_pit_out_lap"]].copy()
    data["position"] = pd.to_numeric(data["position"], errors="coerce")
    data["pit"] = data["pit_stop"].astype(bool) | data["is_pit_out_lap"].fillna(False).astype(bool)
    by_lap = {lap: frame.set_index("driver_number") for lap, frame in data.groupby("lap_number")}
    made, lost = {}, {}
    for lap in sorted(by_lap):
        if lap - 1 not in by_lap:
            continue
        now, before = by_lap[lap], by_lap[lap - 1]
        common = now.index.intersection(before.index)
        ok = [d for d in common if now.at[d, "is_green_lap"] and not now.at[d, "pit"] and not before.at[d, "pit"]
              and pd.notna(now.at[d, "position"]) and pd.notna(before.at[d, "position"])]
        for i in ok:
            for j in ok:
                if before.at[i, "position"] > before.at[j, "position"] and now.at[i, "position"] < now.at[j, "position"]:
                    made[i] = made.get(i, 0) + 1
                    lost[j] = lost.get(j, 0) + 1
    drivers = sorted(data["driver_number"].unique())
    return pd.DataFrame({
        "driver_number": drivers,
        "passes_made": [made.get(d, 0) for d in drivers],
        "passes_lost": [lost.get(d, 0) for d in drivers],
    })


def driver_info(race_folder, session_key) -> pd.DataFrame:
    path = RAW_ROOT / race_folder / "drivers.json"
    drivers = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    drivers = drivers[drivers["session_key"] == session_key].drop_duplicates("driver_number")
    return drivers[["driver_number", "name_acronym", "team_name", "team_colour"]].rename(columns={"name_acronym": "driver"})


def extract_race(race_folder: str, session_key: int) -> dict[str, pd.DataFrame]:
    dataset = load_race_dataset(race_folder, session_key)
    laps, grid, channels = lap_traces(race_folder, clean_laps(dataset))
    reference, corners = reference_profile(grid, channels)
    reference = reference.merge(corners[["apex_m", "corner", "class"]], left_on="distance", right_on="apex_m", how="left").drop(columns="apex_m")
    tables = {
        "corners": corner_metrics(laps, channels, corners),
        "laps": lap_metrics(laps, channels, reference),
        "starts": start_metrics(race_folder, session_key, dataset),
        "stints": stint_metrics(dataset),
        "passes": pass_counts(dataset),
        "reference": reference,
    }
    info = driver_info(race_folder, session_key)
    for name, table in tables.items():
        if name != "reference":
            tables[name] = table.merge(info, on="driver_number", how="left")
        tables[name]["race"] = race_folder
    return tables


def load_or_extract(race_folder: str, session_key: int, refresh: bool = False) -> dict[str, pd.DataFrame]:
    folder = OUTPUT_DIR / race_folder
    meta = folder / "meta.json"
    if not refresh and meta.exists() and json.loads(meta.read_text())["version"] == VERSION:
        return {name: pd.read_parquet(folder / f"{name}.parquet") for name in TABLES}
    tables = extract_race(race_folder, session_key)
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(folder / f"{name}.parquet", index=False)
    meta.write_text(json.dumps({"version": VERSION, "race": race_folder}), encoding="utf-8")
    return tables


def races_with_telemetry() -> pd.DataFrame:
    """Local races with a car_data file for every driver (downloads may be in progress)."""
    races = list_local_races()
    complete = []
    for race in races.itertuples():
        drivers = json.loads((RAW_ROOT / race.race / "drivers.json").read_text(encoding="utf-8"))
        numbers = {d["driver_number"] for d in drivers if d.get("session_key") == race.session_key}
        complete.append(bool(numbers) and all(
            (RAW_ROOT / race.race / f"car_data_driver_{n}.json").exists() for n in numbers
        ))
    return races[complete]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--race", help="Local race folder, e.g. bahrain_2023")
    group.add_argument("--all", action="store_true", help="Every local race with car_data")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached results")
    args = parser.parse_args()
    races = races_with_telemetry()
    if args.race:
        races = races[races["race"] == args.race]
    for race in races.itertuples():
        try:
            tables = load_or_extract(race.race, int(race.session_key), refresh=args.refresh)
            print(f"{race.race}: {tables['laps'].shape[0]} laps, {tables['corners']['corner'].nunique()} corners", flush=True)
        except (RaceDataError, ValueError, KeyError) as error:
            print(f"{race.race}: skipped ({error})", flush=True)


if __name__ == "__main__":
    main()
