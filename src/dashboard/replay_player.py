"""F1 session replay: telemetry, timeline, leaderboard and track."""
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.dashboard.circuit_info import circuit_rotation, rotate_xy
from src.dashboard.replay_data import ReplayData

SAMPLE_INTERVAL_SECONDS = 0.1
MAX_INTERPOLATION_GAP_SECONDS = 2.0


def format_elapsed_time(seconds):
    """Format elapsed time as 01:24:35."""
    if seconds is None or pd.isna(seconds):
        return "–"
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"

def format_lap_time(seconds):
    """Format lap time: 83.456 -> 1:23.456."""
    if seconds is None or pd.isna(seconds):
        return "–"
    minutes = int(seconds // 60)
    remaining_seconds = seconds - minutes * 60
    return f"{minutes}:{remaining_seconds:06.3f}"

def _seconds(value):
    """Convert a duration to seconds."""
    if value is None or pd.isna(value):
        return np.nan
    return pd.to_timedelta(
        value,
        errors="coerce",
    ).total_seconds()

def _resample_driver(driver_data, timeline):
    """Resample one driver's telemetry onto the shared timeline.

    Does not interpolate across laps or over long telemetry gaps.
    """
    sampled_parts = []
    numeric_columns = [
        "X",
        "Y",
        "Distance",
        "Speed",
    ]
    discrete_columns = [
        "LapNumber",
        "Compound",
        "TyreLife",
        "Stint",
        "DRS",
        "nGear",
        "Throttle",
        "Brake",
    ]
    for lap_number, segment in driver_data.groupby(
        "LapNumber",
        sort=True,
    ):
        segment = (
            segment.sort_values("TimeSeconds")
            .drop_duplicates("TimeSeconds", keep="last")
            .reset_index(drop=True)
        )
        if len(segment) < 2:
            continue
        times = segment["TimeSeconds"].to_numpy(dtype=float)
        start = times[0]
        end = times[-1]
        mask = (timeline >= start) & (timeline <= end)
        sample_times = timeline[mask]
        if len(sample_times) == 0:
            continue
        right = np.searchsorted(times, sample_times, side="left")
        right = np.clip(right, 0, len(times) - 1)
        left = np.maximum(right - 1, 0)
        # Interpolate only over short gaps.
        gaps = times[right] - times[left]
        valid = (
            (sample_times >= times[0])
            & (sample_times <= times[-1])
            & (gaps <= MAX_INTERPOLATION_GAP_SECONDS)
        )
        if not valid.any():
            continue
        sample_times = sample_times[valid]
        left = left[valid]
        right = right[valid]
        result = pd.DataFrame({
            "TimeSeconds": sample_times,
            "Driver": segment["Driver"].iloc[0],
            "LapNumber": int(lap_number),
        })
        for column in numeric_columns:
            if column not in segment.columns:
                result[column] = np.nan
                continue
            values = pd.to_numeric(
                segment[column],
                errors="coerce",
            ).to_numpy(dtype=float)
            result[column] = np.interp(
                sample_times,
                times,
                values,
            )
        # Discrete values come from the latest known sample.
        for column in discrete_columns:
            if column not in segment.columns:
                result[column] = np.nan
                continue
            result[column] = (
                segment[column]
                .iloc[left]
                .to_numpy()
            )
        sampled_parts.append(result)
    if not sampled_parts:
        return pd.DataFrame()
    return (
        pd.concat(sampled_parts, ignore_index=True)
        .sort_values("TimeSeconds")
        .drop_duplicates(
            ["Driver", "TimeSeconds"],
            keep="last",
        )
        .reset_index(drop=True)
    )

DEFAULT_DRIVER_COLOR = "#808080"
# Track map styling (dark, broadcast-like).
TRACK_MAP_BACKGROUND = "#111218"
TRACK_MAP_HEIGHT = 560
# Gaussian time smoothing of displayed car positions; OpenF1 location samples
# arrive every 0.16-0.38 s, which makes linearly interpolated motion surge.
POSITION_SMOOTHING_SECONDS = 0.2
# Minimum pause in lap starts that marks a red-flag stoppage.
RED_FLAG_PAUSE_SECONDS = 120.0
STINT_COLUMNS = ["Driver", "stint_number", "lap_start", "lap_end", "compound", "tyre_age_at_start"]


def build_track_status(race_control, start_time, lap_start_seconds=None):
    """SC, VSC and red-flag periods from race control messages, in replay seconds.

    A period starts with its deployment message and ends with the next
    track-level TRACK CLEAR/green flag, a session (re)start or the chequered
    flag. 'SAFETY CAR IN THIS LAP' and 'VSC ENDING' only mark it as ending.
    A new deployment (e.g. SC -> red flag) replaces the current period.
    Under a red flag the cars stop, so when the restart message is missing
    from the feed the red flag ends at the first lap start after a pause of
    at least RED_FLAG_PAUSE_SECONDS.
    """
    if race_control is None or race_control.empty or "message" not in race_control.columns:
        return []
    events = race_control.copy()
    events["date"] = pd.to_datetime(events["date"], utc=True, format="mixed")
    events = events.dropna(subset=["date"]).sort_values("date")
    periods, current = [], None

    def close(at):
        nonlocal current
        if current is not None:
            current["end"] = at
            periods.append(current)
            current = None

    for _, event in events.iterrows():
        at = (event["date"] - start_time).total_seconds()
        message = str(event.get("message", "")).upper()
        category = str(event.get("category", ""))
        flag = str(event.get("flag", "")).upper()
        scope = str(event.get("scope", ""))
        status = None
        if category == "SafetyCar":
            if "VIRTUAL SAFETY CAR DEPLOYED" in message or "VSC DEPLOYED" in message:
                status = "VSC"
            elif "SAFETY CAR DEPLOYED" in message:
                status = "SC"
            elif current is not None and ("ENDING" in message or "IN THIS LAP" in message):
                current["ending_from"] = at
        elif flag == "RED" or (category == "SessionStatus" and "ABORTED" in message):
            # An aborted session is a red-flag stoppage even without a RED FLAG message.
            status = "RED"
        elif flag == "CHEQUERED" or (category == "SessionStatus" and "FINISHED" in message):
            # Messages after the finish (e.g. a post-race red flag) are not race status.
            close(at)
            break
        elif (
            "TRACK CLEAR" in message
            or (flag in ("CLEAR", "GREEN") and scope == "Track")
            or (category == "SessionStatus" and "STARTED" in message)
        ):
            close(at)
        if status and (current is None or current["status"] != status):
            close(at)
            current = {"status": status, "start": at, "end": None, "ending_from": None}
    close(None)
    starts = sorted(lap_start_seconds or [])
    for period in periods:
        if period["status"] == "RED":
            # Cars on track still finish their lap after the flag; the restart
            # is the first lap start that follows a long pause.
            later = [t for t in starts if t > period["start"]]
            restart = next(
                (t for previous, t in zip([period["start"]] + later, later)
                 if t - previous >= RED_FLAG_PAUSE_SECONDS),
                None,
            )
            if restart is not None and period["end"] is None:
                period["end"] = restart
    return periods


def track_status_at(periods, current_seconds):
    """The SC/VSC/red-flag period active at a replay time, or None for green."""
    for period in periods or []:
        if period["start"] <= current_seconds and (period["end"] is None or current_seconds < period["end"]):
            ending = period["ending_from"] is not None and current_seconds >= period["ending_from"]
            return {"status": period["status"], "ending": ending}
    return None


def find_retirements(laps, finish_seconds=None):
    """Drivers who stopped before the chequered flag: Driver -> {lap, out_from, hidden_from}.

    A driver retired if their last lap ended (or, if unfinished, started)
    before the finish. Lapped finishers end their last lap after it.
    out_from: the next race lap start; the leaderboard marks the retirement.
    hidden_from: one race lap later; the car stays visible on track until then.
    """
    if laps is None or laps.empty:
        return {}
    laps = laps.dropna(subset=["LapNumber", "LapStartTime"]).copy()
    if finish_seconds is None:
        finish_seconds = float(laps["TimeSeconds"].max())
    # Start time of every race lap: the first car to begin it.
    lap_starts = laps.groupby("LapNumber")["LapStartTime"].min().sort_values()
    # A lap cannot start after the field is 5 laps further on; such timestamps
    # are feed errors and are replaced by the race start of that lap.
    by_lap = lap_starts.sort_index()
    limit = laps["LapNumber"].map(lambda n: by_lap.get(n + 5, np.inf))
    implausible = laps["LapStartTime"] > limit
    laps.loc[implausible, "LapStartTime"] = laps.loc[implausible, "LapNumber"].map(by_lap)
    laps.loc[implausible, "TimeSeconds"] = np.nan
    ends = laps["TimeSeconds"].where(laps["TimeSeconds"].notna(), laps["LapStartTime"])
    retirements = {}
    for driver, driver_laps in laps.assign(end=ends).groupby("Driver"):
        last = driver_laps.sort_values("LapNumber").iloc[-1]
        if last["end"] >= finish_seconds - 1.0:
            continue
        later = lap_starts[lap_starts > last["end"]].tolist()
        out_from = later[0] if later else float(last["end"]) + 60.0
        hidden_from = later[1] if len(later) > 1 else out_from + 120.0
        retirements[str(driver)] = {
            "lap": int(last["LapNumber"]),
            "out_from": float(out_from),
            "hidden_from": float(hidden_from),
        }
    return retirements


def tyre_state(stints, current_laps):
    """Compound, tyre age and stint number for each driver's lap in progress.

    current_laps maps Driver -> lap number. The stint covering that lap is
    used (the later one if two overlap); tyre age is tyre_age_at_start plus
    the laps completed since the stint started. After the last recorded
    stint the age keeps counting on the same set.
    """
    rows = []
    if stints is None or stints.empty:
        return pd.DataFrame(columns=["Driver", "Compound", "TyreLife", "Stint"])
    for driver, lap in current_laps.items():
        driver_stints = stints.loc[
            (stints["Driver"] == driver) & (stints["lap_start"] <= lap)
        ].sort_values("stint_number")
        if pd.isna(lap) or driver_stints.empty:
            continue
        covering = driver_stints.loc[driver_stints["lap_end"].isna() | (driver_stints["lap_end"] >= lap)]
        stint = (covering if not covering.empty else driver_stints).iloc[-1]
        age = stint["tyre_age_at_start"] + (int(lap) - int(stint["lap_start"]))
        rows.append({
            "Driver": driver,
            "Compound": stint["compound"],
            "TyreLife": int(age) if pd.notna(age) else None,
            "Stint": int(stint["stint_number"]),
        })
    return pd.DataFrame(rows, columns=["Driver", "Compound", "TyreLife", "Stint"])


def _fastf1_stints(session_laps):
    """Build an OpenF1-style stint table from FastF1 laps (TyreLife counts the lap itself)."""
    laps = session_laps.dropna(subset=["Stint", "LapNumber"])
    if laps.empty:
        return pd.DataFrame(columns=STINT_COLUMNS)
    grouped = laps.sort_values("LapNumber").groupby(["Driver", "Stint"], as_index=False)
    stints = grouped.agg(
        lap_start=("LapNumber", "min"),
        lap_end=("LapNumber", "max"),
        compound=("Compound", "first"),
        tyre_life_first=("TyreLife", "first"),
    )
    stints["tyre_age_at_start"] = (stints["tyre_life_first"] - 1).clip(lower=0)
    stints = stints.rename(columns={"Stint": "stint_number"})
    return stints[STINT_COLUMNS]


def _rotate_replay(frames, track, degrees):
    """Rotate car positions and the track line to the official map orientation."""
    if not degrees:
        return frames, track
    frames = frames.copy()
    frames["X"], frames["Y"] = rotate_xy(frames["X"], frames["Y"], degrees)
    if track is not None and not track.empty:
        track = track.copy()
        track["X"], track["Y"] = rotate_xy(track["X"], track["Y"], degrees)
    return frames, track


def _team_color(value):
    """Return a '#RRGGBB' colour from an OpenF1/FastF1 hex value, else None."""
    text = str(value or "").strip().lstrip("#")
    if len(text) == 6 and all(c in "0123456789abcdefABCDEF" for c in text):
        return f"#{text.upper()}"
    return None


def _prepare_track_coordinates(data, x_column="X", y_column="Y"):
    """Return clean X/Y coordinates for drawing the track."""
    if data is None or data.empty or not {x_column, y_column}.issubset(data.columns):
        return pd.DataFrame(columns=["X", "Y"])

    track = data[[x_column, y_column]].copy()
    track = track.rename(columns={x_column: "X", y_column: "Y"})
    track["X"] = pd.to_numeric(track["X"], errors="coerce")
    track["Y"] = pd.to_numeric(track["Y"], errors="coerce")
    return (
        track.dropna(subset=["X", "Y"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _build_fastf1_track(session, raw_telemetry):
    """Build the track line from FastF1 data, with a fallback source."""
    try:
        fastest_lap = session.laps.pick_fastest()
        track = _prepare_track_coordinates(fastest_lap.get_telemetry())
    except (KeyError, ValueError, AttributeError):
        track = pd.DataFrame(columns=["X", "Y"])

    if track.empty:
        track = _prepare_track_coordinates(raw_telemetry)
    return track


def _build_local_track(replay):
    """Pick the track line from a single complete lap.

    Skips lap 1 (grid start) and in/out laps (pit lane), then picks the lap
    whose closed outline has the smallest maximum gap between samples.
    OpenF1 x/y is not in metres, so no absolute distance thresholds are used.
    """
    candidates = []

    for number, driver_laps in replay.laps.items():
        location_df = replay.locations.get(number)
        if location_df is None or location_df.empty:
            continue

        location_df = location_df.sort_values("date")
        driver_laps = driver_laps.sort_values("lap_number")
        if "is_pit_out_lap" in driver_laps.columns:
            pit_out = driver_laps["is_pit_out_lap"].fillna(False).astype(bool)
        else:
            pit_out = pd.Series(False, index=driver_laps.index)
        # The in-lap ends in the pit lane.
        pit_in = pit_out.shift(-1, fill_value=False)

        for index, lap in driver_laps.iterrows():
            if lap["lap_number"] <= 1 or pit_out[index] or pit_in[index]:
                continue
            duration = pd.to_numeric(lap["lap_duration"], errors="coerce")
            if pd.isna(lap["date_start"]) or pd.isna(duration):
                continue
            if duration < 30 or duration > 200:
                continue
            lap_end = lap["date_start"] + pd.to_timedelta(
                float(duration), unit="s"
            )
            lap_locations = location_df.loc[
                (location_df["date"] >= lap["date_start"])
                & (location_df["date"] <= lap_end)
            ]
            points = _prepare_track_coordinates(
                lap_locations, x_column="x", y_column="y"
            )
            if len(points) < 50:
                continue

            xy = points[["X", "Y"]].to_numpy(dtype=float)
            closed_xy = np.vstack([xy, xy[:1]])
            step_lengths = np.hypot(*np.diff(closed_xy, axis=0).T)
            candidates.append((step_lengths.max(), points))

    if not candidates:
        raise ValueError(
            "Nem sikerült pályavonalat előállítani a helyi adatokból: "
            "nincs olyan teljes kör, amelyhez elegendő koordináta tartozik."
        )

    _, track = min(candidates, key=lambda candidate: candidate[0])

    # Repeat the first point to close the one-sample gap on the main straight.
    return pd.concat([track, track.iloc[[0]]], ignore_index=True)


def load_session_data(year, event_name, session_name):
    """Load a FastF1 session and build the replay data.

    Returns a dict with "laps", "frames" (telemetry on the timeline) and "track".
    """
    session = fastf1.get_session(
        int(year),
        event_name,
        session_name,
    )
    # Position telemetry is required too.
    session.load(
        laps=True,
        telemetry=True,
        weather=False,
        messages=False,
    )
    print("Session betöltve.")
    print("Köradatok száma:", len(session.laps))
    print("Telemetria tesztelése...")
    lap = session.laps.iloc[0]
    print("Versenyző:", lap["Driver"])
    print("Kör:", lap["LapNumber"])
    telemetry = lap.get_telemetry()
    print("Telemetria sorainak száma:", len(telemetry))
    print("Telemetria oszlopai:", telemetry.columns.tolist())
    laps = session.laps.copy()
    if laps.empty:
        raise ValueError(
            "Ehhez a sessionhöz nem érhető el köradat."
        )
    # --- Prepare lap data ---
    lap_columns = [
        "Driver",
        "DriverNumber",
        "LapNumber",
        "LapTime",
        "Time",
        "LapStartTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "Compound",
        "TyreLife",
        "Stint",
        "Position",
        "IsPersonalBest",
    ]
    laps = laps[
        [column for column in lap_columns if column in laps.columns]
    ].copy()
    duration_columns = [
        "LapTime",
        "Time",
        "LapStartTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
    ]
    for column in duration_columns:
        if column in laps.columns:
            laps[column] = pd.to_timedelta(
                laps[column],
                errors="coerce",
            ).dt.total_seconds()
    laps = laps.dropna(subset=["Driver", "LapNumber"])
    laps["LapNumber"] = laps["LapNumber"].astype(int)
    # --- Load telemetry per lap ---
    telemetry_parts = []
    for _, lap in session.laps.iterlaps():
        try:
            telemetry = lap.get_telemetry()
        except (KeyError, ValueError, AttributeError):
            # Some laps may lack position telemetry.
            continue
        if telemetry is None or telemetry.empty:
            continue
        required = {"SessionTime", "X", "Y"}
        if not required.issubset(telemetry.columns):
            continue
        telemetry = telemetry.copy()
        telemetry["TimeSeconds"] = (
            pd.to_timedelta(
                telemetry["SessionTime"],
                errors="coerce",
            ).dt.total_seconds()
        )
        telemetry = telemetry.dropna(
            subset=["TimeSeconds", "X", "Y"]
        )
        if telemetry.empty:
            continue
        telemetry["Driver"] = str(lap["Driver"])
        telemetry["LapNumber"] = int(lap["LapNumber"])
        for column in [
            "Compound",
            "TyreLife",
            "Stint",
        ]:
            telemetry[column] = lap.get(column, np.nan)
        for column in [
            "Distance",
            "Speed",
            "DRS",
            "nGear",
            "Throttle",
            "Brake",
        ]:
            if column not in telemetry.columns:
                telemetry[column] = np.nan
        telemetry_parts.append(
            telemetry[
                [
                    "Driver",
                    "LapNumber",
                    "TimeSeconds",
                    "X",
                    "Y",
                    "Distance",
                    "Speed",
                    "Compound",
                    "TyreLife",
                    "Stint",
                    "DRS",
                    "nGear",
                    "Throttle",
                    "Brake",
                ]
            ]
        )
    if not telemetry_parts:
        raise ValueError(
            "A session betöltődött, de nem található "
            "feldolgozható pozíciótelemetria."
        )
    raw_telemetry = pd.concat(
        telemetry_parts,
        ignore_index=True,
    )
    raw_telemetry = raw_telemetry.sort_values(
        "TimeSeconds"
    ).reset_index(drop=True)
    # Shared timeline starting at zero.
    time_origin = float(
        raw_telemetry["TimeSeconds"].min()
    )
    raw_telemetry["TimeSeconds"] -= time_origin
    # Align lap timestamps to the same timeline.
    for column in ["Time", "LapStartTime"]:
        if column in laps.columns:
            laps[column] -= time_origin
    if "Time" in laps.columns:
        laps["TimeSeconds"] = laps["Time"]
    else:
        laps["TimeSeconds"] = np.nan
    if "LapStartTime" in laps.columns:
        laps["LapStartSeconds"] = laps["LapStartTime"]
    # --- Build shared timeline ---
    duration = float(
        raw_telemetry["TimeSeconds"].max()
    )
    timeline = np.arange(
        0,
        duration + SAMPLE_INTERVAL_SECONDS,
        SAMPLE_INTERVAL_SECONDS,
    )
    frame_parts = []
    for driver, driver_data in raw_telemetry.groupby("Driver"):
        sampled = _resample_driver(
            driver_data,
            timeline,
        )
        if not sampled.empty:
            frame_parts.append(sampled)
    if not frame_parts:
        raise ValueError(
            "Nem sikerült telemetria-képkockákat előállítani."
        )
    frames = (
        pd.concat(frame_parts, ignore_index=True)
        .sort_values(["TimeSeconds", "Driver"])
        .reset_index(drop=True)
    )
    # Track building lives in its own function.
    track = _build_fastf1_track(session, raw_telemetry)
    try:
        rotation = float(session.get_circuit_info().rotation)
    except Exception:  # noqa: BLE001 - circuit info is optional; draw unrotated
        rotation = 0.0
    frames, track = _rotate_replay(frames, track, rotation)
    driver_colors = {}
    try:
        for _, result in session.results.iterrows():
            color = _team_color(result.get("TeamColor"))
            if color and result.get("Abbreviation"):
                driver_colors[str(result["Abbreviation"])] = color
    except (AttributeError, KeyError):
        pass
    try:
        stints = _fastf1_stints(session.laps)
    except (AttributeError, KeyError):
        stints = pd.DataFrame(columns=STINT_COLUMNS)
    return {
        "laps": laps,
        "frames": frames,
        "track": track,
        "driver_colors": driver_colors,
        "stints": stints,
        "track_status": None,
        "retirements": find_retirements(laps),
    }

def load_local_session_data(
    race_name="australia_2025",
    session_key=9693,
):
    """Convert local OpenF1 data into the replay format.

    Without a session_key, the race session is picked from the local
    drivers.json/laps.json, so no API calls are needed.
    """
    if session_key is None:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / race_name
        drivers_path = data_dir / "drivers.json"
        laps_path = data_dir / "laps.json"
        if not drivers_path.exists() or not laps_path.exists():
            raise FileNotFoundError(
                f"A helyi futamfájlok hiányoznak: {data_dir}. "
                "Ellenőrizd a drivers.json és laps.json fájlokat."
            )
        drivers_df = pd.read_json(drivers_path)
        if "session_key" not in drivers_df.columns:
            raise ValueError(f"A {drivers_path} fájlból hiányzik a session_key oszlop.")
        session_keys = pd.to_numeric(drivers_df["session_key"], errors="coerce").dropna().unique()
        if len(session_keys) == 1:
            session_key = int(session_keys[0])
        elif len(session_keys) > 1:
            laps_df = pd.read_json(laps_path)
            if not {"session_key", "lap_number"}.issubset(laps_df.columns):
                raise ValueError(
                    f"Több session található a {drivers_path} fájlban, "
                    "de a laps.json alapján nem választható ki automatikusan a futam."
                )
            laps_df["session_key"] = pd.to_numeric(laps_df["session_key"], errors="coerce")
            laps_df["lap_number"] = pd.to_numeric(laps_df["lap_number"], errors="coerce")
            max_laps = laps_df.groupby("session_key")["lap_number"].max().dropna()
            matching = max_laps.loc[max_laps.index.isin(session_keys)]
            if matching.empty:
                raise ValueError(f"Nem választható ki session a {data_dir} adataiból.")
            session_key = int(matching.idxmax())
        else:
            raise ValueError(f"Nem található session_key a {drivers_path} fájlban.")

    replay = ReplayData(
        race_name=race_name,
        session_key=int(session_key),
    ).load()
    duration = (replay.end_time - replay.start_time).total_seconds()
    if duration <= 0:
        raise ValueError("A verseny időtartama nem megfelelő.")
    timeline = np.arange(
        0,
        int(duration) + 1,
        SAMPLE_INTERVAL_SECONDS,
    )
    frame_parts = []
    laps_parts = []
    track = pd.DataFrame()
    for number, driver in replay.drivers.items():
        driver_name = driver.get("name_acronym", str(number))
        location_df = replay.locations.get(number)
        # --- Car positions on the shared timeline ---
        if location_df is not None and not location_df.empty:
            location = location_df.copy().sort_values("date")
            # Allow coordinates up to 2 s before the race start.
            location = location.loc[
                (location["date"] >= replay.start_time
                 - pd.Timedelta(seconds=2))
                & (location["date"] <= replay.end_time)
            ].copy()
            if not location.empty:
                location["TimeSeconds"] = (
                    location["date"] - replay.start_time
                ).dt.total_seconds()
                dx = location["x"].diff().fillna(0)
                dy = location["y"].diff().fillna(0)
                location["Distance"] = np.sqrt(
                    dx ** 2 + dy ** 2
                ).cumsum()
                target_times = timeline
                source = (
                    location[["TimeSeconds", "x", "y", "Distance"]]
                    .dropna(subset=["TimeSeconds", "x", "y"])
                    .sort_values("TimeSeconds")
                    .drop_duplicates(subset=["TimeSeconds"])
                    .reset_index(drop=True)
                )
                source_times = source["TimeSeconds"].to_numpy()
                print(
                    f"{driver_name}: "
                    f"location sorok={len(location)}, "
                    f"source sorok={len(source)}, "
                    f"timeline={timeline.min():.1f}–{timeline.max():.1f}"
                )
                if len(source_times) > 0:
                    print(
                        f"  source időtartomány="
                        f"{source_times.min():.2f}–{source_times.max():.2f}, "
                        f"NaN x={source['x'].isna().sum()}, "
                        f"NaN y={source['y'].isna().sum()}"
                    )
                if len(source) >= 2:
                    right_idx = np.searchsorted(
                        source_times,
                        target_times,
                        side="left",
                    )
                    exact = np.zeros(len(target_times), dtype=bool)
                    inside = right_idx < len(source_times)
                    exact[inside] = (
                        source_times[right_idx[inside]] == target_times[inside]
                    )
                    left_idx = np.clip(right_idx - 1, 0, len(source_times) - 1)
                    safe_right_idx = np.clip(
                        right_idx, 0, len(source_times) - 1
                    )
                    gaps = (
                        source_times[safe_right_idx]
                        - source_times[left_idx]
                    )
                    valid = exact | (
                        (right_idx > 0)
                        & (right_idx < len(source_times))
                        & (gaps <= 10.0)
                    )
                    print(
                        f"  érvényes interpolációs pontok: "
                        f"{valid.sum()} / {len(valid)}"
                    )
                    sampled = pd.DataFrame({"TimeSeconds": target_times})
                    for column in ["x", "y", "Distance"]:
                        sampled[column] = np.interp(
                            target_times,
                            source_times,
                            source[column].to_numpy(),
                        )
                        sampled.loc[~valid, column] = np.nan
                    sampled = sampled.rename(
                        columns={"x": "X", "y": "Y"}
                    )
                else:
                    sampled = pd.DataFrame({
                        "TimeSeconds": target_times,
                        "X": np.nan,
                        "Y": np.nan,
                        "Distance": np.nan,
                    })
                sampled["Driver"] = driver_name
                # Lap number = latest started lap.
                driver_laps = replay.laps.get(number)
                if driver_laps is not None and not driver_laps.empty:
                    lap_starts = driver_laps["date_start"]
                    valid_starts = lap_starts.notna()
                    lap_data = driver_laps.loc[
                        valid_starts
                    ].sort_values("date_start")
                    start_seconds = (
                        lap_data["date_start"] - replay.start_time
                    ).dt.total_seconds().to_numpy()
                    lap_numbers = pd.to_numeric(
                        lap_data["lap_number"],
                        errors="coerce",
                    ).to_numpy()
                    indexes = np.searchsorted(
                        start_seconds,
                        sampled["TimeSeconds"].to_numpy(),
                        side="right",
                    ) - 1
                    sampled["LapNumber"] = np.where(
                        indexes >= 0,
                        lap_numbers[np.maximum(indexes, 0)],
                        np.nan,
                    )
                else:
                    sampled["LapNumber"] = np.nan
                # Not available in the local location files.
                for column in [
                    "Speed",
                    "Compound",
                    "TyreLife",
                    "Stint",
                    "DRS",
                    "nGear",
                    "Throttle",
                    "Brake",
                ]:
                    sampled[column] = np.nan
                sampled = sampled.dropna(subset=["X", "Y"])
                print(
                    f"{driver_name}: sampled sorok={len(sampled)}, "
                    f"oszlopok={sampled.columns.tolist()}, "
                    f"X NaN={sampled['X'].isna().sum()}, "
                    f"Y NaN={sampled['Y'].isna().sum()}"
                )
                if not sampled.empty:
                    frame_parts.append(sampled)
                    print(f"{driver_name}: frame_parts elemek={len(frame_parts)}")
        # --- Convert lap data ---
        driver_laps = replay.laps.get(number)
        if driver_laps is not None and not driver_laps.empty:
            laps = driver_laps.copy()
            laps["Driver"] = driver_name
            laps["DriverNumber"] = number
            laps["LapTime"] = pd.to_numeric(
                laps["lap_duration"],
                errors="coerce",
            )
            laps["LapNumber"] = pd.to_numeric(
                laps["lap_number"],
                errors="coerce",
            )
            laps["LapStartTime"] = (
                laps["date_start"] - replay.start_time
            ).dt.total_seconds()
            laps["TimeSeconds"] = (
                laps["LapStartTime"] + laps["LapTime"]
            )
            for source, target_column in [
                ("duration_sector_1", "Sector1Time"),
                ("duration_sector_2", "Sector2Time"),
                ("duration_sector_3", "Sector3Time"),
            ]:
                laps[target_column] = pd.to_numeric(
                    laps[source],
                    errors="coerce",
                )
            for column in [
                "Compound",
                "TyreLife",
                "Stint",
                "Position",
            ]:
                laps[column] = np.nan
            laps["IsPersonalBest"] = False
            laps_parts.append(laps[[
                "Driver",
                "DriverNumber",
                "LapNumber",
                "LapTime",
                "TimeSeconds",
                "LapStartTime",
                "Sector1Time",
                "Sector2Time",
                "Sector3Time",
                "Compound",
                "TyreLife",
                "Stint",
                "Position",
                "IsPersonalBest",
            ]])
    if not frame_parts:
        raise ValueError(
            "Nem sikerült használható autópozíciókat előállítani."
        )
    frames = (
        pd.concat(frame_parts, ignore_index=True)
        .sort_values(["TimeSeconds", "Driver"])
        .reset_index(drop=True)
    )
    laps = (
        pd.concat(laps_parts, ignore_index=True)
        if laps_parts
        else pd.DataFrame()
    )
    # Track line from the cleanest full lap (see _build_local_track).
    track = _build_local_track(replay)
    meeting_key = next(iter(replay.drivers.values()), {}).get("meeting_key")
    frames, track = _rotate_replay(
        frames, track, circuit_rotation(meeting_key, replay.start_time.year)
    )
    driver_colors = {
        str(driver.get("name_acronym", number)): color
        for number, driver in replay.drivers.items()
        if (color := _team_color(driver.get("team_colour")))
    }
    stints_path = replay.data_dir / "stints.json"
    stints = pd.read_json(stints_path) if stints_path.exists() else pd.DataFrame()
    if {"session_key", "driver_number", "stint_number", "lap_start"}.issubset(stints.columns):
        stints = stints.loc[pd.to_numeric(stints["session_key"], errors="coerce") == replay.session_key].copy()
        stints["Driver"] = stints["driver_number"].map(
            lambda number: str(replay.drivers.get(int(number), {}).get("name_acronym", number))
        )
        stints = stints.reindex(columns=STINT_COLUMNS)
    else:
        stints = pd.DataFrame(columns=STINT_COLUMNS)
    control_path = replay.data_dir / "race_control.json"
    race_control = pd.read_json(control_path) if control_path.exists() else pd.DataFrame()
    if "session_key" in race_control.columns:
        race_control = race_control.loc[
            pd.to_numeric(race_control["session_key"], errors="coerce") == replay.session_key
        ]
    finish_seconds = None
    if {"flag", "date"}.issubset(race_control.columns):
        chequered = pd.to_datetime(
            race_control.loc[race_control["flag"] == "CHEQUERED", "date"], utc=True, format="mixed"
        )
        if not chequered.empty:
            finish_seconds = (chequered.min() - replay.start_time).total_seconds()
    return {
        "laps": laps,
        "frames": frames,
        "track": track,
        "driver_colors": driver_colors,
        "stints": stints,
        "track_status": build_track_status(
            race_control, replay.start_time, laps["LapStartTime"].dropna().tolist()
        ),
        "retirements": find_retirements(laps, finish_seconds),
    }

def build_current_snapshot(frames_by_driver, current_time_seconds):
    """Per-driver state at the current time."""
    rows = []
    current_time = float(current_time_seconds)
    for driver_frames in frames_by_driver.values():
        times = driver_frames["TimeSeconds"].to_numpy()
        index = np.searchsorted(
            times,
            current_time,
            side="right",
        ) - 1
        if index < 0:
            continue
        row = driver_frames.iloc[index]
        if current_time - float(row["TimeSeconds"]) > 3.0:
            continue
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)

def build_leaderboard(current_frames, laps, current_time_seconds, stints=None, retirements=None):
    """Build the leaderboard from the current replay state.

    Order is estimated from lap number, then distance within the lap;
    not an official FIA result.
    """
    columns = [
        "Position",
        "Driver",
        "Lap",
        "LastLapTime",
        "Interval",
        "Compound",
        "TyreLife",
        "Stint",
        "Status",
    ]
    if (
        current_frames is None
        or current_frames.empty
        or current_time_seconds is None
    ):
        return pd.DataFrame(columns=columns)
    current_time = float(current_time_seconds)
    # Retired drivers leave the running order and are listed at the bottom.
    retired = {
        driver: info for driver, info in (retirements or {}).items()
        if info["out_from"] <= current_time
    }
    latest = current_frames.loc[~current_frames["Driver"].isin(list(retired))].copy()
    if latest.empty:
        return pd.DataFrame(columns=columns)
    latest["LapNumber"] = pd.to_numeric(
        latest["LapNumber"],
        errors="coerce",
    ).fillna(0)
    latest["Distance"] = pd.to_numeric(
        latest["Distance"],
        errors="coerce",
    ).fillna(0)
    latest = latest.sort_values(
        ["LapNumber", "Distance"],
        ascending=[False, False],
    ).reset_index(drop=True)
    latest["Position"] = np.arange(1, len(latest) + 1)
    # Last completed lap time per driver.
    if (
        laps is not None
        and not laps.empty
        and "TimeSeconds" in laps.columns
    ):
        completed = laps.loc[
            (laps["TimeSeconds"] <= current_time)
            & laps["LapTime"].notna()
        ].copy()
        if not completed.empty:
            last_laps = (
                completed.sort_values("TimeSeconds")
                .groupby("Driver", as_index=False)
                .tail(1)[["Driver", "LapTime"]]
                .rename(columns={"LapTime": "LastLapTime"})
            )
            latest = latest.merge(
                last_laps,
                on="Driver",
                how="left",
            )
        else:
            latest["LastLapTime"] = np.nan
    else:
        latest["LastLapTime"] = np.nan
    latest["LastLapTime"] = latest["LastLapTime"].map(
        format_lap_time
    )
    latest["Interval"] = "–"
    # Rough interval: distance gap / following car's speed. Unofficial; inaccurate at lap boundaries.
    for index in range(1, len(latest)):
        leader = latest.iloc[index - 1]
        driver_row = latest.iloc[index]
        distance_gap = (
            float(leader["Distance"])
            - float(driver_row["Distance"])
        )
        speed = driver_row.get("Speed")
        if pd.notna(speed) and float(speed) > 0:
            gap_seconds = max(0, distance_gap) / (
                float(speed) / 3.6
            )
            latest.loc[index, "Interval"] = (
                f"+{gap_seconds:.1f}s"
            )
    result = pd.DataFrame({
        "Position": latest["Position"],
        "Driver": latest["Driver"],
        "Lap": latest["LapNumber"].astype(int),
        "LastLapTime": latest["LastLapTime"],
        "Interval": latest["Interval"],
        "Compound": latest["Compound"],
        "TyreLife": latest["TyreLife"],
        "Stint": latest["Stint"] if "Stint" in latest.columns else np.nan,
        "Status": "",
    })
    if retired:
        last_times = {}
        if laps is not None and not laps.empty:
            finished = laps.dropna(subset=["LapTime"]).sort_values("LapNumber")
            last_times = finished.groupby("Driver")["LapTime"].last().to_dict()
        out = pd.DataFrame([
            {
                "Position": None,
                "Driver": driver,
                "Lap": info["lap"],
                "LastLapTime": format_lap_time(last_times.get(driver)),
                "Interval": "–",
                "Compound": np.nan,
                "TyreLife": np.nan,
                "Stint": np.nan,
                "Status": f"Kiesett ({info['lap']}. kör)",
            }
            # Later retirements rank higher, as in the official classification.
            for driver, info in sorted(retired.items(), key=lambda item: -item[1]["out_from"])
        ])
        result = pd.concat([result, out], ignore_index=True)
    # Tyre data from the stint table, counted per lap from tyre_age_at_start.
    if stints is not None and not stints.empty:
        tyres = tyre_state(stints, dict(zip(result["Driver"], result["Lap"])))
        result = result.drop(columns=["Compound", "TyreLife", "Stint"]).merge(
            tyres, on="Driver", how="left"
        )
    return result[columns].reset_index(drop=True)

def build_track_figure(track, revision=None):
    """Track map: static track line plus an empty car layer that the browser animates.

    The car layer is the last trace; assets/replay_animation.js updates only
    that trace, so the figure itself is sent once per loaded session. The view
    is framed on the track with a fixed 1:1 aspect; `revision` (one per
    session) resets any zoom when another session is loaded.
    """
    figure = go.Figure()
    x_range = y_range = None
    if track is not None and not track.empty:
        x, y = track["X"].to_numpy(dtype=float), track["Y"].to_numpy(dtype=float)
        # Two layers give the track an outlined, broadcast-style look.
        for width, color in ((16, "#2b2d3a"), (7, "#c9ced8")):
            figure.add_trace(go.Scatter(
                x=x, y=y, mode="lines", hoverinfo="skip",
                line={"width": width, "color": color, "shape": "spline", "smoothing": 0.6},
            ))
        # Start/finish line: a short white bar across the track at the lap start.
        span = max(x.max() - x.min(), y.max() - y.min())
        direction = np.array([x[min(3, len(x) - 1)] - x[0], y[min(3, len(y) - 1)] - y[0]])
        norm = np.hypot(*direction) or 1.0
        normal = np.array([-direction[1], direction[0]]) / norm * span * 0.018
        figure.add_trace(go.Scatter(
            x=[x[0] - normal[0], x[0] + normal[0]], y=[y[0] - normal[1], y[0] + normal[1]],
            mode="lines", hoverinfo="skip", line={"width": 4, "color": "#ffffff"},
        ))
        pad = span * 0.07
        x_range = [x.min() - pad, x.max() + pad]
        y_range = [y.min() - pad, y.max() + pad]
    figure.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers+text",
            text=[],
            textposition="top center",
            textfont={"color": "#ffffff", "size": 11, "family": "Inter, Segoe UI, Arial, sans-serif"},
            name="Versenyzők",
            marker={
                "size": 15,
                # Team colours with a white outline so dark colours stay visible.
                "color": [],
                "line": {"width": 2, "color": "#ffffff"},
            },
            hovertemplate="%{text}<extra></extra>",
        )
    )
    figure.update_layout(
        height=TRACK_MAP_HEIGHT,
        paper_bgcolor=TRACK_MAP_BACKGROUND,
        plot_bgcolor=TRACK_MAP_BACKGROUND,
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1, "range": x_range,
               "autorange": x_range is None},
        yaxis={"visible": False, "range": y_range, "autorange": y_range is None},
        showlegend=False,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        dragmode="pan",
        hoverlabel={"bgcolor": "#22232e", "font": {"color": "#ffffff"}},
        uirevision=revision or "track",
    )
    return figure


def _smooth_segments(values, sigma):
    """Gaussian smoothing within each run of consecutive non-missing samples."""
    if sigma <= 0:
        return values
    radius = int(np.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    result = values.copy()
    present = ~np.isnan(values)
    edges = np.flatnonzero(np.diff(np.r_[False, present, False].astype(int)))
    for start, stop in zip(edges[::2], edges[1::2]):
        segment = values[start:stop]
        padded = np.pad(segment, radius, mode="edge")
        result[start:stop] = np.convolve(padded, kernel, mode="valid")
    return result


def build_position_window(frames_by_driver, driver_colors, start_seconds, end_seconds, retirements=None):
    """Car positions on the common timeline between two replay times, for the browser.

    Values are on the SAMPLE_INTERVAL_SECONDS grid starting at t0; missing
    samples are None so the browser hides the car instead of guessing.
    Positions are smoothed in time (display only) to even out the irregular
    OpenF1 sampling.
    """
    t0 = max(0.0, round(start_seconds / SAMPLE_INTERVAL_SECONDS) * SAMPLE_INTERVAL_SECONDS)
    count = int(round((end_seconds - t0) / SAMPLE_INTERVAL_SECONDS)) + 1
    sigma = POSITION_SMOOTHING_SECONDS / SAMPLE_INTERVAL_SECONDS
    margin = int(np.ceil(3 * sigma))
    drivers, colors, xs, ys = [], [], [], []
    for driver, frames in frames_by_driver.items():
        # Read a margin on both sides so smoothing near the window edges uses real samples.
        first = t0 - margin * SAMPLE_INTERVAL_SECONDS
        total = count + 2 * margin
        times = frames["TimeSeconds"].to_numpy(dtype=float)
        index = np.rint((times - first) / SAMPLE_INTERVAL_SECONDS).astype(int)
        valid = (index >= 0) & (index < total)
        x = np.full(total, np.nan)
        y = np.full(total, np.nan)
        x[index[valid]] = frames["X"].to_numpy(dtype=float)[valid]
        y[index[valid]] = frames["Y"].to_numpy(dtype=float)[valid]
        x = _smooth_segments(x, sigma)[margin:margin + count]
        y = _smooth_segments(y, sigma)[margin:margin + count]
        hidden_from = (retirements or {}).get(str(driver), {}).get("hidden_from")
        if hidden_from is not None:
            # Retired cars leave the track from the next race lap.
            grid = t0 + np.arange(count) * SAMPLE_INTERVAL_SECONDS
            x[grid >= hidden_from] = np.nan
            y[grid >= hidden_from] = np.nan
        drivers.append(str(driver))
        colors.append((driver_colors or {}).get(driver, DEFAULT_DRIVER_COLOR))
        xs.append([None if np.isnan(v) else round(float(v), 1) for v in x])
        ys.append([None if np.isnan(v) else round(float(v), 1) for v in y])
    return {
        "t0": t0,
        "dt": SAMPLE_INTERVAL_SECONDS,
        "start": t0,
        "end": t0 + (count - 1) * SAMPLE_INTERVAL_SECONDS,
        "drivers": drivers,
        "colors": colors,
        "x": xs,
        "y": ys,
    }
