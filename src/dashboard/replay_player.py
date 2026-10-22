"""F1 session replay: telemetria, idővonal, leaderboard és pálya."""
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.dashboard.replay_data import ReplayData

SAMPLE_INTERVAL_SECONDS = 0.1
MAX_INTERPOLATION_GAP_SECONDS = 2.0


def format_elapsed_time(seconds):
    """Eltelt idő formázása: 01:24:35."""
    if seconds is None or pd.isna(seconds):
        return "–"
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"

def format_lap_time(seconds):
    """Köridő formázása: 83.456 -> 1:23.456."""
    if seconds is None or pd.isna(seconds):
        return "–"
    minutes = int(seconds // 60)
    remaining_seconds = seconds - minutes * 60
    return f"{minutes}:{remaining_seconds:06.3f}"

def _seconds(value):
    """Egy időtartam átalakítása másodpercre."""
    if value is None or pd.isna(value):
        return np.nan
    return pd.to_timedelta(
        value,
        errors="coerce",
    ).total_seconds()

def _resample_driver(driver_data, timeline):
    """
    Egy versenyző telemetriáját közös idővonalra mintavételezi.
    Nem interpolál két különböző kör között, és nem köt össze
    hosszabb telemetriahiányokat.
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
        # Csak rövid, telemetria alapján áthidalható időközöket
        # interpolálunk.
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
        # A diszkrét értékeket a legutóbbi ismert mintából vesszük.
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

def _prepare_track_coordinates(data, x_column="X", y_column="Y"):
    """Egységes, tiszta X/Y koordinátákat ad vissza a pályarajzoláshoz."""
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
    """Pályavonalat készít FastF1-adatokból, tartalék forrással."""
    try:
        fastest_lap = session.laps.pick_fastest()
        track = _prepare_track_coordinates(fastest_lap.get_telemetry())
    except (KeyError, ValueError, AttributeError):
        track = pd.DataFrame(columns=["X", "Y"])

    if track.empty:
        track = _prepare_track_coordinates(raw_telemetry)
    return track


def _build_local_track(replay, sample_count=800):
    """Robusztus pályavonalat épít több teljes kör koordinátáinak mediánjából.

    Minden használható kört a megtett geometriai távolság mentén azonos
    számú pontra mintavételezünk. Az azonos relatív pályapozíciók X/Y
    koordinátáinak mediánja csökkenti az egyedi hiányok és a boxkiállásos
    körök hatását.
    """
    normalized_laps = []

    for number, driver_laps in replay.laps.items():
        location_df = replay.locations.get(number)
        if location_df is None or location_df.empty:
            continue

        location_df = location_df.sort_values("date")
        valid_laps = driver_laps.dropna(
            subset=["date_start", "lap_duration"]
        )

        for _, lap in valid_laps.iterrows():
            lap_start = lap["date_start"]
            duration = pd.to_numeric(lap["lap_duration"], errors="coerce")
            if pd.isna(duration) or duration < 30 or duration > 200:
                continue
            lap_end = lap_start + pd.to_timedelta(float(duration), unit="s")
            lap_locations = location_df.loc[
                (location_df["date"] >= lap_start)
                & (location_df["date"] <= lap_end)
            ]
            points = _prepare_track_coordinates(
                lap_locations, x_column="x", y_column="y"
            )
            if len(points) < 50:
                continue

            # Csak egymást követő azonos pontokat távolítunk el, hogy a
            # körvonal sorrendje megmaradjon.
            xy = points[["X", "Y"]].to_numpy(dtype=float)
            step_lengths = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
            keep = np.r_[True, step_lengths > 0.01]
            xy = xy[keep]
            if len(xy) < 50:
                continue

            step_lengths = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
            cumulative_distance = np.r_[0.0, np.cumsum(step_lengths)]
            total_distance = cumulative_distance[-1]
            # Kiszűrjük az egyértelműen hibás vagy töredékes köröket.
            if not 1000 <= total_distance <= 20000:
                continue
            if np.any(np.diff(cumulative_distance) <= 0):
                continue

            target_distance = np.linspace(0.0, total_distance, sample_count)
            resampled = np.column_stack((
                np.interp(target_distance, cumulative_distance, xy[:, 0]),
                np.interp(target_distance, cumulative_distance, xy[:, 1]),
            ))
            normalized_laps.append(resampled)

    if not normalized_laps:
        # Tartalék megoldás: ha a mediános rekonstrukció szigorú
        # feltételei nem teljesülnek, használjuk a korábbi, egyszerű
        # módszert, amely az első elegendően mintázott körből rajzol.
        for number, driver_laps in replay.laps.items():
            location_df = replay.locations.get(number)
            if location_df is None or location_df.empty:
                continue

            valid_laps = driver_laps.dropna(
                subset=["date_start", "lap_duration"]
            )
            for _, lap in valid_laps.iterrows():
                lap_start = lap["date_start"]
                duration = pd.to_numeric(
                    lap["lap_duration"], errors="coerce"
                )
                if pd.isna(duration) or duration <= 0:
                    continue
                lap_end = lap_start + pd.to_timedelta(
                    float(duration), unit="s"
                )
                lap_locations = location_df.loc[
                    (location_df["date"] >= lap_start)
                    & (location_df["date"] <= lap_end)
                ]
                track = _prepare_track_coordinates(
                    lap_locations, x_column="x", y_column="y"
                )
                if len(track) >= 20:
                    return track.reset_index(drop=True)

        raise ValueError(
            "Nem sikerült pályavonalat előállítani a helyi adatokból. "
            "A mediános rekonstrukció és a korábbi tartalék módszer is "
            "üres eredményt adott."
        )

    # A koordinátánkénti medián robusztusabb, mint egyetlen kör kiválasztása:
    # a több körben közös pályavonal megmarad, az egyedi eltérések mérséklődnek.
    stacked = np.stack(normalized_laps, axis=0)
    median_xy = np.median(stacked, axis=0)
    track = pd.DataFrame(median_xy, columns=["X", "Y"])
    track = track.dropna().drop_duplicates().reset_index(drop=True)

    if len(track) < 50:
        # A mediános eredmény túl rövid: próbáljuk meg az eredeti módszert.
        for number, driver_laps in replay.laps.items():
            location_df = replay.locations.get(number)
            if location_df is None or location_df.empty:
                continue
            valid_laps = driver_laps.dropna(
                subset=["date_start", "lap_duration"]
            )
            for _, lap in valid_laps.iterrows():
                lap_start = lap["date_start"]
                duration = pd.to_numeric(
                    lap["lap_duration"], errors="coerce"
                )
                if pd.isna(duration) or duration <= 0:
                    continue
                lap_end = lap_start + pd.to_timedelta(
                    float(duration), unit="s"
                )
                lap_locations = location_df.loc[
                    (location_df["date"] >= lap_start)
                    & (location_df["date"] <= lap_end)
                ]
                fallback_track = _prepare_track_coordinates(
                    lap_locations, x_column="x", y_column="y"
                )
                if len(fallback_track) >= 20:
                    return fallback_track.reset_index(drop=True)
        raise ValueError("A pályavonal túl rövid, és a tartalék módszer sem talált használható kört.")
    return track


def load_session_data(year, event_name, session_name):
    """
    Betölti a FastF1-sessiont és előállítja a replay adatait.
    Visszatérés:
        {
            "laps": körrekordok DataFrame,
            "frames": idővonalra mintavételezett telemetria DataFrame,
            "track": pályavonal DataFrame,
        }
    """
    session = fastf1.get_session(
        int(year),
        event_name,
        session_name,
    )
    # A pozíciótelemetria betöltése is szükséges.
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
    # ----------------------------------------------
    # Köradatok előkészítése
    # ----------------------------------------------
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
    # ----------------------------------------------
    # Telemetria betöltése körönként
    # ----------------------------------------------
    telemetry_parts = []
    for _, lap in session.laps.iterlaps():
        try:
            telemetry = lap.get_telemetry()
        except (KeyError, ValueError, AttributeError):
            # Egyes körökhöz hiányozhat a pozíciótelemetria.
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
    # Egységes, nulláról induló idővonal.
    time_origin = float(
        raw_telemetry["TimeSeconds"].min()
    )
    raw_telemetry["TimeSeconds"] -= time_origin
    # A körrekordok időpontját ugyanahhoz az idővonalhoz igazítjuk.
    for column in ["Time", "LapStartTime"]:
        if column in laps.columns:
            laps[column] -= time_origin
    if "Time" in laps.columns:
        laps["TimeSeconds"] = laps["Time"]
    else:
        laps["TimeSeconds"] = np.nan
    if "LapStartTime" in laps.columns:
        laps["LapStartSeconds"] = laps["LapStartTime"]
    # ----------------------------------------------
    # Közös idővonal létrehozása
    # ----------------------------------------------
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
    # A pályavonal felépítése külön függvényben van, így a
    # koordináták tisztítása nem keveredik a session feldolgozásával.
    track = _build_fastf1_track(session, raw_telemetry)
    return {
        "laps": laps,
        "frames": frames,
        "track": track,
    }

def load_local_session_data(
    race_name="australia_2025",
    session_key=9693,
):
    """Helyi OpenF1-adatok átalakítása a replay által használt formátumra.

    Ha a session_key nincs megadva, a helyi drivers.json/laps.json fájlokból
    választjuk ki a futam sessionjét. Így a letöltött adatokat nem kell újra
    külső API-ról lekérni.
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
        # --------------------------------------------------
        # Autópozíciók előállítása közös idővonalon
        # --------------------------------------------------
        if location_df is not None and not location_df.empty:
            location = location_df.copy().sort_values("date")
            # A verseny kezdeténél legfeljebb 2 másodperccel
            # korábbi koordinátát is felhasználhatunk.
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
                # A körszámot a legutóbb megkezdett körből
                # határozzuk meg.
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
                # Ezeket az adatokat a helyi pozíciófájlok
                # önmagukban nem tartalmazzák.
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
        # --------------------------------------------------
        # Köradatok átalakítása
        # --------------------------------------------------
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
    # A pályát nem az első elfogadható körből vesszük: az összes
    # elérhető kör közül a legtöbb érvényes koordinátát tartalmazót választjuk.
    track = _build_local_track(replay)
    return {
        "laps": laps,
        "frames": frames,
        "track": track,
    }

def build_current_snapshot(frames_by_driver, current_time_seconds):
    """Az aktuális időponthoz tartozó állapot versenyzőnként."""
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

def build_leaderboard(current_frames, laps, current_time_seconds):
    """
    Leaderboard készítése a lejátszás pillanatnyi állapotából.
    A sorrendet a körszám, majd a körön belüli megtett
    távolság alapján becsüljük. Ez nem hivatalos FIA-eredmény.
    """
    columns = [
        "Position",
        "Driver",
        "Lap",
        "LastLapTime",
        "Interval",
        "Compound",
        "TyreLife",
    ]
    if (
        current_frames is None
        or current_frames.empty
        or current_time_seconds is None
    ):
        return pd.DataFrame(columns=columns)
    current_time = float(current_time_seconds)
    latest = current_frames.copy()
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
    # Utolsó befejezett köridő versenyzőnként.
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
    # Az intervallum első, egyszerű becslése: a megtett
    # távolságkülönbség osztva a követő autó sebességével.
    # Nem hivatalos időmérés, és körhatáron pontatlan lehet.
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
    })
    return result[columns].reset_index(drop=True)

def build_track_figure(current_frames, track, current_time_seconds):
    """Pályavonal és az aktuális autópozíciók Plotly-ábrája."""
    figure = go.Figure()
    if track is not None and not track.empty:
        figure.add_trace(
            go.Scatter(
                x=track["X"],
                y=track["Y"],
                mode="lines",
                name="Pálya",
                line={"width": 3, "color": "#777777"},
                hoverinfo="skip",
            )
        )
    if (
        current_frames is not None
        and not current_frames.empty
        and current_time_seconds is not None
    ):
        current = (
            current_frames.copy()
            if current_frames is not None
            else pd.DataFrame()
        )
        if not current.empty:
            current = (
                current.sort_values("TimeSeconds")
                .groupby("Driver", as_index=False)
                .tail(1)
            )
            current = current.loc[
                current["X"].notna()
                & current["Y"].notna()
                & (
                    float(current_time_seconds)
                    - current["TimeSeconds"]
                    <= 3.0
                )
            ]
            if not current.empty:
                figure.add_trace(
                    go.Scatter(
                        x=current["X"],
                        y=current["Y"],
                        mode="markers+text",
                        text=current["Driver"],
                        textposition="top center",
                        name="Versenyzők",
                        marker={"size": 10},
                        customdata=current[
                            ["LapNumber", "Speed"]
                        ].to_numpy(),
                        hovertemplate=(
                            "%{text}<br>"
                            "Kör: %{customdata[0]}<br>"
                            "Sebesség: %{customdata[1]:.0f} km/h"
                            "<extra></extra>"
                        ),
                    )
                )
    figure.update_layout(
        title="Pálya és versenyzőpozíciók",
        xaxis={"visible": False, "scaleanchor": "y"},
        yaxis={"visible": False},
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return figure
