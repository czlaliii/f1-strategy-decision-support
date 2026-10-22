"""Historical race replay data loader using locally stored OpenF1 data."""
import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
# Maximum age of a position sample, in seconds.
POSITION_MAX_AGE_SECONDS = 2.0

def _read_json(path: Path) -> pd.DataFrame:
    """Read a JSON array into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return pd.DataFrame(data)

def _prepare_timestamps(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Parse timestamps and sort records chronologically."""
    if df.empty:
        return df.copy()
    result = df.copy()
    result[column] = pd.to_datetime(
        result[column],
        utc=True,
        errors="coerce",
    )
    result = result.dropna(subset=[column])
    return result.sort_values(column).reset_index(drop=True)

class ReplayData:
    """Load and query replay data for one historical race session."""
    def __init__(
        self,
        race_name: str,
        session_key: int,
        data_dir: Path | None = None,
    ):
        self.race_name = race_name
        self.session_key = session_key
        self.data_dir = data_dir or RAW_DATA_DIR / race_name
        self.drivers = {}
        self.locations = {}
        self.positions = {}
        self.intervals = {}
        self.laps = {}
        self.start_time = None
        self.end_time = None
    def load(self) -> "ReplayData":
        """Load local data and prepare the replay timeline."""
        drivers_df = _read_json(self.data_dir / "drivers.json")
        drivers_df = drivers_df[
            drivers_df["session_key"] == self.session_key
        ].copy()
        if drivers_df.empty:
            raise ValueError(
                f"No drivers found for session {self.session_key}"
            )
        for _, row in drivers_df.iterrows():
            number = int(row["driver_number"])
            self.drivers[number] = row.to_dict()
        laps_df = _read_json(self.data_dir / "laps.json")
        laps_df = laps_df[
            laps_df["session_key"] == self.session_key
        ].copy()
        laps_df = _prepare_timestamps(laps_df, "date_start")
        if laps_df.empty:
            raise ValueError(
                f"No lap data found for session {self.session_key}"
            )
        laps_df["driver_number"] = laps_df["driver_number"].astype(int)
        laps_df["lap_number"] = pd.to_numeric(
            laps_df["lap_number"], errors="coerce"
        )
        laps_df["lap_duration"] = pd.to_numeric(
            laps_df["lap_duration"], errors="coerce"
        )
        # Use lap starts and durations to determine the actual race window.
        valid_laps = laps_df.dropna(
            subset=["date_start", "lap_duration"]
        )
        valid_laps = valid_laps[valid_laps["lap_duration"] > 0]
        if valid_laps.empty:
            raise ValueError("No valid lap durations found")
        lap_ends = (
            valid_laps["date_start"]
            + pd.to_timedelta(valid_laps["lap_duration"], unit="s")
        )
        self.start_time = laps_df["date_start"].min()
        self.end_time = lap_ends.max()
        for number, group in laps_df.groupby("driver_number"):
            self.laps[int(number)] = group.sort_values(
                "date_start"
            ).reset_index(drop=True)
        # Load sparse position-change events.
        positions_df = _read_json(self.data_dir / "position.json")
        positions_df = positions_df[
            positions_df["session_key"] == self.session_key
        ].copy()
        positions_df = _prepare_timestamps(positions_df, "date")
        if not positions_df.empty:
            positions_df["driver_number"] = (
                positions_df["driver_number"].astype(int)
            )
            for number, group in positions_df.groupby("driver_number"):
                self.positions[int(number)] = group.reset_index(drop=True)
        # Load gap and interval samples.
        intervals_df = _read_json(self.data_dir / "intervals.json")
        intervals_df = intervals_df[
            intervals_df["session_key"] == self.session_key
        ].copy()
        intervals_df = _prepare_timestamps(intervals_df, "date")
        if not intervals_df.empty:
            intervals_df["driver_number"] = (
                intervals_df["driver_number"].astype(int)
            )
            for number, group in intervals_df.groupby("driver_number"):
                self.intervals[int(number)] = group.reset_index(drop=True)
        # Load the coordinate file for each driver in this session.
        for number in self.drivers:
            path = self.data_dir / f"location_driver_{number}.json"
            if not path.exists():
                continue
            location_df = _read_json(path)
            location_df = location_df[
                location_df["session_key"] == self.session_key
            ].copy()
            location_df = _prepare_timestamps(location_df, "date")
            if location_df.empty:
                continue
            for column in ["x", "y", "z"]:
                location_df[column] = pd.to_numeric(
                    location_df[column], errors="coerce"
                )
            location_df = location_df.dropna(subset=["x", "y"])
            location_df = location_df[
                (location_df["x"] != 0)
                | (location_df["y"] != 0)
            ].reset_index(drop=True)
            self.locations[number] = location_df
        if not self.locations:
            raise ValueError("No usable location data found")
        return self
    @staticmethod
    def _latest_record(df, timestamp, column="date"):
        """Return the last record at or before the requested time."""
        if df is None or df.empty:
            return None
        index = df[column].searchsorted(timestamp, side="right") - 1
        if index < 0:
            return None
        return df.iloc[index]
    def snapshot(self, elapsed_seconds: float) -> dict:
        """Return the race state at the requested elapsed time."""
        if self.start_time is None or self.end_time is None:
            raise RuntimeError("Call load() before requesting a snapshot")
        if elapsed_seconds < 0:
            raise ValueError("Elapsed time cannot be negative")
        timestamp = self.start_time + pd.to_timedelta(
            elapsed_seconds, unit="s"
        )
        timestamp = min(timestamp, self.end_time)
        cars = []
        for number, driver in self.drivers.items():
            location_df = self.locations.get(number)
            location = self._latest_record(
                location_df, timestamp
            )
            x = y = z = None
            if location is not None:
                age = (
                    timestamp - location["date"]
                ).total_seconds()
                if 0 <= age <= POSITION_MAX_AGE_SECONDS:
                    x = float(location["x"])
                    y = float(location["y"])
                    z = float(location["z"])
            position_record = self._latest_record(
                self.positions.get(number), timestamp
            )
            interval_record = self._latest_record(
                self.intervals.get(number), timestamp
            )
            lap_df = self.laps.get(number)
            lap_number = None
            last_lap_time = None
            if lap_df is not None and not lap_df.empty:
                started_laps = lap_df[
                    lap_df["date_start"] <= timestamp
                ]
                if not started_laps.empty:
                    lap_number = int(
                        started_laps["lap_number"].max()
                    )
                completed_laps = lap_df[
                    lap_df["lap_duration"].notna()
                    & (
                        lap_df["date_start"]
                        + pd.to_timedelta(
                            lap_df["lap_duration"], unit="s"
                        )
                        <= timestamp
                    )
                ]
                if not completed_laps.empty:
                    last_lap_time = float(
                        completed_laps.iloc[-1]["lap_duration"]
                    )
            cars.append({
                "driver_number": number,
                "name": driver.get("name_acronym", str(number)),
                "full_name": driver.get("full_name", ""),
                "team": driver.get("team_name", ""),
                "team_colour": driver.get("team_colour", "808080"),
                "x": x,
                "y": y,
                "z": z,
                "position": (
                    int(position_record["position"])
                    if position_record is not None
                    and pd.notna(position_record["position"])
                    else None
                ),
                "gap_to_leader": (
                    interval_record["gap_to_leader"]
                    if interval_record is not None
                    else None
                ),
                "interval": (
                    interval_record["interval"]
                    if interval_record is not None
                    else None
                ),
                "lap_number": lap_number,
                "last_lap_time": last_lap_time,
            })
        # Sort classified drivers first by their latest known position.
        cars.sort(
            key=lambda car: (
                car["position"] is None,
                car["position"] or float("inf"),
                car["driver_number"],
            )
        )
        return {
            "race_name": self.race_name,
            "session_key": self.session_key,
            "timestamp": timestamp.isoformat(),
            "elapsed_seconds": (
                timestamp - self.start_time
            ).total_seconds(),
            "cars": cars,
        }
