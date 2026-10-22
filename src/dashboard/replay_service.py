"""Shared replay state and timeline helpers used by replay callbacks."""
import pandas as pd

# In-memory cache avoids sending large telemetry DataFrames to the browser.
REPLAY_CACHE: dict[str, dict] = {}

def get_lap_start_times(replay_id: str) -> pd.Series:
    """Return the best available elapsed-time estimate for each lap start."""
    replay = REPLAY_CACHE.get(replay_id)
    if replay is None:
        return pd.Series(dtype=float)
    laps = replay["laps"]
    if (
        not laps.empty
        and "LapNumber" in laps.columns
        and "LapStartTime" in laps.columns
    ):
        starts = laps[["LapNumber", "LapStartTime"]].copy()
        starts["LapNumber"] = pd.to_numeric(starts["LapNumber"], errors="coerce")
        starts["LapStartTime"] = pd.to_numeric(starts["LapStartTime"], errors="coerce")
        starts = starts.dropna()
        if not starts.empty:
            return starts.groupby("LapNumber")["LapStartTime"].min().sort_values()
    frames = replay["frames"]
    if frames.empty:
        return pd.Series(dtype=float)
    fallback = frames[["LapNumber", "TimeSeconds"]].copy()
    fallback["LapNumber"] = pd.to_numeric(fallback["LapNumber"], errors="coerce")
    fallback["TimeSeconds"] = pd.to_numeric(fallback["TimeSeconds"], errors="coerce")
    fallback = fallback.dropna()
    return fallback.groupby("LapNumber")["TimeSeconds"].min().sort_values()

def lap_to_time(replay_id: str, lap_number: int) -> float:
    """Convert a lap number to its elapsed-time position."""
    starts = get_lap_start_times(replay_id)
    if starts.empty:
        return 0.0
    lap_number = int(lap_number)
    if lap_number in starts.index:
        return float(starts.loc[lap_number])
    earlier = starts.loc[starts.index <= lap_number]
    if not earlier.empty:
        return float(earlier.iloc[-1])
    return float(starts.iloc[0])

def time_to_lap(replay_id: str, seconds: float) -> int:
    """Convert an elapsed-time position to the latest lap start."""
    starts = get_lap_start_times(replay_id)
    if starts.empty:
        return 1
    eligible = starts.loc[starts <= float(seconds)]
    if eligible.empty:
        return int(starts.index[0])
    return int(eligible.index[-1])
