import numpy as np
import pandas as pd

from src.dashboard.callbacks.replay_callbacks import lap_time_ticks
from src.dashboard.replay_player import STINT_COLUMNS, enrich_lap_table, fastest_laps


def test_enrich_lap_table_fills_tyres_and_position():
    laps = pd.DataFrame({
        "Driver": ["NOR"] * 4,
        "LapNumber": [1, 2, 3, 4],
        "LapTime": [90.0, 84.0, 85.0, 83.0],
        "TimeSeconds": [90.0, 174.0, 259.0, 342.0],
    })
    stints = pd.DataFrame(
        [["NOR", 1, 1, 2, "MEDIUM", 0], ["NOR", 2, 3, 4, "HARD", 2]], columns=STINT_COLUMNS
    )
    positions = {"NOR": pd.DataFrame({"TimeSeconds": [0.0, 100.0, 300.0], "position": [5, 3, 1]})}
    result = enrich_lap_table(laps, stints, positions)

    assert result["Compound"].tolist() == ["MEDIUM", "MEDIUM", "HARD", "HARD"]
    # Used HARD set: tyre_age_at_start 2, so 2 then 3 laps on the set.
    assert result["TyreLife"].tolist() == [0, 1, 2, 3]
    assert result["Stint"].tolist() == [1, 1, 2, 2]
    # Last known position at each lap end.
    assert result["Position"].tolist() == [5, 3, 3, 1]


def test_fastest_laps_one_personal_best_per_driver_up_to_replay_time():
    laps = pd.DataFrame({
        "Driver": ["NOR", "NOR", "NOR", "PIA", "PIA", "PIA"],
        "LapNumber": [1, 2, 3, 1, 2, 3],
        "LapTime": [90.0, 82.0, 82.0, 91.0, 81.5, 80.0],
        "TimeSeconds": [90.0, 172.0, 254.0, 91.0, 172.5, 252.5],
    })
    # At 200 s: PIA's lap 3 is not finished yet.
    overall, personal = fastest_laps(laps, 200.0)
    assert overall == {"Driver": "PIA", "LapNumber": 2, "LapTime": 81.5}
    assert personal == {"NOR": 2, "PIA": 2}
    # Later: NOR's tie on lap 3 keeps the earlier lap; PIA takes the overall fastest.
    overall, personal = fastest_laps(laps, 300.0)
    assert overall["LapNumber"] == 3 and overall["Driver"] == "PIA"
    assert personal == {"NOR": 2, "PIA": 3}
    assert fastest_laps(laps, 10.0) == (None, {})


def test_lap_time_ticks_are_minutes_and_seconds():
    values, labels = lap_time_ticks(80.9, 82.1)
    assert labels[0] == "1:21.000" and labels[-1] == "1:22.000"
    assert np.allclose(np.diff(values), 0.2)
    values, labels = lap_time_ticks(95.0, 130.0)
    assert labels[:2] == ["1:35.000", "1:40.000"] and "2:00.000" in labels


def test_replay_view_handles_driver_without_completed_laps():
    """Regression: right after loading, the selected driver has no completed lap."""
    from src.dashboard.callbacks import replay_callbacks
    from src.dashboard.replay_service import REPLAY_CACHE

    class FakeApp:
        def __init__(self):
            self.functions = {}

        def callback(self, *args, **kwargs):
            def register(function):
                self.functions[function.__name__] = function
                return function
            return register

        def clientside_callback(self, *args, **kwargs):
            pass

    app = FakeApp()
    replay_callbacks.register_replay_callbacks(app)
    laps = pd.DataFrame({
        "Driver": ["NOR", "PIA"], "DriverNumber": [4, 81], "LapNumber": [1, 1],
        "LapTime": [90.0, 89.0], "TimeSeconds": [90.0, 89.0], "LapStartTime": [0.0, 0.0],
        "Sector1Time": np.nan, "Sector2Time": np.nan, "Sector3Time": np.nan,
        "Compound": "MEDIUM", "TyreLife": 0, "Stint": 1, "Position": [2, 1], "IsPersonalBest": np.nan,
    })
    frames = pd.DataFrame({
        "Driver": ["NOR", "PIA"], "TimeSeconds": [0.0, 0.0], "LapNumber": [1, 1],
        "X": [0.0, 1.0], "Y": [0.0, 1.0], "Distance": [0.0, 1.0], "Speed": np.nan,
        "Compound": np.nan, "TyreLife": np.nan,
    })
    REPLAY_CACHE["empty-test"] = {
        "laps": laps, "frames": frames, "frames_by_driver": {d: g for d, g in frames.groupby("Driver")},
    }
    try:
        before = app.functions["update_replay_view"]({"replay_id": "empty-test"}, 0.0, "NOR", None)
        assert before[5] == []  # no completed laps yet
        after = app.functions["update_replay_view"]({"replay_id": "empty-test"}, 95.0, "PIA", None)
        assert "⏱" in after[5][0]["LapTime"]
    finally:
        REPLAY_CACHE.pop("empty-test", None)
