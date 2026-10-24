import pandas as pd

from src.dashboard.race_progression import build_progression_figure, progression_data
from src.dashboard.replay_player import STINT_COLUMNS


def sample():
    laps = pd.DataFrame({
        "Driver": ["NOR", "NOR", "NOR", "PIA", "PIA", "PIA", "VER"],
        "LapNumber": [1, 2, 3, 1, 2, 3, 1],
        "Position": [2, 1, 1, 1, 2, 2, 3],
        "TimeSeconds": [90.0, 180.0, 270.0, 89.0, 181.0, 271.0, 92.0],
    })
    stints = pd.DataFrame(
        [["NOR", 1, 1, 2, "MEDIUM", 0], ["NOR", 2, 3, 3, "HARD", 0], ["PIA", 1, 1, 3, "MEDIUM", 0]],
        columns=STINT_COLUMNS,
    )
    return progression_data(
        laps,
        grid={"NOR": 3, "PIA": 1, "VER": 2},
        stints=stints,
        retirements={"VER": {"lap": 1, "out_from": 150.0, "hidden_from": 240.0}},
        track_status=[{"status": "SC", "start": 100.0, "end": 200.0, "ending_from": None}],
        lap_of=lambda seconds: int(seconds // 90) + 1,
    )


def test_progression_data_grid_pits_and_bands():
    data = sample()
    grid = data["positions"][data["positions"]["Lap"] == 0].set_index("Driver")["Position"]
    assert grid.to_dict() == {"NOR": 3, "PIA": 1, "VER": 2}
    # NOR's first stint ends on lap 2: that is the in-lap of the stop.
    assert data["pit_laps"] == {"NOR": [2], "PIA": []}
    assert data["retired"] == {"VER": 1}
    assert data["bands"] == [{"status": "SC", "start": 2, "end": 3}]


def test_figure_shows_only_finished_laps_and_styles_drivers():
    data = sample()
    colors = {"NOR": "#ff8000", "PIA": "#ff8000", "VER": "#3671c6"}
    figure = build_progression_figure(data, 185.0, colors, team_of=colors, highlight="NOR")
    lines = {trace.name: trace for trace in figure.data if trace.mode == "lines"}
    # At 185 s NOR and PIA have finished lap 2; lap 3 is not shown yet.
    assert list(lines["NOR"].x) == [0, 1, 2] and list(lines["PIA"].x) == [0, 1, 2]
    # Teammates share a colour; the second (alphabetical) driver of the team is dashed.
    assert lines["NOR"].line.dash == "solid" and lines["PIA"].line.dash == "dash"
    # The highlighted driver is bold and drawn last (on top).
    assert lines["NOR"].line.width > lines["PIA"].line.width
    assert [t.name for t in figure.data if t.mode == "lines"][-1] == "NOR"
    # Retired driver ends with a cross; NOR's stop on lap 2 is a marker.
    symbols = [t.marker.symbol for t in figure.data if t.mode == "markers"]
    assert "x" in symbols
    labels = {a.text for a in figure.layout.annotations}
    assert {"NOR", "PIA", "VER", "SC"} <= labels


def test_safety_car_pit_lane_pass_is_not_a_pit_stop():
    laps = pd.DataFrame({
        "Driver": ["NOR"] * 4, "LapNumber": [1, 2, 3, 4],
        "Position": [1, 1, 1, 1], "TimeSeconds": [90.0, 180.0, 270.0, 360.0],
    })
    stints = pd.DataFrame(
        [["NOR", 1, 1, 2, "INTERMEDIATE", 0], ["NOR", 2, 3, 3, "INTERMEDIATE", 0], ["NOR", 3, 4, 4, "HARD", 0]],
        columns=STINT_COLUMNS,
    )
    track_status = [{"status": "SC", "start": 100.0, "end": 400.0, "ending_from": None, "pit_lane_from": 150.0}]
    data = progression_data(laps, stints=stints, track_status=track_status, lap_of=lambda s: int(s // 90) + 1)
    # Lap 2: same compound under "through the pit lane" -> pass. Lap 3: real change to HARD.
    assert data["pit_laps"] == {"NOR": [3]}
