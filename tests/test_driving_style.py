import numpy as np
import pandas as pd

from src.analysis.driving_style import STEP_M, _brake_onset, corner_class, pass_counts, zone_labels
from src.analysis.style_report import car_adjusted


def test_corner_classes():
    assert [corner_class(v) for v in (80, 150, 240)] == ["slow", "medium", "fast"]


def test_brake_onset_is_interpolated_between_grid_points():
    brake = np.zeros(100)
    brake[60:70] = 100
    # Brake value crosses 50 halfway between grid points 59 and 60 if point 59 is at 0 and 60 at 100.
    onset = _brake_onset(brake, apex=80, lo=40)
    assert np.isclose(onset, (80 - 59.5) * STEP_M)


def test_brake_onset_uses_the_last_braking_run_before_the_apex():
    brake = np.zeros(100)
    brake[45:50] = 100  # previous corner
    brake[70:78] = 100
    onset = _brake_onset(brake, apex=80, lo=40)
    assert np.isclose(onset, (80 - 69.5) * STEP_M)


def test_zone_labels_follow_nearest_corner_where_not_flat_out():
    grid = np.arange(0, 100, 10.0)
    throttle = np.array([100, 100, 40, 40, 100, 100, 100, 30, 30, 100])
    corners = pd.DataFrame({"apex_m": [25.0, 75.0], "class": ["slow", "fast"]})
    labels = zone_labels(grid, throttle, corners)
    assert list(labels) == ["full", "full", "slow", "slow", "full", "full", "full", "fast", "fast", "full"]


def test_pass_counts_ignore_position_changes_from_pit_stops():
    rows = []
    # Lap 1: A P1, B P2, C P3. Lap 2: B passes A on track; C stays. Lap 3: C gains on A only because A pits.
    for lap, order, pits in [(1, ["A", "B", "C"], set()), (2, ["B", "A", "C"], set()), (3, ["B", "C", "A"], {"A"})]:
        for position, driver in enumerate(order, start=1):
            rows.append({"driver_number": driver, "lap_number": lap, "position": position,
                         "is_green_lap": True, "pit_stop": driver in pits, "is_pit_out_lap": False})
    passes = pass_counts(pd.DataFrame(rows)).set_index("driver_number")
    assert passes.loc["B", "passes_made"] == 1 and passes.loc["A", "passes_lost"] == 1
    assert passes.loc["C", "passes_made"] == 0


def test_car_adjusted_removes_the_team_level():
    table = pd.DataFrame({
        "race": ["r"] * 3, "team_name": ["X", "X", "Y"], "driver_number": [1, 2, 3], "m": [5.0, 3.0, 9.0],
    })
    adjusted = car_adjusted(table, ["m"])
    assert adjusted["m"].tolist()[:2] == [1.0, -1.0]
    # No teammate in the data: the car effect cannot be removed.
    assert np.isnan(adjusted["m"].iloc[2])
