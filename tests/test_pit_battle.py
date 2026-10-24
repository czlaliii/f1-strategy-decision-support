import numpy as np

from src.simulator import pit_battle
from src.simulator.pit_battle import Car


class TyreModel:
    """Base pace plus 0.1 s per lap of tyre age; a fixed offset makes one car faster."""

    feature_names_in_ = ["compound", "tyre_age"]
    fuel_effect_per_lap = 0.0

    def __init__(self, base=90.0, degradation=0.1):
        self.base, self.degradation = base, degradation

    def predict(self, X):
        return self.base + self.degradation * X["tyre_age"].to_numpy(dtype=float)


def car(name, tyre_age=20, base=90.0):
    return Car(name, TyreModel(base), {"lap_number": 10, "compound": "MEDIUM", "tyre_age": float(tyre_age)}, "HARD")


def test_undercut_gain_equals_the_fresh_tyre_advantage():
    # Pitting one lap earlier: on the lap the rival is still out, the driver is on new tyres.
    rows = pit_battle.undercut(car("A"), car("B"), gap=1.0, current_lap=10, last_lap=50, pit_loss=20)
    first = rows[0]
    # Lap 12: A on age 0 (90.0) vs B on age 21 (92.1); lap 13: A age 1 vs B age 0 (+0.1 for A).
    assert np.isclose(first["gain"], 2.1 - 0.1)
    assert first["gap_after"] < 0
    # A later response gives the undercut more time to work.
    assert rows[1]["gain"] > rows[0]["gain"]


def test_overcut_loses_against_degrading_tyres():
    rows = pit_battle.overcut(car("A"), car("B"), gap=1.0, current_lap=10, last_lap=50, pit_loss=20)
    assert all(row["gap_after"] > 1.0 for row in rows)


def test_threat_from_a_faster_car_behind():
    fast = car("C", base=89.0)
    rows = pit_battle.undercut_threat(car("A"), fast, gap=0.5, current_lap=10, last_lap=50, pit_loss=20)
    assert rows[0]["gap_after"] < 0
    assert pit_battle.verdict(rows, attacking=False).startswith("Veszély")


def test_no_scenarios_beyond_the_race_end():
    # Lap 47 of 50: only a one-lap response still ends (lap 49 + 1) before the flag.
    rows = pit_battle.undercut(car("A"), car("B"), 1.0, current_lap=47, last_lap=50, pit_loss=20)
    assert [row["response"] for row in rows] == [1]


def test_verdicts():
    rows = [{"response": r, "gap_after": g} for r, g in [(1, 0.3), (2, -0.2), (3, -0.6)]]
    assert pit_battle.verdict(rows) == "Csak akkor működik, ha a rivális legalább 2 kört kint marad."
    assert pit_battle.verdict(rows[:1]).startswith("Nem elég")


def test_rejoin_position_counts_cars_within_the_pit_loss():
    gaps = {"A": 0.0, "B": 5.0, "C": 12.0, "D": 30.0}
    rejoin = pit_battle.rejoin_position(gaps, "B", pit_loss=20.0)
    assert rejoin["position"] == 3 and rejoin["ahead"] == "C" and np.isclose(rejoin["gap_ahead"], 13.0)
    assert rejoin["behind"] == "D" and np.isclose(rejoin["gap_behind"], 5.0)
