"""Undercut and overcut estimates between two cars, from model simulations.

Both cars start from their state at the end of the current lap and are
simulated with their own (anchored) lap-time model. One car pits on the next
lap, the other `response` laps later; the gap is read at the end of the lap
after the later stop, when both cars have changed tyres. Each car fits the
next compound of its own best strategy.

Gaps are in seconds: positive = the chasing car is still behind. The
estimate ignores traffic after the stop and the rival's real intentions.
"""
from dataclasses import dataclass

import numpy as np

from src.simulator.race_simulator import RaceSimulator, Strategy

# Laps the other car waits before reacting to a stop.
RESPONSE_LAPS = (1, 2, 3)
# A car on tyres younger than this has just stopped; no undercut duel is shown.
MIN_TYRE_AGE = 3


@dataclass(frozen=True)
class Car:
    name: str
    predictor: object
    state: dict
    next_compound: str


def window_time(car: Car, pit_lap: int, end_lap: int, pit_loss: float) -> float:
    """Time from the current state to the end of end_lap, stopping on pit_lap."""
    strategy = Strategy((pit_lap,), (str(car.state["compound"]).upper(), car.next_compound))
    simulator = RaceSimulator(car.predictor, car.state, race_last_lap=end_lap, pit_loss_seconds=pit_loss)
    return float(simulator.total_times([strategy])[0])


def gap_after(chaser: Car, leader: Car, gap: float, chaser_pit: int, leader_pit: int, pit_loss: float) -> float:
    """Chaser's gap to the leader after both stops (negative = chaser ahead)."""
    end = max(chaser_pit, leader_pit) + 1
    return gap + window_time(chaser, chaser_pit, end, pit_loss) - window_time(leader, leader_pit, end, pit_loss)


def _scenarios(chaser, leader, gap, chaser_first, current_lap, last_lap, pit_loss):
    rows = []
    for response in RESPONSE_LAPS:
        first, later = current_lap + 1, current_lap + 1 + response
        if later + 1 > last_lap:
            break
        chaser_pit, leader_pit = (first, later) if chaser_first else (later, first)
        after = gap_after(chaser, leader, gap, chaser_pit, leader_pit, pit_loss)
        rows.append({"response": response, "gap_after": after, "gain": gap - after})
    return rows


def undercut(driver: Car, ahead: Car, gap, current_lap, last_lap, pit_loss):
    """Driver stops next lap, the car ahead `response` laps later."""
    return _scenarios(driver, ahead, gap, True, current_lap, last_lap, pit_loss)


def overcut(driver: Car, ahead: Car, gap, current_lap, last_lap, pit_loss):
    """The car ahead stops next lap, the driver stays out `response` laps longer."""
    return _scenarios(driver, ahead, gap, False, current_lap, last_lap, pit_loss)


def undercut_threat(driver: Car, behind: Car, gap, current_lap, last_lap, pit_loss):
    """The car behind stops next lap, the driver reacts `response` laps later (chaser = car behind)."""
    return _scenarios(behind, driver, gap, True, current_lap, last_lap, pit_loss)


def verdict(rows, attacking=True) -> str:
    """Short reading of a scenario list: the latest response that still flips the order."""
    if not rows:
        return "Nincs elég kör a verseny végéig."
    flips = [row["response"] for row in rows if row["gap_after"] < 0]
    if attacking:
        if not flips:
            best = min(rows, key=lambda row: row["gap_after"])
            return f"Nem elég: a legjobb esetben is {best['gap_after']:.1f} s marad a hátrány."
        if flips[0] == rows[0]["response"]:
            return "Működik: azonnali reakció mellett is elé kerülsz."
        return f"Csak akkor működik, ha a rivális legalább {flips[0]} kört kint marad."
    if not flips:
        return "Nincs veszély: ha reagálsz, előtte maradsz."
    safe = [row["response"] for row in rows if row["gap_after"] >= 0]
    if not safe:
        return "Veszély: még azonnali reakcióval is elé kerülhet."
    return f"Veszély: ha {safe[-1] + 1} vagy több kört vársz a reakcióval, elé kerül."


def rejoin_position(gaps_to_leader: dict, driver, pit_loss: float):
    """Estimated place if the driver stopped now and nobody else did: (position, car ahead, gap, car behind, gap)."""
    own = gaps_to_leader[driver] + pit_loss
    others = {name: gap for name, gap in gaps_to_leader.items() if name != driver and np.isfinite(gap)}
    ahead = {name: gap for name, gap in others.items() if gap <= own}
    behind = {name: gap for name, gap in others.items() if gap > own}
    car_ahead = max(ahead, key=ahead.get) if ahead else None
    car_behind = min(behind, key=behind.get) if behind else None
    return {
        "position": len(ahead) + 1,
        "ahead": car_ahead,
        "gap_ahead": None if car_ahead is None else own - ahead[car_ahead],
        "behind": car_behind,
        "gap_behind": None if car_behind is None else behind[car_behind] - own,
    }
