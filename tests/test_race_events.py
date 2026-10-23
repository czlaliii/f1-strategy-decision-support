import pandas as pd

from src.dashboard.replay_player import (
    build_leaderboard,
    build_track_status,
    find_retirements,
    track_status_at,
)

START = pd.Timestamp("2025-01-01 14:00:00", tz="UTC")


def control(*events):
    """(seconds, category, flag, scope, message) -> race_control DataFrame."""
    return pd.DataFrame([
        {"date": (START + pd.Timedelta(seconds=t)).isoformat(), "category": c, "flag": f, "scope": s, "message": m}
        for t, c, f, s, m in events
    ])


def test_safety_car_escalates_to_red_flag_and_ends_on_track_clear():
    periods = build_track_status(control(
        (100, "SafetyCar", None, None, "SAFETY CAR DEPLOYED"),
        (150, "Flag", "CLEAR", "Sector", "CLEAR IN TRACK SECTOR 3"),  # sector-level: ignored
        (200, "Flag", "RED", "Track", "RED FLAG"),
        (900, "Flag", "CLEAR", "Track", "TRACK CLEAR"),
        (1000, "SafetyCar", None, None, "VIRTUAL SAFETY CAR DEPLOYED"),
        (1060, "SafetyCar", None, None, "VIRTUAL SAFETY CAR ENDING"),
        (1070, "Flag", "CLEAR", "Track", "TRACK CLEAR"),
    ), START)
    assert [(p["status"], p["start"], p["end"]) for p in periods] == [
        ("SC", 100, 200), ("RED", 200, 900), ("VSC", 1000, 1070),
    ]
    assert track_status_at(periods, 150) == {"status": "SC", "ending": False}
    assert track_status_at(periods, 1065) == {"status": "VSC", "ending": True}
    assert track_status_at(periods, 950) is None


def test_messages_after_the_finish_are_ignored():
    periods = build_track_status(control(
        (500, "Flag", "CHEQUERED", "Track", "CHEQUERED FLAG"),
        (600, "Flag", "RED", "Track", "RED FLAG"),
    ), START)
    assert periods == []


def test_red_flag_without_restart_message_ends_after_pause_in_laps():
    periods = build_track_status(
        control((300, "SessionStatus", None, None, "SESSION ABORTED")),
        START,
        # Cars still finish their lap under the flag, then a long stop, then the restart.
        lap_start_seconds=[250, 310, 330, 1500, 1590],
    )
    assert [(p["status"], p["start"], p["end"]) for p in periods] == [("RED", 300, 1500)]


def replay_laps(rows):
    return pd.DataFrame(rows, columns=["Driver", "LapNumber", "LapStartTime", "TimeSeconds", "LapTime"])


def test_retirements_exclude_lapped_finishers_and_fix_bad_timestamps():
    rows = []
    for lap in range(1, 11):
        start = (lap - 1) * 90.0
        rows.append(("WIN", lap, start, start + 90, 90.0))
        if lap <= 9:  # lapped finisher: last lap ends after the leader's finish
            rows.append(("LAP", lap, start + 95, start + 195, 100.0))
        if lap <= 4:  # retires during lap 4
            rows.append(("DNF", lap, start + 1, start + 91 if lap < 4 else None, 90.0 if lap < 4 else None))
    rows.append(("BAD", 1, 5000.0, None, None))  # feed error: lap 1 timestamped after the race
    retirements = find_retirements(replay_laps(rows), finish_seconds=900.0)

    assert set(retirements) == {"DNF", "BAD"}
    # DNF started lap 4 at 271 s: out of the classification when race lap 5
    # begins (360 s) and off the track one lap later (450 s).
    assert retirements["DNF"] == {"lap": 4, "out_from": 360.0, "hidden_from": 450.0}
    assert retirements["BAD"] == {"lap": 1, "out_from": 90.0, "hidden_from": 180.0}


def test_leaderboard_lists_retired_drivers_last():
    frames = pd.DataFrame({
        "Driver": ["WIN", "DNF"], "LapNumber": [6, 4], "Distance": [10.0, 5.0],
        "TimeSeconds": [500.0, 500.0], "Compound": None, "TyreLife": None, "Speed": 200.0,
    })
    laps = replay_laps([("WIN", 5, 360.0, 450.0, 90.0), ("DNF", 3, 181.0, 271.0, 90.0)])
    board = build_leaderboard(
        frames, laps, 500.0, retirements={"DNF": {"lap": 4, "out_from": 360.0, "hidden_from": 450.0}}
    )
    assert board["Driver"].tolist() == ["WIN", "DNF"]
    assert board.iloc[0]["Status"] == "" and board.iloc[0]["Position"] == 1
    assert board.iloc[1]["Status"] == "Kiesett (4. kör)" and pd.isna(board.iloc[1]["Position"])
