import pandas as pd
import pytest
from dash import no_update

from src.dashboard import strategy_service
from src.dashboard.callbacks import replay_strategy_callbacks as callbacks
from src.dashboard.replay_service import REPLAY_CACHE
from src.features.race_dataset import RAW_ROOT


class FakeApp:
    def __init__(self):
        self.functions = {}

    def callback(self, *args, **kwargs):
        def register(function):
            self.functions[function.__name__] = function
            return function
        return register


@pytest.fixture
def app(monkeypatch):
    submitted, results = [], {}
    monkeypatch.setattr(strategy_service, "submit", lambda key, fn, *a: submitted.append(key))
    monkeypatch.setattr(strategy_service, "result", lambda key: results.get(key, (False, None)))
    REPLAY_CACHE["test"] = {
        "laps": pd.DataFrame({
            "Driver": ["NOR"] * 3, "DriverNumber": [4] * 3,
            "LapNumber": [1, 2, 3], "LapStartTime": [0.0, 90.0, 180.0],
        }),
        "race_folder": "some_race",
        "session_key": 1,
    }
    fake = FakeApp()
    callbacks.register_replay_strategy_callbacks(fake)
    yield fake, submitted, results
    REPLAY_CACHE.pop("test", None)


def test_request_is_queued_only_when_lap_or_driver_changes(app):
    fake, submitted, _ = app
    request = fake.functions["request_strategy"]

    first = request({"replay_id": "test"}, 100.0, "NOR", None)
    assert first["key"] == "some_race|1|4|2" and first["lap"] == 2
    # Later ticks in the same lap only re-check; nothing new is queued.
    assert request({"replay_id": "test"}, 120.0, "NOR", first) is no_update
    second = request({"replay_id": "test"}, 185.0, "NOR", first)
    assert second["lap"] == 3
    jobs = [key for key in submitted if not key.startswith("warm|")]
    assert jobs == ["some_race|1|4|2", "some_race|1|4|3"]
    assert "warm|some_race|1" in submitted


def test_render_shows_pending_then_result_once(app):
    fake, _, results = app
    render = fake.functions["render_strategy"]
    request = {"key": "k", "driver": "NOR", "lap": 2}

    pending = render(0, request, {"rendered": "old"})
    assert "számítás folyamatban" in pending[0]
    assert all(value is no_update for value in pending[1:7])
    assert render(1, request, pending[7]) == (no_update,) * 8

    results["k"] = (True, {"status": "message", "message": "kész"})
    done = render(2, request, pending[7])
    assert done[0] == "kész" and done[7] == {"rendered": "k"}
    assert render(3, request, done[7]) == (no_update,) * 8


def test_message_requests_render_immediately(app):
    fake, _, _ = app
    message = fake.functions["request_strategy"](None, 0.0, None, None)
    rendered = fake.functions["render_strategy"](0, message, {})
    assert rendered[0] == message["message"]


@pytest.mark.skipif(not (RAW_ROOT / "hungarian_2025").exists(), reason="local race data not available")
def test_worker_result_is_plain_data_and_matches_advisor():
    import pickle

    from src.dashboard import strategy_worker
    from src.features.race_dataset import resolve_session_key

    result = strategy_worker.compute_strategy("hungarian_2025", resolve_session_key("hungarian_2025"), 4, 20)
    pickle.dumps(result)  # must be able to cross the process boundary
    assert result["status"] == "ok"
    assert result["best"]["compounds"] == ["MEDIUM", "HARD"]
    start, end = result["pit_window"]
    assert start <= result["best"]["pit_laps"][0] <= end
    assert callbacks._render(result, "NOR")[0].startswith("NOR · aktuális kör: 20/")


@pytest.mark.skipif(not (RAW_ROOT / "hungarian_2025").exists(), reason="local race data not available")
def test_what_if_for_completed_stop_only():
    from src.dashboard import strategy_worker
    from src.features.race_dataset import resolve_session_key

    session_key = resolve_session_key("hungarian_2025")
    # NOR pitted at the end of lap 31 (MEDIUM -> HARD).
    before = strategy_worker.compute_strategy("hungarian_2025", session_key, 4, 31)
    assert before["what_if"]["analyses"] == []
    after = strategy_worker.compute_strategy("hungarian_2025", session_key, 4, 40)
    (analysis,) = after["what_if"]["analyses"]
    assert (analysis["stop_lap"], analysis["from"], analysis["to"]) == (31, "MEDIUM", "HARD")
    assert analysis["decision_lap"] == 26 and analysis["delta"] >= 0
    # Actual stints shown only up to the current lap.
    assert after["what_if"]["actual_segments"][-1]["end"] == 40
    section = str(callbacks._what_if_section(after["what_if"]))
    assert "1. KIÁLLÁS" in section and "Modell javaslata" in section
