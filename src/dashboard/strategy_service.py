"""Run strategy computations in one long-lived worker process.

The replay callbacks stay in the web server process; heavy pandas/sklearn
work runs elsewhere so it cannot hold the GIL and stall playback. A single
worker keeps datasets and models loaded between laps.
"""
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import threading

_LOCK = threading.Lock()
_EXECUTOR: ProcessPoolExecutor | None = None
# Request key -> Future; only the latest few are kept.
_JOBS: dict[str, object] = {}
MAX_KEPT_JOBS = 20


def _executor() -> ProcessPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ProcessPoolExecutor(max_workers=1)
    return _EXECUTOR


def submit(key: str, function, *args) -> None:
    """Queue a job once per key; pending (not yet started) older jobs are cancelled."""
    global _EXECUTOR
    with _LOCK:
        if key in _JOBS:
            return
        for old_key, future in list(_JOBS.items()):
            # A queued lap that never started is no longer worth computing.
            if not old_key.startswith("warm|") and future.cancel():
                del _JOBS[old_key]
        try:
            _JOBS[key] = _executor().submit(function, *args)
        except BrokenProcessPool:
            _EXECUTOR = None
            _JOBS.clear()
            _JOBS[key] = _executor().submit(function, *args)
        while len(_JOBS) > MAX_KEPT_JOBS:
            del _JOBS[next(iter(_JOBS))]


def result(key: str):
    """(done, value) for a submitted job; value is an error message dict on failure."""
    with _LOCK:
        future = _JOBS.get(key)
    if future is None:
        return True, {"status": "message", "message": "A számítás elveszett; lépj egy kört a frissítéshez."}
    if not future.done():
        return False, None
    if future.cancelled():
        return True, {"status": "message", "message": "A számítás megszakadt."}
    try:
        return True, future.result()
    except BrokenProcessPool:
        global _EXECUTOR
        with _LOCK:
            _EXECUTOR = None
            _JOBS.clear()
        return True, {"status": "message", "message": "A háttérfolyamat leállt; lépj egy kört az újraindításhoz."}
    except Exception as error:  # noqa: BLE001 - surfaced to the user instead of crashing the UI
        return True, {"status": "message", "message": f"A stratégia nem számítható ki: {error}"}
