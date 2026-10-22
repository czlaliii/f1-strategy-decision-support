import pandas as pd
import pytest

from src.features.race_dataset import RAW_ROOT, build_race_dataset, list_local_races
from src.models.expanding_window_model import races_before
from src.preprocessing.build_lap_data import build_stint_mapping
from src.simulator.race_simulator import MODEL_FEATURES

REAL_RACE = "hungarian_2025"


def test_stint_mapping_covers_every_stint():
    stints = pd.DataFrame({
        "driver_number": [1, 1, 44],
        "stint_number": [1, 2, 1],
        "lap_start": [1, 20, 1],
        "lap_end": [20, 30, 30],
        "compound": ["MEDIUM", "HARD", "SOFT"],
        "tyre_age_at_start": [0, 0, 3],
    })
    mapping = build_stint_mapping(stints)
    assert len(mapping) == 60
    # At a shared boundary lap the later stint wins.
    lap_20 = mapping[(mapping.driver_number == 1) & (mapping.lap_number == 20)]
    assert lap_20["compound"].item() == "HARD"
    assert mapping[mapping.driver_number == 44]["tyre_age"].max() == 32


@pytest.mark.skipif(not (RAW_ROOT / REAL_RACE).exists(), reason="local race data not available")
def test_features_do_not_use_later_laps(monkeypatch):
    races = list_local_races()
    session_key = int(races.loc[races.race == REAL_RACE, "session_key"].item())
    full = build_race_dataset(REAL_RACE, session_key)

    cutoff = 30
    import src.features.race_dataset as race_dataset

    original = race_dataset._read_session_json

    def truncated(race_dir, filename, key):
        df = original(race_dir, filename, key)
        if filename == "laps.json":
            df = df[df["lap_number"] <= cutoff]
        return df

    monkeypatch.setattr(race_dataset, "_read_session_json", truncated)
    partial = build_race_dataset(REAL_RACE, session_key)

    keys = ["driver_number", "lap_number"]
    features = [f for f in MODEL_FEATURES if f not in keys]
    before = full[full.lap_number <= cutoff].set_index(keys).sort_index()[features]
    after = partial.set_index(keys).sort_index()[features]
    pd.testing.assert_frame_equal(before, after.loc[before.index], check_dtype=False)


@pytest.mark.skipif(not (RAW_ROOT / REAL_RACE).exists(), reason="local race data not available")
def test_training_races_are_strictly_earlier():
    races = list_local_races()
    selected_date = races.loc[races.race == REAL_RACE, "race_date"].item()
    earlier = races_before(REAL_RACE)
    assert REAL_RACE not in set(earlier.race)
    assert (earlier.race_date < selected_date).all()
