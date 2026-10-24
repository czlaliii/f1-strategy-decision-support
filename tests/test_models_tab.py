import numpy as np
import pandas as pd

from src.analysis.style_report import CAR_METRICS, DRIVER_METRICS
from src.dashboard import model_results as results


def test_car_profile_sorted_by_slow_corner_gain_with_values_as_text():
    teams = pd.DataFrame({"season": 2023, "team_name": ["A", "B", "C"], "races": 3,
                          **{m: [0.1, 0.3, -0.2] for m in CAR_METRICS}})
    figure = results.car_profile_figure(teams, 2023)
    assert list(figure.data[0].y) == ["B", "A", "C"]
    assert figure.data[0].text[0][0] == "+0.30"


def test_cluster_scatter_has_one_trace_per_cluster():
    rng = np.random.default_rng(0)
    clusters = pd.DataFrame({m: rng.normal(size=6) for m in DRIVER_METRICS})
    clusters = clusters.assign(driver=list("ABCDEF"), season=2023, races=5, cluster=[0, 0, 0, 1, 1, 1])
    meta = {"clip_z": 2.5, "cluster_names": {"0": "x", "1": "y"}}
    figure = results.cluster_scatter_figure(clusters, meta)
    assert [len(trace.x) for trace in figure.data] == [3, 3]


def test_missing_results_name_the_script(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "LAP_MODEL_DIR", tmp_path)
    try:
        results.load_lap_model_results()
    except results.MissingResults as error:
        assert "compare_lap_time_models" in str(error)
    else:
        raise AssertionError("MissingResults expected")
