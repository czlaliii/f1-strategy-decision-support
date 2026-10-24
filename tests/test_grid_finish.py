import numpy as np
import pandas as pd
import pytest

from src.analysis.grid_finish import race_grid_and_result, rank_correlation
from src.features.race_dataset import RAW_ROOT, resolve_session_key


def table(grid, finish, status=None):
    n = len(grid)
    return pd.DataFrame({
        "driver_number": range(n), "Driver": [f"D{i}" for i in range(n)], "team_colour": "#000000",
        "grid": grid, "finish": finish, "status": status or ["Célba ért"] * n,
    })


def test_rank_correlation_extremes_and_retirements():
    assert rank_correlation(table([1, 2, 3, 4], [1, 2, 3, 4]))["spearman"] == pytest.approx(1.0)
    assert rank_correlation(table([1, 2, 3, 4], [4, 3, 2, 1]))["spearman"] == pytest.approx(-1.0)
    # A retired driver does not enter the correlation.
    with_dnf = rank_correlation(table([1, 2, 3, 4, 5], [1, 2, 3, 4, np.nan], ["Célba ért"] * 4 + ["DNF"]))
    assert with_dnf["n"] == 4 and with_dnf["spearman"] == pytest.approx(1.0)
    assert np.isnan(rank_correlation(table([1, 2], [2, 1]))["spearman"])


@pytest.mark.skipif(not (RAW_ROOT / "chinese_2024" / "starting_grid.json").exists(), reason="results not downloaded")
def test_sprint_weekend_uses_race_grid_not_sprint_grid():
    result = race_grid_and_result("chinese_2024", resolve_session_key("chinese_2024"))
    # 2024 Chinese GP: Verstappen on race pole (Norris took sprint pole).
    assert result.sort_values("grid")["Driver"].head(2).tolist() == ["VER", "PER"]
    assert result.sort_values("finish")["Driver"].head(3).tolist() == ["VER", "NOR", "PER"]


@pytest.mark.skipif(not (RAW_ROOT / "australia_2025" / "session_result.json").exists(), reason="results not downloaded")
def test_official_status_marks_dnf_and_dns():
    result = race_grid_and_result("australia_2025", resolve_session_key("australia_2025")).set_index("Driver")
    assert result.loc["HAD", "status"] == "DNS"
    assert result.loc["ALO", "status"] == "DNF" and np.isnan(result.loc["ALO", "finish"])
    assert result.loc["NOR", "grid"] == 1 and result.loc["NOR", "finish"] == 1
