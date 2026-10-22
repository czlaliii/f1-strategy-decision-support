import json
import time
from pathlib import Path

import pandas as pd

RACES = [
    "australia_2025",
    "bahrain_2023",
    "canada_2024",
    "hungary_2026",
    "imola_2024",
    "monaco_2023",
    "sao_paulo_2024",
    "silverstone_2024",
    "silverstone_2025",
    "zandvoort_2023",
]


def load_json(path: Path) -> pd.DataFrame:
    """JSON fájl betöltése Pandas DataFrame-be."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return pd.DataFrame(data)


def build_lap_data(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    pit: pd.DataFrame,
) -> pd.DataFrame:
    """
    A Pandas alapú lap-adat feldolgozás.
    Ugyanazt a logikát használja, mint az eredeti preprocessing.
    """

    # ---------------------------------------------------------
    # STINT MAPPING
    # ---------------------------------------------------------

    stint_rows = []

    for _, stint in stints.iterrows():

        driver_number = stint["driver_number"]
        stint_number = stint["stint_number"]
        compound = stint["compound"]

        lap_start_value = stint["lap_start"]
        lap_end_value = stint["lap_end"]

        # Ha nincs érvényes lap-tartomány, ezt a stintet kihagyjuk.
        if (
            pd.isna(lap_start_value)
            or pd.isna(lap_end_value)
        ):
            continue

        lap_start = int(lap_start_value)
        lap_end = int(lap_end_value)

        tyre_age_at_start = stint["tyre_age_at_start"]

        for lap_number in range(
            lap_start,
            lap_end + 1,
        ):
            stint_rows.append(
                {
                    "driver_number": driver_number,
                    "lap_number": lap_number,
                    "stint_number": stint_number,
                    "compound": compound,
                    "tyre_age": (
                        tyre_age_at_start
                        + lap_number
                        - lap_start
                    ),
                }
            )

    stint_mapping = pd.DataFrame(stint_rows)

    # A közös határköröknél a későbbi stintet tartjuk meg.
    stint_mapping = (
        stint_mapping
        .sort_values(["driver_number", "lap_number", "stint_number"])
        .drop_duplicates(
            subset=["driver_number", "lap_number"],
            keep="last",
        )
    )

    # ---------------------------------------------------------
    # LAP + STINT
    # ---------------------------------------------------------

    lap_data = laps.merge(
        stint_mapping,
        on=[
            "driver_number",
            "lap_number",
        ],
        how="left",
    )

    # ---------------------------------------------------------
    # PIT STOP FLAG
    # ---------------------------------------------------------

    if (
        not pit.empty
        and "driver_number" in pit.columns
        and "lap_number" in pit.columns
    ):

        pit_flags = (
            pit[
                [
                    "driver_number",
                    "lap_number",
                ]
            ]
            .drop_duplicates()
            .assign(
                pit_stop=True
            )
        )

    else:

        pit_flags = pd.DataFrame(
            columns=[
                "driver_number",
                "lap_number",
                "pit_stop",
            ]
        )

    lap_data = lap_data.merge(
        pit_flags,
        on=[
            "driver_number",
            "lap_number",
        ],
        how="left",
    )

    lap_data["pit_stop"] = (
        lap_data["pit_stop"]
        .fillna(False)
        .astype(bool)
    )

    # ---------------------------------------------------------
    # SORT
    # ---------------------------------------------------------

    lap_data = lap_data.sort_values(
        [
            "driver_number",
            "lap_number",
        ]
    ).reset_index(
        drop=True
    )

    return lap_data


def main() -> None:

    results = []

    print("=== PANDAS MULTI-RACE PROCESSING ===")

    for race in RACES:

        print(
            f"\nProcessing: {race}"
        )

        data_dir = (
            Path("data/raw")
            / race
        )

        output_dir = (
            Path("data/processed")
            / race
            / "pandas"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        laps_path = (
            data_dir / "laps.json"
        )

        stints_path = (
            data_dir / "stints.json"
        )

        pit_path = (
            data_dir / "pit.json"
        )

        try:

            start = time.perf_counter()

            laps = load_json(
                laps_path
            )

            stints = load_json(
                stints_path
            )

            pit = load_json(
                pit_path
            )

            lap_data = build_lap_data(
                laps=laps,
                stints=stints,
                pit=pit,
            )

            output_path = (
                output_dir
                / "lap_data.parquet"
            )

            lap_data.to_parquet(
                output_path,
                index=False,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            results.append(
                {
                    "race": race,
                    "status": "OK",
                    "time_seconds": elapsed,
                    "rows": len(lap_data),
                    "columns": len(
                        lap_data.columns
                    ),
                }
            )

            print(
                f"{race:<20} "
                f"OK     "
                f"{elapsed:.3f} s     "
                f"rows={len(lap_data)}"
            )

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - start
            )

            results.append(
                {
                    "race": race,
                    "status": "ERROR",
                    "time_seconds": elapsed,
                    "rows": None,
                    "columns": None,
                }
            )

            print(
                f"{race:<20} "
                f"ERROR  "
                f"{elapsed:.3f} s"
            )

            print(
                f"Error: {exc}"
            )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    benchmark = pd.DataFrame(
        results
    )

    successful = benchmark[
        benchmark["status"] == "OK"
    ]

    total_time = (
        successful["time_seconds"]
        .sum()
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "=== SUMMARY ==="
    )

    print(
        "=" * 60
    )

    for _, row in benchmark.iterrows():

        print(
            f"{row['race']:<20} "
            f"{row['status']:<6} "
            f"{row['time_seconds']:.3f} s"
        )

    print(
        "-" * 60
    )

    print(
        "Total Pandas processing time: "
        f"{total_time:.3f} s"
    )

    output_path = Path(
        "data/processed/"
        "pandas_multi_race_benchmark.csv"
    )

    benchmark.to_csv(
        output_path,
        index=False,
    )

    print(
        "Benchmark saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()