import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw/hungary_2026")
OUTPUT_DIR = Path("data/processed/hungary_2026")

def load_json(filename: str) -> pd.DataFrame:

    filepath = DATA_DIR / filename

    with filepath.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return pd.DataFrame(data)



def build_stint_mapping(stints: pd.DataFrame) -> pd.DataFrame:
    """
    Creates one row for every driver/lap combination covered by a stint.
    At shared boundaries, the later stint takes precedence.
    """

    rows = []

    stints = stints.dropna(
        subset=["driver_number", "stint_number", "lap_start", "lap_end"]
    ).sort_values(["driver_number", "stint_number"])

    for _, stint in stints.iterrows():
        lap_start = int(stint["lap_start"])
        lap_end = int(stint["lap_end"])

        for lap in range(lap_start, lap_end + 1):
            rows.append({
                "driver_number": stint["driver_number"],
                "lap_number": lap,
                "stint_number": stint["stint_number"],
                "compound": stint["compound"],
                "tyre_age": (
                    stint["tyre_age_at_start"]
                    + lap
                    - lap_start
                ),
            })

    mapping = pd.DataFrame(
        rows,
        columns=["driver_number", "lap_number", "stint_number", "compound", "tyre_age"],
    )

    mapping = (
        mapping
        .sort_values(["driver_number", "lap_number", "stint_number"])
        .drop_duplicates(
            subset=["driver_number", "lap_number"],
            keep="last",
        )
    )

    return mapping


def build_lap_data(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    pit: pd.DataFrame,
) -> pd.DataFrame:
    """Join laps with stint compound/tyre age and a pit stop flag."""

    # --- STINT MAPPING ---

    stint_mapping = build_stint_mapping(stints)

    # --- LAP + STINT ---

    lap_data = laps.merge(
        stint_mapping,
        on=["driver_number", "lap_number"],
        how="left"
    )

    # --- PIT STOP FLAG ---

    if {"driver_number", "lap_number"}.issubset(pit.columns):
        pit_flags = (
            pit[
                ["driver_number", "lap_number"]
            ]
            .drop_duplicates()
            .assign(pit_stop=True)
        )

        lap_data = lap_data.merge(
            pit_flags,
            on=["driver_number", "lap_number"],
            how="left"
        )

        lap_data["pit_stop"] = (
            lap_data["pit_stop"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
    else:
        lap_data["pit_stop"] = False

    # --- SORT ---

    return lap_data.sort_values(
        [
            "driver_number",
            "lap_number"
        ]
    ).reset_index(drop=True)


def main():

    print("Loading data...")

    laps = load_json("laps.json")
    stints = load_json("stints.json")
    pit = load_json("pit.json")

    print(f"Laps: {len(laps)}")
    print(f"Stints: {len(stints)}")
    print(f"Pit stops: {len(pit)}")

    lap_data = build_lap_data(laps, stints, pit)

    # --- SAVE ---

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = OUTPUT_DIR / "lap_data.parquet"

    lap_data.to_parquet(
        output_file,
        index=False
    )

    print("\nProcessed data:")
    print(lap_data.head(20))

    print("\nShape:")
    print(lap_data.shape)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()