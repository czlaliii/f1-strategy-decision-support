from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed/hungary_2026")
RACE_STATE_FILE = PROCESSED_DIR / "race_state.parquet"


def main():
    print("Loading race state...")

    race_state = pd.read_parquet(RACE_STATE_FILE)

    # Csak az érdekes körök
    data = race_state[
        race_state["lap_number"].between(53, 58)
    ].copy()

    # Érintett / lassú körök
    data["slow_lap"] = data["lap_duration"] > 95

    print("\n=== SLOW LAPS (>95s) ===")

    slow = data[data["slow_lap"]].copy()

    columns = [
        "driver_number",
        "lap_number",
        "lap_duration",
        "position",
        "gap_to_leader",
        "interval",
        "laps_down",
        "compound",
        "tyre_age",
        "pit_stop",
    ]

    print(
        slow[columns]
        .sort_values(["lap_number", "lap_duration"], ascending=[True, False])
        .to_string(index=False)
    )

    print("\n=== RACE STATE: LAPS 53-58 ===")

    print(
        data[columns]
        .sort_values(["lap_number", "position"])
        .to_string(index=False)
    )

    print("\n=== SUMMARY BY LAP ===")

    summary = (
        data.groupby("lap_number")
        .agg(
            drivers=("driver_number", "count"),
            mean_lap_time=("lap_duration", "mean"),
            slow_laps=("slow_lap", "sum"),
            mean_gap=("gap_to_leader", "mean"),
            mean_interval=("interval", "mean"),
        )
        .round(3)
    )

    print(summary)


if __name__ == "__main__":
    main()