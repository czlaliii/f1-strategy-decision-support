import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/hungary_2026")
PROCESSED_DIR = Path("data/processed/hungary_2026")

RACE_CONTROL_FILE = RAW_DIR / "race_control.json"
LAP_DATA_FILE = PROCESSED_DIR / "lap_data.parquet"


def main():
    print("Loading race control data...")
    
    with open(RACE_CONTROL_FILE, "r", encoding="utf-8") as file:
        race_control = json.load(file)

    # VSC események
    vsc_events = []

    for event in race_control:
        message = str(event.get("message", "")).upper()

        if "VSC DEPLOYED" in message or "VSC ENDING" in message:
            vsc_events.append(event)

    print("\n=== VSC EVENTS ===")

    for event in vsc_events:
        print(
            f"Lap {event.get('lap_number')}: "
            f"{event.get('message')} "
            f"({event.get('date')})"
        )

    # Lap adatok
    print("\nLoading lap data...")

    laps = pd.read_parquet(LAP_DATA_FILE)

    # Csak az érintett körök
    affected_laps = laps[
        laps["lap_number"].between(53, 58)
    ].copy()

    print("\n=== LAP TIMES: LAPS 53-58 ===")

    summary = (
        affected_laps
        .groupby("lap_number")["lap_duration"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            min="min",
            max="max",
        )
        .round(3)
    )

    print(summary)

    print("\n=== INDIVIDUAL LAP TIMES ===")

    columns = [
        "driver_number",
        "lap_number",
        "lap_duration",
        "compound",
        "tyre_age",
        "pit_stop",
    ]

    print(
        affected_laps[columns]
        .sort_values(["lap_number", "lap_duration"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()