import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/hungary_2026")
PROCESSED_DIR = Path("data/processed/hungary_2026")

RACE_CONTROL_FILE = RAW_DIR / "race_control.json"
RACE_STATE_FILE = PROCESSED_DIR / "race_state.parquet"


def load_race_control():
    with open(RACE_CONTROL_FILE, "r", encoding="utf-8") as file:
        events = json.load(file)

    df = pd.DataFrame(events)

    df["date"] = pd.to_datetime(
        df["date"],
        utc=True,
        format="mixed"
    )

    return df


def load_race_state():
    df = pd.read_parquet(RACE_STATE_FILE)

    df["date_start"] = pd.to_datetime(
        df["date_start"],
        utc=True
    )

    df["date_end"] = pd.to_datetime(
        df["date_end"],
        utc=True
    )

    return df


def main():
    print("Loading race control events...")
    events = load_race_control()

    print("Loading race state...")
    race_state = load_race_state()

    # --- 1. Relevant time window only ---

    events = events[
        events["lap_number"].between(50, 58, inclusive="both")
    ].copy()

    race_state = race_state[
        race_state["lap_number"].between(53, 58, inclusive="both")
    ].copy()

    print()
    print("=== RELEVANT RACE CONTROL EVENTS ===")

    for _, event in events.sort_values("date").iterrows():
        print(
            f"{event['date']} | "
            f"Lap {event.get('lap_number')} | "
            f"{event.get('category')} | "
            f"{event.get('flag')} | "
            f"{event.get('message')}"
        )

    # --- 2. Select significant track events ---

    important_events = events[
        events["message"]
        .fillna("")
        .str.upper()
        .str.contains(
            "YELLOW|DOUBLE YELLOW|VSC|MARSHALS ON TRACK|TRACK CLEAR"
        )
    ].copy()

    print()
    print("=== IMPORTANT TRACK EVENTS ===")

    for _, event in important_events.sort_values("date").iterrows():
        print(
            f"{event['date']} | "
            f"Lap {event.get('lap_number')} | "
            f"{event.get('message')}"
        )

    # --- 3. Assign events to laps ---
    # An event affects a lap if it falls between the lap's date_start and date_end.

    print()
    print("=== EVENT / LAP OVERLAP ===")

    overlaps = []

    for _, event in important_events.sort_values("date").iterrows():

        event_time = event["date"]

        affected = race_state[
            (race_state["date_start"] <= event_time)
            & (race_state["date_end"] >= event_time)
        ].copy()

        if affected.empty:
            print(
                f"\n{event_time} | {event['message']}"
            )
            print("No lap overlap found.")
            continue

        print()
        print(
            f"EVENT: {event_time} | "
            f"Lap {event.get('lap_number')} | "
            f"{event['message']}"
        )

        for _, lap in affected.sort_values("driver_number").iterrows():

            print(
                f"  Driver {int(lap['driver_number']):>2} | "
                f"Lap {int(lap['lap_number']):>2} | "
                f"Start {lap['date_start']} | "
                f"End {lap['date_end']} | "
                f"Lap time {lap['lap_duration']:.3f}s"
            )

            overlaps.append({
                "event_time": event_time,
                "event_lap": event.get("lap_number"),
                "event_message": event.get("message"),
                "event_category": event.get("category"),
                "event_flag": event.get("flag"),
                "driver_number": lap["driver_number"],
                "lap_number": lap["lap_number"],
                "lap_duration": lap["lap_duration"],
            })

    # --- 4. Summary ---

    if overlaps:
        overlap_df = pd.DataFrame(overlaps)

        print()
        print("=== AFFECTED DRIVERS SUMMARY ===")

        summary = (
            overlap_df
            .groupby(
                ["event_message", "lap_number"]
            )
            .agg(
                affected_drivers=("driver_number", "count"),
                mean_lap_time=("lap_duration", "mean"),
                min_lap_time=("lap_duration", "min"),
                max_lap_time=("lap_duration", "max"),
            )
            .round(3)
        )

        print(summary)

    print()
    print("Analysis completed.")


if __name__ == "__main__":
    main()