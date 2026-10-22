from pathlib import Path

import pandas as pd

RAW_DIR = Path(
    "data/raw/hungary_2026"
)
PROCESSED_DIR = Path(
    "data/processed/hungary_2026"
)
RACE_STATE_FILE = PROCESSED_DIR / "race_state.parquet"
RACE_CONTROL_FILE = RAW_DIR / "race_control.json"
OUTPUT_FILE = PROCESSED_DIR / "race_state_with_conditions.parquet"

def load_data():
    race_state = pd.read_parquet(RACE_STATE_FILE)
    race_control = pd.read_json(RACE_CONTROL_FILE)

    race_state["date_start"] = pd.to_datetime(
        race_state["date_start"],
        utc=True,
        format="mixed",
    )

    race_state["date_end"] = pd.to_datetime(
        race_state["date_end"],
        utc=True,
        format="mixed",
    )

    race_control["date"] = pd.to_datetime(
        race_control["date"],
        utc=True,
        format="mixed",
    )

    return race_state, race_control

def build_race_condition_intervals(race_control):
    events = race_control.sort_values("date").copy()

    intervals = []
    current_start = None

    for _, event in events.iterrows():
        message = str(event["message"])
        timestamp = event["date"]

        if "VSC DEPLOYED" in message.upper():
            current_start = timestamp

        elif "VSC ENDING" in message.upper():
            if current_start is not None:
                intervals.append(
                    (current_start, timestamp)
                )
                current_start = None

    return intervals

def calculate_total_overlap_seconds(
    lap_start,
    lap_end,
    condition_intervals,
):
    total_overlap = 0.0

    for condition_start, condition_end in condition_intervals:

        overlap_start = max(
            lap_start,
            condition_start,
        )

        overlap_end = min(
            lap_end,
            condition_end,
        )

        if overlap_end > overlap_start:
            total_overlap += (
                overlap_end - overlap_start
            ).total_seconds()

    return total_overlap

def is_condition_active_at_time(timestamp, intervals):
    for start, end in intervals:
        if start <= timestamp < end:
            return 1

    return 0

def add_conditions(race_state, intervals):
    df = race_state.copy()

    # VSC duration during current lap

    df["vsc_duration_in_lap"] = df.apply(
        lambda row: calculate_total_overlap_seconds(
            row["date_start"],
            row["date_end"],
            intervals,
        ),
        axis=1,
    )

    # Was VSC active when the lap ended?

    df["vsc_active_at_end"] = df["date_end"].apply(
        lambda timestamp: is_condition_active_at_time(
            timestamp,
            intervals,
        )
    )

    # Was the lap affected by VSC at all?

    df["vsc_in_lap"] = (
        df["vsc_duration_in_lap"] > 0
    ).astype(int)

    # Fraction of lap completed under VSC

    df["vsc_lap_fraction"] = (
        df["vsc_duration_in_lap"]
        / df["lap_duration"]
    ).clip(lower=0, upper=1)

    return df

def main():
    print("Loading race state and race control data...")

    race_state, race_control = load_data()

    print(f"Race state shape: {race_state.shape}")
    print(f"Race control events: {len(race_control)}")

    intervals = build_race_condition_intervals(race_control)

    print("\n=== RACE CONDITION INTERVALS ===")

    for i, (start, end) in enumerate(intervals, start=1):
        duration = (end - start).total_seconds()

        print(
            f"VSC #{i}: "
            f"{start} -> {end} "
            f"({duration:.1f} seconds)"
        )

    race_state = add_conditions(
        race_state,
        intervals,
    )

    print("\n=== CONDITION FEATURES ===")

    condition_columns = [
        "driver_number",
        "lap_number",
        "lap_duration",
        "date_start",
        "date_end",
        "vsc_duration_in_lap",
        "vsc_in_lap",
        "vsc_lap_fraction",
        "vsc_active_at_end",
    ]

    print(
        race_state[
            condition_columns
        ]
        .query("lap_number >= 53 and lap_number <= 58")
        .to_string(index=False)
    )

    print("\n=== VSC SUMMARY ===")

    print(
        race_state[
            [
                "vsc_duration_in_lap",
                "vsc_in_lap",
                "vsc_lap_fraction",
                "vsc_active_at_end",
            ]
        ].describe()
    )

    race_state.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\n=== DONE ===")
    print(f"Shape: {race_state.shape}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()