import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


RACES = {
    "silverstone_2025": [
        (10, 23),
    ],
    "australia_2025": [
        (1, 7),
        (30, 51),
    ],
    "zandvoort_2023": [
        (58, 68),
    ],
    "sao_paulo_2024": [
        (25, 35),
    ],
}


def find_race_control_file(race):
    candidates = [
        RAW_DIR / race / "race_control.json",
        RAW_DIR / race / "race_control" / "race_control.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(
        RAW_DIR.rglob(
            "race_control.json"
        )
    )

    for path in matches:
        if race in str(path):
            return path

    return None


def load_race_control(race):
    path = find_race_control_file(race)

    if path is None:
        print(
            f"\nWARNING: race_control.json not found "
            f"for {race}"
        )
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    print(
        f"\nLoaded race control: {path}"
    )

    print(
        f"Rows: {len(df)}"
    )

    return df


def print_columns(df):
    print("\nAvailable columns:")
    print(
        ", ".join(
            str(column)
            for column in df.columns
        )
    )


def print_relevant_events(
    df,
    race,
    min_lap,
    max_lap,
):
    print("\n" + "=" * 78)
    print(
        f"{race} | laps {min_lap}-{max_lap}"
    )
    print("=" * 78)

    if "lap_number" in df.columns:
        events = df[
            df["lap_number"].between(
                min_lap,
                max_lap,
            )
        ].copy()

    else:
        print(
            "No lap_number column available."
        )
        print_columns(df)
        return

    if len(events) == 0:
        print(
            "No race-control events "
            "in this lap range."
        )
        return

    columns = [
        "lap_number",
        "category",
        "flag",
        "scope",
        "sector",
        "message",
        "driver_number",
        "date",
        "date_start",
        "date_end",
    ]

    available = [
        column
        for column in columns
        if column in events.columns
    ]

    print(
        events[available]
        .sort_values(
            ["lap_number"]
        )
        .to_string(
            index=False
        )
    )


def print_event_summary(df, race):
    print("\n" + "-" * 78)
    print(f"{race} | EVENT SUMMARY")
    print("-" * 78)

    for column in [
        "category",
        "flag",
        "scope",
    ]:
        if column in df.columns:
            print(f"\n{column}:")
            print(
                df[column]
                .fillna("UNKNOWN")
                .value_counts()
                .to_string()
            )

    if "message" in df.columns:
        print("\nMessages containing important keywords:")

        keywords = (
            "SAFETY CAR",
            "VSC",
            "RED FLAG",
            "YELLOW",
            "GREEN",
            "CHEQUERED",
            "FORMATION",
            "MEDICAL",
        )

        mask = (
            df["message"]
            .fillna("")
            .str.upper()
            .apply(
                lambda value:
                any(
                    keyword in value
                    for keyword in keywords
                )
            )
        )

        relevant = df[mask]

        columns = [
            "lap_number",
            "category",
            "flag",
            "scope",
            "sector",
            "message",
            "date",
        ]

        available = [
            column
            for column in columns
            if column in relevant.columns
        ]

        print(
            relevant[available]
            .sort_values(
                "lap_number"
            )
            .to_string(
                index=False
            )
        )


def main():
    print("=" * 78)
    print("TRANSITION ↔ RACE CONTROL ANALYSIS")
    print("=" * 78)

    for race, ranges in RACES.items():
        df = load_race_control(race)

        if df is None:
            continue

        print_event_summary(
            df,
            race,
        )

        for min_lap, max_lap in ranges:
            print_relevant_events(
                df,
                race,
                min_lap,
                max_lap,
            )


if __name__ == "__main__":
    main()