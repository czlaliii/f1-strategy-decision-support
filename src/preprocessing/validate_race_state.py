from pathlib import Path

import pandas as pd

RACE_STATE_PATH = Path(
    "data/processed/hungary_2026/race_state.parquet"
)


def main():
    print("Loading race state...")

    race_state = pd.read_parquet(RACE_STATE_PATH)

    print(f"Shape: {race_state.shape}")

    # --- 1. BASIC INFORMATION ---

    print("\n=== BASIC INFORMATION ===")

    print(f"Rows: {len(race_state)}")
    print(f"Drivers: {race_state['driver_number'].nunique()}")
    print(
        f"Lap range: "
        f"{race_state['lap_number'].min()} - "
        f"{race_state['lap_number'].max()}"
    )

    # --- 2. DUPLICATE DRIVER/LAP COMBINATIONS ---

    print("\n=== DUPLICATES ===")

    duplicates = race_state.duplicated(
        subset=["driver_number", "lap_number"]
    ).sum()

    print(
        f"Duplicate (driver_number, lap_number) rows: "
        f"{duplicates}"
    )

    # --- 3. ROWS PER DRIVER ---

    print("\n=== ROWS PER DRIVER ===")

    rows_per_driver = (
        race_state
        .groupby("driver_number")
        .size()
        .sort_index()
    )

    print(rows_per_driver.to_string())

    # --- 4. LAP NUMBER CONSISTENCY ---

    print("\n=== LAP NUMBER CONSISTENCY ===")

    inconsistent_drivers = []

    for driver_number, driver_data in race_state.groupby(
        "driver_number"
    ):
        laps = driver_data["lap_number"].sort_values()

        duplicate_laps = laps.duplicated().sum()

        if duplicate_laps > 0:
            inconsistent_drivers.append(
                (
                    driver_number,
                    "duplicate laps",
                    duplicate_laps
                )
            )

        missing_laps = set(
            range(
                int(laps.min()),
                int(laps.max()) + 1
            )
        ) - set(laps)

        if missing_laps:
            inconsistent_drivers.append(
                (
                    driver_number,
                    "missing laps",
                    sorted(missing_laps)
                )
            )

    if inconsistent_drivers:
        for item in inconsistent_drivers:
            print(item)
    else:
        print("No lap-number inconsistencies found.")

    # --- 5. DATE VALIDATION ---

    print("\n=== DATE VALIDATION ===")

    invalid_date_start = race_state[
        "date_start"
    ].isna().sum()

    invalid_date_end = race_state[
        "date_end"
    ].isna().sum()

    print(
        f"Missing date_start: {invalid_date_start}"
    )

    print(
        f"Missing date_end: {invalid_date_end}"
    )

    invalid_order = (
        race_state["date_end"]
        < race_state["date_start"]
    ).sum()

    print(
        f"date_end before date_start: {invalid_order}"
    )

    # --- 6. LAP DURATION VALIDATION ---

    print("\n=== LAP DURATION ===")

    print(
        f"Missing lap_duration: "
        f"{race_state['lap_duration'].isna().sum()}"
    )

    print(
        f"Minimum lap duration: "
        f"{race_state['lap_duration'].min():.3f} s"
    )

    print(
        f"Maximum lap duration: "
        f"{race_state['lap_duration'].max():.3f} s"
    )

    print(
        f"Average lap duration: "
        f"{race_state['lap_duration'].mean():.3f} s"
    )

    # --- 7. POSITION ---

    print("\n=== POSITION ===")

    missing_position = race_state[
        "position"
    ].isna().sum()

    print(
        f"Missing position: {missing_position}"
    )

    print(
        f"Minimum position: "
        f"{race_state['position'].min()}"
    )

    print(
        f"Maximum position: "
        f"{race_state['position'].max()}"
    )

    invalid_position = race_state[
        (
            race_state["position"].notna()
        )
        & (
            (race_state["position"] < 1)
            | (race_state["position"] > 22)
        )
    ]

    print(
        f"Invalid position values: "
        f"{len(invalid_position)}"
    )

    # --- 8. GAP / INTERVAL ---

    print("\n=== GAP / INTERVAL ===")
    print(
        f"Missing gap_to_leader: "
        f"{race_state['gap_to_leader'].isna().sum()}"
    )
    print(
        f"Missing interval: "
        f"{race_state['interval'].isna().sum()}"
    )

    print(
        f"Negative gap_to_leader values: "
        f"{(race_state['gap_to_leader'] < 0).sum()}"
    )

    print(
        f"Negative interval values: "
        f"{(race_state['interval'] < 0).sum()}"
    )


    print("\n=== LAPS DOWN ===")
    print(
        race_state["laps_down"]
        .value_counts(dropna=False)
        .sort_index()
    )

    # --- 9. TYRE DATA ---

    print("\n=== TYRE DATA ===")

    print(
        "Compounds:"
    )

    print(
        race_state["compound"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        f"\nMissing compound: "
        f"{race_state['compound'].isna().sum()}"
    )

    negative_tyre_age = (
        race_state["tyre_age"] < 0
    ).sum()

    print(
        f"Negative tyre_age: {negative_tyre_age}"
    )

    # --- 10. PIT STOPS ---

    print("\n=== PIT STOPS ===")

    print(
        "Pit stops per driver:"
    )

    pit_stops = (
        race_state
        .groupby("driver_number")["pit_stop"]
        .sum()
        .sort_index()
    )

    print(pit_stops.to_string())

    # --- 11. WEATHER ---

    print("\n=== WEATHER ===")

    weather_columns = [
        "wind_direction",
        "rainfall",
        "air_temperature",
        "humidity",
        "pressure",
        "track_temperature",
        "wind_speed",
    ]

    weather_missing = (
        race_state[weather_columns]
        .isna()
        .sum()
    )

    print(weather_missing.to_string())

    # --- 12. FINAL SUMMARY ---

    negative_gap = race_state[
        race_state["gap_to_leader"] < 0
    ]

    negative_interval = race_state[
        race_state["interval"] < 0
    ]

    print("\n=== VALIDATION SUMMARY ===")

    checks = {
        "No duplicate driver/lap rows": duplicates == 0,
        "No missing date_start": invalid_date_start == 0,
        "No missing date_end": invalid_date_end == 0,
        "No invalid date order": invalid_order == 0,
        "No invalid positions": len(invalid_position) == 0,
        "No negative gaps": len(negative_gap) == 0,
        "No negative intervals": len(negative_interval) == 0,
        "No negative tyre ages": negative_tyre_age == 0,
    }

    for check, result in checks.items():
        status = "PASS" if result else "FAIL"

        print(
            f"[{status}] {check}"
        )

    print("\nValidation finished.")


if __name__ == "__main__":
    main()