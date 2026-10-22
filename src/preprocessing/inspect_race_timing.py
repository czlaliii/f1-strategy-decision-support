import json
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/hungary_2026")


def load_json(filename):
    with open(RAW_PATH / filename, "r", encoding="utf-8") as file:
        return pd.DataFrame(json.load(file))


def main():
    laps = load_json("laps.json")
    position = load_json("position.json")
    intervals = load_json("intervals.json")
    weather = load_json("weather.json")

    # Convert timestamps to datetime
    laps["date_start"] = pd.to_datetime(
        laps["date_start"],
        utc=True,
        format="mixed"
    )

    position["date"] = pd.to_datetime(
        position["date"],
        utc=True,
        format="mixed"
    )

    intervals["date"] = pd.to_datetime(
        intervals["date"],
        utc=True,
        format="mixed"
    )

    weather["date"] = pd.to_datetime(
        weather["date"],
        utc=True,
        format="mixed"
    )

    print("\n=== LAPS ===")
    print("Első időpont:", laps["date_start"].min())
    print("Utolsó időpont:", laps["date_start"].max())

    print("\nElső 10 lap:")
    print(
        laps[
            ["driver_number", "lap_number", "date_start", "lap_duration"]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n=== POSITION ===")
    print("Első időpont:", position["date"].min())
    print("Utolsó időpont:", position["date"].max())

    print("\nElső 20 position adat:")
    print(
        position[
            ["date", "driver_number", "position"]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n=== INTERVALS ===")
    print("Első időpont:", intervals["date"].min())
    print("Utolsó időpont:", intervals["date"].max())

    print("\nElső 20 interval adat:")
    print(
        intervals[
            ["date", "driver_number", "gap_to_leader", "interval"]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n=== WEATHER ===")
    print("Első időpont:", weather["date"].min())
    print("Utolsó időpont:", weather["date"].max())

    print("\nElső 10 weather adat:")
    print(
        weather[
            [
                "date",
                "air_temperature",
                "track_temperature",
                "humidity",
                "rainfall",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()