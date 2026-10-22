from pathlib import Path

import pandas as pd

RACE = "hungary_2026"

PATH = (
    Path("data/raw")
    / RACE
    / "position.json"
)


def main():

    df = pd.read_json(PATH)

    df["date"] = pd.to_datetime(
        df["date"],
        utc=True,
    )

    df = df[
        df["driver_number"] == 11
    ].sort_values("date")

    print("=== DRIVER 11 POSITION DATA ===")

    print(
        df[
            [
                "driver_number",
                "date",
                "position",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()