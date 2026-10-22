import json
from pathlib import Path

DATA_DIR = Path("data/raw/hungary_2026")


FILES = [
    "drivers.json",
    "laps.json",
    "stints.json",
    "pit.json",
    "intervals.json",
    "race_control.json",
    "weather.json",
]


def inspect_file(filename: str):
    filepath = DATA_DIR / filename

    with filepath.open("r", encoding="utf-8") as file:
        data = json.load(file)

    print("\n" + "=" * 60)
    print(filename)
    print("=" * 60)

    print(f"Records: {len(data)}")

    if data:
        print("\nFields:")
        for field in data[0].keys():
            print(f"  - {field}")

        print("\nFirst record:")
        print(json.dumps(data[0], indent=2, ensure_ascii=False))


def main():
    for filename in FILES:
        inspect_file(filename)


if __name__ == "__main__":
    main()