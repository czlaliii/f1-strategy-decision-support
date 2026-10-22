import json
from pathlib import Path

RAW_DIR = Path("data/raw/hungary_2026")
RACE_CONTROL_FILE = RAW_DIR / "race_control.json"


def main():
    print("Loading race control data...")

    with open(RACE_CONTROL_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Total events: {len(data)}")
    print()

    # Csak az 50-60. kör környéki események
    events = []

    for event in data:
        lap_number = event.get("lap_number")

        if lap_number is None:
            continue

        if 50 <= lap_number <= 60:
            events.append(event)

    print("=== RACE CONTROL EVENTS: LAPS 50-60 ===")

    if not events:
        print("No race control events found.")
        return

    for event in events:
        print("-" * 80)

        print(f"Date:          {event.get('date')}")
        print(f"Lap:           {event.get('lap_number')}")
        print(f"Driver:        {event.get('driver_number')}")
        print(f"Category:      {event.get('category')}")
        print(f"Flag:          {event.get('flag')}")
        print(f"Scope:         {event.get('scope')}")
        print(f"Sector:        {event.get('sector')}")
        print(f"Message:       {event.get('message')}")

    print("-" * 80)
    print()
    print(f"Events found: {len(events)}")


if __name__ == "__main__":
    main()