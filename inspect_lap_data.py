
import json
from pathlib import Path

path = Path("data/raw/australia_2025/laps.json")

with path.open("r", encoding="utf-8") as file:
    data = json.load(file)

print("Köradatok ellenőrzése")
print("=" * 75)
print(f"Rekordok száma: {len(data)}")

if data:
    print("\nElérhető mezők:")
    print(list(data[0].keys()))

    print("\nElső három rekord:")
    for row in data[:3]:
        print(json.dumps(row, ensure_ascii=False, indent=2))

    print("\nLegutolsó három rekord:")
    for row in data[-3:]:
        print(json.dumps(row, ensure_ascii=False, indent=2))
