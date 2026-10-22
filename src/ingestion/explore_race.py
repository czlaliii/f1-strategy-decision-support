import requests

BASE_URL = "https://api.openf1.org/v1"

def get_2026_races():
    response = requests.get(
        f"{BASE_URL}/sessions",
        params={
            "year": 2026,
            "session_name": "Race"
        },
        timeout=30
    )

    response.raise_for_status()

    return response.json()

if __name__ == "__main__":
    races = get_2026_races()

    print(f"Talált versenyek száma: {len(races)}\n")

    for race in races:
        print(
            f"{race['session_key']} | "
            f"{race['meeting_key']} | "
            f"{race['country_name']} | "
            f"{race['circuit_short_name']} | "
            f"{race['date_start']}"
        )