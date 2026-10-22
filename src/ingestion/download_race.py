"""Resumable downloader for historical OpenF1 Race sessions.

Examples (run from the project root):
    python -m src.ingestion.download_race --list
    python -m src.ingestion.download_race --year 2023
    python -m src.ingestion.download_race --year 2023 --event "Dutch Grand Prix"
    python -m src.ingestion.download_race --all
    python -m src.ingestion.download_race --all --telemetry full

Default mode downloads the location data used by the replay; --telemetry full
also fetches car_data. Valid existing files are skipped, so downloads can resume.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

from src.ingestion.api_client import OpenF1Client

CATALOG_PATH = Path("data/metadata/race_catalog.csv")
OUTPUT_BASE_DIR = Path("data/raw")
NORMAL_SESSION_ENDPOINTS = (
    "laps",
    "stints",
    "pit",
    "intervals",
    "race_control",
    "weather",
)

# Preserve the folder names already used by this project when starting fresh.
KNOWN_FOLDER_NAMES = {
    (2023, "Bahrain Grand Prix"): "bahrain_2023",
    (2023, "Monaco Grand Prix"): "monaco_2023",
    (2023, "Dutch Grand Prix"): "zandvoort_2023",
    (2024, "Emilia Romagna Grand Prix"): "imola_2024",
    (2024, "Canadian Grand Prix"): "canada_2024",
    (2024, "British Grand Prix"): "silverstone_2024",
    (2024, "São Paulo Grand Prix"): "sao_paulo_2024",
    (2025, "Australian Grand Prix"): "australia_2025",
    (2025, "British Grand Prix"): "silverstone_2025",
    (2026, "Hungarian Grand Prix"): "hungary_2026",
}


def save_json(output_dir: Path, filename: str, data: list) -> None:
    """Write JSON atomically to avoid leaving a partial file after interruption."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    temporary_path = filepath.with_suffix(filepath.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)
    temporary_path.replace(filepath)
    print(f"Saved: {filepath} ({len(data)} records)")


def read_valid_json(path: Path):
    """Return a JSON list if the file is complete and valid, otherwise None."""
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else None
    except (OSError, json.JSONDecodeError):
        return None


def get_or_download(
    client: OpenF1Client,
    output_dir: Path,
    filename: str,
    endpoint: str,
    params: dict,
    *,
    optional_404: bool = False,
    force: bool = False,
) -> list:
    """Reuse an existing valid file, otherwise fetch and save it."""
    path = output_dir / filename
    if not force:
        existing = read_valid_json(path)
        if existing is not None:
            print(f"Already present, skipping: {path} ({len(existing)} records)")
            return existing

    try:
        data = client.get(endpoint, params)
    except requests.HTTPError as exc:
        if optional_404 and exc.response is not None and exc.response.status_code == 404:
            print(f"No {endpoint} data for this session; saving an empty list.")
            data = []
        else:
            raise

    save_json(output_dir, filename, data)
    return data


def load_catalog() -> pd.DataFrame:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Race catalog not found: {CATALOG_PATH}. "
            "Run `python -m src.ingestion.catalog_races` first."
        )
    catalog = pd.read_csv(CATALOG_PATH)
    required = {"year", "meeting_name", "meeting_key", "session_key"}
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"The race catalog is missing columns: {', '.join(sorted(missing))}")
    return catalog


def slugify(value: str) -> str:
    value = value.lower().replace("ã", "a").replace("á", "a").replace("é", "e")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    value = re.sub(r"_+", "_", value)
    # Keep directory names readable without repeating "grand_prix".
    value = value.replace("_grand_prix", "")
    return value or "race"


def existing_folder_for_session(session_key: int) -> Path | None:
    """Find a previously downloaded folder by its session_key, whatever its name."""
    if not OUTPUT_BASE_DIR.exists():
        return None
    for folder in OUTPUT_BASE_DIR.iterdir():
        if not folder.is_dir():
            continue
        for filename in ("drivers.json", "laps.json"):
            records = read_valid_json(folder / filename)
            if records and any(
                str(row.get("session_key")) == str(session_key)
                for row in records
                if isinstance(row, dict) and row.get("session_key") is not None
            ):
                return folder
    return None


def output_folder_for_race(year: int, meeting_name: str, session_key: int) -> Path:
    existing = existing_folder_for_session(session_key)
    if existing is not None:
        return existing
    folder_name = KNOWN_FOLDER_NAMES.get(
        (year, meeting_name), f"{slugify(meeting_name)}_{year}"
    )
    return OUTPUT_BASE_DIR / folder_name


def select_races(catalog: pd.DataFrame, year: int | None, event: str | None) -> pd.DataFrame:
    races = catalog.copy()
    races["year"] = pd.to_numeric(races["year"], errors="coerce")
    races["session_key"] = pd.to_numeric(races["session_key"], errors="coerce")
    races = races.dropna(subset=["year", "session_key", "meeting_name"])
    races = races.drop_duplicates(subset=["session_key"]).copy()
    if year is not None:
        races = races[races["year"].astype(int) == year]
    if event:
        normalized = event.casefold().strip()
        exact = races[races["meeting_name"].astype(str).str.casefold() == normalized]
        if not exact.empty:
            races = exact
        else:
            races = races[races["meeting_name"].astype(str).str.casefold().str.contains(normalized, regex=False)]
    return races.sort_values(["year", "race_date"] if "race_date" in races.columns else ["year", "meeting_name"])


def download_race(
    client: OpenF1Client,
    race: pd.Series,
    telemetry_mode: str = "location",
    force: bool = False,
) -> None:
    year = int(race["year"])
    meeting_name = str(race["meeting_name"])
    meeting_key = int(race["meeting_key"])
    session_key = int(race["session_key"])
    output_dir = output_folder_for_race(year, meeting_name, session_key)

    print("\n" + "=" * 72)
    print(f"RACE: {year} - {meeting_name}")
    print(f"Session key: {session_key} | Meeting key: {meeting_key}")
    print(f"Output: {output_dir}")
    print(f"Telemetry mode: {telemetry_mode}")
    print("=" * 72)

    drivers = get_or_download(
        client, output_dir, "drivers.json", "drivers",
        {"session_key": session_key}, force=force,
    )
    driver_numbers = sorted({
        int(driver["driver_number"])
        for driver in drivers
        if isinstance(driver, dict) and driver.get("driver_number") is not None
    })
    if not driver_numbers:
        raise RuntimeError(f"No driver numbers found for {year} - {meeting_name}; telemetry cannot be downloaded.")
    print(f"Drivers found: {driver_numbers}")

    for endpoint in NORMAL_SESSION_ENDPOINTS:
        get_or_download(
            client, output_dir, f"{endpoint}.json", endpoint,
            {"session_key": session_key},
            optional_404=(endpoint == "pit"), force=force,
        )

    get_or_download(
        client, output_dir, "position.json", "position",
        {"meeting_key": meeting_key}, force=force,
    )

    telemetry_endpoints = ["location"]
    if telemetry_mode == "full":
        telemetry_endpoints.append("car_data")
    elif telemetry_mode == "none":
        telemetry_endpoints = []

    for endpoint in telemetry_endpoints:
        for index, driver_number in enumerate(driver_numbers, start=1):
            filename = f"{endpoint}_driver_{driver_number}.json"
            print(f"[{index}/{len(driver_numbers)}] {endpoint} for driver {driver_number}")
            get_or_download(
                client, output_dir, filename, endpoint,
                {"session_key": session_key, "driver_number": driver_number},
                force=force,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable OpenF1 downloader for historical Race sessions."
    )
    parser.add_argument("--all", action="store_true", help="Download every race in the catalog.")
    parser.add_argument("--year", type=int, help="Download all catalog races from a year, e.g. 2023.")
    parser.add_argument("--event", help="Download a race by meeting name (partial names are accepted). Can be combined with --year.")
    parser.add_argument("--list", action="store_true", help="List races in the catalog without downloading.")
    parser.add_argument(
        "--telemetry", choices=("none", "location", "full"), default="location",
        help="Telemetry to download: none, location (default; replay support), or full (also car_data).",
    )
    parser.add_argument("--force", action="store_true", help="Redownload and overwrite files, including valid existing ones.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.all and (args.year is not None or args.event):
        print("--all cannot be combined with --year or --event.")
        return 2
    if not args.list and not (args.all or args.year is not None or args.event):
        print("Choose --all, --year YEAR, or --event NAME. Use --list to inspect the catalog.")
        return 2

    catalog = load_catalog()
    races = select_races(catalog, args.year, args.event)
    if races.empty:
        print("No matching races found in the catalog.")
        return 1

    if args.list:
        columns = [column for column in ("year", "meeting_name", "country_name", "race_date", "session_key") if column in races.columns]
        print(races[columns].to_string(index=False))
        print(f"\nMatching races: {len(races)}")
        return 0

    print(f"Races selected: {len(races)}")
    print("Existing valid files will be skipped; interrupted downloads can be resumed.")
    failed = []
    for index, (_, race) in enumerate(races.iterrows(), start=1):
        print(f"\n### RACE {index}/{len(races)} ###")
        try:
            download_race(client=OpenF1Client(), race=race, telemetry_mode=args.telemetry, force=args.force)
        except Exception as exc:
            label = f"{int(race['year'])} - {race['meeting_name']}"
            failed.append((label, str(exc)))
            print(f"ERROR: {label}: {exc}")
            print("Continuing with the next race; rerun the same command to resume this one.")

    print("\n" + "=" * 72)
    print(f"Download run finished. Successful: {len(races) - len(failed)} | Failed: {len(failed)}")
    for label, error in failed:
        print(f"  FAILED: {label} — {error}")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
