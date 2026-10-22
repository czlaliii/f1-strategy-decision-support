"""Validate downloaded historical OpenF1 race data without changing source files.

Run from the project root:
    python -m src.ingestion.validate_race_data
    python -m src.ingestion.validate_race_data --year 2023
    python -m src.ingestion.validate_race_data --race zandvoort_2023

Writes CSV reports under data/processed/ by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RAW_DIR = Path("data/raw")
CATALOG_PATH = Path("data/metadata/race_catalog.csv")
DEFAULT_REPORT = Path("data/processed/race_data_validation.csv")
DEFAULT_MISSING_REPORT = Path("data/processed/race_data_missing_from_disk.csv")
BASE_FILES = ("drivers.json", "laps.json", "stints.json", "pit.json", "intervals.json", "race_control.json", "weather.json", "position.json")


def read_json_list(path: Path) -> tuple[list[Any] | None, str]:
    if not path.exists():
        return None, "missing"
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid_json:{type(exc).__name__}"
    if not isinstance(value, list):
        return None, "not_a_list"
    return value, "ok"


def as_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_catalog() -> list[dict[str, str]]:
    if not CATALOG_PATH.exists():
        return []
    try:
        with CATALOG_PATH.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def get_session_keys(records: list[Any] | None) -> set[int]:
    keys: set[int] = set()
    for row in records or []:
        if isinstance(row, dict):
            key = as_int(row.get("session_key"))
            if key is not None:
                keys.add(key)
    return keys


def inspect_folder(folder: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "folder": folder.name,
        "year_from_folder": _year_from_name(folder.name),
        "session_key": "",
        "drivers": 0,
        "laps": 0,
        "stints": 0,
        "pit": 0,
        "intervals": 0,
        "race_control": 0,
        "weather": 0,
        "position": 0,
        "location_files": 0,
        "location_records": 0,
        "car_data_files": 0,
        "car_data_records": 0,
        "missing_base_files": "",
        "invalid_files": "",
        "empty_files": "",
        "session_key_mismatches": "",
        "drivers_without_location": "",
        "laps_missing_duration": 0,
        "duplicate_driver_lap_keys": 0,
        "status": "",
        "notes": "",
    }
    records_by_name: dict[str, list[Any] | None] = {}
    missing, invalid, empty, mismatches, notes = [], [], [], [], []
    for filename in BASE_FILES:
        records, state = read_json_list(folder / filename)
        records_by_name[filename] = records
        stem = filename[:-5]
        if state == "missing":
            missing.append(filename)
        elif state != "ok":
            invalid.append(f"{filename}({state})")
        else:
            result[stem] = len(records or [])
            if not records:
                empty.append(filename)
    for filename in missing:
        # pit data may legitimately be unavailable for a session; a file is still expected
        pass

    # Collect session keys from primary session-scoped files and check consistency.
    for filename in ("drivers.json", "laps.json", "stints.json", "intervals.json", "race_control.json", "weather.json"):
        records = records_by_name.get(filename)
        keys = get_session_keys(records)
        if len(keys) > 1:
            mismatches.append(f"{filename}:multiple_keys={sorted(keys)}")
    primary_keys = get_session_keys(records_by_name.get("drivers.json")) | get_session_keys(records_by_name.get("laps.json"))
    if len(primary_keys) == 1:
        result["session_key"] = next(iter(primary_keys))
    elif len(primary_keys) > 1:
        mismatches.append(f"drivers/laps keys={sorted(primary_keys)}")

    drivers = records_by_name.get("drivers.json") or []
    driver_numbers = {str(row.get("driver_number")) for row in drivers if isinstance(row, dict) and row.get("driver_number") is not None}
    laps = records_by_name.get("laps.json") or []
    lap_keys = []
    for row in laps:
        if not isinstance(row, dict):
            continue
        duration = row.get("lap_duration")
        try:
            valid_duration = duration is not None and float(duration) > 0
        except (TypeError, ValueError):
            valid_duration = False
        if not valid_duration:
            result["laps_missing_duration"] += 1
        driver, lap_no = row.get("driver_number"), row.get("lap_number")
        if driver is not None and lap_no is not None:
            lap_keys.append((str(driver), str(lap_no)))
    counts = Counter(lap_keys)
    result["duplicate_driver_lap_keys"] = sum(count - 1 for count in counts.values() if count > 1)

    location_drivers: set[str] = set()
    for path in sorted(folder.glob("location_driver_*.json")):
        match = re.search(r"location_driver_(\d+)\.json$", path.name)
        if match:
            location_drivers.add(match.group(1))
        records, state = read_json_list(path)
        if state != "ok":
            invalid.append(f"{path.name}({state})")
            continue
        result["location_files"] += 1
        result["location_records"] += len(records or [])
        if not records:
            empty.append(path.name)
        keys = get_session_keys(records)
        if len(keys) > 1:
            mismatches.append(f"{path.name}:multiple_keys={sorted(keys)}")

    for path in sorted(folder.glob("car_data_driver_*.json")):
        records, state = read_json_list(path)
        if state != "ok":
            invalid.append(f"{path.name}({state})")
            continue
        result["car_data_files"] += 1
        result["car_data_records"] += len(records or [])
        if not records:
            empty.append(path.name)

    result["drivers_without_location"] = ";".join(sorted(driver_numbers - location_drivers))
    result["missing_base_files"] = ";".join(missing)
    result["invalid_files"] = ";".join(sorted(set(invalid)))
    result["empty_files"] = ";".join(sorted(set(empty)))
    result["session_key_mismatches"] = ";".join(mismatches)
    if not drivers:
        notes.append("no_driver_records")
    if not laps:
        notes.append("no_lap_records")
    if not result["location_files"]:
        notes.append("no_location_files_replay_may_not_work")
    if result["laps_missing_duration"]:
        notes.append("some_laps_have_missing_or_nonpositive_duration")
    if result["duplicate_driver_lap_keys"]:
        notes.append("duplicate_driver_lap_number_pairs")
    if missing:
        notes.append("missing_base_files")
    if invalid:
        notes.append("invalid_json_or_unexpected_format")
    if mismatches:
        notes.append("session_key_inconsistency")
    result["notes"] = ";".join(notes)
    result["status"] = "CHECK" if notes else "OK"
    return result


def _year_from_name(name: str) -> str:
    match = re.search(r"(?:19|20)\d{2}", name)
    return match.group(0) if match else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local OpenF1 race files; does not modify downloaded data.")
    parser.add_argument("--year", type=int, help="Only inspect folders whose names contain this year.")
    parser.add_argument("--race", help="Inspect one data/raw subfolder by its folder name.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT, help=f"CSV report path (default: {DEFAULT_REPORT}).")
    parser.add_argument("--missing-output", type=Path, default=DEFAULT_MISSING_REPORT, help=f"Catalog races missing locally report (default: {DEFAULT_MISSING_REPORT}).")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"ERROR: raw-data directory not found: {RAW_DIR}. Run this command from the project root.")
        return 2
    folders = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    if args.race:
        folders = [p for p in folders if p.name.casefold() == args.race.casefold()]
    if args.year is not None:
        folders = [p for p in folders if str(args.year) in p.name]
    if not folders:
        print("No matching race folders found.")
        return 1

    rows = [inspect_folder(folder) for folder in folders]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Compare local session keys with the catalog to identify catalog entries not downloaded.
    catalog = load_catalog()
    # Use every local folder for catalog comparison, even when --race narrows the detailed audit.
    all_local_keys: set[int] = set()
    for local_folder in (p for p in RAW_DIR.iterdir() if p.is_dir()):
        for filename in ("drivers.json", "laps.json"):
            local_records, state = read_json_list(local_folder / filename)
            if state == "ok":
                all_local_keys.update(get_session_keys(local_records))
    missing_rows = []
    for item in catalog:
        year = as_int(item.get("year"))
        session_key = as_int(item.get("session_key"))
        if args.year is not None and year != args.year:
            continue
        if session_key is not None and session_key not in all_local_keys:
            missing_rows.append({"year": item.get("year", ""), "meeting_name": item.get("meeting_name", ""), "session_key": item.get("session_key", ""), "race_date": item.get("race_date", ""), "status": "NOT_FOUND_LOCALLY"})
    args.missing_output.parent.mkdir(parents=True, exist_ok=True)
    with args.missing_output.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["year", "meeting_name", "session_key", "race_date", "status"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(missing_rows)

    ok_count = sum(row["status"] == "OK" for row in rows)
    check_count = len(rows) - ok_count
    print("\n=== LOCAL OPENF1 DATA VALIDATION ===")
    print(f"Folders checked: {len(rows)} | OK: {ok_count} | Need review: {check_count}")
    print(f"Race-folder report: {args.output}")
    print(f"Catalog sessions not matched to inspected local folders: {len(missing_rows)}")
    print(f"Missing-session report: {args.missing_output}")
    print("\nPer-race summary:")
    for row in rows:
        print(f"- {row['folder']}: {row['status']} | drivers={row['drivers']} laps={row['laps']} location_files={row['location_files']} | {row['notes'] or 'no obvious issues'}")
    if missing_rows:
        print("\nNote: catalog entries absent locally are not automatically errors; cancelled races or non-race sessions may appear in a catalog.")
    return 0 if check_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
