import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RACES = [
    "australia_2025",
    "bahrain_2023",
    "canada_2024",
    "hungary_2026",
    "imola_2024",
    "monaco_2023",
    "sao_paulo_2024",
    "silverstone_2024",
    "silverstone_2025",
    "zandvoort_2023",
]


def main() -> None:
    benchmark_dir = Path("data/processed")
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    output_csv = benchmark_dir / "spark_race_state_benchmark.csv"

    env = os.environ.copy()

    env["PYSPARK_SUBMIT_ARGS"] = (
        "--conf "
        "spark.driver.extraJavaOptions=-Djava.library.path=C:/hadoop/bin "
        "pyspark-shell"
    )

    results = []

    print("=== SPARK RACE STATE - ALL RACES ===\n")

    total_start = time.perf_counter()

    for race in RACES:
        print(f"--- {race} ---")

        start = time.perf_counter()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.spark.build_race_state_spark",
                race,
            ],
            env=env,
            capture_output=True,
            text=True,
        )

        elapsed = time.perf_counter() - start

        rows = None
        columns = None

        row_match = re.search(
            r"Rows:\s*(\d+)",
            result.stdout,
        )

        column_match = re.search(
            r"Columns:\s*(\d+)",
            result.stdout,
        )

        if row_match:
            rows = int(row_match.group(1))

        if column_match:
            columns = int(column_match.group(1))

        success = result.returncode == 0

        results.append(
            {
                "race": race,
                "status": "OK" if success else "FAILED",
                "rows": rows,
                "columns": columns,
                "elapsed_seconds": round(elapsed, 3),
            }
        )

        if success:
            print(
                f"OK       "
                f"{elapsed:.3f} s   "
                f"rows={rows}, columns={columns}"
            )
        else:
            print(f"FAILED   {elapsed:.3f} s")
            print("\n--- STDOUT ---")
            print(result.stdout)
            print("\n--- STDERR ---")
            print(result.stderr)

        print()

    total_elapsed = time.perf_counter() - total_start

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "race",
                "status",
                "rows",
                "columns",
                "elapsed_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print("=== SUMMARY ===")

    successful = [
        result
        for result in results
        if result["status"] == "OK"
    ]

    for result in results:
        print(
            f"{result['race']:<20}"
            f"{result['status']:<10}"
            f"{result['elapsed_seconds']:>8.3f} s"
        )

    print()
    print(f"Successful races: {len(successful)}/{len(RACES)}")
    print(f"Total wall-clock time: {total_elapsed:.3f} s")
    print(f"Benchmark saved to: {output_csv}")


if __name__ == "__main__":
    main()