import os
import subprocess
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


def main():
    env = os.environ.copy()

    env["PYSPARK_SUBMIT_ARGS"] = (
        "--conf spark.driver.extraJavaOptions="
        "-Djava.library.path=C:/hadoop/bin pyspark-shell"
    )

    results = []

    print("=== SPARK MULTI-RACE PROCESSING ===")

    for race in RACES:
        print(f"\n{'=' * 60}")
        print(f"Processing: {race}")
        print(f"{'=' * 60}")

        start = time.perf_counter()

        try:
            subprocess.run(
                [
                    "python",
                    "-m",
                    "src.spark.build_lap_data_spark",
                    race,
                ],
                env=env,
                check=True,
            )

            elapsed = time.perf_counter() - start

            results.append(
                {
                    "race": race,
                    "status": "OK",
                    "time_seconds": round(elapsed, 3),
                }
            )

            print(f"\n{race}: OK ({elapsed:.3f} s)")

        except subprocess.CalledProcessError:
            elapsed = time.perf_counter() - start

            results.append(
                {
                    "race": race,
                    "status": "ERROR",
                    "time_seconds": round(elapsed, 3),
                }
            )

            print(f"\n{race}: ERROR")

    print("\n" + "=" * 60)
    print("=== SUMMARY ===")
    print("=" * 60)

    total_time = 0.0

    for result in results:
        print(
            f"{result['race']:20s} "
            f"{result['status']:6s} "
            f"{result['time_seconds']:.3f} s"
        )

        if result["status"] == "OK":
            total_time += result["time_seconds"]

    print("-" * 60)
    print(f"Total Spark processing time: {total_time:.3f} s")

    output_path = Path("data/processed/spark_multi_race_benchmark.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("race,status,time_seconds\n")

        for result in results:
            f.write(
                f"{result['race']},"
                f"{result['status']},"
                f"{result['time_seconds']}\n"
            )

    print(f"Benchmark saved to: {output_path}")


if __name__ == "__main__":
    main()