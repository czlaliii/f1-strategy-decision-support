import pytest
from pyspark.sql import functions as F

from src.spark.process_telemetry import aggregate_to_laps, load_telemetry
from src.spark.session import get_spark_session


@pytest.fixture(scope="module")
def spark():
    session = get_spark_session("F1SparkTelemetryTests")
    yield session
    session.stop()


@pytest.fixture
def synthetic_telemetry_and_laps(spark):
    telemetry = spark.createDataFrame(
        [
            (14, 11342, "car", "2026-07-26 14:00:01", 300.0, 90.0, 0.0, 11000.0, None, None, None),
            (14, 11342, "car", "2026-07-26 14:00:02", 310.0, 100.0, 0.0, 11200.0, None, None, None),
            (14, 11342, "location", "2026-07-26 14:00:01", None, None, None, None, 1.0, 2.0, 3.0),
        ],
        "driver_number int, session_key long, source string, timestamp string, speed double, throttle double, brake double, rpm double, x double, y double, z double",
    ).withColumn("timestamp", F.to_timestamp("timestamp"))
    laps = spark.createDataFrame(
        [(14, 11342, 1, "2026-07-26 14:00:00", 5.0)],
        "driver_number int, session_key long, lap_number int, lap_timestamp_start string, lap_duration_seconds double",
    ).withColumn("lap_timestamp_start", F.to_timestamp("lap_timestamp_start")).withColumn(
        "lap_timestamp_end", F.expr("lap_timestamp_start + INTERVAL 5 SECONDS")
    )
    return telemetry, laps


def test_spark_loads_raw_telemetry_with_expected_schema(spark):
    telemetry = load_telemetry(spark)
    assert {"timestamp", "driver_number", "session_key", "source", "speed"}.issubset(telemetry.columns)
    # Creating this DataFrame verifies that the glob paths and explicit schemas
    # are accepted without scanning the full 1.5M-record historical dataset.
    assert telemetry.schema["source"].dataType.simpleString() == "string"


def test_telemetry_schema_normalizes_timestamp_and_numeric_fields(spark):
    telemetry = load_telemetry(spark)
    fields = {field.name: field.dataType.simpleString() for field in telemetry.schema.fields}
    assert fields["timestamp"] == "timestamp"
    assert fields["driver_number"] == "int"
    assert fields["speed"] == "double"


def test_lap_aggregation_keeps_correct_driver_and_lap(synthetic_telemetry_and_laps):
    telemetry, laps = synthetic_telemetry_and_laps
    output = aggregate_to_laps(telemetry, laps).collect()
    assert len(output) == 1
    assert output[0]["driver_number"] == 14
    assert output[0]["lap_number"] == 1


def test_lap_aggregation_has_no_duplicate_driver_lap_session_keys(synthetic_telemetry_and_laps):
    telemetry, laps = synthetic_telemetry_and_laps
    output = aggregate_to_laps(telemetry, laps)
    assert output.count() == output.select("driver_number", "lap_number", "session_key").distinct().count()


def test_lap_aggregation_contains_required_telemetry_fields(synthetic_telemetry_and_laps):
    telemetry, laps = synthetic_telemetry_and_laps
    output = aggregate_to_laps(telemetry, laps)
    assert {
        "driver_number", "lap_number", "session_key", "telemetry_timestamp_start",
        "telemetry_timestamp_end", "avg_speed", "max_speed", "car_sample_count",
    }.issubset(output.columns)
