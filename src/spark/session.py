import os
import subprocess
import tempfile
from pathlib import Path

import pyspark
from pyspark.sql import SparkSession


def _configure_windows_spark_home() -> None:
    """Avoid the Windows launcher issue caused by `!` in an installed path."""
    package_home = Path(pyspark.__file__).resolve().parent
    if os.name != "nt" or "!" not in str(package_home):
        return

    link_path = Path(tempfile.gettempdir()) / "f1_strategy_pyspark"
    if not link_path.exists():
        result = subprocess.run(
            ["cmd", "/d", "/v:off", "/c", "mklink", "/J", str(link_path), str(package_home)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Could not create temporary Spark junction: {result.stderr}")
    os.environ["SPARK_HOME"] = str(link_path)


def get_spark_session(app_name: str = "F1StrategyDecisionSupport") -> SparkSession:
    """Create the local Spark session used by offline historical processing."""
    _configure_windows_spark_home()
    return (
        SparkSession.builder.appName(app_name)
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
