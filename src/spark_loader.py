from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


EXPECTED_COLUMNS = [
    "timestamp",
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6",
    "target",
]

FEATURE_COLUMNS = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6"]
TEMPORAL_FEATURES = [
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "quarter",
]


class ElectricityDataLoader:
    """Load, validate, and prepare the electricity forecasting dataset."""

    def __init__(self, data_path: str | Path = "data/electricity.csv", app_name: str = "AdaptiveElectricityForecasting"):
        self.data_path = Path(data_path)
        self.app_name = app_name
        self.spark = self._create_spark_session()

    def _create_spark_session(self) -> SparkSession:
        return (
            SparkSession.builder
            .appName(self.app_name)
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )

    def _validate_columns(self, df) -> None:
        missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(
                "The electricity dataset is missing required columns: " + ", ".join(missing)
            )

    def _coerce_numeric_columns(self, df):
        for column_name in FEATURE_COLUMNS + ["target"]:
            df = df.withColumn(column_name, F.col(column_name).cast("double"))
        return df

    def _parse_timestamp(self, df):
        df = df.withColumn("timestamp_raw", F.col("timestamp").cast("string"))
        df = df.withColumn("timestamp", F.to_date(F.col("timestamp_raw"), "dd-MM-yyyy"))
        invalid_rows = df.filter(F.col("timestamp").isNull())
        if invalid_rows.limit(1).count() > 0:
            raise ValueError(
                "Invalid timestamp values found. Expected dates in dd-MM-yyyy format."
            )
        return df.drop("timestamp_raw")

    def _validate_values(self, df):
        problematic = []
        for column_name in FEATURE_COLUMNS + ["target"]:
            null_count = df.filter(F.col(column_name).isNull()).count()
            if null_count:
                problematic.append(f"{column_name}:{null_count}")
        if problematic:
            raise ValueError("Missing numeric values detected in: " + ", ".join(problematic))
        return df

    def _add_temporal_features(self, df):
        return (
            df.withColumn("year", F.year(F.col("timestamp")))
            .withColumn("month", F.month(F.col("timestamp")))
            .withColumn("day", F.dayofmonth(F.col("timestamp")))
            .withColumn("day_of_week", F.dayofweek(F.col("timestamp")))
            .withColumn("day_of_year", F.dayofyear(F.col("timestamp")))
            .withColumn("week_of_year", F.weekofyear(F.col("timestamp")))
            .withColumn("quarter", F.quarter(F.col("timestamp")))
            .orderBy("timestamp")
        )

    def load_dataset(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Electricity dataset not found at: {self.data_path}")

        if self.data_path.stat().st_size == 0:
            raise ValueError(
                f"Electricity dataset is empty at: {self.data_path}. Place a valid CSV with columns: "
                + ", ".join(EXPECTED_COLUMNS)
            )

        df = self.spark.read.option("header", True).option("inferSchema", False).csv(str(self.data_path))
        if not df.columns or len(df.columns) == 0:
            raise ValueError(f"Electricity dataset at {self.data_path} is empty or malformed.")

        self._validate_columns(df)
        df = self._coerce_numeric_columns(df)
        df = self._parse_timestamp(df)
        df = self._validate_values(df)
        df = self._add_temporal_features(df)

        if df.rdd.isEmpty():
            raise ValueError("Electricity dataset is empty and cannot be used for training or forecasting.")

        return df

    def describe(self) -> Dict[str, Any]:
        df = self.load_dataset()
        stats = df.describe().toPandas().to_dict(orient="records")
        row_count = df.count()
        missing_values = {
            column_name: int(df.filter(F.col(column_name).isNull()).count())
            for column_name in EXPECTED_COLUMNS
        }
        return {
            "row_count": row_count,
            "columns": df.columns,
            "missing_values": missing_values,
            "summary": stats,
            "feature_columns": FEATURE_COLUMNS + TEMPORAL_FEATURES,
        }

    def close(self) -> None:
        self.spark.stop()


if __name__ == "__main__":
    loader = ElectricityDataLoader()
    try:
        df = loader.load_dataset()
        print("Loaded rows:", df.count())
        df.show(5, truncate=False)
        print(loader.describe())
    finally:
        loader.close()
