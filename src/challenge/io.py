import os
from typing import Any, List, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .schemas import atc_schema, clicks_schema, impressions_schema, orders_schema


def _read_json_with_schema(
    spark: SparkSession,
    path: str,
    schema: Any,
    corrupt_col: Optional[str] = "_corrupt_record",
) -> Tuple[DataFrame, Optional[DataFrame]]:
    df = (
        spark.read.schema(schema)
        .option("mode", "PERMISSIVE")
        .option("multiline", "true")  # support JSON arrays (single file with [] is fine)
        .option("columnNameOfCorruptRecord", corrupt_col)
        .json(path)
    )
    bad: Optional[DataFrame] = None
    if corrupt_col and corrupt_col in df.columns:
        bad = df.where(F.col(corrupt_col).isNotNull())
        df = df.where(F.col(corrupt_col).isNull()).drop(corrupt_col)
    return df, bad


def read_impressions(spark: SparkSession, path: str) -> Tuple[DataFrame, Optional[DataFrame]]:
    return _read_json_with_schema(spark, path, impressions_schema)


def read_clicks(spark: SparkSession, path: str) -> Tuple[DataFrame, Optional[DataFrame]]:
    return _read_json_with_schema(spark, path, clicks_schema)


def read_atc(spark: SparkSession, path: str) -> Tuple[DataFrame, Optional[DataFrame]]:
    return _read_json_with_schema(spark, path, atc_schema)


def read_orders(spark: SparkSession, path: str) -> Tuple[DataFrame, Optional[DataFrame]]:
    return _read_json_with_schema(spark, path, orders_schema)


def write_parquet(
    df: DataFrame,
    out_path: str,
    *,
    mode: str = "overwrite",
    partition_by: Optional[List[str]] = None,
) -> None:
    (df.write.mode(mode).partitionBy(*(partition_by or [])).parquet(out_path))


def write_quarantine(df: Optional[DataFrame], out_dir: str, name: str) -> None:
    if df is None:
        return
    path = os.path.join(out_dir, f"_quarantine_{name}")
    df.write.mode("overwrite").json(path)
