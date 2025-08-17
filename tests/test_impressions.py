import os
import sys

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql import types as T

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from challenge.impressions import explode_impressions, normalize_impressions


def _sample_impressions_raw(spark):
    rows = [
        Row(
            dt="2025-08-15",
            ranking_id="r1",
            customer_id=101,
            impressions=[Row(item_id=1, is_order=True), Row(item_id=2, is_order=False)],
        ),
        Row(
            dt="dt=2025-08-16",
            ranking_id="r2",
            customer_id=102,
            impressions=[Row(item_id=3, is_order=False)],
        ),
    ]
    return spark.createDataFrame(rows)


def _sample_impressions_norm(spark):
    rows = [
        Row(
            as_of_date="2025-08-10",
            ranking_id="r3",
            customer_id=201,
            impressions=[Row(item_id=7, is_order=False)],
        ),
    ]
    df = spark.createDataFrame(rows)
    return df.withColumn("as_of_date", F.to_date("as_of_date"))


def test_normalize_impressions_from_dt_and_dt_equals_prefixed(spark):
    df = _sample_impressions_raw(spark)
    out = normalize_impressions(df)
    cols = set(out.columns)
    assert {"as_of_date", "ranking_id", "customer_id", "impressions"} <= cols

    c = out.collect()
    # as_of_date should be parsed from both plain and "dt=YYYY-MM-DD"
    assert str(c[0]["as_of_date"]) == "2025-08-15"
    assert str(c[1]["as_of_date"]) == "2025-08-16"
    assert isinstance(c[0]["customer_id"], int)
    assert isinstance(c[0]["ranking_id"], str)


def test_normalize_impressions_idempotent_when_as_of_date_present(spark):
    df = _sample_impressions_norm(spark)
    out = normalize_impressions(df)
    c = out.collect()
    assert str(c[0]["as_of_date"]) == "2025-08-10"
    assert c[0]["ranking_id"] == "r3"
    assert c[0]["impressions"][0]["item_id"] == 7


def test_explode_impressions_schema_and_counts(spark):
    df = _sample_impressions_raw(spark)
    exploded = explode_impressions(df)
    cols = set(exploded.columns)
    assert {
        "as_of_date",
        "ranking_id",
        "customer_id",
        "impression_item_id",
        "is_order",
    } <= cols

    # total rows = total impression items across all records (2 + 1)
    assert exploded.count() == 3


def test_normalize_impressions_raises_when_no_dt_or_as_of_date(spark):
    """
    If an impressions DF has neither 'dt' nor 'as_of_date', normalize_impressions
    must raise a ValueError. Use an explicit schema to avoid NullType inference.
    """
    schema = T.StructType(
        [
            T.StructField("ranking_id", T.StringType(), True),
            T.StructField("customer_id", T.LongType(), True),
            T.StructField(
                "impressions",
                T.ArrayType(
                    T.StructType(
                        [
                            T.StructField("item_id", T.IntegerType(), True),
                            T.StructField("is_order", T.BooleanType(), True),
                        ]
                    )
                ),
                True,
            ),
        ]
    )

    df = spark.createDataFrame([], schema)  # no 'dt' or 'as_of_date' on purpose

    with pytest.raises(ValueError, match="expected either 'dt' or 'as_of_date'"):
        normalize_impressions(df)
