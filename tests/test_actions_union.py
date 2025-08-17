import os
import sys

from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql import types as T

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from challenge.actions_union import build_actions_union


def _clicks_df(spark):
    rows = [
        Row(
            dt="2025-08-10",
            customer_id=1,
            item_id=123,
            click_time="2025-08-09 10:00:00",
        ),
        Row(
            dt="2025-08-11",
            customer_id=2,
            item_id=456,
            click_time="2025-08-10 22:00:00",
        ),
    ]
    return spark.createDataFrame(rows)


def _atc_df(spark):
    # IMPORTANT: Give a schema because simple_id is None and Spark can't infer its type.
    schema = T.StructType(
        [
            T.StructField("dt", T.StringType(), True),
            T.StructField("customer_id", T.LongType(), True),
            T.StructField("config_id", T.IntegerType(), True),
            T.StructField("simple_id", T.StringType(), True),  # stays unused, but typed
            T.StructField("occurred_at", T.StringType(), True),
        ]
    )
    rows = [
        ("2025-08-10", 1, 999, None, "2025-08-08 09:00:00"),
    ]
    return spark.createDataFrame(rows, schema=schema)


def _orders_df(spark):
    rows = [
        Row(order_date="2025-08-07", customer_id=1, config_id=321),
    ]
    return spark.createDataFrame(rows)


def test_actions_union_schema_and_action_date(spark):
    clicks = _clicks_df(spark)
    atc = _atc_df(spark)
    orders = _orders_df(spark)

    out = build_actions_union(clicks, atc, orders)
    cols = set(out.columns)
    assert {
        "customer_id",
        "item_id",
        "action_type",
        "occurred_at",
        "action_date",
        "source",
    } <= cols

    # action_date must equal to_date(occurred_at)
    sample = out.select(F.to_date("occurred_at").alias("d"), "action_date").limit(1).collect()[0]
    assert sample["d"] == sample["action_date"]

    # one of each type present (1,2,3)
    types = {r["action_type"] for r in out.select("action_type").distinct().collect()}
    assert {1, 2, 3} <= types
