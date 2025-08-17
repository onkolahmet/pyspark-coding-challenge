import os
import sys

from pyspark.sql import Row
from pyspark.sql import functions as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from challenge.training_examples import (
    build_training_examples,
    join_impressions_with_histories,
)


def _impressions_raw(spark):
    rows = [
        Row(
            dt="2025-08-10",
            ranking_id="r-1",
            customer_id=7,
            impressions=[Row(item_id=1, is_order=False), Row(item_id=2, is_order=True)],
        )
    ]
    return spark.createDataFrame(rows)


def _histories_min(spark):
    rows = [
        Row(
            as_of_date="2025-08-10",
            customer_id=7,
            actions=[5, 0, 0],
            action_types=[1, 0, 0],
        ),
    ]
    return (
        spark.createDataFrame(rows)
        .withColumn("as_of_date", F.to_date("as_of_date"))
        .withColumn("customer_id", F.col("customer_id").cast("bigint"))
    )


def test_join_impressions_with_histories_accepts_raw_and_norm(spark):
    impr = _impressions_raw(spark)
    hist = _histories_min(spark)

    base = join_impressions_with_histories(impr, hist)
    cols = set(base.columns)
    assert {
        "as_of_date",
        "ranking_id",
        "customer_id",
        "impressions",
        "actions",
        "action_types",
    } <= cols
    assert base.count() == 1


def test_build_training_examples_explode_and_nonexplode(spark):
    impr = _impressions_raw(spark)
    hist = _histories_min(spark)

    nonexploded = build_training_examples(impr, hist, explode=False)
    assert nonexploded.count() == 1

    exploded = build_training_examples(impr, hist, explode=True)
    # 2 impression items -> 2 rows
    assert exploded.count() == 2
    ecols = set(exploded.columns)
    assert {"impression_item_id", "is_order"} <= ecols
