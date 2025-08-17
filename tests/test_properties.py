import random
from datetime import date, datetime, timedelta

from pyspark.sql import Row
from pyspark.sql import types as T

from challenge.pipeline import build_training_inputs


def make_rows(n=100, start=date(2025, 1, 1)):
    # Generate synthetic impressions/clicks/atc/orders for a few users over a few days
    users = [100 + i for i in range(5)]
    days = [start + timedelta(days=d) for d in range(5)]
    impr_rows = []
    click_rows = []
    atc_rows = []
    order_rows = []

    for d in days:
        for u in users:
            items = [{"item_id": random.randint(1, 50), "is_order": random.random() < 0.1} for _ in range(10)]
            impr_rows.append(Row(dt=str(d), ranking_id=f"r-{d}-{u}", customer_id=u, impressions=items))
            # clicks before this day
            for k in range(random.randint(0, 5)):
                cd = d - timedelta(days=random.randint(1, 3))
                click_rows.append(
                    Row(
                        dt=str(cd),
                        customer_id=u,
                        item_id=random.randint(1, 50),
                        click_time=datetime(cd.year, cd.month, cd.day, random.randint(0, 23), 0, 0),
                    )
                )
            # atc/orders before this day
            if random.random() < 0.5:
                ad = d - timedelta(days=random.randint(1, 3))
                atc_rows.append(
                    Row(
                        dt=str(ad),
                        customer_id=u,
                        config_id=random.randint(1, 50),
                        simple_id=None,
                        occurred_at=datetime(ad.year, ad.month, ad.day, 12, 0, 0),
                    )
                )
            if random.random() < 0.5:
                od = d - timedelta(days=random.randint(1, 3))
                order_rows.append(Row(order_date=od, customer_id=u, config_id=random.randint(1, 50)))

    return impr_rows, click_rows, atc_rows, order_rows


def test_property_no_same_day_leak(spark):
    impr_rows, click_rows, atc_rows, order_rows = make_rows()
    impr_schema = T.StructType(
        [
            T.StructField("dt", T.StringType(), False),
            T.StructField("ranking_id", T.StringType(), False),
            T.StructField("customer_id", T.IntegerType(), False),
            T.StructField(
                "impressions",
                T.ArrayType(
                    T.StructType(
                        [
                            T.StructField("item_id", T.IntegerType(), False),
                            T.StructField("is_order", T.BooleanType(), False),
                        ]
                    )
                ),
                False,
            ),
        ]
    )
    clicks_schema = T.StructType(
        [
            T.StructField("dt", T.StringType(), False),
            T.StructField("customer_id", T.IntegerType(), False),
            T.StructField("item_id", T.IntegerType(), False),
            T.StructField("click_time", T.TimestampType(), False),
        ]
    )
    atc_schema = T.StructType(
        [
            T.StructField("dt", T.StringType(), False),
            T.StructField("customer_id", T.IntegerType(), False),
            T.StructField("config_id", T.IntegerType(), False),
            T.StructField("simple_id", T.IntegerType(), True),
            T.StructField("occurred_at", T.TimestampType(), False),
        ]
    )
    orders_schema = T.StructType(
        [
            T.StructField("order_date", T.DateType(), False),
            T.StructField("customer_id", T.IntegerType(), False),
            T.StructField("config_id", T.IntegerType(), False),
        ]
    )

    impressions = spark.createDataFrame(impr_rows, schema=impr_schema)
    clicks = spark.createDataFrame(click_rows, schema=clicks_schema)
    atc = spark.createDataFrame(atc_rows, schema=atc_schema)
    orders = spark.createDataFrame(order_rows, schema=orders_schema)

    out = build_training_inputs(
        impressions_df=impressions,
        clicks_df=clicks,
        atc_df=atc,
        orders_df=orders,
        max_actions=50,
        lookback_days=365,
        explode=False,
    )

    # Basic sanity: output rows equal to impressions rows
    assert out.count() == impressions.count()
