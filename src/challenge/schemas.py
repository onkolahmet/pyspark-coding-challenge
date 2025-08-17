# src/challenge/schemas.py
"""
Canonical Spark schemas for the challenge pipeline.

Conventions
-----------
- customer_id : LongType (bigint)
- item_id     : IntegerType
- Dates       : DateType
- Timestamps  : TimestampType
"""

from pyspark.sql import types as T


def arr(elem: T.DataType, contains_null: bool = True) -> T.ArrayType:
    return T.ArrayType(elem, containsNull=contains_null)


# ──────────────────────────────────────────────────────────────────────────────
# Common element structs
# ──────────────────────────────────────────────────────────────────────────────

# Element inside impressions array
impression_struct = T.StructType(
    [
        T.StructField("item_id", T.IntegerType(), nullable=True),
        T.StructField("is_order", T.BooleanType(), nullable=True),
    ]
)

# ──────────────────────────────────────────────────────────────────────────────
# Raw input schemas (used by challenge.io to read JSON from data/)
# ──────────────────────────────────────────────────────────────────────────────

# Mock impressions JSON rows:
# { "dt": "2025-08-15", "ranking_id": "r-...", "customer_id": 123,
#   "impressions": [ {"item_id": 1, "is_order": false}, ... ] }
impressions_schema = T.StructType(
    [
        T.StructField("dt", T.StringType(), nullable=True),
        T.StructField("ranking_id", T.StringType(), nullable=True),
        T.StructField("customer_id", T.LongType(), nullable=True),
        T.StructField("impressions", arr(impression_struct), nullable=True),
    ]
)

# Mock clicks JSON rows:
# { "dt": "2025-08-10", "customer_id": 1, "item_id": 123, "click_time": "2025-08-09 10:00:00" }
clicks_schema = T.StructType(
    [
        T.StructField("dt", T.StringType(), nullable=True),
        T.StructField("customer_id", T.LongType(), nullable=True),
        T.StructField("item_id", T.IntegerType(), nullable=True),
        T.StructField("click_time", T.StringType(), nullable=True),  # parsed to timestamp in transforms
    ]
)

# Mock add-to-cart JSON rows:
# { "dt": "2025-08-10", "customer_id": 1, "config_id": 999, "simple_id": null, "occurred_at": "2025-08-08 09:00:00" }
# Note: simple_id can be absent or non-numeric in some sources; keep it StringType for leniency.
atc_schema = T.StructType(
    [
        T.StructField("dt", T.StringType(), nullable=True),
        T.StructField("customer_id", T.LongType(), nullable=True),
        T.StructField("config_id", T.IntegerType(), nullable=True),
        T.StructField("simple_id", T.StringType(), nullable=True),
        T.StructField("occurred_at", T.StringType(), nullable=True),  # parsed to timestamp in transforms
    ]
)

# Mock orders JSON rows:
# { "order_date": "2025-08-07", "customer_id": 1, "config_id": 321 }
orders_schema = T.StructType(
    [
        T.StructField("order_date", T.StringType(), nullable=True),
        T.StructField("customer_id", T.LongType(), nullable=True),
        T.StructField("config_id", T.IntegerType(), nullable=True),
    ]
)

# ──────────────────────────────────────────────────────────────────────────────
# Canonical/normalized schemas (used within the pipeline)
# ──────────────────────────────────────────────────────────────────────────────

# Final training output schema (what scripts/run.py writes; non-exploded)
training_output_schema = T.StructType(
    [
        T.StructField("ranking_id", T.StringType(), nullable=True),
        T.StructField("customer_id", T.LongType(), nullable=True),  # bigint canonical
        T.StructField("impressions", arr(impression_struct), nullable=True),
        T.StructField("actions", arr(T.IntegerType()), nullable=True),
        T.StructField("action_types", arr(T.IntegerType()), nullable=True),
        T.StructField("as_of_date", T.DateType(), nullable=True),
    ]
)

# Unified actions after normalization (output of actions_union)
# 1=click, 2=atc, 3=order, 0=pad
actions_union_schema = T.StructType(
    [
        T.StructField("customer_id", T.LongType(), nullable=True),  # bigint canonical
        T.StructField("item_id", T.IntegerType(), nullable=True),
        T.StructField("action_type", T.IntegerType(), nullable=True),
        T.StructField("occurred_at", T.TimestampType(), nullable=True),
        T.StructField("action_date", T.DateType(), nullable=True),
        T.StructField("source", T.StringType(), nullable=True),
    ]
)
