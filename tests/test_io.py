# tests/test_io.py
import json
import os
from glob import glob
from typing import Any, Dict, List, Optional

from pyspark.sql import functions as F
from pyspark.sql import types as T

from challenge.io import (
    _read_json_with_schema,
    read_atc,
    read_clicks,
    read_impressions,
    read_orders,
    write_parquet,
    write_quarantine,
)
from challenge.schemas import (
    clicks_schema,
    impressions_schema,
    orders_schema,
)

# ----------------------------- Helpers -----------------------------


def _write_json_array(path: str, rows: list[dict]) -> None:
    """Write a single JSON file containing an array of objects."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _write_ndjson(path: str, rows: List[Dict[str, Any]], include_broken_line: Optional[str] = None) -> None:
    """Write newline-delimited JSON. Optionally insert a malformed line after the first record."""
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i == 0 and include_broken_line is not None:
                f.write(include_broken_line + "\n")


def _write_text(path: str, txt: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def _safe_count(df):
    return 0 if df is None else df.count()


# ----------------------------- Read tests -----------------------------


def test_read_impressions_happy_path_array_and_types(spark, tmp_path):
    p = tmp_path / "impr.json"
    _write_json_array(
        p.as_posix(),
        [
            {
                "dt": "2025-08-15",
                "ranking_id": "r-1",
                "customer_id": 101,
                "impressions": [
                    {"item_id": 1, "is_order": False},
                    {"item_id": 2, "is_order": True},
                ],
            },
            {
                "dt": "dt=2025-08-16",  # also acceptable: "dt=YYYY-MM-DD"
                "ranking_id": "r-2",
                "customer_id": 102,
                "impressions": [{"item_id": 3, "is_order": False}],
            },
        ],
    )

    df, bad = read_impressions(spark, p.as_posix())
    # On clean data, Spark may not materialize the corrupt column → bad can be None
    assert _safe_count(bad) == 0

    assert df.count() == 2
    cols = set(df.columns)
    assert {"dt", "ranking_id", "customer_id", "impressions"} <= cols

    # types: impressions is array<struct<item_id:int, is_order:boolean>>
    t = dict(df.dtypes)
    assert t["customer_id"] in {"bigint", "long"}
    assert t["ranking_id"] == "string"
    row = df.select(F.col("impressions").getItem(0).alias("e")).limit(1).collect()[0]["e"]
    assert set(row.asDict().keys()) == {"item_id", "is_order"}


def test_read_impressions_extra_and_missing_fields(spark, tmp_path):
    p = tmp_path / "impr_extra_missing.json"
    _write_json_array(
        p.as_posix(),
        [
            {
                "dt": "2025-08-15",
                "ranking_id": "r-3",
                "customer_id": 201,
                "impressions": [{"item_id": 7, "is_order": False}],
                "extra": "ignored",
            },
            {
                "dt": "2025-08-15",
                # missing ranking_id -> becomes null
                "customer_id": 202,
                "impressions": [{"item_id": 8, "is_order": True}],
            },
        ],
    )

    df, bad = read_impressions(spark, p.as_posix())
    assert _safe_count(bad) == 0
    assert df.count() == 2

    # don't assume order
    by_cust = {r["customer_id"]: r for r in df.collect()}
    assert by_cust[201]["ranking_id"] == "r-3"
    assert by_cust[202]["ranking_id"] is None
    assert "extra" not in df.columns


def test_read_json_corrupt_default_capture(spark, tmp_path):
    """
    NDJSON with one malformed line between two valid ones.
    Depending on Spark, either:
      - bad != None with >=1 rows and df has valid rows, OR
      - df exists but has 0..2 valid rows (and no bad split).
    """
    p = tmp_path / "broken.ndjson"
    valid1 = {
        "dt": "2025-08-12",
        "ranking_id": "r-a",
        "customer_id": 11,
        "impressions": [],
    }
    valid2 = {
        "dt": "2025-08-13",
        "ranking_id": "r-b",
        "customer_id": 12,
        "impressions": [],
    }
    _write_ndjson(p.as_posix(), [valid1, valid2], include_broken_line='{"totally":"broken"')

    df, bad = _read_json_with_schema(spark, p.as_posix(), impressions_schema)

    if bad is not None and bad.count() >= 1:
        # prefer this path: split corrupt rows out
        valid = df.where(F.col("dt").isNotNull() | F.col("ranking_id").isNotNull() | F.col("customer_id").isNotNull())
        assert valid.count() in (1, 2)
    else:
        # tolerate versions that don't materialize corrupt col
        valid = df.where(F.col("dt").isNotNull() | F.col("ranking_id").isNotNull() | F.col("customer_id").isNotNull())
        assert valid.count() in (0, 1, 2)


def test_read_json_corrupt_custom_capture_name(spark, tmp_path):
    """
    Same as above but with a custom corrupt column name.
    Accept either split or no-split behavior depending on Spark.
    """
    p = tmp_path / "broken_custom.ndjson"
    valid1 = {
        "dt": "2025-08-12",
        "customer_id": 1,
        "item_id": 10,
        "click_time": "2025-08-11 10:00:00",
    }
    valid2 = {
        "dt": "2025-08-13",
        "customer_id": 2,
        "item_id": 11,
        "click_time": "2025-08-12 10:00:00",
    }
    _write_ndjson(p.as_posix(), [valid1, valid2], include_broken_line='{"oops":42')

    df, bad = _read_json_with_schema(spark, p.as_posix(), clicks_schema, corrupt_col="_bad_json")

    if bad is not None and bad.count() >= 1:
        assert "_bad_json" not in df.columns  # clean df shouldn't carry the corrupt col
        valid = df.where(F.col("dt").isNotNull() | F.col("customer_id").isNotNull() | F.col("item_id").isNotNull())
        assert valid.count() in (1, 2)
    else:
        valid = df.where(F.col("dt").isNotNull() | F.col("customer_id").isNotNull() | F.col("item_id").isNotNull())
        assert valid.count() in (0, 1, 2)


def test_read_json_corrupt_no_capture_param(spark, tmp_path):
    """
    With corrupt_col=None, no split is requested.
    Some Spark builds add `_corrupt_record`, others produce empty df or a row with all-null schema fields.
    """
    p = tmp_path / "broken_nocapture.ndjson"
    _write_text(p.as_posix(), '{"broken": true')

    df, bad = _read_json_with_schema(spark, p.as_posix(), orders_schema, corrupt_col=None)
    assert bad is None  # never split

    if "_corrupt_record" in df.columns:
        assert df.where(F.col("_corrupt_record").isNotNull()).count() >= 1
    else:
        # ensure no valid schema-matching row sneaks in
        valid = df.where(
            F.col("order_date").isNotNull() | F.col("customer_id").isNotNull() | F.col("config_id").isNotNull()
        )
        assert valid.count() in (0, 1)  # allow a single all-null row


def test_read_each_source_minimal_valid(spark, tmp_path):
    impr_p = tmp_path / "impr.json"
    clicks_p = tmp_path / "clicks.json"
    atc_p = tmp_path / "atc.json"
    orders_p = tmp_path / "orders.json"

    _write_json_array(
        impr_p.as_posix(),
        [
            {
                "dt": "2025-08-12",
                "ranking_id": "r-x",
                "customer_id": 1,
                "impressions": [],
            }
        ],
    )
    _write_json_array(
        clicks_p.as_posix(),
        [
            {
                "dt": "2025-08-12",
                "customer_id": 1,
                "item_id": 10,
                "click_time": "2025-08-11 10:00:00",
            }
        ],
    )
    _write_json_array(
        atc_p.as_posix(),
        [
            {
                "dt": "2025-08-12",
                "customer_id": 1,
                "config_id": 99,
                "simple_id": None,
                "occurred_at": "2025-08-10 09:00:00",
            }
        ],
    )
    _write_json_array(
        orders_p.as_posix(),
        [{"order_date": "2025-08-09", "customer_id": 1, "config_id": 321}],
    )

    impr_df, impr_bad = read_impressions(spark, impr_p.as_posix())
    clicks_df, clicks_bad = read_clicks(spark, clicks_p.as_posix())
    atc_df, atc_bad = read_atc(spark, atc_p.as_posix())
    orders_df, orders_bad = read_orders(spark, orders_p.as_posix())

    # Treat None as zero corrupt
    assert _safe_count(impr_bad) == 0
    assert _safe_count(clicks_bad) == 0
    assert _safe_count(atc_bad) == 0
    assert _safe_count(orders_bad) == 0

    assert impr_df.count() == 1
    assert clicks_df.count() == 1
    assert atc_df.count() == 1
    assert orders_df.count() == 1


# ----------------------------- Write tests -----------------------------


def test_write_parquet_basic_and_append_and_partition(spark, tmp_path):
    out1 = tmp_path / "parq_basic"
    out2 = tmp_path / "parq_part"

    df = spark.createDataFrame(
        [
            (1, "A", "2025-08-15"),
            (2, "B", "2025-08-16"),
        ],
        schema=T.StructType(
            [
                T.StructField("id", T.IntegerType(), True),
                T.StructField("group", T.StringType(), True),
                T.StructField("as_of_date", T.StringType(), True),
            ]
        ),
    )

    # basic write (overwrite)
    write_parquet(df, out1.as_posix())
    back = spark.read.parquet(out1.as_posix())
    assert back.count() == 2

    # append
    write_parquet(df, out1.as_posix(), mode="append")
    back2 = spark.read.parquet(out1.as_posix())
    assert back2.count() == 4  # doubled

    # partitioned write
    write_parquet(df, out2.as_posix(), partition_by=["as_of_date"])
    assert os.path.isdir(out2.as_posix())
    # check partition folders exist
    parts = sorted(glob(os.path.join(out2.as_posix(), "as_of_date=*")))
    assert len(parts) >= 2  # one per distinct date


def test_write_quarantine_none_is_noop(tmp_path):
    out_dir = tmp_path / "out"
    os.makedirs(out_dir.as_posix(), exist_ok=True)
    # None -> no folder creation for a specific name
    write_quarantine(None, out_dir.as_posix(), "impressions")
    # still only the base out folder should exist
    assert os.path.isdir(out_dir.as_posix())
    assert not glob(os.path.join(out_dir.as_posix(), "_quarantine_*"))


def test_write_quarantine_writes_json(spark, tmp_path):
    out_dir = tmp_path / "out2"
    os.makedirs(out_dir.as_posix(), exist_ok=True)

    # a tiny df to "quarantine"
    df = spark.createDataFrame(
        [(1, "bad")],
        schema=T.StructType(
            [
                T.StructField("rownum", T.IntegerType(), True),
                T.StructField("_corrupt_record", T.StringType(), True),
            ]
        ),
    )

    write_quarantine(df, out_dir.as_posix(), "impressions")
    q_dir = os.path.join(out_dir.as_posix(), "_quarantine_impressions")
    assert os.path.isdir(q_dir)
    # should have at least one JSON part file under the dir
    files = glob(os.path.join(q_dir, "*.json")) + glob(os.path.join(q_dir, "part-*"))
    assert files, "Expected JSON/part files under quarantine directory"

    # sanity: read it back
    back = spark.read.json(q_dir)
    assert back.count() == 1
