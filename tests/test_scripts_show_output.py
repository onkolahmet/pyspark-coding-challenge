# tests/test_scripts_show_output.py
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import pytest
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Import the module to test helper functions directly
# (we also run it as a script in subprocess for integration tests)
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
import show_output as S

# ----------------------------- Helpers -----------------------------


def _project_root() -> Path:
    # tests/.. is project root
    return Path(__file__).resolve().parents[1]


def _run_show(args: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run scripts/show_output.py in a subprocess with provided args, capturing stdout/stderr."""
    if cwd is None:
        cwd = _project_root()
    exe = sys.executable
    cmd = [exe, str(_project_root() / "scripts" / "show_output.py"), *args]
    return subprocess.run(cmd, check=False, cwd=cwd, text=True, capture_output=True)


def _make_training_like_df(spark):
    """
    Create a small DataFrame with the same columns and types the show script expects.
    """
    from datetime import date as d

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
            T.StructField("actions", T.ArrayType(T.IntegerType()), True),
            T.StructField("action_types", T.ArrayType(T.IntegerType()), True),
            T.StructField("as_of_date", T.DateType(), True),
        ]
    )

    rows = [
        (
            "r-2025-08-15-101",
            101,
            [{"item_id": 10, "is_order": False}, {"item_id": 11, "is_order": True}],
            [10, 20, 30, 40],
            [1, 2, 3, 0],  # click, atc, order, pad
            d(2025, 8, 15),  # real date object
        ),
        (
            "r-2025-08-16-202",
            202,
            [{"item_id": 99, "is_order": False}],
            [99, 88, 77],
            [3, 2, 1],  # order, atc, click
            d(2025, 8, 16),  # real date object
        ),
    ]

    return spark.createDataFrame(rows, schema)


def _write_run_under_base(spark, base_dir: Path, bucket: str, run_name: str, empty: bool = False) -> tuple[str, Path]:
    """
    Write training-like parquet into base_dir/bucket/run_name and return:
      (relative_subdir, absolute_path)
    """
    out = base_dir / bucket / run_name
    out.mkdir(parents=True, exist_ok=True)

    df = _make_training_like_df(spark)
    if empty:
        df = df.limit(0)
    df.write.mode("overwrite").parquet(out.as_posix())

    # touch mtime so tests can control newest selection
    now = time.time()
    os.utime(out, (now, now))
    rel = f"{bucket}/{run_name}"
    return rel, out


def _stdout_has(stdout: str, needle: str) -> bool:
    return needle in stdout


# ----------------------------- Unit tests (helpers) -----------------------------


def test_action_type_name_function(spark):
    """
    Ensure action_type_name maps 1→click, 2→atc, 3→order, 0→pad, else→unk.
    """
    src = spark.createDataFrame([(i,) for i in [0, 1, 2, 3, 9]], "t int")
    out = src.select(S.action_type_name(F.col("t")).alias("name")).collect()
    got = [r["name"] for r in out]
    assert got == ["pad", "click", "atc", "order", "unk"]


def test_newest_leaf_dir_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.newest_leaf_dir(tmp_path.as_posix())


def test_newest_leaf_dir_picks_latest(tmp_path, spark):
    base = tmp_path / "out"
    # two buckets, two runs; make the second one newer
    _, p1 = _write_run_under_base(spark, base, "bucketA", "run1")
    time.sleep(0.05)
    _, p2 = _write_run_under_base(spark, base, "bucketB", "run2")
    assert p2.as_posix() == S.newest_leaf_dir(base.as_posix())


# ----------------------------- Integration tests (script) -----------------------------


def test_show_output_main_with_subdir_nonempty(tmp_path, spark):
    base = tmp_path / "out"
    subdir, _ = _write_run_under_base(spark, base, "default_training_outputs", "ts1")

    cp = _run_show(
        [
            "--base",
            base.as_posix(),
            "--subdir",
            subdir,
            "--limit",
            "3",
            "--actions-head",
            "4",
        ]
    )
    assert cp.returncode == 0, f"stderr:\n{cp.stderr}"
    # Sections present
    assert _stdout_has(cp.stdout, "=== SCHEMA ===")
    assert _stdout_has(cp.stdout, "=== AGGREGATES ===")
    assert _stdout_has(cp.stdout, "=== PER-CUSTOMER SNAPSHOT")
    assert _stdout_has(cp.stdout, "=== SAMPLE ACTIONS (humanized, top 4)")
    assert _stdout_has(cp.stdout, "=== SAMPLE IMPRESSIONS (exploded, diversified)")
    # Humanized tokens appear
    # (We wrote action_types with 1,2,3,0 → click/atc/order/pad)
    assert "click" in cp.stdout
    assert "atc" in cp.stdout
    assert "order" in cp.stdout
    assert "pad" in cp.stdout
    # Some columns from sizes preview
    assert "n_actions" in cp.stdout
    assert "n_impr" in cp.stdout


def test_show_output_main_without_subdir_picks_newest(tmp_path, spark):
    base = tmp_path / "out"
    # Write an older run then a newer run
    _write_run_under_base(spark, base, "bucketX", "old")
    time.sleep(0.05)
    newest_rel, newest_abs = _write_run_under_base(spark, base, "bucketY", "new")

    cp = _run_show(["--base", base.as_posix(), "--limit", "1"])
    assert cp.returncode == 0, f"stderr:\n{cp.stderr}"

    # Since we don't print the path in the script, just sanity-check
    # that the output has all the sections (meaning it read something)
    assert _stdout_has(cp.stdout, "=== SCHEMA ===")
    assert _stdout_has(cp.stdout, "=== AGGREGATES ===")
    # Double-check newest_leaf_dir agrees with our expectation
    assert newest_abs.as_posix() == S.newest_leaf_dir(base.as_posix())
    # And that the rel path we created is of the expected two-level form base/*/*
    assert newest_rel.count("/") == 1


def test_show_output_main_handles_empty_dataset(tmp_path, spark):
    base = tmp_path / "out"
    subdir, _ = _write_run_under_base(spark, base, "default_training_outputs", "empty_ts", empty=True)
    cp = _run_show(
        [
            "--base",
            base.as_posix(),
            "--subdir",
            subdir,
            "--limit",
            "5",
            "--actions-head",
            "5",
        ]
    )
    # Should still succeed and print sections (aggregates will be nulls)
    assert cp.returncode == 0, f"stderr:\n{cp.stderr}"
    assert _stdout_has(cp.stdout, "=== SCHEMA ===")
    assert _stdout_has(cp.stdout, "=== AGGREGATES ===")
    assert _stdout_has(cp.stdout, "=== PER-CUSTOMER SNAPSHOT")
    assert _stdout_has(cp.stdout, "=== SAMPLE ACTIONS")
    assert _stdout_has(cp.stdout, "=== SAMPLE IMPRESSIONS")


def test_show_output_limits_and_actions_head(tmp_path, spark):
    base = tmp_path / "out"
    subdir, _ = _write_run_under_base(spark, base, "custom_outputs", "tsA")

    # Small limit and action head of 2; header should reflect it
    cp = _run_show(
        [
            "--base",
            base.as_posix(),
            "--subdir",
            subdir,
            "--limit",
            "1",
            "--actions-head",
            "2",
        ]
    )
    assert cp.returncode == 0, f"stderr:\n{cp.stderr}"
    assert _stdout_has(cp.stdout, "=== SAMPLE ACTIONS (humanized, top 2)")

    # We should still see humanized tokens in the output somewhere
    assert "click" in cp.stdout or "atc" in cp.stdout or "order" in cp.stdout or "pad" in cp.stdout


def test_can_import_show_output():
    # Keep the original smoke test: module import shouldn't crash
    import importlib

    m = importlib.import_module("show_output")
    assert hasattr(m, "main")
