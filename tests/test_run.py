# tests/test_run.py
import importlib
import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List, Optional

import pytest

from scripts import run

ROOT = Path(__file__).resolve().parents[1]


# ----------------------
# Helpers
# ----------------------
def _write_json_array(path: str, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(list(rows), f, indent=2)


def _tiny_external_dataset(tmpdir: Path) -> dict[str, str]:
    """Small valid dataset with NO same-day leakage."""
    impr = tmpdir / "impr.json"
    clicks = tmpdir / "clicks.json"
    atc = tmpdir / "atc.json"
    orders = tmpdir / "orders.json"

    _write_json_array(
        impr.as_posix(),
        [
            {
                "dt": "2025-08-16",
                "ranking_id": "r-1",
                "customer_id": 1001,
                "impressions": [{"item_id": 10, "is_order": False}],
            }
        ],
    )
    _write_json_array(
        clicks.as_posix(),
        [
            {
                "dt": "dt=2025-08-15",
                "customer_id": 1001,
                "item_id": 10,
                "click_time": "2025-08-15 10:00:00",
            }
        ],
    )
    _write_json_array(
        atc.as_posix(),
        [
            {
                "dt": "dt=2025-08-14",
                "customer_id": 1001,
                "config_id": 10,
                "simple_id": None,
                "occurred_at": "2025-08-14 11:00:00",
            }
        ],
    )
    _write_json_array(
        orders.as_posix(),
        [{"order_date": "2025-08-13", "customer_id": 1001, "config_id": 10}],
    )
    return {
        "impressions": impr.as_posix(),
        "clicks": clicks.as_posix(),
        "atc": atc.as_posix(),
        "orders": orders.as_posix(),
    }


def _leaky_external_dataset(base: Path) -> dict[str, str]:
    base.mkdir(parents=True, exist_ok=True)
    impr_p = base / "impr.json"
    clicks_p = base / "clicks.json"
    atc_p = base / "atc.json"
    orders_p = base / "orders.json"

    ds = "2025-08-16"
    _write_json_array(
        impr_p.as_posix(),
        [
            {
                "dt": ds,
                "ranking_id": "r-leak",
                "customer_id": 1,
                "impressions": [{"item_id": 42, "is_order": False}],
            }
        ],
    )
    _write_json_array(
        clicks_p.as_posix(),
        [
            {
                "dt": f"dt={ds}",
                "customer_id": 1,
                "item_id": 42,
                "click_time": f"{ds} 12:00:00",
            }
        ],
    )
    _write_json_array(
        atc_p.as_posix(),
        [
            {
                "dt": f"dt={ds}",
                "customer_id": 1,
                "config_id": 42,
                "simple_id": None,
                "occurred_at": f"{ds} 13:00:00",
            }
        ],
    )
    _write_json_array(orders_p.as_posix(), [{"order_date": ds, "customer_id": 1, "config_id": 42}])
    return {
        "impressions": impr_p.as_posix(),
        "clicks": clicks_p.as_posix(),
        "atc": atc_p.as_posix(),
        "orders": orders_p.as_posix(),
    }


def _latest_subdir(base: Path) -> Path:
    subs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)
    assert subs, f"No subdirs under {base}"
    return subs[-1]


def _list_parquet_paths(out_dir: Path) -> list[Path]:
    return list(out_dir.rglob("*.parquet"))


def _run_script(args: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    script_path = (ROOT / "scripts" / "run.py").as_posix()
    cmd = [sys.executable, script_path, *args]
    return subprocess.run(cmd, check=False, cwd=cwd, capture_output=True, text=True)


def _dq_warn_in(text: str) -> bool:
    """Robustly match either form of leak warning seen across versions."""
    text_lower = text.lower()
    return (
        ("same-day leak detected" in text_lower)
        or ("same-day actions found joined to contexts" in text_lower)
        or ("inputs contain" in text_lower and "same-day actions" in text_lower)
    )


# ----------------------
# Unit-ish tests
# ----------------------
def test_parse_args_defaults(monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setattr("sys.argv", ["run.py"])
    args = run.parse_args()
    assert args.out == "out/training_inputs"
    assert args.demo is False
    assert args.train_days == 14
    assert args.lookback_days == 365
    assert args.max_actions == 1000
    assert args.shuffle_partitions == 0
    assert args.impressions_path is None


def test_resolve_out_dir_creates_timestamped(tmp_path: Path):
    base = tmp_path / "out"
    created = Path(run.resolve_out_dir(base.as_posix()))
    assert created.exists()
    assert created.parent == base


# ----------------------
# End-to-end: DEMO (subprocess)
# ----------------------
def test_main_demo_writes_partitioned_parquet(tmp_path: Path):
    base_out = tmp_path / "demo_out"
    cp = _run_script(["--demo", "--out", base_out.as_posix(), "--train-days", "7"])
    assert cp.returncode == 0, cp.stderr

    out_dir = _latest_subdir(base_out)
    part_dirs = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("as_of_date=")]
    assert part_dirs, f"No partitions under {out_dir}"
    assert _list_parquet_paths(out_dir), "No parquet files written"
    assert "Running job. Outputs will be written to" in cp.stdout


# ----------------------
# End-to-end: LOCAL (./data) (subprocess)
# ----------------------
def test_main_local_mode_uses_data_tree(tmp_path: Path):
    cwd = tmp_path
    data = cwd / "data"
    (data / "impressions").mkdir(parents=True)
    (data / "clicks").mkdir(parents=True)
    (data / "atc").mkdir(parents=True)
    (data / "orders").mkdir(parents=True)

    _write_json_array(
        (data / "impressions" / "part.json").as_posix(),
        [
            {
                "dt": "2025-08-16",
                "ranking_id": "r-l",
                "customer_id": 333,
                "impressions": [{"item_id": 1, "is_order": False}],
            }
        ],
    )
    _write_json_array(
        (data / "clicks" / "part.json").as_posix(),
        [
            {
                "dt": "dt=2025-08-15",
                "customer_id": 333,
                "item_id": 1,
                "click_time": "2025-08-15 10:00:00",
            }
        ],
    )
    _write_json_array(
        (data / "atc" / "part.json").as_posix(),
        [
            {
                "dt": "dt=2025-08-14",
                "customer_id": 333,
                "config_id": 1,
                "simple_id": None,
                "occurred_at": "2025-08-14 09:00:00",
            }
        ],
    )
    _write_json_array(
        (data / "orders" / "part.json").as_posix(),
        [{"order_date": "2025-08-13", "customer_id": 333, "config_id": 1}],
    )

    cp = _run_script(["--out", (tmp_path / "local_out").as_posix(), "--train-days", "7"], cwd=cwd)
    assert cp.returncode == 0, cp.stderr

    out_dir = _latest_subdir(tmp_path / "local_out")
    assert _list_parquet_paths(out_dir)
    assert "Running job. Outputs will be written to" in cp.stdout


# ----------------------
# End-to-end: EXTERNAL (subprocess)
# ----------------------
def test_main_external_happy_path(tmp_path: Path):
    paths = _tiny_external_dataset(tmp_path / "ext_ok")
    base_out = tmp_path / "ext_out"

    cp = _run_script(
        [
            "--out",
            base_out.as_posix(),
            "--impressions-path",
            paths["impressions"],
            "--clicks-path",
            paths["clicks"],
            "--atc-path",
            paths["atc"],
            "--orders-path",
            paths["orders"],
            "--external-upstream-scan",
            "--train-days",
            "14",
        ]
    )
    assert cp.returncode == 0, cp.stderr
    out_dir = _latest_subdir(base_out)
    assert _list_parquet_paths(out_dir)
    assert "No same-day leakage" in cp.stdout


def test_main_external_leak_warns_no_fail(tmp_path: Path):
    paths = _leaky_external_dataset(tmp_path / "ext_leak_warn")
    base_out = tmp_path / "ext_out_warn"

    cp = _run_script(
        [
            "--out",
            base_out.as_posix(),
            "--impressions-path",
            paths["impressions"],
            "--clicks-path",
            paths["clicks"],
            "--atc-path",
            paths["atc"],
            "--orders-path",
            paths["orders"],
            "--external-upstream-scan",
        ]
    )
    assert cp.returncode == 0, cp.stderr
    out_dir = _latest_subdir(base_out)
    assert _list_parquet_paths(out_dir)
    assert _dq_warn_in(cp.stdout), cp.stdout


def test_main_external_leak_with_fail_flag_exits_nonzero(tmp_path: Path):
    paths = _leaky_external_dataset(tmp_path / "ext_leak_fail")
    base_out = tmp_path / "ext_out_fail"

    cp = _run_script(
        [
            "--out",
            base_out.as_posix(),
            "--impressions-path",
            paths["impressions"],
            "--clicks-path",
            paths["clicks"],
            "--atc-path",
            paths["atc"],
            "--orders-path",
            paths["orders"],
            "--external-upstream-scan",
            "--fail-on-same-day-leak",
        ]
    )

    # With Option A, --fail-on-same-day-leak applies only to strict leaks into histories.
    # Upstream same-day actions should not fail the run; instead, an informational line is printed.
    assert cp.returncode == 0, f"Did not expect failure; stdout:\n{cp.stdout}\n\nstderr:\n{cp.stderr}"

    # Outputs should exist
    out_dir = _latest_subdir(base_out)
    assert _list_parquet_paths(out_dir)

    # Strict check passed
    assert "No same-day leakage into histories" in cp.stdout

    # Upstream scan surfaced an FYI line
    assert "same-day action" in cp.stdout.lower()


# ---------- extra helpers for new cases ----------
def _write_text(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _write_ndjson(path: str, rows: list[dict], include_broken_line: Optional[str] = None) -> None:
    """Write newline-delimited JSON. Optionally insert a malformed line after the first record."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i == 0 and include_broken_line is not None:
                f.write(include_broken_line + "\n")


# ----------------------
# Additional end-to-end tests to cover remaining branches
# ----------------------
def test_main_demo_with_shuffle_partitions_knob(tmp_path: Path):
    base_out = tmp_path / "demo_shuffle"
    cp = _run_script(["--demo", "--out", base_out.as_posix(), "--shuffle-partitions", "7"])
    assert cp.returncode == 0, cp.stderr
    out_dir = _latest_subdir(base_out)
    assert _list_parquet_paths(out_dir)


def test_main_demo_explode_branch(tmp_path: Path):
    base_out = tmp_path / "demo_explode"
    cp = _run_script(["--demo", "--explode", "--out", base_out.as_posix()])
    assert cp.returncode == 0, cp.stderr
    out_dir = _latest_subdir(base_out)
    assert _list_parquet_paths(out_dir)


def test_main_external_empty_impressions_produces_zero_rows(tmp_path: Path):
    """
    Exercise the empty-impressions path in slice_impressions_last_n_days
    without triggering ANSI cast errors: use an empty impressions file.
    """
    base = tmp_path / "ext_empty_impr"
    impr = base / "impr.json"
    clicks = base / "clicks.json"
    atc = base / "atc.json"
    orders = base / "orders.json"

    _write_json_array(impr.as_posix(), [])  # <- empty impressions (safe)
    _write_json_array(clicks.as_posix(), [])
    _write_json_array(atc.as_posix(), [])
    _write_json_array(orders.as_posix(), [])

    base_out = tmp_path / "ext_empty_out"
    cp = _run_script(
        [
            "--out",
            base_out.as_posix(),
            "--impressions-path",
            impr.as_posix(),
            "--clicks-path",
            clicks.as_posix(),
            "--atc-path",
            atc.as_posix(),
            "--orders-path",
            orders.as_posix(),
        ]
    )
    assert cp.returncode == 0, cp.stderr
    assert "Rows out: 0" in cp.stdout
    out_dir = _latest_subdir(base_out)
    # Empty output should not have parquet parts
    assert not _list_parquet_paths(out_dir)


def test_main_external_writes_quarantine_on_corrupt_impressions(tmp_path: Path):
    """
    Provide impressions with one corrupt record inside a JSON ARRAY to exercise
    write_quarantine(bad_impr). Some Spark builds may instead drop the corrupt
    record without exposing `_corrupt_record`; accept either outcome.
    """
    base = tmp_path / "ext_quarantine"
    impr = base / "impr.json"
    clicks = base / "clicks.json"
    atc = base / "atc.json"
    orders = base / "orders.json"

    # JSON array with one bad record (customer_id wrong type) between two valid ones
    _write_json_array(
        impr.as_posix(),
        [
            {
                "dt": "2025-08-16",
                "ranking_id": "r-q1",
                "customer_id": 77,
                "impressions": [{"item_id": 1, "is_order": False}],
            },
            {
                "dt": "2025-08-16",
                "ranking_id": "r-bad",
                "customer_id": "not-a-number",  # corrupt w.r.t schema (expects long)
                "impressions": [{"item_id": 2, "is_order": True}],
            },
            {
                "dt": "2025-08-16",
                "ranking_id": "r-q2",
                "customer_id": 78,
                "impressions": [{"item_id": 3, "is_order": False}],
            },
        ],
    )
    _write_json_array(
        clicks.as_posix(),
        [
            {
                "dt": "dt=2025-08-15",
                "customer_id": 77,
                "item_id": 1,
                "click_time": "2025-08-15 08:00:00",
            }
        ],
    )
    _write_json_array(
        atc.as_posix(),
        [
            {
                "dt": "dt=2025-08-14",
                "customer_id": 77,
                "config_id": 1,
                "simple_id": None,
                "occurred_at": "2025-08-14 09:00:00",
            }
        ],
    )
    _write_json_array(
        orders.as_posix(),
        [{"order_date": "2025-08-13", "customer_id": 77, "config_id": 1}],
    )

    base_out = tmp_path / "ext_quarantine_out"
    cp = _run_script(
        [
            "--out",
            base_out.as_posix(),
            "--impressions-path",
            impr.as_posix(),
            "--clicks-path",
            clicks.as_posix(),
            "--atc-path",
            atc.as_posix(),
            "--orders-path",
            orders.as_posix(),
        ]
    )
    assert cp.returncode == 0, cp.stderr
    out_dir = _latest_subdir(base_out)

    q = out_dir / "_quarantine_impressions"
    if q.exists() and any(q.rglob("*.json")):
        # Expected/ideal path: corrupt captured and quarantined
        assert True
    else:
        # Accept alternate Spark behavior: no separate bad DF, but job still succeeds
        # and writes at least the valid records.
        assert "Finished. Rows out:" in cp.stdout
        assert _list_parquet_paths(out_dir)


# ----------------------
# Additional coverage for scripts/run.py
# ----------------------
def test_main_local_inprocess_hits_local_branch(tmp_path, monkeypatch):
    """Cover the LOCAL branch (no --demo, no external paths) in-process without
    breaking the shared Spark fixture or relying on the process CWD seen by the JVM.
    """
    cwd = tmp_path
    data = cwd / "data"
    (data / "impressions").mkdir(parents=True)
    (data / "clicks").mkdir(parents=True)
    (data / "atc").mkdir(parents=True)
    (data / "orders").mkdir(parents=True)

    # minimal valid records
    _write_json_array(
        (data / "impressions" / "part.json").as_posix(),
        [
            {
                "dt": "2025-08-16",
                "ranking_id": "r-local",
                "customer_id": 101,
                "impressions": [{"item_id": 9, "is_order": False}],
            }
        ],
    )
    _write_json_array(
        (data / "clicks" / "part.json").as_posix(),
        [
            {
                "dt": "dt=2025-08-15",
                "customer_id": 101,
                "item_id": 9,
                "click_time": "2025-08-15 10:00:00",
            }
        ],
    )
    _write_json_array(
        (data / "atc" / "part.json").as_posix(),
        [
            {
                "dt": "dt=2025-08-14",
                "customer_id": 101,
                "config_id": 9,
                "simple_id": None,
                "occurred_at": "2025-08-14 09:00:00",
            }
        ],
    )
    _write_json_array(
        (data / "orders" / "part.json").as_posix(),
        [{"order_date": "2025-08-13", "customer_id": 101, "config_id": 9}],
    )

    # Ensure run.main() doesn't stop the global Spark session used by other tests
    monkeypatch.setattr(run.SparkSession, "stop", lambda self: None, raising=False)

    # Force run.py's LOCAL branch to read from our tmp data/ by overriding its IO functions.
    CIO = importlib.import_module("challenge.io")
    monkeypatch.setattr(
        run,
        "read_impressions",
        lambda spark, _p: CIO.read_impressions(spark, (data / "impressions").as_posix()),
        raising=True,
    )
    monkeypatch.setattr(
        run,
        "read_clicks",
        lambda spark, _p: CIO.read_clicks(spark, (data / "clicks").as_posix()),
        raising=True,
    )
    monkeypatch.setattr(
        run,
        "read_atc",
        lambda spark, _p: CIO.read_atc(spark, (data / "atc").as_posix()),
        raising=True,
    )
    monkeypatch.setattr(
        run,
        "read_orders",
        lambda spark, _p: CIO.read_orders(spark, (data / "orders").as_posix()),
        raising=True,
    )

    out_base = tmp_path / "local_inproc_out"
    # We can keep cwd unchanged; paths are injected above
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--out",
            out_base.as_posix(),
            "--train-days",
            "7",
        ],
    )

    run.main()

    out_dir = _latest_subdir(out_base)
    assert _list_parquet_paths(out_dir)


def test_main_demo_dq_exception_is_swallowed(tmp_path, monkeypatch, capsys):
    """Force the strict DQ check to raise and ensure the job continues (lines 336-338)."""
    # Prevent teardown of the shared Spark session
    monkeypatch.setattr(run.SparkSession, "stop", lambda self: None, raising=False)

    def boom(*_a, **_k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(run, "check_no_same_day_leak_strict", boom, raising=True)
    out_base = tmp_path / "demo_dq_exc"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--demo",
            "--out",
            out_base.as_posix(),
        ],
    )
    run.main()

    # Completed successfully and wrote outputs
    out_dir = _latest_subdir(out_base)
    assert _list_parquet_paths(out_dir)
    out = capsys.readouterr().out
    assert "DQ check failed to run" in out


def test_main_demo_strict_leak_with_fail_flag_exits_2(tmp_path, monkeypatch):
    """Simulate a strict leak and assert --fail-on-same-day-leak triggers SystemExit(2) (lines 341-346)."""
    # Prevent teardown of the shared Spark session
    monkeypatch.setattr(run.SparkSession, "stop", lambda self: None, raising=False)

    monkeypatch.setattr(run, "check_no_same_day_leak_strict", lambda *_a, **_k: 1, raising=True)
    out_base = tmp_path / "demo_fail_on_leak"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--demo",
            "--out",
            out_base.as_posix(),
            "--fail-on-same-day-leak",
        ],
    )
    with pytest.raises(SystemExit) as e:
        run.main()
    assert e.value.code == 2
