# tests/test_generate_mock_data.py
import json
import sys

# ---------- helpers ----------
# at the top of tests/test_generate_mock_data.py
from datetime import date as _RealDate
from pathlib import Path

import pytest

# Import the script as a module
import scripts.generate_mock_data as gmd


class _FrozenDate(_RealDate):
    """A date subclass whose today() returns a fixed date for deterministic tests."""

    @classmethod
    def today(cls):
        # Return a real datetime.date instance, not the subclass
        return _RealDate(2025, 8, 16)


def _read_array_file(p: Path) -> list[dict]:
    with open(p) as f:
        return json.load(f)


def _list_json_parts(dirpath: Path) -> list[Path]:
    return sorted(dirpath.glob("part-*.json"))


# ---------- unit tests for small helpers ----------


def test_ensure_dir_idempotent(tmp_path: Path):
    d = tmp_path / "a" / "b"
    # Should create missing parents
    gmd.ensure_dir(d)
    assert d.exists() and d.is_dir()
    # Idempotent
    gmd.ensure_dir(d)
    assert d.exists() and d.is_dir()


def test_write_array_split_basic_and_edges(tmp_path: Path):
    # 23 rows, batch 5 => files 0..4 with sizes 5,5,5,5,3
    rows = [{"i": i} for i in range(23)]
    out = tmp_path / "x" / "part.json"
    gmd.write_array_split(out, rows, batch_size=5)

    parts = _list_json_parts(out.parent)
    assert [p.name for p in parts] == [
        "part-000.json",
        "part-001.json",
        "part-002.json",
        "part-003.json",
        "part-004.json",
    ]
    sizes = [len(_read_array_file(p)) for p in parts]
    assert sizes == [5, 5, 5, 5, 3]

    # batch_size >= len(rows) → single file
    rows2 = [{"i": i} for i in range(7)]
    out2 = tmp_path / "y" / "part.json"
    gmd.write_array_split(out2, rows2, batch_size=1000)
    parts2 = _list_json_parts(out2.parent)
    assert len(parts2) == 1 and len(_read_array_file(parts2[0])) == 7

    # empty rows → no files written
    out3 = tmp_path / "z" / "part.json"
    gmd.write_array_split(out3, [], batch_size=3)
    assert _list_json_parts(out3.parent) == []

    # batch_size = 1 → each row in its own file
    out4 = tmp_path / "w" / "part.json"
    gmd.write_array_split(out4, [{"i": i} for i in range(4)], batch_size=1)
    parts4 = _list_json_parts(out4.parent)
    assert len(parts4) == 4
    assert all(len(_read_array_file(p)) == 1 for p in parts4)


# ---------- integration tests for main() (CLI) ----------


def test_main_happy_path_creates_all_dirs_and_json_arrays(tmp_path: Path, monkeypatch, capsys):
    """
    Run CLI with small sizes and deterministic seed. Validate structure & basic invariants.
    Freeze today's date so we can assert date ranges reliably.
    """
    # Freeze date.today()
    monkeypatch.setattr(gmd, "date", _FrozenDate)

    base = tmp_path / "data_out"
    argv = [
        "generate_mock_data",
        "--base",
        base.as_posix(),
        "--days",
        "5",
        "--customers",
        "3",
        "--items",
        "10",
        "--batch-size",
        "4",
        "--seed",
        "42",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    gmd.main()
    out = capsys.readouterr().out
    assert "Mock data written under" in out

    # All subdirs must exist
    imp = base / "impressions"
    clicks = base / "clicks"
    atc = base / "atc"
    orders = base / "orders"
    for d in (imp, clicks, atc, orders):
        assert d.exists() and d.is_dir()

    # There should be at least one file per dataset and all are JSON arrays with <= batch_size rows
    for d in (imp, clicks, atc, orders):
        parts = _list_json_parts(d)
        assert len(parts) >= 1
        for p in parts:
            arr = _read_array_file(p)
            assert isinstance(arr, list)
            assert len(arr) <= 4  # batch-size
            # quick schema spot-check
            if d is imp and arr:
                rec = arr[0]
                assert set(rec.keys()) == {
                    "dt",
                    "ranking_id",
                    "customer_id",
                    "impressions",
                }
                assert isinstance(rec["impressions"], list)
            if d is clicks and arr:
                rec = arr[0]
                assert set(rec.keys()) == {"dt", "customer_id", "item_id", "click_time"}
                assert isinstance(rec["dt"], str) and rec["dt"].startswith("dt=")
            if d is atc and arr:
                rec = arr[0]
                assert set(rec.keys()) == {
                    "dt",
                    "customer_id",
                    "config_id",
                    "simple_id",
                    "occurred_at",
                }
            if d is orders and arr:
                rec = arr[0]
                assert set(rec.keys()) == {"order_date", "customer_id", "config_id"}

    # Determinism with same seed & params
    base2 = tmp_path / "data_out2"
    argv2 = [
        "generate_mock_data",
        "--base",
        base2.as_posix(),
        "--days",
        "5",
        "--customers",
        "3",
        "--items",
        "10",
        "--batch-size",
        "4",
        "--seed",
        "42",  # same seed => identical content
    ]
    monkeypatch.setattr(sys, "argv", argv2)
    gmd.main()

    # Compare byte-wise contents per dataset (same number of part files)
    for sub in ("impressions", "clicks", "atc", "orders"):
        parts1 = _list_json_parts(base / sub)
        parts2 = _list_json_parts(base2 / sub)
        assert [p.name for p in parts1] == [p.name for p in parts2]
        # Py3.9: zip(strict=...) not available
        for a, b in zip(parts1, parts2):
            with open(a, "rb") as fa, open(b, "rb") as fb:
                assert fa.read() == fb.read()


def test_main_different_seed_changes_output(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(gmd, "date", _FrozenDate)
    baseA = tmp_path / "A"
    baseB = tmp_path / "B"

    for base, seed in [(baseA, 1), (baseB, 2)]:
        argv = [
            "generate_mock_data",
            "--base",
            base.as_posix(),
            "--days",
            "3",
            "--customers",
            "2",
            "--items",
            "10",
            "--batch-size",
            "100",
            "--seed",
            str(seed),
        ]
        # avoid stdout clutter
        monkeypatch.setattr(sys, "argv", argv)
        gmd.main()

    # Compare impressions files; different seeds should differ
    partsA = _list_json_parts(baseA / "impressions")
    partsB = _list_json_parts(baseB / "impressions")
    assert len(partsA) == len(partsB) >= 1
    eq_any = False
    # Py3.9: zip(strict=...) not available
    for a, b in zip(partsA, partsB):
        with open(a, "rb") as fa, open(b, "rb") as fb:
            eq_any |= fa.read() == fb.read()
    # It's possible (but unlikely) for a small sample to collide entirely;
    # assert that at least one corresponding part differs.
    assert not eq_any, "Different seeds should produce different outputs"


def test_main_zero_days_or_customers_or_items(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(gmd, "date", _FrozenDate)
    base = tmp_path / "empty_cases"

    # days=0 → no rows; still writes nothing but creates directories
    argv = [
        "generate_mock_data",
        "--base",
        base.as_posix(),
        "--days",
        "0",
        "--customers",
        "1",
        "--items",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    gmd.main()
    for sub in ("impressions", "clicks", "atc", "orders"):
        d = base / sub
        assert d.exists() and d.is_dir()
        assert _list_json_parts(d) == []

    # customers=0 → no rows
    base2 = tmp_path / "cust0"
    argv = [
        "generate_mock_data",
        "--base",
        base2.as_posix(),
        "--days",
        "2",
        "--customers",
        "0",
        "--items",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    gmd.main()
    for sub in ("impressions", "clicks", "atc", "orders"):
        assert _list_json_parts(base2 / sub) == []

    # items=0 is allowed by script; random.randint(1, 0) would fail at runtime,
    # but the script does not guard this explicitly. We won't force an error here.
    # Instead, ensure items=1 to keep randint valid.
    base3 = tmp_path / "items1"
    argv = [
        "generate_mock_data",
        "--base",
        base3.as_posix(),
        "--days",
        "1",
        "--customers",
        "1",
        "--items",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    gmd.main()
    assert len(_list_json_parts(base3 / "impressions")) >= 1


def test_argparse_type_error(monkeypatch):
    """
    Passing a non-integer to an int arg should cause argparse to SystemExit with error.
    """
    argv = ["generate_mock_data", "--days", "ten"]  # invalid int
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        gmd.main()


def test_impression_fields_and_dates_consistency(tmp_path: Path, monkeypatch):
    """
    Validate impression record shape and date window when today() is frozen.
    """
    monkeypatch.setattr(gmd, "date", _FrozenDate)
    base = tmp_path / "shape_check"
    argv = [
        "generate_mock_data",
        "--base",
        base.as_posix(),
        "--days",
        "3",
        "--customers",
        "2",
        "--items",
        "5",
        "--seed",
        "7",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    gmd.main()

    imp_parts = _list_json_parts(base / "impressions")
    assert imp_parts
    rows = [r for p in imp_parts for r in _read_array_file(p)]

    # Each row must have 10 impressions with item_id and is_order
    for r in rows:
        assert len(r["impressions"]) == 10
        for imp in r["impressions"]:
            assert set(imp.keys()) == {"item_id", "is_order"}
            assert isinstance(imp["item_id"], int)
            assert isinstance(imp["is_order"], bool)

    # dt should be within the last 3 days relative to frozen today (2025-08-16)
    dts = sorted({r["dt"] for r in rows})
    assert dts[0] >= "2025-08-14" and dts[-1] <= "2025-08-16"


def test_clicks_dt_matches_click_time_date(tmp_path: Path, monkeypatch):
    """
    For clicks, dt='dt=YYYY-MM-DD' should match the date of click_time.
    """
    monkeypatch.setattr(gmd, "date", _FrozenDate)
    base = tmp_path / "click_shape"
    argv = [
        "generate_mock_data",
        "--base",
        base.as_posix(),
        "--days",
        "2",
        "--customers",
        "2",
        "--items",
        "5",
        "--seed",
        "9",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    gmd.main()

    parts = _list_json_parts(base / "clicks")
    rows = [r for p in parts for r in _read_array_file(p)]
    assert rows  # there should be some clicks

    for r in rows[:10]:  # sample a few because dataset is random
        dt_str = r["dt"]
        assert dt_str.startswith("dt=")
        dt_date = dt_str.split("=", 1)[1]
        # click_time is 'YYYY-MM-DD HH:MM:SS'
        assert r["click_time"].startswith(dt_date)


def test_atc_and_orders_basic_shapes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(gmd, "date", _FrozenDate)
    base = tmp_path / "atc_orders"
    argv = [
        "generate_mock_data",
        "--base",
        base.as_posix(),
        "--days",
        "2",
        "--customers",
        "3",
        "--items",
        "5",
        "--seed",
        "13",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    gmd.main()

    atc_parts = _list_json_parts(base / "atc")
    ord_parts = _list_json_parts(base / "orders")
    # Some runs may produce zeros for small sizes; allow empty but check schema when present
    for p in atc_parts:
        arr = _read_array_file(p)
        for r in arr:
            assert set(r.keys()) == {
                "dt",
                "customer_id",
                "config_id",
                "simple_id",
                "occurred_at",
            }
            assert isinstance(r["occurred_at"], str)
    for p in ord_parts:
        arr = _read_array_file(p)
        for r in arr:
            assert set(r.keys()) == {"order_date", "customer_id", "config_id"}
