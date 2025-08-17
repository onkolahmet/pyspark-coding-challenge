#!/usr/bin/env python3
"""
Generate mock data for the PySpark coding challenge.

Features:
- Writes JSON arrays for impressions, clicks, atc, orders
- Each dataset is split into multiple part-*.json files
- Configurable customers, items, days, and batch size
- Detailed timing breakdown for generation & writing
"""

import argparse
import json
import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_array_split(path: Path, rows, batch_size: int = 10000):
    """Write rows into multiple JSON array files, each up to batch_size rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        out_path = path.with_name(f"{path.stem}-{i//batch_size:03}.json")
        with open(out_path, "w") as f:
            json.dump(batch, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Generate mock JSON data for impressions/clicks/atc/orders")
    ap.add_argument("--base", default="data", help="output base dir")
    ap.add_argument("--days", type=int, default=30, help="number of days of data")
    ap.add_argument("--customers", type=int, default=50, help="number of customers")
    ap.add_argument("--items", type=int, default=200, help="item catalog size")
    ap.add_argument("--seed", type=int, default=7, help="random seed")
    ap.add_argument("--batch-size", type=int, default=1000, help="rows per JSON file")
    args = ap.parse_args()

    t0 = time.time()
    random.seed(args.seed)
    base = Path(args.base)
    imp_dir, clicks_dir, atc_dir, orders_dir = (
        base / "impressions",
        base / "clicks",
        base / "atc",
        base / "orders",
    )
    for d in (imp_dir, clicks_dir, atc_dir, orders_dir):
        ensure_dir(d)

    today = date.today()
    days = [today - timedelta(days=i) for i in range(args.days)][::-1]
    users = [100 + i for i in range(args.customers)]
    items = args.items

    impr_rows, click_rows, atc_rows, order_rows = [], [], [], []

    # -------- Generate synthetic rows --------
    t_gen0 = time.time()
    for day in days:
        ds = day.strftime("%Y-%m-%d")
        # impressions
        for u in users:
            imps = [
                {
                    "item_id": random.randint(1, items),
                    "is_order": random.random() < 0.15,
                }
                for _ in range(10)
            ]
            impr_rows.append(
                {
                    "dt": ds,
                    "ranking_id": f"r-{ds}-{u}",
                    "customer_id": u,
                    "impressions": imps,
                }
            )

        def past_ts(max_back=3, hour=None):
            d = day - timedelta(days=random.randint(1, max_back))
            h = hour if hour is not None else random.randint(0, 23)
            return datetime(d.year, d.month, d.day, h, 0, 0)

        for u in users:  # clicks
            for _ in range(random.randint(1, 4)):
                t = past_ts()
                click_rows.append(
                    {
                        "dt": f"dt={t.strftime('%Y-%m-%d')}",
                        "customer_id": u,
                        "item_id": random.randint(1, items),
                        "click_time": t.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
        for u in users:  # atc
            if random.random() < 0.6:
                t = past_ts(5, hour=11)
                atc_rows.append(
                    {
                        "dt": f"dt={t.strftime('%Y-%m-%d')}",
                        "customer_id": u,
                        "config_id": random.randint(1, items),
                        "simple_id": None,
                        "occurred_at": t.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
        for u in users:  # orders
            if random.random() < 0.4:
                od = day - timedelta(days=random.randint(2, 7))
                order_rows.append(
                    {
                        "order_date": od.strftime("%Y-%m-%d"),
                        "customer_id": u,
                        "config_id": random.randint(1, items),
                    }
                )
    t_gen1 = time.time()

    # -------- Write to JSON --------
    t_wr0 = time.time()
    write_array_split(imp_dir / "part.json", impr_rows, batch_size=args.batch_size)
    write_array_split(clicks_dir / "part.json", click_rows, batch_size=args.batch_size)
    write_array_split(atc_dir / "part.json", atc_rows, batch_size=args.batch_size)
    write_array_split(orders_dir / "part.json", order_rows, batch_size=args.batch_size)
    t_wr1 = time.time()

    t1 = time.time()
    print(f"\n✅ Mock data written under '{base}/' as JSON arrays (split into {args.batch_size}-row chunks)\n")
    print("⏱️ Data generation time breakdown:")
    print(f"   • Generate rows in memory: {t_gen1 - t_gen0:.2f}s")
    print(f"   • Write JSON files to disk: {t_wr1 - t_wr0:.2f}s")
    print(f"   • TOTAL data generation:   {t1 - t0:.2f}s")


if __name__ == "__main__":
    main()
