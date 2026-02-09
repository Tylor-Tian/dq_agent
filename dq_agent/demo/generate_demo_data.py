from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def generate_demo_data(output_dir: Path, seed: Optional[int] = 42) -> Path:
    rng = np.random.default_rng(seed)
    rows = 500
    order_ids = np.arange(1, rows + 1)
    user_ids = rng.integers(1000, 1100, size=rows).astype(str)
    amounts = rng.normal(loc=120.0, scale=30.0, size=rows).round(2)
    statuses = rng.choice(["PAID", "REFUND", "CANCEL", "PENDING"], size=rows)
    created_at = pd.date_range("2024-01-01", periods=rows, freq="h")

    df = pd.DataFrame(
        {
            "order_id": order_ids.astype(str),
            "user_id": user_ids,
            "amount": amounts,
            "status": statuses,
            "created_at": created_at,
        }
    )

    df.loc[0, "amount"] = -100.0
    df.loc[1, "amount"] = 999999.0
    df.loc[2:5, "user_id"] = None
    df.loc[6, "order_id"] = df.loc[7, "order_id"]
    df.loc[8, "status"] = "UNKNOWN"

    # Prefer Parquet (faster) but gracefully fall back to CSV if parquet
    # engines (pyarrow/fastparquet) aren't installed.
    parquet_path = output_dir / "orders.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = output_dir / "orders.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
