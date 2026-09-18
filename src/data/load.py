"""Read the raw CSVs, join identity to transactions, and write one typed parquet.

Numeric columns are stored as float32 except the amount (float64, so cents are
exact) and the integer keys. This keeps the merged frame near 1 GB in memory.
"""
from __future__ import annotations

import re

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from src.config import paths, pipeline_config

INT_COLS = {"TransactionID": pa.int64(), "TransactionDT": pa.int64(), "isFraud": pa.int8(),
            "card1": pa.int32()}
F64_COLS = {"TransactionAmt": pa.float64()}
STR_PATTERN = re.compile(r"^(ProductCD|card4|card6|P_emaildomain|R_emaildomain|M\d+|DeviceType|DeviceInfo"
                         r"|id_(12|15|16|23|27|28|29|30|31|33|34|35|36|37|38))$")


def _read_csv(path) -> pa.Table:
    with open(path, encoding="utf-8") as fh:
        cols = fh.readline().strip().split(",")
    types = {}
    for c in cols:
        if c in INT_COLS:
            types[c] = INT_COLS[c]
        elif c in F64_COLS:
            types[c] = F64_COLS[c]
        elif STR_PATTERN.match(c):
            types[c] = pa.string()
        else:
            types[c] = pa.float32()
    return pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(column_types=types))


def build_merged(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or pipeline_config()
    p = paths(cfg).ensure()
    out = p.interim / "merged.parquet"
    if out.exists():
        return pd.read_parquet(out)
    tx = _read_csv(p.raw / "train_transaction.csv").to_pandas()
    idn = _read_csv(p.raw / "train_identity.csv").to_pandas()
    idn.columns = [c.replace("-", "_") for c in idn.columns]
    n_before = len(tx)
    df = tx.merge(idn, on="TransactionID", how="left", validate="one_to_one")
    assert len(df) == n_before, "identity join changed the row count"
    del tx, idn
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("string"):
            df[c] = df[c].astype("string")
    df.to_parquet(out, index=False)
    return df


def load_base(cfg: dict | None = None) -> pd.DataFrame:
    """Merged data with the split column attached."""
    p = paths(cfg or pipeline_config())
    return pd.read_parquet(p.interim / "base.parquet")
