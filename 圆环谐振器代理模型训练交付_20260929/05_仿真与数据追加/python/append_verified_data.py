from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = PACKAGE_ROOT / "01_训练数据" / "training_q_user_added.csv"
GEOMETRY_COLUMNS = [
    "a_ring_um", "h_ring_um",
    *[name for k in range(1, 9) for name in (f"C{k}_um", f"S{k}_um")],
]
REQUIRED_LABELS = ["selected_Q", "selected_freq_MHz", "valid_for_Q"]


def append_verified_dataframe(new_data: pd.DataFrame, target: Path = DEFAULT_TARGET) -> tuple[int, int]:
    missing = [c for c in ["a_ring_um", "h_ring_um", *REQUIRED_LABELS] if c not in new_data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = new_data.copy()
    for col in GEOMETRY_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["selected_Q"] = pd.to_numeric(df["selected_Q"], errors="coerce")
    df["selected_freq_MHz"] = pd.to_numeric(df["selected_freq_MHz"], errors="coerce")
    valid_flag = df["valid_for_Q"].astype(str).str.lower().isin(["1", "true"])
    df = df[
        valid_flag
        & np.isfinite(df["selected_Q"])
        & (df["selected_Q"] > 0)
        & np.isfinite(df["selected_freq_MHz"])
    ].copy()
    if df.empty:
        raise ValueError("No valid rows remain after valid_for_Q/Q/frequency checks.")

    before = 0
    if target.exists():
        old = pd.read_csv(target)
        before = len(old)
        combined = pd.concat([old, df], ignore_index=True, sort=False)
    else:
        combined = df

    # Geometry is the design identity. Rounding prevents text-format differences
    # from creating duplicate designs.
    key = combined[GEOMETRY_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).round(10)
    combined = combined.loc[~key.duplicated(keep="last")].reset_index(drop=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target, index=False, encoding="utf-8-sig")
    return len(df), len(combined) - before


def main() -> None:
    parser = argparse.ArgumentParser(description="Append already verified rows to the training data folder.")
    parser.add_argument("--input", required=True, help="CSV containing strict-mode-verified rows.")
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    args = parser.parse_args()

    accepted, net_added = append_verified_dataframe(pd.read_csv(args.input), Path(args.target))
    print(f"Accepted verified rows: {accepted}")
    print(f"Net new geometries after deduplication: {net_added}")
    print(f"Training data file: {Path(args.target).resolve()}")


if __name__ == "__main__":
    main()
