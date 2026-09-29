from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "01_训练数据"
DEFAULT_DATA = sorted(DATA_DIR.glob("training_q_*.csv"))

RAW_FEATURES = [
    "a_ring_um", "h_ring_um",
    *[name for k in range(1, 9) for name in (f"C{k}_um", f"S{k}_um")],
]
WIDTH_MIN_LIMIT = 6.5
WIDTH_MAX_LIMIT = 13.5


def finite_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def load_data(paths: list[Path] | None = None) -> pd.DataFrame:
    paths = paths or DEFAULT_DATA
    frames = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["dataset_file"] = path.name
        frames.append(frame)
    if not frames:
        raise RuntimeError("未找到训练 CSV。")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["valid_for_Q"].astype(str).str.lower().isin(["1", "true"])].copy()
    df["selected_Q"] = finite_float(df["selected_Q"])
    df["selected_freq_MHz"] = finite_float(df["selected_freq_MHz"])
    df = df[
        np.isfinite(df["selected_Q"])
        & (df["selected_Q"] > 0)
        & np.isfinite(df["selected_freq_MHz"])
    ].copy()

    for col in RAW_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = finite_float(df[col]).fillna(0.0)
    df["log10_Q"] = np.log10(df["selected_Q"])
    return df.reset_index(drop=True)


def add_geometry_features(df: pd.DataFrame, n_theta: int = 360) -> pd.DataFrame:
    df = df.copy()
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    widths = np.tile(df["h_ring_um"].to_numpy(float)[:, None], (1, n_theta))
    d1 = np.zeros_like(widths)
    d2 = np.zeros_like(widths)
    amp_cols = []

    for k in range(1, 9):
        c = df[f"C{k}_um"].to_numpy(float)[:, None]
        s = df[f"S{k}_um"].to_numpy(float)[:, None]
        cos = np.cos(k * theta)[None, :]
        sin = np.sin(k * theta)[None, :]
        widths += c * cos + s * sin
        d1 += -k * c * sin + k * s * cos
        d2 += -(k**2) * c * cos - (k**2) * s * sin
        col = f"amp{k}_um"
        df[col] = np.sqrt(df[f"C{k}_um"] ** 2 + df[f"S{k}_um"] ** 2)
        amp_cols.append(col)

    df["width_mean_calc_um"] = widths.mean(axis=1)
    df["width_min_calc_um"] = widths.min(axis=1)
    df["width_max_calc_um"] = widths.max(axis=1)
    if "width_min_um" not in df.columns:
        df["width_min_um"] = df["width_min_calc_um"]
    if "width_max_um" not in df.columns:
        df["width_max_um"] = df["width_max_calc_um"]
    df["width_p05_um"] = np.percentile(widths, 5, axis=1)
    df["width_p95_um"] = np.percentile(widths, 95, axis=1)
    df["width_range_um"] = df["width_max_calc_um"] - df["width_min_calc_um"]
    df["width_std_um"] = widths.std(axis=1)
    df["width_cv"] = df["width_std_um"] / np.maximum(np.abs(df["width_mean_calc_um"]), 1e-9)
    df["min_width_margin_um"] = df["width_min_calc_um"] - WIDTH_MIN_LIMIT
    df["max_width_margin_um"] = WIDTH_MAX_LIMIT - df["width_max_calc_um"]
    df["boundary_margin_um"] = np.minimum(df["min_width_margin_um"], df["max_width_margin_um"])
    df["dwidth_rms_um"] = np.sqrt(np.mean(d1**2, axis=1))
    df["dwidth_max_abs_um"] = np.max(np.abs(d1), axis=1)
    df["d2width_rms_um"] = np.sqrt(np.mean(d2**2, axis=1))
    df["d2width_max_abs_um"] = np.max(np.abs(d2), axis=1)

    amps = df[amp_cols].to_numpy(float)
    df["amp_total_l2_um"] = np.sqrt(np.sum(amps**2, axis=1))
    df["amp_total_l1_um"] = np.sum(np.abs(amps), axis=1)
    df["amp_low_l2_um"] = np.sqrt(np.sum(df[["amp1_um", "amp2_um"]].to_numpy(float) ** 2, axis=1))
    df["amp_mid_l2_um"] = np.sqrt(np.sum(df[["amp3_um", "amp4_um"]].to_numpy(float) ** 2, axis=1))
    df["amp_high_l2_um"] = np.sqrt(np.sum(df[["amp5_um", "amp6_um", "amp7_um", "amp8_um"]].to_numpy(float) ** 2, axis=1))
    denom = np.maximum(df["amp_total_l2_um"], 1e-9)
    df["amp_high_frac"] = df["amp_high_l2_um"] / denom
    df["amp_mid_frac"] = df["amp_mid_l2_um"] / denom

    energy = np.maximum(np.sum(amps**2, axis=1), 1e-12)
    df["d2_symmetry_error"] = np.sum(df[["amp1_um", "amp3_um", "amp5_um", "amp7_um"]].to_numpy(float) ** 2, axis=1) / energy
    df["d4_symmetry_error"] = np.sum(df[["amp1_um", "amp2_um", "amp3_um", "amp5_um", "amp6_um", "amp7_um"]].to_numpy(float) ** 2, axis=1) / energy
    df["d8_symmetry_error"] = np.sum(df[[f"amp{k}_um" for k in range(1, 8)]].to_numpy(float) ** 2, axis=1) / energy
    df["mirror_symmetry_error"] = np.sum(df[[f"S{k}_um" for k in range(1, 9)]].to_numpy(float) ** 2, axis=1) / energy
    df["d4_energy_frac"] = (df["amp4_um"] ** 2 + df["amp8_um"] ** 2) / energy
    df["d8_energy_frac"] = df["amp8_um"] ** 2 / energy
    df["radius_width_ratio"] = df["a_ring_um"] / np.maximum(df["h_ring_um"], 1e-9)
    df["width_range_over_h"] = df["width_range_um"] / np.maximum(df["h_ring_um"], 1e-9)
    df["roughness_over_h"] = df["dwidth_rms_um"] / np.maximum(df["h_ring_um"], 1e-9)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    derived = [
        c for c in df.columns
        if c.startswith("amp")
        or c.startswith("width_")
        or c.endswith("_margin_um")
        or c in {
            "boundary_margin_um", "dwidth_rms_um", "dwidth_max_abs_um",
            "d2width_rms_um", "d2width_max_abs_um", "radius_width_ratio",
            "width_range_over_h", "roughness_over_h", "d2_symmetry_error",
            "d4_symmetry_error", "d8_symmetry_error", "mirror_symmetry_error",
            "d4_energy_frac", "d8_energy_frac",
        }
    ]
    return RAW_FEATURES + derived
