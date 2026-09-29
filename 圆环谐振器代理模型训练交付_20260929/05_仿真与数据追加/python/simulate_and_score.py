from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from append_verified_data import append_verified_dataframe


SIM_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((SIM_ROOT / "config" / "project_config.json").read_text(encoding="utf-8"))
REFERENCE_FILE = SIM_ROOT / "reference" / "strict_mac_initial_vectors.csv"
REFERENCE_LABELS = ["apodized_d1_m10", "apodized_d2_m10", "apodized_d3_m10", "width2_d1_m8"]


def feature_names(order: int) -> list[str]:
    names = ["a_ring_um", "h_ring_um"]
    for k in range(1, order + 1):
        names.extend([f"C{k}_um", f"S{k}_um"])
    return names


def all_geometry_names() -> list[str]:
    return ["a_ring_um", "h_ring_um", *[n for k in range(1, 9) for n in (f"C{k}_um", f"S{k}_um")]]


def encode_arg(value: float) -> str:
    text = f"{value:.12g}"
    return "m" + text[1:] if text.startswith("-") else text


def width_range(row: dict, order: int, n_theta: int = 720) -> tuple[float, float]:
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    width = np.full(n_theta, float(row["h_ring_um"]))
    for k in range(1, order + 1):
        width += float(row[f"C{k}_um"]) * np.cos(k * theta)
        width += float(row[f"S{k}_um"]) * np.sin(k * theta)
    return float(width.min()), float(width.max())


def read_designs(path: Path, order: int) -> list[dict]:
    frame = pd.read_csv(path)
    required = ["a_ring_um", "h_ring_um"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Input design CSV is missing: {missing}")
    rows = []
    width_min = float(CONFIG["geometry"]["width_min_um"])
    width_max = float(CONFIG["geometry"]["width_max_um"])
    for index, source in frame.iterrows():
        row = {"source_kind": str(source.get("source_kind", "user_added"))}
        for name in all_geometry_names():
            default = 0.0
            value = source.get(name, default)
            row[name] = float(value) if pd.notna(value) else default
        lo, hi = width_range(row, order)
        row["width_min_um"] = lo
        row["width_max_um"] = hi
        if lo < width_min or hi > width_max:
            raise ValueError(
                f"Design row {index + 1} violates width limits: min={lo:.6g}, max={hi:.6g}, "
                f"allowed=[{width_min}, {width_max}] um"
            )
        rows.append(row)
    if not rows:
        raise ValueError("Input design CSV contains no rows.")
    return rows


def read_reference_vectors() -> dict[str, list[float]]:
    with REFERENCE_FILE.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    refs = {}
    for label in REFERENCE_LABELS:
        selected = sorted((r for r in rows if r["label"] == label), key=lambda r: int(r["sample_index"]))
        vector = []
        for row in selected:
            vector.extend([float(row["u_um"]), float(row["v_um"]), float(row["w_um"])])
        if len(vector) != 192:
            raise RuntimeError(f"Reference vector {label} has {len(vector)} values; expected 192.")
        refs[label] = vector
    return refs


def mac(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    denom = float(np.dot(av, av) * np.dot(bv, bv))
    if denom <= 0:
        return float("nan")
    return float(np.dot(av, bv) ** 2 / denom)


def runner_paths(order: int) -> tuple[Path, Path, Path]:
    model = SIM_ROOT / "models" / f"plain_width{order}_model_manual10.mph"
    java = SIM_ROOT / "java" / f"ScanWidth{order}ModesForStrictMacArgs.java"
    clazz = SIM_ROOT / "java" / f"ScanWidth{order}ModesForStrictMacArgs.class"
    return model, java, clazz


def compile_runner(order: int) -> None:
    _, source, _ = runner_paths(order)
    compile_exe = Path(CONFIG["comsol"]["compile_exe"])
    if not compile_exe.exists():
        raise FileNotFoundError(f"COMSOL compiler not found: {compile_exe}")
    subprocess.run([str(compile_exe), str(source)], cwd=source.parent, check=True)


def run_comsol(order: int, designs: list[dict], raw_log: Path, compile_java: bool) -> None:
    batch_exe = Path(CONFIG["comsol"]["batch_exe"])
    model, _, runner = runner_paths(order)
    if compile_java:
        compile_runner(order)
    for path, label in [(batch_exe, "COMSOL batch executable"), (model, "COMSOL model"), (runner, "compiled Java runner")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    args = [
        str(batch_exe), "-graphics", "-nosave", "-inputfile", str(runner), str(model),
        str(CONFIG["comsol"].get("study_tag", "std3")), str(CONFIG["comsol"].get("z_um", 0.0)),
    ]
    names = feature_names(order)
    for design in designs:
        args.extend(encode_arg(float(design[name])) for name in names)
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    with raw_log.open("wb") as stream:
        result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"COMSOL returned exit code {result.returncode}. Check {raw_log}")


def decode_log(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-16", "utf-8-sig", "utf-8", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_modes(path: Path) -> list[dict]:
    modes = []
    for line in decode_log(path).splitlines():
        if not line.startswith("MODE,") or line.startswith("MODE,index"):
            continue
        parts = line.split(",", 12)
        if len(parts) < 13:
            continue
        try:
            vector = [float(x) for x in parts[12].split(";") if x]
            if len(vector) != 192:
                continue
            modes.append({
                "index": int(parts[1]), "mode_index": int(parts[2]),
                "freq_MHz": float(parts[3]), "Q": float(parts[4]),
                "same_sign_fraction": float(parts[5]), "breathing_uniformity": float(parts[6]),
                "vector": vector,
            })
        except (ValueError, IndexError):
            continue
    return modes


def score_modes(order: int, designs: list[dict], modes: list[dict], batch_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    refs = read_reference_vectors()
    strict_mac_min = float(CONFIG["mode_identification"]["strict_mac_min"])
    same_sign_min = float(CONFIG["mode_identification"]["same_sign_fraction_min"])
    uniformity_min = float(CONFIG["mode_identification"]["breathing_uniformity_min"])
    grouped: dict[int, list[dict]] = {}
    score_rows = []

    for mode in modes:
        scores = {label: mac(mode["vector"], vector) for label, vector in refs.items()}
        best_label = max(scores, key=scores.get)
        best_mac = scores[best_label]
        valid = (
            best_mac >= strict_mac_min
            and mode["same_sign_fraction"] >= same_sign_min
            and mode["breathing_uniformity"] >= uniformity_min
        )
        item = {**mode, "best_strict_mac": best_mac, "best_ref_label": best_label, "valid": valid, "scores": scores}
        grouped.setdefault(mode["index"], []).append(item)
        score_rows.append({
            "order": order, "batch": batch_id, "index": mode["index"], "mode_index": mode["mode_index"],
            "freq_MHz": mode["freq_MHz"], "Q": mode["Q"],
            "same_sign_fraction": mode["same_sign_fraction"], "breathing_uniformity": mode["breathing_uniformity"],
            "best_strict_mac": best_mac, "best_ref_label": best_label,
            "valid_target": int(valid), **{f"mac_vs_{k}": v for k, v in scores.items()},
        })

    summary_rows = []
    for idx, design in enumerate(designs, start=1):
        candidates = grouped.get(idx, [])
        candidates.sort(
            key=lambda m: (m["valid"], m["best_strict_mac"], m["same_sign_fraction"], m["breathing_uniformity"]),
            reverse=True,
        )
        base = {
            "order": order, "batch": batch_id, "index": idx,
            **{name: design[name] for name in all_geometry_names()},
            "source_kind": design["source_kind"],
            "width_min_um": design["width_min_um"], "width_max_um": design["width_max_um"],
        }
        if not candidates:
            summary_rows.append({**base, "selected_mode_index": "", "selected_freq_MHz": "", "selected_Q": "", "selected_Q_candidate": "", "same_sign_fraction": "", "breathing_uniformity": "", "best_strict_mac": "", "best_ref_label": "", "valid_for_Q": 0})
            continue
        best = candidates[0]
        summary_rows.append({
            **base,
            "selected_mode_index": best["mode_index"], "selected_freq_MHz": best["freq_MHz"],
            "selected_Q": best["Q"] if best["valid"] else "", "selected_Q_candidate": best["Q"],
            "same_sign_fraction": best["same_sign_fraction"], "breathing_uniformity": best["breathing_uniformity"],
            "best_strict_mac": best["best_strict_mac"], "best_ref_label": best["best_ref_label"],
            "valid_for_Q": int(best["valid"]),
        })
    return pd.DataFrame(score_rows), pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COMSOL and strict displacement-vector mode identification.")
    parser.add_argument("--order", type=int, choices=[6, 8], required=True)
    parser.add_argument("--input", required=True, help="Design CSV path.")
    parser.add_argument("--label", required=True, help="Output folder and batch label.")
    parser.add_argument("--append", action="store_true", help="Append valid rows to 01_训练数据/training_q_user_added.csv.")
    parser.add_argument("--compile", action="store_true", help="Recompile the Java runner before simulation.")
    args = parser.parse_args()

    output = SIM_ROOT / "output" / args.label
    output.mkdir(parents=True, exist_ok=True)
    designs = read_designs(Path(args.input), args.order)
    print(f"Designs accepted by geometry limits: {len(designs)}", flush=True)
    raw_log = output / "comsol_raw.log"
    print("Starting COMSOL...", flush=True)
    run_comsol(args.order, designs, raw_log, args.compile)
    print("COMSOL finished. Parsing displacement vectors...", flush=True)
    modes = parse_modes(raw_log)
    mode_scores, summary = score_modes(args.order, designs, modes, args.label)
    valid = summary[summary["valid_for_Q"].astype(str).isin(["1", "True", "true"])].copy()

    mode_scores.to_csv(output / "mode_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    valid.to_csv(output / "training_valid.csv", index=False, encoding="utf-8-sig")
    print(f"Parsed modes: {len(modes)}; valid designs: {len(valid)}/{len(summary)}", flush=True)
    print(f"Output directory: {output.resolve()}", flush=True)

    if args.append and not valid.empty:
        accepted, net_added = append_verified_dataframe(valid)
        print(f"Appended verified rows: accepted={accepted}, net_new={net_added}", flush=True)
        print(f"Training file: {(PACKAGE_ROOT / '01_训练数据' / 'training_q_user_added.csv').resolve()}", flush=True)
    elif args.append:
        print("No rows passed the strict mode criteria; the training data was not changed.", flush=True)


if __name__ == "__main__":
    main()
