from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from feature_engineering import RAW_FEATURES, WIDTH_MAX_LIMIT, WIDTH_MIN_LIMIT, add_geometry_features, load_data
from train_final_mlp import MLP


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "05_仿真与数据追加" / "config" / "design_search_config.json"


def order_features(order: int) -> list[str]:
    return ["a_ring_um", "h_ring_um", *[name for k in range(1, order + 1) for name in (f"C{k}_um", f"S{k}_um")]]


def resolve_model_dir(requested: str) -> Path:
    if requested != "auto":
        return Path(requested).resolve()
    delivered = ROOT / "03_最终模型"
    retrained = ROOT / "04_训练输出" / "final_mlp"
    if (retrained / "torch_mlp.pt").exists():
        return retrained
    return delivered


class Surrogate:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.prep = joblib.load(model_dir / "preprocessing.joblib")
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        config = metadata["model_configs"]["torch_mlp"]
        self.model = MLP(
            len(self.prep["features"]),
            hidden=tuple(config["hidden"]),
            dropout=config["dropout"],
            n_out=config["n_out"],
        )
        self.model.load_state_dict(torch.load(model_dir / "torch_mlp.pt", map_location="cpu", weights_only=True))
        self.model.eval()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        featured = add_geometry_features(frame)
        raw = featured[self.prep["features"]].astype(float).to_numpy()
        x = self.prep["scaler_x"].transform(self.prep["imputer"].transform(raw))
        with torch.no_grad():
            scaled = self.model(torch.tensor(x, dtype=torch.float32)).numpy()
        return self.prep["scaler_y"].inverse_transform(scaled)


def lhs(n: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    values = np.empty((n, dimensions), dtype=float)
    for j in range(dimensions):
        column = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(column)
        values[:, j] = column
    return values


def training_bounds(existing: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lows, highs = [], []
    for name in names:
        values = existing[name].to_numpy(float)
        if name in ("a_ring_um", "h_ring_um"):
            low, high = np.quantile(values, [0.005, 0.995])
        else:
            limit = float(np.quantile(np.abs(values), 0.995))
            limit = max(limit, 0.02)
            low, high = -limit, limit
        if high - low < 1e-8:
            high = low + 1e-8
        lows.append(float(low))
        highs.append(float(high))
    return np.asarray(lows), np.asarray(highs)


def generate_valid_pool(
    existing: pd.DataFrame,
    order: int,
    pool_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    names = order_features(order)
    lower, upper = training_bounds(existing, names)
    anchors = existing[names].to_numpy(float)
    scale = np.maximum(upper - lower, 1e-8)
    accepted: list[pd.DataFrame] = []
    accepted_count = 0

    for _ in range(30):
        if accepted_count >= pool_size:
            break
        batch_size = min(max(2000, pool_size // 2), 6000)
        n_global = batch_size // 2
        global_values = lower + lhs(n_global, len(names), rng) * scale

        anchor_indices = rng.integers(0, len(anchors), size=batch_size - n_global)
        local_values = anchors[anchor_indices] + rng.normal(0.0, 0.075, size=(batch_size - n_global, len(names))) * scale
        local_values = np.clip(local_values, lower, upper)
        values = np.vstack([global_values, local_values])

        frame = pd.DataFrame(values, columns=names)
        for name in RAW_FEATURES:
            if name not in frame.columns:
                frame[name] = 0.0
        featured = add_geometry_features(frame, n_theta=720)
        valid = featured[
            (featured["width_min_calc_um"] >= WIDTH_MIN_LIMIT)
            & (featured["width_max_calc_um"] <= WIDTH_MAX_LIMIT)
        ].copy()
        if not valid.empty:
            accepted.append(valid[RAW_FEATURES + ["width_min_calc_um", "width_max_calc_um"]])
            accepted_count += len(valid)

    if not accepted:
        raise RuntimeError("未生成满足环宽约束的候选点，请检查训练数据范围。")
    pool = pd.concat(accepted, ignore_index=True)
    pool = pool.drop_duplicates(subset=names).head(pool_size).reset_index(drop=True)
    if len(pool) < max(100, pool_size // 10):
        raise RuntimeError(f"有效候选点只有 {len(pool)} 个，数量不足。")
    return pool


def candidate_novelty(pool: pd.DataFrame, existing: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    existing_x = existing[names].to_numpy(float)
    pool_x = pool[names].to_numpy(float)
    center = np.median(existing_x, axis=0)
    scale = np.quantile(np.abs(existing_x - center), 0.75, axis=0)
    fallback = np.std(existing_x, axis=0)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(scale > 1e-8, scale, 1.0)
    existing_z = (existing_x - center) / scale
    pool_z = (pool_x - center) / scale
    nearest = NearestNeighbors(n_neighbors=1).fit(existing_z)
    distance = nearest.kneighbors(pool_z, return_distance=True)[0][:, 0]
    return distance, existing_z, pool_z


def select_mixed_batch(
    pool: pd.DataFrame,
    existing: pd.DataFrame,
    order: int,
    n_points: int,
    coverage_fraction: float,
    min_distance: float,
) -> pd.DataFrame:
    names = order_features(order)
    novelty, _, pool_z = candidate_novelty(pool, existing, names)
    eligible = novelty >= min_distance
    if int(eligible.sum()) < n_points:
        eligible = np.ones(len(pool), dtype=bool)

    q_rank = pd.Series(pool["pred_log10_Q"]).rank(pct=True).to_numpy(float)
    selected: list[int] = []
    available = eligible.copy()
    live_distance = novelty.copy()
    n_coverage = min(n_points, max(0, int(round(n_points * coverage_fraction))))

    def take(score: np.ndarray, kind: str) -> None:
        masked = np.where(available, score, -np.inf)
        index = int(np.argmax(masked))
        if not np.isfinite(masked[index]):
            return
        selected.append(index)
        pool.loc[index, "source_kind"] = kind
        available[index] = False
        distance_to_selected = np.linalg.norm(pool_z - pool_z[index], axis=1)
        np.minimum(live_distance, distance_to_selected, out=live_distance)

    for _ in range(n_coverage):
        take(live_distance, "coverage")
    while len(selected) < n_points:
        score = q_rank + 0.08 * np.minimum(live_distance, 3.0)
        before = len(selected)
        take(score, "surrogate_highQ")
        if len(selected) == before:
            break

    result = pool.loc[selected].copy().reset_index(drop=True)
    result["novelty_score"] = novelty[selected]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="根据现有训练数据和最终 MLP 自动寻找下一批设计点")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--order", type=int, choices=[6, 8])
    parser.add_argument("--n-points", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--coverage-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-dir", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    order = args.order or int(config["order"])
    n_points = args.n_points or int(config["n_points"])
    pool_size = args.pool_size or int(config["pool_size"])
    coverage_fraction = args.coverage_fraction if args.coverage_fraction is not None else float(config["coverage_fraction"])
    seed = args.seed if args.seed is not None else int(config["seed"])
    min_distance = float(config.get("min_normalized_distance", 0.04))
    min_predicted_q = float(config.get("min_predicted_q", 1.0e6))
    output = Path(args.output) if args.output else ROOT / "05_仿真与数据追加" / "input" / f"designs_width{order}_auto.csv"

    if not 0.0 <= coverage_fraction <= 1.0:
        raise ValueError("coverage_fraction 必须在 0–1 之间。")
    if n_points < 1 or pool_size < n_points:
        raise ValueError("n_points 至少为 1，pool_size 不能小于 n_points。")

    rng = np.random.default_rng(seed)
    data = load_data()
    relevant = data.copy()
    if order == 6:
        relevant = relevant[(relevant[["C7_um", "S7_um", "C8_um", "S8_um"]].abs().max(axis=1) < 1e-10)].copy()
    model_dir = resolve_model_dir(args.model_dir)
    scorer = Surrogate(model_dir)

    report_dir = ROOT / "04_训练输出" / "candidate_search"
    report_dir.mkdir(parents=True, exist_ok=True)
    history_path = report_dir / f"width{order}_proposal_history.csv"
    distance_existing = relevant.copy()
    prior_proposed = 0
    if history_path.exists():
        history = pd.read_csv(history_path)
        for name in RAW_FEATURES:
            if name not in history.columns:
                history[name] = 0.0
        prior_proposed = len(history)
        distance_existing = pd.concat([distance_existing, history[RAW_FEATURES]], ignore_index=True)

    print(f"有效训练数据：{len(data):,} 行；用于 {order} 阶距离计算：{len(relevant):,} 行")
    if prior_proposed:
        print(f"另排除以前已提出的设计：{prior_proposed:,} 行")
    print(f"代理模型：{model_dir}")
    print(f"正在生成并筛选 {pool_size:,} 个候选点……")
    pool = generate_valid_pool(relevant, order, pool_size, rng)
    prediction = scorer.predict(pool[RAW_FEATURES].copy())
    pool["pred_log10_Q"] = prediction[:, 0]
    pool["pred_Q"] = np.power(10.0, prediction[:, 0])
    pool["pred_freq_MHz"] = prediction[:, 1]
    pool["source_kind"] = ""
    pool = pool[pool["pred_Q"] >= min_predicted_q].copy().reset_index(drop=True)
    if len(pool) < n_points:
        raise RuntimeError(f"预测 Q≥{min_predicted_q:g} 的候选点不足：{len(pool)} 个。")

    selected = select_mixed_batch(pool, distance_existing, order, n_points, coverage_fraction, min_distance)
    selected = selected.rename(columns={"width_min_calc_um": "width_min_um", "width_max_calc_um": "width_max_um"})
    names = order_features(order)
    output_columns = names + [
        "source_kind", "width_min_um", "width_max_um", "pred_log10_Q", "pred_Q", "pred_freq_MHz", "novelty_score"
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    selected[output_columns].to_csv(output, index=False, encoding="utf-8-sig")

    ranking = pool.sort_values("pred_Q", ascending=False).head(200).copy()
    ranking.to_csv(report_dir / f"width{order}_candidate_ranking_top200.csv", index=False, encoding="utf-8-sig")
    history_add = selected[output_columns].copy()
    history_add.insert(0, "generated_at", datetime.now().isoformat(timespec="seconds"))
    if history_path.exists():
        history_add = pd.concat([pd.read_csv(history_path), history_add], ignore_index=True)
    history_add = history_add.drop_duplicates(subset=names, keep="first")
    history_add.to_csv(history_path, index=False, encoding="utf-8-sig")
    summary = {
        "strategy": "mixed_coverage_majority",
        "order": order,
        "training_rows": int(len(data)),
        "relevant_existing_rows": int(len(relevant)),
        "prior_proposed_rows_excluded": int(prior_proposed),
        "candidate_pool_valid": int(len(pool)),
        "min_predicted_q": min_predicted_q,
        "selected_points": int(len(selected)),
        "coverage_points": int((selected["source_kind"] == "coverage").sum()),
        "surrogate_highQ_points": int((selected["source_kind"] == "surrogate_highQ").sum()),
        "model_dir": str(model_dir.resolve()),
        "output": str(output.resolve()),
        "seed": seed,
    }
    (report_dir / "latest_search_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n已选设计：")
    print(selected[["source_kind", "pred_Q", "pred_freq_MHz", "novelty_score"]].to_string(index=False))
    print(f"\n下一批 COMSOL 输入：{output.resolve()}")
    print(f"搜索报告：{report_dir.resolve()}")


if __name__ == "__main__":
    main()
