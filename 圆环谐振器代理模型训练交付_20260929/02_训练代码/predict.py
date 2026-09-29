from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from feature_engineering import RAW_FEATURES, add_geometry_features
from train_final_mlp import MLP


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="使用最终 MLP 预测 Q 和目标模态频率")
    parser.add_argument("--input", default=str(ROOT / "example_input.csv"))
    parser.add_argument("--model-dir", default=str(ROOT / "03_最终模型"))
    parser.add_argument("--output", default=str(ROOT / "04_训练输出" / "predictions.csv"))
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    prep = joblib.load(model_dir / "preprocessing.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["model_configs"]["torch_mlp"]

    frame = pd.read_csv(args.input)
    for col in RAW_FEATURES:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame = add_geometry_features(frame)

    x_raw = frame[prep["features"]].astype(float).to_numpy()
    x = prep["scaler_x"].transform(prep["imputer"].transform(x_raw))
    model = MLP(len(prep["features"]), hidden=tuple(config["hidden"]), dropout=config["dropout"], n_out=config["n_out"])
    model.load_state_dict(torch.load(model_dir / "torch_mlp.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        pred = prep["scaler_y"].inverse_transform(model(torch.tensor(x, dtype=torch.float32)).numpy())

    result = pd.read_csv(args.input)
    result["pred_log10_Q"] = pred[:, 0]
    result["pred_Q"] = np.power(10.0, pred[:, 0])
    result["pred_freq_MHz"] = pred[:, 1]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(result[["pred_Q", "pred_freq_MHz"]].to_string(index=False))
    print(f"\n输出文件：{output.resolve()}")


if __name__ == "__main__":
    main()
