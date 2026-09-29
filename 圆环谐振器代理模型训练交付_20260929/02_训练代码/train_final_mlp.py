from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feature_engineering import DEFAULT_DATA, add_geometry_features, feature_columns, load_data


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "04_训练输出" / "final_mlp"


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden=(256, 128, 64), n_out: int = 2, dropout: float = 0.05):
        super().__init__()
        layers = []
        last = n_in
        for width in hidden:
            layers.extend([nn.Linear(last, width), nn.ReLU(), nn.BatchNorm1d(width), nn.Dropout(dropout)])
            last = width
        layers.append(nn.Linear(last, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def q_metrics(true_logq: np.ndarray, pred_logq: np.ndarray) -> dict[str, float]:
    q_true = 10.0 ** true_logq
    q_pred = 10.0 ** pred_logq
    rel = np.abs(q_pred - q_true) / np.maximum(q_true, 1e-12)
    return {
        "logQ_MAE": float(mean_absolute_error(true_logq, pred_logq)),
        "logQ_RMSE": float(math.sqrt(mean_squared_error(true_logq, pred_logq))),
        "logQ_R2": float(r2_score(true_logq, pred_logq)),
        "Q_mean_rel_err": float(np.mean(rel)),
        "Q_median_rel_err": float(np.median(rel)),
        "Q_p90_rel_err": float(np.quantile(rel, 0.90)),
        "Q_p95_rel_err": float(np.quantile(rel, 0.95)),
    }


def freq_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.abs(y_pred - y_true)
    return {
        "freq_MAE_MHz": float(mean_absolute_error(y_true, y_pred)),
        "freq_RMSE_MHz": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "freq_R2": float(r2_score(y_true, y_pred)),
        "freq_p90_abs_err_MHz": float(np.quantile(err, 0.90)),
    }


def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size, seed, log):
    loss_weights = torch.tensor([1.0, 0.7], dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_x = torch.tensor(x_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=18, factor=0.5)
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            pred = model(xb)
            loss = (((pred - yb) ** 2) * loss_weights).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            val_loss = float(((((model(val_x) - val_y) ** 2) * loss_weights).mean()).detach())
        train_loss = float(np.mean(batch_losses))
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            log(f"Epoch {epoch:03d}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= 55:
            log(f"Early stopping at epoch {epoch}; best val_loss={best_loss:.6f}")
            break

    if best_state is None:
        raise RuntimeError("训练未产生有效模型。")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="训练最终三层 MLP：几何参数 -> log10(Q) + 频率")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--min-q", type=float, default=1e6)
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    log_path = outdir / "training.log"
    log_path.write_text("", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    log("Loading training CSV files...")
    df = load_data(DEFAULT_DATA)
    log(f"Loaded {len(df):,} valid rows from {len(DEFAULT_DATA)} files.")
    df = df[df["selected_Q"] >= args.min_q].copy()
    log(f"Rows after Q >= {args.min_q:g} filter: {len(df):,}")
    log("Building Fourier-derived geometry features...")
    df = add_geometry_features(df).reset_index(drop=True)
    features = feature_columns(df)
    log(f"Feature count: {len(features)}")
    x_raw = df[features].astype(float).to_numpy()
    y_raw = np.column_stack([df["log10_Q"].to_numpy(float), df["selected_freq_MHz"].to_numpy(float)])

    indices = np.arange(len(df))
    q_bins = pd.qcut(df["log10_Q"], q=5, labels=False, duplicates="drop")
    h_bins = pd.qcut(df["h_ring_um"], q=4, labels=False, duplicates="drop")
    strat = (q_bins.astype(str) + "_" + h_bins.astype(str)).to_numpy()
    trainval_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=args.seed, stratify=strat)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=0.18, random_state=args.seed + 1)

    imputer = SimpleImputer(strategy="median")
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train = scaler_x.fit_transform(imputer.fit_transform(x_raw[train_idx]))
    x_val = scaler_x.transform(imputer.transform(x_raw[val_idx]))
    x_test = scaler_x.transform(imputer.transform(x_raw[test_idx]))
    y_train = scaler_y.fit_transform(y_raw[train_idx])
    y_val = scaler_y.transform(y_raw[val_idx])

    log(
        f"Split sizes: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}. "
        f"Starting MLP training for at most {args.epochs} epochs..."
    )
    model = MLP(len(features))
    model, history, best_val_loss = train_model(
        model, x_train, y_train, x_val, y_val, args.epochs, args.batch_size, args.seed, log
    )
    log("Training finished. Saving model and evaluating the test split...")
    torch.save(model.state_dict(), outdir / "torch_mlp.pt")
    history.to_csv(outdir / "torch_mlp_history.csv", index=False, encoding="utf-8-sig")
    joblib.dump(
        {"features": features, "imputer": imputer, "scaler_x": scaler_x, "scaler_y": scaler_y},
        outdir / "preprocessing.joblib",
    )

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(x_test, dtype=torch.float32)).numpy()
    pred = scaler_y.inverse_transform(pred_scaled)
    metrics = {
        "model": "torch_mlp",
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "epochs_run": len(history), "best_val_loss_scaled": best_val_loss,
        **q_metrics(y_raw[test_idx, 0], pred[:, 0]),
        **freq_metrics(y_raw[test_idx, 1], pred[:, 1]),
    }
    pd.DataFrame([metrics]).to_csv(outdir / "deep_model_summary.csv", index=False, encoding="utf-8-sig")

    predictions = df.iloc[test_idx].copy().reset_index(drop=True)
    predictions["pred_logQ"] = pred[:, 0]
    predictions["pred_Q"] = 10.0 ** pred[:, 0]
    predictions["pred_freq_MHz"] = pred[:, 1]
    predictions.to_csv(outdir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "features": features,
        "targets": ["log10_Q", "selected_freq_MHz"],
        "n_samples": len(df),
        "min_q_filter": args.min_q,
        "data_files": [p.name for p in DEFAULT_DATA],
        "splits": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "model_configs": {"torch_mlp": {"hidden": [256, 128, 64], "dropout": 0.05, "n_out": 2}},
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(pd.DataFrame([metrics]).to_string(index=False), flush=True)
    log(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
