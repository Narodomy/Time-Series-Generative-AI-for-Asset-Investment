import pandas as pd
import torch
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

import json
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from datasets import MarketDataset


def load_processed_store(
    store_name: str,
    saved_dir: Path,
    batch_size: int = None,  # override meta batch_size
    batch_size_test: int = 1,
    shuffle_train: bool = True,
):
    """
    Returns
    -------
    parts   : dict  { "train": {...}, "val": {...}, "test": {...} }  raw arrays
    datasets: dict  { "train": MarketDataset, ... }
    loaders : dict  { "train": DataLoader, ... }
    meta    : dict
    """
    saved_dir = saved_dir / store_name

    with open(saved_dir / "meta.json") as f:
        meta = json.load(f)

    bs = batch_size or meta["batch_size"]
    bs_test = batch_size_test

    # ── Load raw arrays ──────────────────────────────────────────────────
    parts = {}
    for split in ["train", "val", "test"]:
        data = np.load(saved_dir / f"{split}.npz", allow_pickle=True)
        parts[split] = {
            "x": data["x"],
            "cond": data["cond"],
            "dates": data["dates"].astype("datetime64[ns]"),
            "close_prices": data["close_prices"],
            "x_prev": data["x_prev"],
            "x_prev_mean": data["x_prev_mean"],  # [N_windows, A, F]  already shifted
            "x_prev_std": data["x_prev_std"],  # [N_windows, A, F]  already shifted
        }

    # ── Build MarketDataset ──────────────────────────────────────────────
    datasets = {}
    strides = {
        "train": meta["stride"],
        "val": meta["stride"],
        "test": meta["test_batch_stride"],  # non-overlap สำหรับ portfolio eval
    }
    for split, part in parts.items():
        ds = MarketDataset(
            **part,
            window_size=meta["window_size"],
            stride=strides[split],
            normalize_window=meta["normalize_window"],
        )
        if meta["normalize_window"]:
            ds = Subset(ds, range(1, len(ds)))  # drop idx=0 (NaN mean/std)
        datasets[split] = ds

    # ── Build DataLoaders ────────────────────────────────────────────────
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=bs, shuffle=shuffle_train, drop_last=True
        ),
        "val": DataLoader(datasets["val"], batch_size=bs, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=bs_test, shuffle=False),
    }

    print(f"✅ Loaded feature store  →  {saved_dir}")
    for split, ds in datasets.items():
        print(f"   {split:5s}: {len(ds)} windows")

    return parts, datasets, loaders, meta


def timestamp_info(df: pd.DataFrame):
    info = (df.index.min(), df.index.max())
    print(f"Data range: {info[0]} to {info[1]}")

    duration = df.index.max() - df.index.min()
    print(f"Total duration: {duration}")


def timestamp_mask(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):
    mask = (df.index >= start_date) & (df.index <= end_date)
    return mask


def inspect(data, name: str = ""):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    _min = np.min(data)
    _max = np.max(data)
    _mean = np.mean(data)
    _std = np.std(data)

    print(f"--- Inspecting: {name} ---")
    print("-" * 36)
    print(f"Shape: {data.shape}")
    print(f"Min:   {_min:.4f}")
    print(f"Max:   {_max:.4f}")
    print(f"Mean:  {_mean:.4f}")
    print(f"Std:   {_std:.4f}")
    print("-" * 36)
    return _min, _max, _mean, _std


def inverse_log_returns(r_log: np.ndarray) -> np.ndarray:
    # R_simple = e^(R_log) - 1
    r_simple = np.exp(r_log) - 1

    return r_simple


def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str


def save_as_html(report: dict, filename: str):
    """
    Requires
    {
        "metrics": {"Mean": 0.5, "Sharpe": 1.2, ...},
        "plots": {"Cumulative Return": "base64...", "Drawdown": "base64..."}
    }
    """
    metrics_html = ""
    if "metrics" in report and report["metrics"]:
        rows = "".join(
            [f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in report["metrics"].items()]
        )
        metrics_html = f"""
        <h3>Metrics</h3>
        <table border="1" style="border-collapse: collapse; width: 50%;">
            <tr><th>Metric</th><th>Value</th></tr>
            {rows}
        </table>
        """

    plots_html = ""
    if "plots" in report and report["plots"]:
        for title, img_str in report["plots"].items():
            plots_html += f"""
            <div style="margin-bottom: 20px;">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{img_str}" style="max-width: 100%; border: 1px solid #ccc;">
            </div>
            """

    html = f"""
    <html>
    <head>
        <title>Simulation Report</title>
        <style>body {{ font-family: sans-serif; padding: 20px; }} h3 {{ color: #444; }}</style>
    </head>
    <body>
        <h1>Inspection Report</h1>
        <p><strong>ID:</strong> {report.get('id', 'N/A')}</p>
        <hr>
        {metrics_html}
        <hr>
        {plots_html}
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
