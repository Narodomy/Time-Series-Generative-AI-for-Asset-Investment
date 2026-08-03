"""
feature_utils.py
────────────────
Helper functions for loading, saving, inverse-transforming,
and feeding feature stores into DataLoaders.

Usage
-----
from feature_utils import load_split, load_scalers, make_dataloader
from feature_utils import inverse_x, inverse_cond, inverse_price
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# ── Save ──────────────────────────────────────────────────────────────────────


def save_split(
    split_data: dict,
    save_dir: Path,
    cfg_name: str,
    variant: str,
    split: str,
) -> Path:
    """
    Save one split dict to .npz.

    Parameters
    ----------
    split_data : dict with keys x, cond, dates, init_date, init_price, ohlcv_raw
    save_dir   : root directory (cfg.save_dir)
    cfg_name   : e.g. "v3_trial01"
    variant    : e.g. "standard", "robust", "none"
    split      : "train" | "val" | "test"

    Returns
    -------
    Path to saved .npz file
    """
    out_dir = save_dir / f"{cfg_name}_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.npz"

    np.savez_compressed(
        path,
        x=split_data["x"].astype(np.float32),
        cond=split_data["cond"].astype(np.float32),
        dates=split_data["dates"],
        init_date=split_data["init_date"],
        init_price=split_data["init_price"].astype(np.float32),
        ohlcv_raw=split_data["ohlcv_raw"].astype(np.float32),
    )
    return path


def save_scalers(
    x_scaler,
    cond_scaler,
    save_dir: Path,
    cfg_name: str,
    variant: str,
) -> Path:
    """
    Save x_scaler and cond_scaler to scalers.pkl.

    Returns
    -------
    Path to saved .pkl file
    """
    out_dir = save_dir / f"{cfg_name}_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scalers.pkl"

    with open(path, "wb") as f:
        pickle.dump({"x": x_scaler, "cond": cond_scaler}, f)

    return path


def save_meta(
    data: dict,
    cfg,
    variant: str,
    save_dir: Path,
    cfg_name: str,
) -> Path:
    """
    Save meta.json — config snapshot + data shapes per split.

    Contents
    --------
    variant, created_at, cfg snapshot, dim_legend, splits shapes + date range
    """
    import json
    from datetime import datetime, timezone

    out_dir = save_dir / f"{cfg_name}_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── cfg snapshot ──────────────────────────────────────────────────────────
    cfg_dict = {
        "name": cfg.name,
        "description": cfg.description,
        "symbols": cfg.symbols,
        "interval": cfg.interval,
        "train_period": list(cfg.train_period),
        "val_period": list(cfg.val_period),
        "test_period": list(cfg.test_period),
        "window_size": cfg.window_size,
        "sequence_depth": cfg.sequence_depth,
        "sequence_mode": cfg.sequence_mode,
        "indicator_shift": cfg.indicator_shift,
        "indicators": [
            {"name": ind.name, "params": ind.params, "norm_type": ind.norm_type}
            for ind in cfg.indicators
        ],
        "x_scaler": cfg.x_scaler,
        "x_scaler_axis": cfg.x_scaler_axis,
        "cond_scaler": cfg.cond_scaler,
        "cond_scaler_axis": cfg.cond_scaler_axis,
        "scaler_exclude_features": cfg.scaler_exclude_features,
        "scaler_clip_ranges": {k: list(v) for k, v in cfg.scaler_clip_ranges.items()},
        "batch_sizes": cfg.batch_sizes,
        "random_seed": cfg.random_seed,
    }

    # ── splits shape + date info ───────────────────────────────────────────────
    splits_info = {}
    for split_name, d in data.items():
        init_date = d["init_date"]
        splits_info[split_name] = {
            "x": list(d["x"].shape),
            "cond": list(d["cond"].shape),
            "ohlcv_raw": list(d["ohlcv_raw"].shape),
            "init_price": list(d["init_price"].shape),
            "dates": list(d["dates"].shape),
            "init_date": list(init_date.shape),
            "date_start": str(np.datetime_as_string(init_date.min(), unit="D")),
            "date_end": str(np.datetime_as_string(init_date.max(), unit="D")),
            "n_windows": int(len(init_date)),
        }

    # ── dim legend ─────────────────────────────────────────────────────────────
    dim_legend = {
        "x": "[T, A, W, D, C]  or  [T, A, W, C] if D=0",
        "cond": "[T, A, W, F]",
        "ohlcv_raw": "[T, A, W, 4]  — raw Close/High/Low/Volume",
        "init_price": "[T, A]  — Close price before window",
        "dates": "[T, W]  — datetime64 per window step",
        "init_date": "[T]     — date before window starts",
        "dims": {
            "T": "n_windows",
            "A": "n_assets",
            "W": "window_size",
            "D": "sequence_depth  (0 = no seq dim)",
            "C": "channels",
            "F": "indicator features",
        },
        "C_channels": [
            "logdiff_close",
            "logdiff_high",
            "logdiff_low",
            "logdiff_volume",
        ],
    }

    meta = {
        "variant": variant,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cfg": cfg_dict,
        "dim_legend": dim_legend,
        "splits": splits_info,
    }

    path = out_dir / "meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

    return path


def load_meta(
    save_dir: Path,
    cfg_name: str,
    variant: str,
) -> dict:
    """
    Load meta.json for a variant.

    Example
    -------
    meta = load_meta(cfg.save_dir, cfg.name, "standard_robust")
    print(meta["splits"]["train"]["x"])   # [2097, 16, 20, 20, 4]
    print(meta["cfg"]["window_size"])     # 20
    print(meta["dim_legend"]["x"])
    """
    import json

    path = save_dir / f"{cfg_name}_{variant}" / "meta.json"
    with open(path) as f:
        return json.load(f)


# ── Load ──────────────────────────────────────────────────────────────────────


def save_variant(
    data: dict,
    x_scaler,
    cond_scaler,
    cfg,
    variant: str,
) -> None:
    """
    Save all splits + scalers + meta.json for one variant.

    Parameters
    ----------
    data    : {"train": dict, "val": dict, "test": dict}
    cfg     : FeatureConfig instance
    variant : e.g. "standard_robust"

    Example
    -------
    save_variant(data_scaled, x_scaler, cond_scaler, cfg, "standard_robust")
    """
    save_dir = cfg.save_dir
    cfg_name = cfg.name

    for split_name, split_data in data.items():
        path = save_split(split_data, save_dir, cfg_name, variant, split_name)
        print(f"  saved {split_name:<6} → {path}")

    scaler_path = save_scalers(x_scaler, cond_scaler, save_dir, cfg_name, variant)
    print(f"  saved scalers  → {scaler_path}")

    meta_path = save_meta(data, cfg, variant, save_dir, cfg_name)
    print(f"  saved meta     → {meta_path}")


def load_split(
    save_dir: Path,
    cfg_name: str,
    variant: str,
    split: str,
) -> dict:
    """
    Load one split from .npz.

    Returns
    -------
    dict with keys: x, cond, dates, init_date, init_price, ohlcv_raw

    Example
    -------
    d = load_split(cfg.save_dir, cfg.name, "standard", "train")
    d["x"].shape   # [T, A, W, D, C]
    """
    path = save_dir / f"{cfg_name}_{variant}" / f"{split}.npz"
    raw = np.load(path, allow_pickle=False)
    return {k: raw[k] for k in raw.files}


def load_scalers(
    save_dir: Path,
    cfg_name: str,
    variant: str,
) -> dict:
    """
    Load scalers from .pkl.

    Returns
    -------
    {"x": x_scaler, "cond": cond_scaler}

    Example
    -------
    scalers = load_scalers(cfg.save_dir, cfg.name, "standard")
    x_scaler    = scalers["x"]
    cond_scaler = scalers["cond"]
    """
    path = save_dir / f"{cfg_name}_{variant}" / "scalers.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_variant(
    save_dir: Path,
    cfg_name: str,
    variant: str,
) -> tuple[dict, dict]:
    """
    Load all splits + scalers for one variant in one call.

    Returns
    -------
    data    : {"train": dict, "val": dict, "test": dict}
    scalers : {"x": x_scaler, "cond": cond_scaler}

    Example
    -------
    data, scalers = load_variant(cfg.save_dir, cfg.name, "standard_robust")
    data["train"]["x"].shape   # [T, A, W, D, C]
    scalers["x"]               # fitted PerChannelScaler
    """
    data = {
        split: load_split(save_dir, cfg_name, variant, split)
        for split in ["train", "val", "test"]
    }
    scalers = load_scalers(save_dir, cfg_name, variant)
    return data, scalers


# ── Inverse Transform ─────────────────────────────────────────────────────────


def inverse_x(
    x_scaled: np.ndarray,
    x_scaler,
) -> np.ndarray:
    """
    Inverse scale x — scaled log diff → log diff.

    Parameters
    ----------
    x_scaled : [..., C]   any leading dims, last dim = channels
    x_scaler : fitted PerChannelScaler (or None → passthrough)

    Returns
    -------
    np.ndarray same shape as input
    """
    if x_scaler is None:
        return x_scaled
    return x_scaler.inverse_transform(x_scaled)


def inverse_cond(
    cond_scaled: np.ndarray,
    cond_scaler,
) -> np.ndarray:
    """
    Inverse scale cond — scaled indicators → original indicators.

    Parameters
    ----------
    cond_scaled : [..., F]   any leading dims, last dim = features
    cond_scaler : fitted PerChannelScaler (or None → passthrough)

    Returns
    -------
    np.ndarray same shape as input
    """
    if cond_scaler is None:
        return cond_scaled
    return cond_scaler.inverse_transform(cond_scaled)


def inverse_price(
    x_scaled: np.ndarray,
    init_price: np.ndarray,
    x_scaler,
    d_idx: int = 0,
) -> np.ndarray:
    """
    Scaled log diff → reconstructed price.

    Steps
    -----
    1. inverse scaler  → log diff  [T, A, W, D, C]
    2. take D=d_idx    → [T, A, W, C]
    3. cumsum along W  → cumulative log return
    4. init_price * exp(cumsum) → price

    Parameters
    ----------
    x_scaled   : [T, A, W, D, C]  scaled log diff
                 or [T, A, W, C]  if D=0 (no seq)
    init_price : [T, A]            Close price before window
    x_scaler   : fitted PerChannelScaler (or None)
    d_idx      : which D slice to use (default 0 = most recent)

    Returns
    -------
    price_recon : [T, A, W, C]
    """
    x_lr = inverse_x(x_scaled, x_scaler)  # [..., C]

    # handle D=0 case (no seq dim)
    if x_lr.ndim == 5:
        x_lr = x_lr[:, :, :, d_idx, :]  # [T, A, W, C]

    x_cum = np.cumsum(x_lr, axis=2)  # [T, A, W, C]
    init = init_price[:, :, None, None]  # [T, A, 1, 1]
    return init * np.exp(x_cum)  # [T, A, W, C]


# ── DataLoader ────────────────────────────────────────────────────────────────


def make_dataloader(
    split_data: dict,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Wrap split dict into a DataLoader via TensorDataset.

    Note: dates and init_date are NOT included (datetime64 not supported
    by TensorDataset). Access them directly from split_data if needed.

    Parameters
    ----------
    split_data : dict from load_split()
    batch_size : int
    shuffle    : True for train, False for val/test
    drop_last  : defaults to shuffle value (drop last incomplete batch only during training)

    Returns
    -------
    DataLoader yielding (x, cond, init_price, ohlcv_raw) per batch

    Example
    -------
    d  = load_split(cfg.save_dir, cfg.name, "standard", "train")
    dl = make_dataloader(d, batch_size=64, shuffle=True)
    x, cond, init_price, ohlcv_raw = next(iter(dl))
    """
    if drop_last is None:
        drop_last = shuffle

    dates_ns = np.array(split_data["dates"], dtype="datetime64[ns]").astype("int64")
    init_date_ns = np.array(split_data["init_date"], dtype="datetime64[ns]").astype(
        "int64"
    )

    ds = TensorDataset(
        torch.tensor(split_data["x"], dtype=torch.float32),
        torch.tensor(split_data["cond"], dtype=torch.float32),
        torch.tensor(split_data["init_price"], dtype=torch.float32),
        torch.tensor(split_data["ohlcv_raw"], dtype=torch.float32),
        torch.tensor(dates_ns, dtype=torch.int64),
        torch.tensor(init_date_ns, dtype=torch.int64),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def make_all_dataloaders(
    save_dir: Path,
    cfg_name: str,
    variant: str,
    batch_sizes: dict,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict:
    """
    Load all splits and return dict of DataLoaders.

    Parameters
    ----------
    batch_sizes : {"train": 64, "val": 64, "test": 1}

    Returns
    -------
    {"train": DataLoader, "val": DataLoader, "test": DataLoader}

    Example
    -------
    loaders = make_all_dataloaders(cfg.save_dir, cfg.name, "standard", cfg.batch_sizes)
    x, cond, init_price, ohlcv_raw = next(iter(loaders["train"]))
    """
    splits = ["train", "val", "test"]
    shuffle = {"train": True, "val": False, "test": False}

    loaders = {}
    for split in splits:
        d = load_split(save_dir, cfg_name, variant, split)
        loaders[split] = make_dataloader(
            d,
            batch_size=batch_sizes[split],
            shuffle=shuffle[split],
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders
