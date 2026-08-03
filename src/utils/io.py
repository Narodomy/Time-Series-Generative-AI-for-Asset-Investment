"""
utils/io.py
────────────
Save / load the feature-engineering output (scaled windows, scalers, meta)
so the exact same Dataset/DataLoader objects can be reconstructed later —
in this notebook, in a different notebook, or in a training script.

On-disk layout (under cfg.save_dir):
    meta.json          human-readable config + shapes + feature names
    scalers.pkl         {"scalers_x": {...}, "scalers_cond": {...}}
    windows.npz          {split}_lr, {split}_cond, {split}_init, {split}_dates
                         for split in ("train", "val", "test")
    window_dates.csv    long-format (split, window_idx, date) — for quick viewing only,
                         not read back by load_data()

Usage
─────
    from utils import save_data, load_data, build_datasets, make_dataloaders

    save_data(
        cfg,
        splits={
            "train": dict(lr=windows_train_lr_scaled, cond=windows_train_cond_scaled,
                           init_price=init_prices_train, dates=window_dates_train),
            "val":   dict(lr=windows_val_lr_scaled,   cond=windows_val_cond_scaled,
                           init_price=init_prices_val,   dates=window_dates_val),
            "test":  dict(lr=windows_test_lr_scaled,  cond=windows_test_cond_scaled,
                           init_price=init_prices_test,  dates=window_dates_test),
        },
        tickers=tickers_lr, features_lr=features_lr, features_cond=features_cond,
        cyclical_idx=cyclical_idx, scale_idx=scale_idx,
        scalers_x=scalers_x, scalers_cond=scalers_cond,
    )

    # ...later, in this notebook or any other one...
    loaded, dataloaders = load_and_make_dataloaders(cfg)
    dl_train, dl_val, dl_test = dataloaders["train"], dataloaders["val"], dataloaders["test"]
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import WindowDataset, make_dataloaders  # noqa: F401  (re-exported)

SPLITS = ("train", "val", "test")


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────
def _resolve_dir(cfg_or_dir) -> Path:
    """Accept either a config object with `.save_dir`, or a path-like directly."""
    save_dir = getattr(cfg_or_dir, "save_dir", cfg_or_dir)
    return Path(save_dir)


def _cfg_to_dict(cfg) -> dict:
    """Best-effort extraction of the config fields we care about. Missing
    attributes are stored as None instead of raising, so this stays
    forward-compatible with config changes."""
    g = lambda name, default=None: getattr(cfg, name, default)
    train_p, val_p, test_p = g("train_period"), g("val_period"), g("test_period")
    return {
        "name": g("name"),
        "group_name": g("group_name"),
        "subgroup_name": g("subgroup_name"),
        "description": g("description"),
        "symbols": list(g("symbols", []) or []),
        "interval": g("interval"),
        "train_period": list(train_p) if train_p is not None else None,
        "val_period": list(val_p) if val_p is not None else None,
        "test_period": list(test_p) if test_p is not None else None,
        "window_size": g("window_size"),
        "window_stride": g("window_stride"),
        "scaler_x": g("scaler_x"),
        "scaler_cond": g("scaler_cond"),
        "random_seed": g("random_seed"),
        "batch_sizes": g("batch_sizes"),
    }


# ──────────────────────────────────────────────────────────────────────────
# save
# ──────────────────────────────────────────────────────────────────────────
def save_data(
    cfg,
    *,
    splits: dict,
    tickers: list,
    features_lr: list,
    features_cond: list,
    cyclical_idx: list,
    scale_idx: list,
    scalers_x: dict,
    scalers_cond: dict,
    save_dir=None,
    extra_meta: dict | None = None,
) -> Path:
    """
    splits : {"train": {"lr":..., "cond":..., "init_price":..., "dates":...}, "val": {...}, "test": {...}}
             lr         (N, W, A, C_x)  scaled log-return windows
             cond       (N, W, A, C_c)  scaled conditioning windows
             init_price (N, A, C_x)     actual price right before the window
             dates      (N,)            numpy datetime64 — last date of each window
    Returns the directory the data was saved to.
    """
    missing = [s for s in SPLITS if s not in splits]
    if missing:
        raise ValueError(f"splits is missing required keys: {missing}")

    save_dir = Path(save_dir) if save_dir is not None else _resolve_dir(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── meta.json ────────────────────────────────────────────────────────
    meta = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "config": _cfg_to_dict(cfg),
        "tickers": list(tickers),
        "features_lr": list(features_lr),
        "features_cond": list(features_cond),
        "cyclical_idx": list(cyclical_idx),
        "scale_idx": list(scale_idx),
        "shapes": {
            s: {
                "lr": list(splits[s]["lr"].shape),
                "cond": list(splits[s]["cond"].shape),
                "init_price": list(splits[s]["init_price"].shape),
                "n_windows": int(len(splits[s]["dates"])),
            }
            for s in SPLITS
        },
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("✓ meta.json")

    # ── scalers.pkl ──────────────────────────────────────────────────────
    with open(save_dir / "scalers.pkl", "wb") as f:
        pickle.dump({"scalers_x": scalers_x, "scalers_cond": scalers_cond}, f)
    print("✓ scalers.pkl")

    # ── windows.npz  (each split keeps its own length — no equal-length
    #    assumption, unlike a single wide DataFrame) ─────────────────────
    npz_payload = {}
    for s in SPLITS:
        npz_payload[f"{s}_lr"] = splits[s]["lr"]
        npz_payload[f"{s}_cond"] = splits[s]["cond"]
        npz_payload[f"{s}_init"] = splits[s]["init_price"]
        npz_payload[f"{s}_dates"] = np.asarray(
            splits[s]["dates"], dtype="datetime64[ns]"
        )
    np.savez_compressed(save_dir / "windows.npz", **npz_payload)
    print("✓ windows.npz")

    # ── window_dates.csv — long format, human-readable only ─────────────
    rows = []
    for s in SPLITS:
        for i, d in enumerate(splits[s]["dates"]):
            rows.append({"split": s, "window_idx": i, "date": pd.Timestamp(d)})
    pd.DataFrame(rows).to_csv(save_dir / "window_dates.csv", index=False)
    print("✓ window_dates.csv")

    # ── summary ──────────────────────────────────────────────────────────
    total_mb = sum(
        (save_dir / fname).stat().st_size / 1024**2
        for fname in ["meta.json", "scalers.pkl", "windows.npz", "window_dates.csv"]
    )
    print(f"\n── Saved to {save_dir}  ({total_mb:.1f} MB total) ──")
    for fname in ["meta.json", "scalers.pkl", "windows.npz", "window_dates.csv"]:
        print(f"  {fname:<20} {(save_dir / fname).stat().st_size / 1024**2:.2f} MB")

    return save_dir


# ──────────────────────────────────────────────────────────────────────────
# load
# ──────────────────────────────────────────────────────────────────────────
def load_data(cfg_or_dir) -> dict:
    """Load everything saved by save_data() back into memory.

    Accepts either the same `cfg` object used to save (reads `cfg.save_dir`)
    or a path/string pointing directly at the save directory — so it works
    fine from a different notebook that only has the path, not the cfg.
    """
    save_dir = _resolve_dir(cfg_or_dir)

    meta_path = save_dir / "meta.json"
    scalers_path = save_dir / "scalers.pkl"
    windows_path = save_dir / "windows.npz"
    for p in (meta_path, scalers_path, windows_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    with open(meta_path) as f:
        meta = json.load(f)

    with open(scalers_path, "rb") as f:
        scaler_bundle = pickle.load(f)

    with np.load(windows_path, allow_pickle=False) as npz:
        splits = {
            s: {
                "lr": npz[f"{s}_lr"],
                "cond": npz[f"{s}_cond"],
                "init_price": npz[f"{s}_init"],
                "dates": npz[f"{s}_dates"],
            }
            for s in SPLITS
        }

    return {
        "save_dir": save_dir,
        "meta": meta,
        "tickers": meta["tickers"],
        "features_lr": meta["features_lr"],
        "features_cond": meta["features_cond"],
        "cyclical_idx": meta["cyclical_idx"],
        "scale_idx": meta["scale_idx"],
        "scalers_x": scaler_bundle["scalers_x"],
        "scalers_cond": scaler_bundle["scalers_cond"],
        "splits": splits,
    }


# ──────────────────────────────────────────────────────────────────────────
# reconstruct Dataset / DataLoader
# ──────────────────────────────────────────────────────────────────────────
def build_datasets(loaded: dict) -> dict:
    """loaded = output of load_data(). Returns {"train": WindowDataset, "val": ..., "test": ...}"""
    tickers = loaded["tickers"]
    features = loaded["features_lr"]
    datasets = {}
    for s in SPLITS:
        sp = loaded["splits"][s]
        datasets[s] = WindowDataset(
            windows_lr=sp["lr"],
            windows_cond=sp["cond"],
            init_prices=sp["init_price"],
            dates=sp["dates"],
            tickers=tickers,
            features=features,
        )
    return datasets


def load_and_make_dataloaders(
    cfg_or_dir,
    batch_sizes: dict | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    One-call convenience, mirrors the load → make-dataloaders pattern:

        loaded, dataloaders = load_and_make_dataloaders(cfg)
        dl_train, dl_val, dl_test = dataloaders["train"], dataloaders["val"], dataloaders["test"]

    batch_sizes defaults to cfg.batch_sizes when cfg_or_dir is a config object
    that has that attribute; otherwise it must be passed explicitly.
    """
    loaded = load_data(cfg_or_dir)

    if batch_sizes is None:
        batch_sizes = getattr(cfg_or_dir, "batch_sizes", None)
    if batch_sizes is None:
        raise ValueError(
            "batch_sizes not provided and cfg_or_dir has no `.batch_sizes` attribute — "
            "pass batch_sizes={'train': .., 'val': .., 'test': ..} explicitly."
        )

    datasets = build_datasets(loaded)
    dataloaders = make_dataloaders(
        datasets, batch_sizes, num_workers=num_workers, pin_memory=pin_memory
    )
    return loaded, dataloaders


# ──────────────────────────────────────────────────────────────────────────
# sanity check
# ──────────────────────────────────────────────────────────────────────────
def verify_roundtrip(
    original_splits: dict, loaded_splits: dict, atol: float = 1e-6
) -> bool:
    """
    Compare the in-memory arrays you saved against what load_data() returned.
    original_splits / loaded_splits both look like:
        {"train": {"lr":..., "cond":..., "init_price":..., "dates":...}, "val": {...}, "test": {...}}
    Prints a per-split, per-array report and returns True iff everything matches.
    """
    ok = True
    print("── Round-trip verification ──────────────────────────────────────")
    for s in SPLITS:
        for key in ("lr", "cond", "init_price"):
            a = np.asarray(original_splits[s][key])
            b = np.asarray(loaded_splits[s][key])
            same_shape = a.shape == b.shape
            same_values = same_shape and np.allclose(a, b, atol=atol, equal_nan=True)
            status = "✓" if (same_shape and same_values) else "✗"
            if not (same_shape and same_values):
                ok = False
            print(
                f"  {status} {s:<5}.{key:<10} shape={b.shape}  match={same_shape and same_values}"
            )

        d_orig = np.asarray(original_splits[s]["dates"], dtype="datetime64[ns]")
        d_load = np.asarray(loaded_splits[s]["dates"], dtype="datetime64[ns]")
        dates_match = d_orig.shape == d_load.shape and np.array_equal(d_orig, d_load)
        if not dates_match:
            ok = False
        print(
            f"  {'✓' if dates_match else '✗'} {s:<5}.dates      shape={d_load.shape}  match={dates_match}"
        )

    print(f"\n{'✓ ALL MATCH' if ok else '✗ MISMATCH FOUND'}")
    return ok
