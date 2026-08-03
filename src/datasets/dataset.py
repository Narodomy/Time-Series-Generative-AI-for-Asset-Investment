import numpy as np
from typing import Annotated, Optional
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view


T = int  # Length
N = int  # Num_windows
A = int  # Assets
F = int  # Features
W = int  # Window_size


class MarketDataset(Dataset):
    def __init__(
        self,
        x: Annotated[np.ndarray, "(T, A, F)"],
        cond: Annotated[np.ndarray, "(T, A, F_cond)"],
        dates: Annotated[np.ndarray, "(T,)"],
        close_prices: Annotated[np.ndarray, "(T, A)"],
        window_size: int,
        stride: int = 1,
        normalize_window: bool = True,
        eps: float = 1e-6,
        # --- precomputed from feature store (optional) ---
        # x_prev      : window of x from the *previous* timestep  [T, A, F]
        # x_prev_mean : mean(x) of the *previous* window, used for inverse-transform  [T, A, F]
        # x_prev_std  : std(x)  of the *previous* window, used for inverse-transform  [T, A, F]
        #
        # Naming convention:
        #   "prev" = computed on window[i-1], applied to window[i]
        #   This avoids leaking future stats when the model generates & inverse-transforms.
        x_prev: Optional[np.ndarray] = None,  # [T, A, F]
        x_prev_mean: Optional[
            np.ndarray
        ] = None,  # [T, A, F]  (squeezed, not [T,A,1,F])
        x_prev_std: Optional[np.ndarray] = None,  # [T, A, F]
        # ---- backward-compat aliases (ignored if the prev_ names are given) ----
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.normalize_window = normalize_window
        self.eps = eps

        # ── Sliding windows ──────────────────────────────────────────────
        self.windows_x = sliding_window_view(x, window_shape=window_size, axis=0)[
            ::stride
        ].transpose(
            0, 1, 3, 2
        )  # [N, A, T, F]

        self.windows_cond = sliding_window_view(cond, window_shape=window_size, axis=0)[
            ::stride
        ].transpose(
            0, 1, 3, 2
        )  # [N, A, T, F_cond]

        self.windows_dates = sliding_window_view(
            dates, window_shape=window_size, axis=0
        )[
            ::stride
        ]  # [N, T]

        self.windows_close_prices = sliding_window_view(
            close_prices, window_shape=window_size, axis=0
        )[
            ::stride
        ]  # [N, T, A]  ← sliding on axis=0

        # ── x_prev windows ───────────────────────────────────────────────
        # x_prev can arrive in two shapes:
        #   raw timeseries  [T, A, F]     → need sliding_window_view
        #   precomputed     [N, A, T, F]  → store directly (from compute_prev_stats)
        if x_prev is not None:
            self.windows_x_prev = np.asarray(x_prev, dtype=np.float32)  # [N, A, W, F]
        else:
            self.windows_x_prev = None

        # ── prev-window normalisation stats ──────────────────────────────
        # Priority: x_prev_mean/x_prev_std > x_mean/x_std (compat) > recompute
        #
        # When loaded from a feature store these are already shifted by 1
        # (i.e. row[i] = stats of window[i-1]), so __getitem__ just indexes
        # directly without the [idx-1] trick.  The first row is NaN → drop
        # idx=0 with Subset as usual.
        #
        # When NOT provided, we fall back to computing mean/std per window
        # on the fly inside __getitem__ (leaky but acceptable for quick runs).
        if normalize_window:
            _pm = x_prev_mean if x_prev_mean is not None else x_mean
            _ps = x_prev_std if x_prev_std is not None else x_std

            if _pm is not None and _ps is not None:
                # Precomputed: shape [N_windows, A, F] — already aligned to windows,
                # already shifted by 1 (row[i] = stats of window[i-1]).
                # Just store directly, no sliding_window_view needed.
                self._prev_mean = np.asarray(_pm, dtype=np.float32)  # [N, A, F]
                self._prev_std = np.asarray(_ps, dtype=np.float32)  # [N, A, F]
            else:
                # Fallback: compute from each window (current-window stats,
                # leaky — only for development / quick sanity runs).
                # Shape: [N, A, 1, F]
                self._prev_mean = self.windows_x.mean(axis=2, keepdims=True)
                self._prev_std = np.maximum(
                    self.windows_x.std(axis=2, keepdims=True), eps
                )

    # ─────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return self.windows_x.shape[0]

    def __getitem__(self, idx: int) -> dict:
        x = self.windows_x[idx].copy().astype(np.float32)  # [A, T, F]
        cond = self.windows_cond[idx].copy().astype(np.float32)  # [A, T, F_cond]
        dates = self.windows_dates[idx].copy().astype(np.int64)  # [T]
        close_prices = (
            self.windows_close_prices[idx].copy().astype(np.float32)
        )  # [T, A]

        # ── x_prev ───────────────────────────────────────────────────────
        if self.windows_x_prev is not None:
            x_prev = self.windows_x_prev[idx].copy().astype(np.float32)  # [A, T, F]
        elif idx == 0:
            x_prev = np.zeros_like(x)
        else:
            x_prev = self.windows_x[idx - 1].copy().astype(np.float32)

        # ── Normalisation ─────────────────────────────────────────────────
        if self.normalize_window:
            if hasattr(self, "_prev_mean"):
                # _prev_mean shape:
                #   precomputed → [N, A, F]  (per-window scalar, already shifted)
                #   fallback    → [N, A, 1, F]
                raw_mean = self._prev_mean[idx]  # [A, F] or [A, 1, F]
                raw_std = self._prev_std[idx]

                # Ensure [A, 1, F] for broadcasting against x [A, T, F]
                if raw_mean.ndim == 2:  # [A, F] → add keepdim
                    mean = raw_mean[:, np.newaxis, :].astype(np.float32)  # [A, 1, F]
                    std = raw_std[:, np.newaxis, :].astype(np.float32)
                else:  # already [A, 1, F]
                    mean = raw_mean[:, :1, :].astype(np.float32)
                    std = raw_std[:, :1, :].astype(np.float32)
            else:
                mean = x.mean(axis=1, keepdims=True)
                std = np.maximum(x.std(axis=1, keepdims=True), self.eps)

            x = (x - mean) / std
        else:
            mean = np.zeros((x.shape[0], 1, x.shape[2]), dtype=np.float32)
            std = np.ones((x.shape[0], 1, x.shape[2]), dtype=np.float32)

        return {
            "x": x,  # [A, T, F]
            "cond": cond,  # [A, T, F_cond]
            "dates": dates,  # [T]
            "close_prices": close_prices,  # [T, A]
            "x_prev_mean": mean.squeeze(1),  # [A, F]  ← renamed from x_mean
            "x_prev_std": std.squeeze(1),  # [A, F]  ← renamed from x_std
            "x_prev": x_prev,  # [A, T, F]
        }
