from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


# ──────────────────────────────────────────────────────────────────────────
# Annual-seasonal scaler (custom — not from sklearn)
# ──────────────────────────────────────────────────────────────────────────
class AnnualSeasonalScaler:
    """
    Annual Seasonal Window Scaling

    ตาม methodology:
      - แบ่งข้อมูล training ตามปี  y ∈ Y
      - ภายในแต่ละปี ทำ sliding window ขนาด T (= window_size) ด้วย stride S=1
        ได้ window index  i = 0, 1, …, W_y−1
        โดยที่  W_y = (rows_in_year − window_size) // stride + 1
      - คำนวณ μ_{y,i}, σ_{y,i}  จาก T rows ใน window นั้น
      - รวมข้าม years:  μ*_i = mean_y(μ_{y,i}),  σ*_i = mean_y(σ_{y,i})

    transform / inverse_transform รับ win_indices (N*W,) ซึ่งบอกว่าแต่ละ row
    อยู่ที่ seasonal position ไหน  → ดึง μ*, σ* ตาม index นั้น (modulo wrap)
    """

    def __init__(self, stride: int = 1):
        self.stride = stride
        self.window_size = None
        self.n_win_per_year = None  # max window slots across years
        self.mu_star = None  # (n_win_per_year, C)
        self.sig_star = None  # (n_win_per_year, C)

    # ──────────────────────────────────────────────────────────────────────
    def fit(
        self, X: np.ndarray, dates: pd.DatetimeIndex, window_size: int
    ) -> "AnnualSeasonalScaler":
        """
        X     : (T, C)  — flattened training tensor สำหรับ 1 asset
        dates : (T,)    — DatetimeIndex ที่ align กับ X
        window_size : int  — T ใน methodology (= cfg.window_size)
        """
        T, C = X.shape
        W = window_size
        S = self.stride
        self.window_size = W

        years = np.array(dates.year)
        unique_years = sorted(set(years))

        # หา n_win_per_year = max W_y across all training years
        # W_y = (rows_in_year - W) // S + 1
        max_n_windows = 0
        for y in unique_years:
            rows_y = int((years == y).sum())
            n_win_y = max(0, (rows_y - W) // S + 1)
            max_n_windows = max(max_n_windows, n_win_y)

        self.n_win_per_year = max_n_windows

        mu_accum = np.zeros((max_n_windows, C), dtype=np.float64)
        sig_accum = np.zeros((max_n_windows, C), dtype=np.float64)
        count = np.zeros(max_n_windows, dtype=np.int64)

        for y in unique_years:
            mask = years == y
            X_y = X[mask]  # (rows_y, C)
            rows_y = len(X_y)
            n_win_y = max(0, (rows_y - W) // S + 1)

            # sliding window ภายในปี  → localized μ_{y,i}, σ_{y,i}
            for i in range(n_win_y):
                start = i * S
                win_data = X_y[start : start + W]  # (W, C)
                mu_yi = win_data.mean(axis=0)  # (C,)
                sig_yi = win_data.std(axis=0)  # (C,)  ddof=0 ตาม paper
                mu_accum[i] += mu_yi
                sig_accum[i] += sig_yi
                count[i] += 1

        # average across years  (count[i] = |Y| สำหรับ i ที่มีข้อมูลครบ)
        safe_count = np.maximum(count[:, None], 1)
        self.mu_star = (mu_accum / safe_count).astype(np.float32)
        self.sig_star = (sig_accum / safe_count).astype(np.float32)
        return self

    # ──────────────────────────────────────────────────────────────────────
    def transform(self, X: np.ndarray, win_indices: np.ndarray) -> np.ndarray:
        """
        X           : (N*W, C)
        win_indices : (N*W,) int  — seasonal position ของแต่ละ row
        Returns     : (N*W, C) scaled
        """
        idx = win_indices % self.n_win_per_year
        mu = self.mu_star[idx]  # (N*W, C)
        sig = self.sig_star[idx]  # (N*W, C)
        return (X - mu) / (sig + 1e-8)

    def inverse_transform(
        self, X_scaled: np.ndarray, win_indices: np.ndarray
    ) -> np.ndarray:
        """
        X_scaled    : (N*W, C)
        win_indices : (N*W,) int
        Returns     : (N*W, C) original scale
        """
        idx = win_indices % self.n_win_per_year
        mu = self.mu_star[idx]
        sig = self.sig_star[idx]
        return X_scaled * (sig + 1e-8) + mu


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────
def create_scaler(scaler_type: str | None, stride: int = 1):
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "robust":
        return RobustScaler()
    elif scaler_type == "minmax":
        return MinMaxScaler()
    elif scaler_type == "annual_seasonal":
        return AnnualSeasonalScaler(stride=stride)
    elif scaler_type is None:
        return None
    raise ValueError(f"Unknown scaler type: {scaler_type}")


# ──────────────────────────────────────────────────────────────────────────
# x  (log-return windows) — transform / inverse on (N, W, A, C)
# ──────────────────────────────────────────────────────────────────────────
def transform_windows_x(
    windows: np.ndarray,  # (N, W, A, C)
    scalers: dict,  # {ticker: scaler}
    tickers: list[str],
    start_indices: np.ndarray | None = None,  # (N,) required for AnnualSeasonalScaler
) -> np.ndarray:
    """
    Scale windows (N, W, A, C) → (N, W, A, C)

    start_indices : seasonal window index ของ row แรกของแต่ละ window
                    ได้จาก compute_start_indices()
                    จำเป็นเฉพาะเมื่อใช้ AnnualSeasonalScaler
    """
    N, W, A, C = windows.shape
    out = windows.copy()

    for a, ticker in enumerate(tickers):
        flat = windows[:, :, a, :].reshape(N * W, C)  # (N*W, C)
        scaler = scalers[ticker]

        if isinstance(scaler, AnnualSeasonalScaler):
            if start_indices is None:
                raise ValueError("start_indices required for AnnualSeasonalScaler")
            # row k ใน flat มาจาก window i=k//W, position j=k%W
            # seasonal index ของ row นั้น = start_indices[i] + j
            i_idx = np.arange(N * W) // W  # window index
            j_idx = np.arange(N * W) % W  # position ภายใน window
            win_indices = start_indices[i_idx] + j_idx  # (N*W,)
            out[:, :, a, :] = scaler.transform(flat, win_indices).reshape(N, W, C)

        elif scaler is not None:
            out[:, :, a, :] = scaler.transform(flat).reshape(N, W, C)
        # scaler is None → passthrough

    return out


def inverse_transform_windows_x(
    windows_scaled: np.ndarray,  # (N, W, A, C)
    scalers: dict,
    tickers: list[str],
    start_indices: np.ndarray | None = None,  # (N,)
) -> np.ndarray:
    """
    Inverse-scale windows (N, W, A, C) scaled → (N, W, A, C) original scale
    """
    N, W, A, C = windows_scaled.shape
    out = windows_scaled.copy()

    for a, ticker in enumerate(tickers):
        flat = windows_scaled[:, :, a, :].reshape(N * W, C)
        scaler = scalers[ticker]

        if isinstance(scaler, AnnualSeasonalScaler):
            if start_indices is None:
                raise ValueError("start_indices required for AnnualSeasonalScaler")
            i_idx = np.arange(N * W) // W
            j_idx = np.arange(N * W) % W
            win_indices = start_indices[i_idx] + j_idx
            out[:, :, a, :] = scaler.inverse_transform(flat, win_indices).reshape(
                N, W, C
            )

        elif scaler is not None:
            out[:, :, a, :] = scaler.inverse_transform(flat).reshape(N, W, C)

    return out


# ──────────────────────────────────────────────────────────────────────────
# cond  (conditioning windows) — transform / inverse on (N, W, A, C_c)
# cyclical channels (sin/cos) always pass through untouched.
# ──────────────────────────────────────────────────────────────────────────
def transform_windows_cond(
    windows: np.ndarray,  # (N, W, A, C_c)
    scalers: dict,
    tickers: list,
    scale_idx: list,
    cyclical_idx: list,
) -> np.ndarray:
    """Scale non-cyclical channels, pass cyclical channels through unchanged."""
    N, W, A, C = windows.shape
    out = windows.copy()

    for a, ticker in enumerate(tickers):
        flat = windows[:, :, a, :].reshape(N * W, C)
        scaler = scalers[ticker]

        if scaler is not None and scale_idx:
            scaled = scaler.transform(flat[:, scale_idx])
            tmp = flat.copy()
            tmp[:, scale_idx] = scaled
            out[:, :, a, :] = tmp.reshape(N, W, C)
        # cyclical_idx → ไม่ทำอะไร ค่าเดิมถูกต้องแล้ว

    return out


def inverse_transform_windows_cond(
    windows_scaled: np.ndarray,  # (N, W, A, C_c)
    scalers: dict,
    tickers: list,
    scale_idx: list,
    cyclical_idx: list,
) -> np.ndarray:
    """Inverse of transform_windows_cond — reconstruct original-scale conditioning values."""
    N, W, A, C = windows_scaled.shape
    out = windows_scaled.copy()

    for a, ticker in enumerate(tickers):
        flat = windows_scaled[:, :, a, :].reshape(N * W, C)
        scaler = scalers[ticker]

        if scaler is not None and scale_idx:
            unscaled = scaler.inverse_transform(flat[:, scale_idx])
            tmp = flat.copy()
            tmp[:, scale_idx] = unscaled
            out[:, :, a, :] = tmp.reshape(N, W, C)
        # cyclical_idx → ไม่ทำอะไร ค่าเดิมถูกต้องแล้ว

    return out


def compute_start_indices(
    dates: pd.DatetimeIndex,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """
    คำนวณ seasonal window index สำหรับแต่ละ window ใน split นี้

    "seasonal index ของ window i" = ตำแหน่ง sliding-window ภายในปี
    ของ *วันแรก* ของ window นั้น

    Parameters
    ----------
    dates       : DatetimeIndex ของ df_split_lr_aligned  (T rows)
    window_size : cfg.window_size
    stride      : cfg.window_stride

    Returns
    -------
    start_indices : np.ndarray shape (N,) dtype int
        N = (T - window_size) // stride + 1
        start_indices[i] = seasonal position ของ window i
                         = index ภายในปีของ *row แรก* ของ window i
    """
    T = len(dates)
    N = (T - window_size) // stride + 1

    years = np.array(dates.year)
    # สำหรับแต่ละ date ใน series นี้ คำนวณ "row index ภายในปี"
    # (row_in_year[t] = ลำดับที่ t อยู่ภายในปีของ t นับจาก 0)
    row_in_year = np.zeros(T, dtype=np.int64)
    for y in sorted(set(years)):
        mask = years == y
        row_in_year[mask] = np.arange(mask.sum())

    # first_row[i] = row ใน series ที่เป็นจุดเริ่มต้นของ window i
    first_rows = np.arange(N) * stride  # (N,)

    # seasonal index = row_in_year ของ first row นั้น หารด้วย stride
    # เพื่อให้ map ไปยัง window slot ภายในปีที่ถูกต้อง
    start_indices = row_in_year[first_rows] // stride  # (N,)

    return start_indices.astype(np.int64)
