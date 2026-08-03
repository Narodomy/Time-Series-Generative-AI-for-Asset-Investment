import numpy as np
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view


class MarketDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        cond: np.ndarray,
        date: np.ndarray,
        price: np.ndarray,
        window_size: int,
        stride: int = 1,
        normalize_window: bool = True,
        normalize_shift: int = 0,
        prefix_x: np.ndarray = None,
        eps: float = 1e-6,
    ):

        self.x = x  # [Length, Assets, Features]
        self.cond = cond  # [Length, Assets, Features_cond]
        self.price = price  # [Length, Assets]
        self.date = date  # [Length]
        self.window_size = window_size
        self.stride = stride
        self.normalize_window = normalize_window
        self.normalize_shift = normalize_shift
        self.eps = eps

        # Shape which we want: [Num_windows as Batchs, Features as Channels, Length, Assets]

        windows_x = sliding_window_view(self.x, window_shape=window_size, axis=0)[
            :: self.stride
        ]  # [Num_Window, Assets, Features, Window_Size]
        windows_cond = sliding_window_view(self.cond, window_shape=window_size, axis=0)[
            :: self.stride
        ]  # [Num_Window, Assets, Features_cond, Window_Size]
        windows_date = sliding_window_view(self.date, window_shape=window_size, axis=0)[
            :: self.stride
        ]  # [Num_Window, Window_Size]
        windows_price = sliding_window_view(
            self.price, window_shape=window_size, axis=0
        )[
            :: self.stride
        ]  # [Num_Window, Assets, Feature price, Window_Size]

        # print(f"Windows_x: {windows_x.shape}")
        # print(f"Windows_cond: {windows_cond.shape}")
        # print(f"Windows_date: {windows_date.shape}")
        # print(f"Windows_price: {windows_price.shape}")

        self.windows_x = windows_x.transpose(
            0, 2, 3, 1
        )  # [Num_windows, Features, Window_size, Assets]
        self.windows_cond = windows_cond.transpose(
            0, 2, 3, 1
        )  # [Num_windows, Features, Window_size, Assets]
        self.windows_date = windows_date
        self.windows_price = windows_price.transpose(
            0, 2, 1
        )  # [Num_Windows, Window_Size, Assets]

        if prefix_x is not None and normalize_shift > 0:
            prefix_windows = sliding_window_view(
                prefix_x, window_shape=window_size, axis=0
            )[::stride]
            self.prefix_windows_x = prefix_windows.transpose(0, 2, 3, 1)[
                -normalize_shift:
            ]
        else:
            self.prefix_windows_x = None

    def __len__(self):
        n_windows, n_features, window_size, n_assets = self.windows_x.shape
        return n_windows

    def __getitem__(self, idx: int):
        window_x = self.windows_x[idx].copy().astype(np.float32)
        window_cond = self.windows_cond[idx].copy().astype(np.float32)
        window_date = self.windows_date[idx].copy().astype(np.int64)
        window_price = self.windows_price[idx].copy().astype(np.float32)

        x_mean = np.zeros_like(window_x)
        x_std = np.ones_like(window_x)

        # Per-Window Scaling
        if self.normalize_window:
            if self.normalize_shift > 0:
                if idx >= self.normalize_shift and self.prefix_windows_x is not None:
                    stat_window = self.windows_x[idx - self.normalize_shift].astype(
                        np.float32
                    )
                elif self.prefix_windows_x is not None:
                    # Use prefix from previous split
                    prefix_idx = max(
                        0, len(self.prefix_windows_x) - self.normalize_shift + idx
                    )
                    stat_window = self.prefix_windows_x[prefix_idx].astype(np.float32)
                else:
                    stat_window = window_x  # fallback

            mean = np.mean(stat_window, axis=1, keepdims=True)
            std = np.std(stat_window, axis=1, keepdims=True)

            # Makesure that's std not equal 0
            std = np.maximum(std, self.eps)

            # Scaling Equation: (x - mean) / std
            window_x = (window_x - mean) / std

            x_mean = mean
            x_std = std

        return {
            "x": window_x,
            "cond": window_cond,
            "date": window_date,
            "price": window_price,
            "x_mean": x_mean,  # For Inverse Transform
            "x_std": x_std,  # For Inverse Transform
        }
