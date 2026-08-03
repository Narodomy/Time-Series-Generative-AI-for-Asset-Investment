import torch
import numpy as np


def simulate_mc_gbm(
    log_returns: torch.Tensor, n_paths: int, horizon: int, dt: float = 1.0
):
    """
    Simulate GBM paths via Monte Carlo with Cholesky-correlated noise.

    Parameters
    ----------
    log_returns : torch.Tensor  [Length, Assets]
        Historical log returns used to estimate mu and sigma per asset.
    n_paths : int
        Number of Monte Carlo paths to simulate.
    horizon : int
        Number of time steps to simulate (e.g. window_size = 30).
    dt : float
        Time step size. 1.0 = daily, 1/252 = annualised daily, etc.

    Returns
    -------
    torch.Tensor  [n_paths, horizon, Assets]
        Simulated log returns.
    """
    device = log_returns.device
    num_assets = log_returns.shape[-1]

    mu = log_returns.mean(dim=0)  # [Assets]
    sigma = log_returns.std(dim=0)  # [Assets]

    # --- Cholesky decomposition to inject inter-asset correlations ---
    # Without this, each asset is simulated independently (zero correlation),
    # which underestimates portfolio risk for correlated assets like tech stocks.
    lr_np = log_returns.detach().cpu().numpy()
    cov_np = np.cov(lr_np, rowvar=False)  # [Assets, Assets]

    # Add small jitter for numerical stability (prevents non-positive-definite issues)
    cov_np += np.eye(num_assets) * 1e-8

    cov = torch.tensor(cov_np, dtype=torch.float32, device=device)
    L = torch.linalg.cholesky(cov)  # [Assets, Assets], lower triangular

    mu = mu.view(1, 1, -1)  # [1, 1, Assets]
    sigma = sigma.view(1, 1, -1)  # [1, 1, Assets]

    # Independent standard normal noise [n_paths, horizon, Assets]
    Z_indep = torch.randn(n_paths, horizon, num_assets, device=device)

    # Inject correlation: Z_corr[i,j] ~ N(0, Sigma)
    # Z_corr = Z_indep @ L.T  →  Cov(Z_corr) = L @ I @ L.T = Sigma
    # L already encodes per-asset variance, so no need to multiply by sigma again
    Z = Z_indep @ L.T  # [n_paths, horizon, Assets]

    # GBM (Geometric Brownian Motion)
    drift_term = (mu - 0.5 * sigma**2) * dt
    diffusion_term = torch.sqrt(torch.tensor(dt, device=device)) * Z

    daily_log_returns = drift_term + diffusion_term

    return daily_log_returns  # [n_paths, horizon, Assets]


def monte_carlo_statistic(
    log_returns: torch.Tensor, n_sims: int, steps: int, dt: float = 1.0
):
    # log_returns shape: [Length, Assets]

    device = log_returns.device
    num_assets = log_returns.shape[-1]

    mu = log_returns.mean(dim=0)  # [Assets]
    sigma = log_returns.std(dim=0)  # [Assets]

    # --- Cholesky decomposition to inject inter-asset correlations ---
    # Without this, each asset is simulated independently (zero correlation),
    # which underestimates portfolio risk for correlated assets like tech stocks.
    lr_np = log_returns.detach().cpu().numpy()
    cov_np = np.cov(lr_np, rowvar=False)  # [Assets, Assets]

    # Add small jitter for numerical stability (prevents non-positive-definite issues)
    cov_np += np.eye(num_assets) * 1e-8

    cov = torch.tensor(cov_np, dtype=torch.float32, device=device)
    L = torch.linalg.cholesky(cov)  # [Assets, Assets], lower triangular

    mu = mu.view(1, 1, -1)  # [1, 1, Assets]
    sigma = sigma.view(1, 1, -1)  # [1, 1, Assets]

    # Independent standard normal noise [N_sim, Steps, Assets]
    Z_indep = torch.randn(n_sims, steps, num_assets, device=device)

    # Inject correlation: Z_corr[i,j] ~ N(0, Sigma)
    # Z_corr = Z_indep @ L.T  →  Cov(Z_corr) = L @ I @ L.T = Sigma
    # L already encodes per-asset variance, so no need to multiply by sigma again
    Z = Z_indep @ L.T  # [N_sim, Steps, Assets]

    # GBM (Geometric Brownian Motion)
    drift_term = (mu - 0.5 * sigma**2) * dt
    diffusion_term = torch.sqrt(torch.tensor(dt, device=device)) * Z

    daily_log_returns = drift_term + diffusion_term

    return daily_log_returns  # [N_sim, Steps, Assets]


def calc_expected_returns(data: np.ndarray) -> np.ndarray:
    """
    คำนวณ Expected Return (Mu)
    Input: data รูปแบบ (Steps, Assets)
    Output: Mu รูปแบบ (Assets,)
    """
    return np.mean(data, axis=0)


def calc_covariance(data: np.ndarray) -> np.ndarray:
    """
    คำนวณ Covariance Matrix (Sigma)
    Input: data รูปแบบ (Steps, Assets)
    Output: Sigma รูปแบบ (Assets, Assets)
    """
    return np.cov(data, rowvar=False)


def calc_volatility(data: np.ndarray) -> np.ndarray:
    """
    คำนวณ Volatility (Standard Deviation) เผื่อใช้ในขั้นตอนอื่น
    Input: data รูปแบบ (Steps, Assets)
    Output: Volatility รูปแบบ (Assets,)
    """
    return np.std(data, axis=0)
