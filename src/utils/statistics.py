import torch

def monte_carlo_statistic(log_returns: torch.Tensor, n_sims: int, steps: int, dt: float = 1.0):
    # log_returns shape: [Length, Assets]

    device = log_returns.device
    num_assets = log_returns.shape[-1]

    
    mu = log_returns.mean(dim=0) # [length, Assets]
    sigma = log_returns.std(dim=0) # [length, Assets]

    # Shape should output: [Batch, N_sim, Future_Steps, Assets]
    
    mu = mu.view(1, 1, -1)   # [ 1, 1, Assets]
    sigma = sigma.view(1, 1, -1) # [1, 1, Assets]


    # Random Noise [N_sim, Future_Steps, Assets]
    Z = torch.randn(n_sims, steps, num_assets, device=device)

    # GBM (Geometric Brownian Motion)
    drift_term = (mu - 0.5 * sigma**2) * dt
    diffusion_term = sigma * torch.sqrt(torch.tensor(dt, device=device)) * Z

    daily_log_returns = drift_term + diffusion_term

    return daily_log_returns # [N_sim, Steps, Assets]
