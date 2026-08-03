# Time-Series Generative AI for Asset Investment

A **Conditional Denoising Diffusion Probabilistic Model (DDPM)**, driven by a causal Transformer with cross-attention over technical/cyclical features, trained to learn the conditional distribution of forward asset returns — with a **Flow Matching** variant implemented as an ablation. Generated path ensembles feed a **Modern Portfolio Theory (MPT)** Layer 2 optimizer to construct and backtest portfolios.

Evaluated on the **S&P 100** (primary) and **SET50, Thailand** (secondary), 2010–2025.

> Master's thesis codebase — Narodom Yatnimit, M.Eng. (Artificial Intelligence and Internet of Things), Dept. of Electrical Engineering, Kasetsart University, AY2025.
> Advisor: Assoc. Prof. Teerasit Kasetkasem, Ph.D. · Co-advisors: Assoc. Prof. Kotaro Funakoshi, Dr.Eng. · Dr. Nuttapong Sanglerdsinlapachai (NSTDA)

## Framework

- **Layer 1 — Path generation**: `DiffusionTransformer` (DDPM) trained with a `DDIM` sampler for fast inference; `FlowMatching` implemented as a comparative ablation. Conditioned on a 14-indicator set spanning trend, momentum, volatility, and volume.
- **Layer 2 — Path aggregation & decision**: the simulated path ensemble is fed into a Sharpe-ratio-maximizing MPT allocator, swept across a grid of rebalancing frequencies and optimization windows.

## Repository structure

```
src/
├── config/       # Dataclass configs: data engineering, modeling (Optuna search space), inferencing
├── datasets/     # MarketDataset — sliding-window (B, W, A, C) tensor construction
├── models/       # DiffusionTransformer, DDIM, FlowMatching, UNet, DiffusionLSTM
├── engine/       # Engine — training loop, checkpointing, loss/metric plots
├── entities/     # Portfolio — MPT optimization, backtesting, performance metrics
└── utils/        # scaling (AnnualSeasonalScaler), feature engineering, I/O, statistics, visualization

03_notebooks/     # v4-* pipeline: data engineering → modeling → inferencing → portfolio backtest
```

## Installation

```bash
pip install -e ".[data,viz,finance,utils]"
```

Optional dependency groups are defined in `pyproject.toml`. Add `dev` if working with notebooks (`pip install -e ".[dev]"`).

> `finance` includes `ta-lib`, which needs the TA-Lib C library installed on your system first.

## Pipeline

Run in order from `03_notebooks/`:

1. **`v4-1-0_data_engineering.ipynb`** — asset universe construction, technical indicator features, train/val/test split
2. **`v4-2-1_modeling.ipynb`** — Optuna hyperparameter search; trains the DDPM-Transformer / Flow Matching models
3. **`v4-3-0_inferencing.ipynb`** — DDIM sampling, decoding generated paths back to price space
4. **`v4-4-2_portfolio.ipynb`** — Layer 2 MPT backtest across the rebalancing-frequency / optimization-window grid
5. **`V4-5-0_stats.ipynb`** — distributional fidelity checks (Wasserstein/KS distance, skewness/kurtosis, correlation)

## Results (summary)

Best-case S&P 100 configuration: **283.48%** cumulative return, **1.25** Sharpe ratio — vs. **59.58%**/0.70 (GBM baseline) and **2.31%**/0.04 (Equal-Weight baseline). SET50 results are more mixed and sensitive to market scale/structure. Full analysis and discussion in the thesis document.

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE).

## Citation

```
Yatnimit, N. (2025). Time-Series Generative AI for Asset Investment.
Master's Thesis, Kasetsart University.
```