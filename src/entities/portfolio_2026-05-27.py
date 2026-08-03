import numpy as np
import pandas as pd
import logging
import os
from datetime import date
from typing import Annotated, Tuple, Optional, List, Literal
import vectorbt as vbt

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pypfopt import EfficientFrontier, plotting
from scipy.optimize import minimize, LinearConstraint, Bounds
import quantstats as qs

from utils import inverse_log_returns
from utils.paths import REPORTS_QS_DIR
from dataclasses import dataclass

logger = logging.getLogger(__name__)

RebalanceSchedule = list[
    tuple[int, int, Annotated[np.ndarray, ("n_assets",)]]
]  # List of (start_idx, end_idx, weights) for each rebalance period.

Strategies = dict[
    str, Annotated[np.ndarray, ("n_paths", "n_lengths", "n_assets")]
]  # Dict of strategy name to MC returns.


@dataclass
class PortfolioResult:
    returns: pd.Series
    portfolio: vbt.Portfolio
    weights: pd.DataFrame  # [n_rebalance, n_assets]
    sigmas: list[np.ndarray]  # [n_rebalance, n_assets, n_assets]
    mus: pd.DataFrame  # [n_rebalance, n_assets]


class Portfolio:
    def __init__(self, annual_risk_free_rate: float, trading_days: int = 252):
        self.annual_risk_free_rate = annual_risk_free_rate
        self.trading_days = trading_days

    @property
    def daily_risk_free_rate(self) -> float:
        return (1 + self.annual_risk_free_rate) ** (1 / self.trading_days) - 1
        # return self.annual_risk_free_rate / self.trading_days

    def calc_mu(self, returns: np.ndarray) -> np.ndarray:
        return np.mean(returns, axis=0)  # mu: (N,) expected returns.

    def calc_sigma(self, returns: np.ndarray) -> np.ndarray:
        return np.cov(returns.T)  # sigma: (N, N) covariance matrix.

    # Empirical Distribution from Monte Carlo Simulation
    def calc_mc_dist(
        self,
        mc_returns: Annotated[np.ndarray, ("n_paths", "n_lengths", "n_assets")],
        method: Literal["Per-Path", "Pooled"],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if method == "Per-Path":
            # Calculate mu and sigma for each path, then average them.
            mus = np.array(
                [self.calc_mu(path) for path in mc_returns]
            )  # (n_paths, n_assets)
            sigmas = np.array(
                [self.calc_sigma(path) for path in mc_returns]
            )  # (n_paths, n_assets, n_assets)

            # Average mu and sigma across paths.
            mu = np.mean(mus, axis=0)  # (n_assets,)
            sigma = np.mean(sigmas, axis=0)  # (n_assets, n_assets)

        elif method == "Pooled":
            # Pool all returns together and calculate mu and sigma once.
            pooled_returns = mc_returns.reshape(-1, mc_returns.shape[-1])
            mu = self.calc_mu(pooled_returns)
            sigma = self.calc_sigma(pooled_returns)

        else:
            raise ValueError(f"Invalid method: {method}.")

        return mu, sigma

    def optimize(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        daily_risk_free_rate: Optional[float] = None,
        solver: Literal["scipy", "pypfopt"] = "scipy",
    ) -> Annotated[np.ndarray, ("n_assets",)]:
        if daily_risk_free_rate is None:
            daily_risk_free_rate = self.daily_risk_free_rate

        if solver == "scipy":
            # Define the objective function (negative Sharpe ratio).
            def objective(weights):
                portfolio_return = np.dot(weights, mu)
                portfolio_volatility = np.sqrt(weights.T @ sigma @ weights)
                sharpe_ratio = (
                    portfolio_return - daily_risk_free_rate
                ) / portfolio_volatility
                return -sharpe_ratio

            # Constraints: weights sum to 1, and weights are between 0 and 1.
            constraint = LinearConstraint(
                np.ones(len(mu)), lb=1.0, ub=1.0
            )  # sum of weights = 1
            bounds = Bounds(0, 1)  # weights between 0 and 1
            # Initial guess (equal weights).
            initial_weights = np.ones(len(mu)) / len(mu)
            # Optimize using SLSQP method.
            result = minimize(
                objective,
                x0=initial_weights,
                method="SLSQP",
                constraints=constraint,
                bounds=bounds,
                options={"disp": False},
            )

            if not result.success:
                logger.warning(f"Optimization failed: {result.message}")
                return initial_weights  # Return equal weights as fallback.

            return result.x  # Optimal weights.

        elif solver == "pypfopt":
            ef = EfficientFrontier(mu, sigma, weight_bounds=(0, 1))
            weights = ef.max_sharpe(risk_free_rate=daily_risk_free_rate)
            return np.array(list(weights.values()))
        else:
            raise ValueError(f"Invalid solver: {solver}.")

    def optimize_window(
        self,
        mc_returns: Annotated[np.ndarray, ("n_paths", "n_lengths", "n_assets")],
        optimize_start: int,
        optimize_end: int,
        method: Literal["Per-Path", "Pooled"] = "Per-Path",
    ) -> Annotated[np.ndarray, ("n_assets")]:
        window = mc_returns[:, optimize_start:optimize_end, :]
        mu, sigma = self.calc_mc_dist(window, method=method)
        weights = self.optimize(mu, sigma)
        return weights, mu, sigma

    def build_rebalance_window(
        self,
        n_periods: int,
        rebalance_freq: int,
        optimize_window: int,
    ) -> list[tuple[int, int, int, int]]:

        windows = []
        for t_start in range(0, n_periods, rebalance_freq):
            t_end = min(t_start + rebalance_freq, n_periods)
            optimize_start = t_start
            optimize_end = min(t_start + optimize_window, n_periods)
            windows.append((t_start, t_end, optimize_start, optimize_end))
        # List of (t_start, t_end, optimize_start, optimize_end) for each rebalance period.
        return windows

    def rebalace_schedule(
        self,
        mc_returns: Annotated[np.ndarray, ("n_paths", "n_lengths", "n_assets")],
        rebalance_freq: int,
        optimize_window: int,
        method: Literal["Per-Path", "Pooled"] = "Per-Path",
    ) -> RebalanceSchedule:
        n_paths, n_periods, n_assets = mc_returns.shape
        assert (
            optimize_window <= n_periods
        ), "Optimize window must be less than or equal to number of periods."

        windows = self.build_rebalance_window(
            n_periods, rebalance_freq, optimize_window
        )

        return [
            (
                t_start,
                t_end,
                self.optimize_window(
                    mc_returns, optimize_start, optimize_end, method=method
                ),
            )
            for t_start, t_end, optimize_start, optimize_end in windows
        ]

    def backtest(
        self,
        returns: Annotated[np.ndarray, ("n_periods", "n_assets")],
        weights: Annotated[np.ndarray, ("n_assets",)],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        transaction_cost: float = 0.0,
    ) -> pd.Series:
        n_periods, n_assets = returns.shape
        assert len(weights) == n_assets, "Weights length must match number of assets."

        if start_date is not None and end_date is not None:
            # Create a date range for the returns index.
            dates = pd.date_range(start=start_date, end=end_date, freq="B")[:n_periods]
        else:
            # If no dates provided, use a simple range index.
            dates = pd.bdate_range(start="2000-01-01", periods=n_periods, freq="B")

        portfolio_returns = returns @ weights
        portfolio_returns[0] -= transaction_cost

        portfolio_returns_df = pd.DataFrame(
            portfolio_returns, index=dates, columns=["Portfolio Returns"]
        )

        # portfolio = vbt.Portfolio.from_returns(
        #     returns=portfolio_returns_df["Portfolio Returns"],
        #     freq="B",
        # )

        # logger.info("\n%s", portfolio.stats())

        return portfolio_returns_df["Portfolio Returns"]

    def evaluate(
        self,
        gt_returns: Annotated[np.ndarray, ("n_periods", "n_assets")],
        mc_returns: Annotated[np.ndarray, ("n_paths", "n_lengths", "n_assets")],
        rebalance_freq: int,
        optimize_window: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        transaction_cost: float = 0.0,
        init_cash: float = 1.0,
        is_log_return: bool = True,
        method: Literal["Per-Path", "Pooled"] = "Per-Path",
    ) -> pd.Series:
        n_periods, n_assets = gt_returns.shape

        # Rebalance Schedule!
        windows = self.build_rebalance_window(
            n_periods=n_periods,
            rebalance_freq=rebalance_freq,
            optimize_window=optimize_window,
        )
        # schedule = self.rebalace_schedule(
        #     mc_returns, rebalance_freq, optimize_window, method=method
        # )
        # pd.tseries.offsets.BusinessDay()

        if start_date is not None and end_date is not None:
            dates = pd.date_range(start=start_date, end=end_date, freq="B")[:n_periods]
        else:
            dates = pd.bdate_range(start="2000-01-01", periods=n_periods, freq="B")

        net_returns = np.zeros(n_periods)
        weights_history = []
        mus_history = []
        sigmas_history = []
        dates_history = []

        for t_start, t_end, optimize_start, optimize_end in windows:
            weights, mu, sigma = self.optimize_window(
                mc_returns, optimize_start, optimize_end, method=method
            )
            series = self.backtest(
                returns=gt_returns[t_start:t_end],
                weights=weights,
                start_date=start_date,
                end_date=end_date,
                transaction_cost=transaction_cost,
            )
            net_returns[t_start:t_end] = series.values
            weights_history.append(weights)
            mus_history.append(mu)
            sigmas_history.append(sigma)
            dates_history.append(dates[t_start])

        returns_series = pd.Series(net_returns, index=dates)

        weights_df = pd.DataFrame(
            weights_history,
            index=dates_history,
            columns=[f"asset_{i}" for i in range(n_assets)],
        )

        mus_df = pd.DataFrame(
            mus_history,
            index=dates_history,
            columns=[f"asset_{i}" for i in range(n_assets)],
        )

        return PortfolioResult(
            returns=returns_series,
            portfolio=self.convert_to_portfolio(
                returns=returns_series, init_cash=init_cash, is_log_return=is_log_return
            ),
            weights=weights_df,
            mus=mus_df,
            sigmas=sigmas_history,
        )

    def experiment(
        self,
        gt_returns: Annotated[np.ndarray, ("n_periods", "n_assets")],
        gt_prices: Annotated[np.ndarray, ("n_periods", "n_assets")],
        strategies: Strategies,
        benchmark_strategy: str,
        rebalance_freq: int,
        optimize_window: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        transaction_cost: float = 0.0,
        init_cash: float = 1.0,
        is_log_return: bool = True,
        method: Literal["Per-Path", "Pooled"] = "Per-Path",
        output_dir: Optional[str] = REPORTS_QS_DIR,
        output_suffix: Optional[str] = None,
    ):
        assert (
            benchmark_strategy in strategies
        ), "Benchmark strategy must be in strategies."

        n_periods = gt_returns.shape[0]

        # --- 1. เช็คเงื่อนไข Window Size = n_lengths ---
        for name, mc_returns in strategies.items():
            # mc_returns.shape คือ (n_paths, n_lengths, n_assets)
            n_lengths = mc_returns.shape[1]
            window_size = n_lengths

            # ใช้ >= เพื่อบังคับว่าต้อง "น้อยกว่าเสมอ" ตามที่คุณบรีฟ
            # (หากต้องการให้น้อยกว่าหรือเท่ากับได้ ให้เปลี่ยนเป็น > นะครับ)
            if optimize_window >= window_size:
                raise ValueError(
                    f"Strategy '{name}': opt. window ({optimize_window}) ต้องน้อยกว่า window size / n_lengths ({window_size}) เสมอ"
                )
            if rebalance_freq >= window_size:
                raise ValueError(
                    f"Strategy '{name}': rebalance freq ({rebalance_freq}) ต้องน้อยกว่า window size / n_lengths ({window_size}) เสมอ"
                )
        # -----------------------------------------------

        results: dict[str, PortfolioResult] = {
            name: self.evaluate(
                gt_returns=gt_returns,
                mc_returns=mc_returns,
                rebalance_freq=rebalance_freq,
                optimize_window=optimize_window,
                start_date=start_date,
                end_date=end_date,
                init_cash=init_cash,
                is_log_return=is_log_return,
                transaction_cost=transaction_cost,
                method=method,
            )
            for name, mc_returns in strategies.items()
        }

        portfolios = {name: r.portfolio for name, r in results.items()}
        returns = {name: r.returns for name, r in results.items()}

        benchmark_returns = returns[benchmark_strategy]
        benchmark_portfolio = portfolios[benchmark_strategy]

        fig, axes = plt.subplots(len(results), 1, figsize=(18, 5 * len(results)))
        if len(results) == 1:
            axes = [axes]

        for ax, (name, result) in zip(axes, results.items()):
            result.weights.plot.bar(
                ax=ax,
                stacked=True,
                title=f"{name} — weights per rebalance",
                width=0.8,
            )
            # ย้าย legend ออกไปทางขวา ไม่ทับกราฟ
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                borderaxespad=0,
                fontsize=8,
                ncol=1,
            )
            ax.set_xticklabels(
                [d.strftime("%Y-%m-%d") for d in result.weights.index],
                rotation=45,
                ha="right",
                fontsize=8,
            )
            ax.set_ylabel("weight")
            ax.set_ylim(0, 1)

        plt.tight_layout()

        # ---------- บันทึกรูป Weights ----------
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            img_filename = f"weights_stacked{output_suffix or ''}.png"
            save_path = os.path.join(output_dir, img_filename)
        else:
            save_path = "weights_stacked.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        # ----------------------------------------

        # --- 2. ส่วนที่แก้ไข: แยก Report ของ QuantStats เป็นราย Window ---
        windows = self.build_rebalance_window(
            n_periods=n_periods,
            rebalance_freq=rebalance_freq,
            optimize_window=optimize_window,
        )

        for name, portfolio in portfolios.items():
            if name == benchmark_strategy:
                continue

            logger.info("Strategy: %s vs Benchmark: %s", name, benchmark_strategy)

            strategy_returns = portfolio.returns()
            benchmark_returns = benchmark_portfolio.returns()

            # วนลูปตาม windows ที่สร้างไว้
            for i, (t_start, t_end, opt_start, opt_end) in enumerate(windows):
                # Slice ช่วงเวลาของ window ปัจจุบันด้วย iloc
                window_strat_returns = strategy_returns.iloc[t_start:t_end]
                window_bench_returns = benchmark_returns.iloc[t_start:t_end]

                title = f"{name} vs {benchmark_strategy} - Window {i+1} ({method}, Rebal: {rebalance_freq}, Window: {optimize_window})"

                if output_dir is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(
                        output_dir,
                        f"{name}_vs_{benchmark_strategy}_window_{i+1}{output_suffix or ''}.html",
                    )
                    qs.reports.html(
                        window_strat_returns,
                        benchmark=window_bench_returns,
                        rf=self.daily_risk_free_rate,
                        output=output_path,
                        title=title,
                    )
                    logger.info("Report for Window %d saved to: %s", i + 1, output_path)
                else:
                    qs.reports.full(
                        window_strat_returns,
                        benchmark=window_bench_returns,
                        rf=self.daily_risk_free_rate,
                        title=title,
                    )
        # -----------------------------------------------------------------

        self.plot_cov_heatmap(results, gt_returns=gt_returns)
        self.plot_corr_heatmap(results, gt_returns=gt_returns)
        self.plot_corr_heatmap(results, gt_returns=gt_returns, rebalance_idx=[0, 5, 10])

        # Plot equity curve + entry/exit markers ต่อ asset
        if gt_prices is not None:
            # ไม่ต้อง build_rebalance_window ซ้ำแล้ว เรียกใช้ตัวแปร windows ข้างบนได้เลย
            if start_date is not None and end_date is not None:
                dates = pd.date_range(start=start_date, end=end_date, freq="B")[
                    :n_periods
                ]
            else:
                dates = pd.bdate_range(start="2000-01-01", periods=n_periods, freq="B")

            self.plot_trades_all_assets(
                gt_prices=gt_prices,
                results=results,
                rebalance_windows=windows,  # ใช้ windows ที่เราสร้างไว้ข้างบนได้เลย
                dates=dates,
                init_cash=init_cash,
                transaction_cost_rate=transaction_cost,
                output_dir=output_dir,
                output_suffix=output_suffix,
            )

        return portfolios

    def plot_cov_heatmap(
        self,
        results: dict[str, "PortfolioResult"],
        gt_returns: Optional[np.ndarray] = None,  # ← เพิ่ม
        asset_idx: Optional[list[int]] = None,
        rebalance_idx: Optional[list[int]] = None,
        output_path: Optional[str] = None,
    ):
        first_result = next(iter(results.values()))
        n_rebalances = len(first_result.sigmas)
        rebal_indices = (
            rebalance_idx if rebalance_idx is not None else list(range(n_rebalances))
        )

        # strategy names + gt ถ้ามี
        col_names = list(results.keys()) + (["gt"] if gt_returns is not None else [])
        n_cols = len(col_names)

        fig, axes = plt.subplots(
            len(rebal_indices),
            n_cols,
            figsize=(7 * n_cols, 6 * len(rebal_indices)),
            squeeze=False,
        )

        for row, r_idx in enumerate(rebal_indices):
            # คำนวณ sigma ของแต่ละ strategy
            all_sigmas = []
            for name, result in results.items():
                sigma = result.sigmas[r_idx]
                idx = (
                    asset_idx if asset_idx is not None else list(range(sigma.shape[0]))
                )
                all_sigmas.append(
                    (name, sigma[np.ix_(idx, idx)], result.weights.index[r_idx])
                )

            # คำนวณ sigma ของ gt ถ้ามี — ใช้ช่วงเดียวกับ rebalance period
            if gt_returns is not None:
                t_start = r_idx * (gt_returns.shape[0] // n_rebalances)
                t_end = min(
                    t_start + gt_returns.shape[0] // n_rebalances, gt_returns.shape[0]
                )
                gt_window = gt_returns[t_start:t_end]
                sigma_gt = self.calc_sigma(gt_window)
                idx = (
                    asset_idx
                    if asset_idx is not None
                    else list(range(sigma_gt.shape[0]))
                )
                all_sigmas.append(("gt", sigma_gt[np.ix_(idx, idx)], None))

            # shared vmin/vmax ทั้ง row
            vmin = min(s.min() for _, s, _ in all_sigmas)
            vmax = max(s.max() for _, s, _ in all_sigmas)

            for col, (name, sigma, rebal_date) in enumerate(all_sigmas):
                ax = axes[row][col]
                idx = (
                    asset_idx if asset_idx is not None else list(range(sigma.shape[0]))
                )
                labels = [f"asset_{i}" for i in idx]

                date_str = (
                    rebal_date.strftime("%Y-%m-%d")
                    if rebal_date is not None
                    else "full period"
                )
                ax.set_title(f"{name} — cov {r_idx} ({date_str})", fontsize=10)

                im = ax.imshow(
                    sigma, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax
                )
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(labels, fontsize=8)

                for i in range(sigma.shape[0]):
                    for j in range(sigma.shape[1]):
                        ax.text(
                            j,
                            i,
                            f"{sigma[i, j]:.3f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="black",
                        )

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle("Covariance matrix — strategies vs GT", fontsize=13, y=1.01)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_corr_heatmap(
        self,
        results: dict[str, "PortfolioResult"],
        gt_returns: Optional[np.ndarray] = None,  # ← เพิ่ม
        asset_idx: Optional[list[int]] = None,
        rebalance_idx: Optional[list[int]] = None,
        output_path: Optional[str] = None,
    ):
        first_result = next(iter(results.values()))
        n_rebalances = len(first_result.sigmas)
        rebal_indices = (
            rebalance_idx if rebalance_idx is not None else list(range(n_rebalances))
        )

        col_names = list(results.keys()) + (["gt"] if gt_returns is not None else [])
        n_cols = len(col_names)

        fig, axes = plt.subplots(
            len(rebal_indices),
            n_cols,
            figsize=(7 * n_cols, 6 * len(rebal_indices)),
            squeeze=False,
        )

        def sigma_to_corr(sigma):
            std = np.sqrt(np.diag(sigma))
            return np.clip(sigma / np.outer(std, std), -1, 1)

        for row, r_idx in enumerate(rebal_indices):
            all_corrs = []
            for name, result in results.items():
                sigma = result.sigmas[r_idx]
                idx = (
                    asset_idx if asset_idx is not None else list(range(sigma.shape[0]))
                )
                corr = sigma_to_corr(sigma[np.ix_(idx, idx)])
                all_corrs.append((name, corr, result.weights.index[r_idx]))

            if gt_returns is not None:
                t_start = r_idx * (gt_returns.shape[0] // n_rebalances)
                t_end = min(
                    t_start + gt_returns.shape[0] // n_rebalances, gt_returns.shape[0]
                )
                sigma_gt = self.calc_sigma(gt_returns[t_start:t_end])
                idx = (
                    asset_idx
                    if asset_idx is not None
                    else list(range(sigma_gt.shape[0]))
                )
                corr_gt = sigma_to_corr(sigma_gt[np.ix_(idx, idx)])
                all_corrs.append(("gt", corr_gt, None))

            for col, (name, corr, rebal_date) in enumerate(all_corrs):
                ax = axes[row][col]
                idx = asset_idx if asset_idx is not None else list(range(corr.shape[0]))
                labels = [f"asset_{i}" for i in idx]

                date_str = (
                    rebal_date.strftime("%Y-%m-%d")
                    if rebal_date is not None
                    else "full period"
                )
                ax.set_title(f"{name} — corr {r_idx} ({date_str})", fontsize=10)

                im = ax.imshow(corr, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(labels, fontsize=8)

                for i in range(corr.shape[0]):
                    for j in range(corr.shape[1]):
                        ax.text(
                            j,
                            i,
                            f"{corr[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="white" if abs(corr[i, j]) > 0.6 else "black",
                        )

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle("Correlation matrix — strategies vs GT", fontsize=13, y=1.01)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()
