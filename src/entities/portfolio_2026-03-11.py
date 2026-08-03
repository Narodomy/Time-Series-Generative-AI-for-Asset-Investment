import numpy as np
import pandas as pd
import logging
import os
from datetime import date
from typing import Tuple, Optional, List, Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pypfopt import EfficientFrontier, plotting
from scipy.optimize import minimize, LinearConstraint, Bounds
import quantstats as qs

from utils import inverse_log_returns
from utils.paths import REPORTS_QS_DIR

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Portfolio optimization and backtesting utility.

    Supports:
    - Mean-Variance optimization via PyPortfolioOpt (EfficientFrontier) or SciPy (SLSQP)
    - NAV simulation with periodic rebalancing and transaction costs
    - QuantStats-based performance reporting
    - Efficient Frontier visualization
    """

    def __init__(
        self,
        risk_free_rate: float,
        weight_bounds: Tuple[float, float] = (0.0, 1.0),
        transaction_cost_rate: float = 0.0,
        start_date: str = str(date.today()),
        save_dir: str = REPORTS_QS_DIR,
        is_daily: bool = False,
    ):
        """
        Args:
            risk_free_rate        : Annual risk-free rate (e.g. 0.02 for 2 %).
                                    Pass daily rf (e.g. 0.02/252) if mu/sigma are daily-scale.
            weight_bounds         : (min_weight, max_weight) per asset. Default (0, 1) = long-only.
            transaction_cost_rate : One-way cost as a fraction of turnover value.
                                    Typical values: 0 (none), 0.001 (10 bps), 0.0025 (25 bps).
            start_date            : Fallback start date for date index when none is supplied.
            save_dir              : Directory for QuantStats HTML reports.
            is_daily              : If True, rf is treated as daily when used in plot_ef.
        """
        self.start_date = start_date
        self.is_daily = is_daily
        self._risk_free_rate = risk_free_rate
        self.weight_bounds = weight_bounds
        self.transaction_cost_rate = transaction_cost_rate
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def risk_free_rate(self) -> float:
        return self._risk_free_rate

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def calc(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean vector and covariance matrix.

        Args:
            returns : (T, N) array of returns.
        Returns:
            mu    : (N,) expected returns.
            sigma : (N, N) covariance matrix.
        """
        mu = np.mean(returns, axis=0)
        sigma = np.cov(returns, rowvar=False)
        return mu, sigma

    def calc_distribution(
        self, all_returns: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stack multiple return arrays and compute pooled mu / sigma.

        Args:
            all_returns : List of (T_i, N) arrays.
        Returns:
            mu    : (N,) pooled expected returns.
            sigma : (N, N) pooled covariance matrix.
        """
        combined = np.vstack(all_returns)
        return self.calc(combined)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _optimize_scipy(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_free_rate: Optional[float] = None,
    ) -> np.ndarray:
        """
        Maximize Sharpe Ratio via SciPy SLSQP.

        Solves:  max  (w'μ - rf) / sqrt(w'Σw)
                 s.t. Σw = 1,  lb ≤ w ≤ ub
        """
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        n = len(mu)

        def neg_sharpe(w: np.ndarray) -> float:
            ret = w @ mu
            vol = np.sqrt(w @ sigma @ w)
            return -(ret - rf) / (vol + 1e-9)

        result = minimize(
            neg_sharpe,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=Bounds(lb=self.weight_bounds[0], ub=self.weight_bounds[1]),
            constraints=LinearConstraint(np.ones(n), lb=1.0, ub=1.0),
        )

        if not result.success:
            logger.warning(f"SciPy optimizer did not converge: {result.message}")

        return result.x

    def _optimize_pypfopt(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_free_rate: Optional[float] = None,
    ) -> np.ndarray:
        """Maximize Sharpe Ratio via PyPortfolioOpt EfficientFrontier."""
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        ef = EfficientFrontier(mu, sigma, weight_bounds=self.weight_bounds)
        ef.max_sharpe(risk_free_rate=rf)
        cleaned = ef.clean_weights()
        return np.array(list(cleaned.values()))

    def optimize_weights(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_free_rate: Optional[float] = None,
        solver: Literal["scipy", "pypfopt"] = "scipy",
    ) -> np.ndarray:
        """
        Find the Sharpe-maximizing (tangency) portfolio weights.

        Args:
            mu             : (N,) expected returns.
            sigma          : (N, N) covariance matrix.
            risk_free_rate : Override instance rf if provided.
            solver         : "scipy"   → SLSQP (continuous weights, no snapping).
                             "pypfopt" → EfficientFrontier (rounds small weights to 0).
        Returns:
            weights : (N,) array that sums to 1, values within weight_bounds.
        """
        if solver == "scipy":
            return self._optimize_scipy(mu, sigma, risk_free_rate)
        elif solver == "pypfopt":
            return self._optimize_pypfopt(mu, sigma, risk_free_rate)
        else:
            raise ValueError(f"Unknown solver '{solver}'. Choose 'scipy' or 'pypfopt'.")

    # ------------------------------------------------------------------
    # NAV simulation with rebalancing
    # ------------------------------------------------------------------

    def balance(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        rebalance_days: int = 1,
        is_log_return: bool = True,
        transaction_cost_rate: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Simulate portfolio NAV growth with periodic rebalancing and TC.

        Transaction cost model (one-way turnover):
          - Day 0 (initial buy)  : cost = tc_rate × 1.0  (full portfolio bought from scratch)
          - Each rebalance event : cost = turnover × tc_rate × NAV
            where turnover = Σ |w_drifted_i − w_target_i|

        Args:
            weights               : (N,) target allocation weights (must sum to 1).
            returns               : (T, N) asset returns.
            rebalance_days        : Rebalance every N days (1=daily, 5=weekly, 21=monthly).
            is_log_return         : If True, converts log → simple before NAV simulation
                                    and returns log returns; otherwise works in simple returns.
            transaction_cost_rate : One-way tc rate. Overrides self.transaction_cost_rate.
        Returns:
            daily_returns         : (T,) portfolio daily returns (same format as input).
            total_tc_paid         : Total transaction costs deducted from NAV (in NAV units).
        """
        tc_rate = (
            transaction_cost_rate
            if transaction_cost_rate is not None
            else self.transaction_cost_rate
        )

        # 1. Convert to simple returns for NAV arithmetic
        simple_returns = np.exp(returns) - 1 if is_log_return else returns.copy()
        n_days, n_assets = simple_returns.shape

        # 2. Initialise NAV
        portfolio_value = np.zeros(n_days + 1)
        portfolio_value[0] = 1.0

        # Initial buy: turnover = 1.0 (buying everything from cash)
        initial_tc = tc_rate * portfolio_value[0]
        portfolio_value[0] -= initial_tc
        total_tc_paid = initial_tc

        current_holdings = weights * portfolio_value[0]

        for i in range(n_days):
            # Apply today's returns
            current_holdings = current_holdings * (1.0 + simple_returns[i])
            total_val = np.sum(current_holdings)

            # Rebalance check
            if (i + 1) % rebalance_days == 0:
                drifted_weights = current_holdings / (total_val + 1e-12)
                turnover = np.sum(np.abs(drifted_weights - weights))  # one-way

                tc = turnover * tc_rate * total_val
                total_val -= tc
                total_tc_paid += tc

                current_holdings = total_val * weights

            portfolio_value[i + 1] = np.sum(current_holdings)

        # 3. Derive daily returns from NAV series
        nav = pd.Series(portfolio_value)
        daily_simple = nav.pct_change().dropna().values

        # 4. Return in the same format as input
        daily_returns = np.log(1.0 + daily_simple) if is_log_return else daily_simple
        return daily_returns, total_tc_paid

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def back_test(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        weights_benchmark: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        rebalance_days: int = 1,
        transaction_cost_rate: Optional[float] = None,
        benchmark_tc_rate: Optional[float] = None,
        is_saved: bool = False,
        is_log_return: bool = True,
        title: str = "GenAI Portfolio Backtest",
        filename: Optional[str] = None,  # FIX 1: was evaluated at import-time
    ) -> pd.DataFrame:
        """
        Run a backtest and return QuantStats metrics.

        Args:
            weights               : (N,) portfolio weights.
            returns               : (T, N) asset returns.
            weights_benchmark     : (N,) benchmark weights. If None, compares vs 'SPY'.
            dates                 : DatetimeIndex for the return series.
            rebalance_days        : Passed to balance().
            transaction_cost_rate : Passed to balance(); overrides self.transaction_cost_rate.
            is_saved              : Save HTML report to save_dir.
            is_log_return         : Format of `returns` (and output).
            title                 : Title for the HTML report.
            filename              : Report filename (no extension).
                                    Defaults to "Portfolio_Report_<today>" evaluated at call time.
        Returns:
            metrics : pd.DataFrame with full QuantStats metric table.
        """
        # FIX 1: default evaluated at call time, not import time
        if filename is None:
            filename = f"Portfolio_Report_{date.today()}"

        save_path = os.path.join(self.save_dir, filename)

        if dates is None:
            n_obs = returns.shape[0]
            dates = pd.date_range(start=self.start_date, periods=n_obs, freq="D")

        logger.debug(f"back_test | returns: {returns.shape} | weights: {weights.shape}")

        ret_portfolio, total_tc_paid = self.balance(
            weights=weights,
            returns=returns,
            rebalance_days=rebalance_days,
            is_log_return=is_log_return,
            transaction_cost_rate=transaction_cost_rate,
        )

        logger.info(f"Total transaction cost paid: {total_tc_paid:.6f}")

        # FIX 2: only convert if balance() returned log returns
        qs_returns = (
            inverse_log_returns(ret_portfolio) if is_log_return else ret_portfolio
        )
        portfolio_series = pd.Series(qs_returns, index=dates)

        logger.debug(
            f"Portfolio return  max={np.max(qs_returns):.4f}  min={np.min(qs_returns):.4f}"
        )

        if weights_benchmark is not None:
            # simple_returns = np.exp(returns) - 1 if is_log_return else returns
            benchmark_tc = (
                benchmark_tc_rate
                if benchmark_tc_rate is not None
                else (
                    transaction_cost_rate
                    if transaction_cost_rate is not None
                    else self.transaction_cost_rate
                )
            )

            ret_benchmark, _ = self.balance(
                weights=weights_benchmark,
                returns=returns,
                rebalance_days=rebalance_days,
                is_log_return=is_log_return,
                transaction_cost_rate=benchmark_tc,
            )
            benchmark_series = pd.Series(
                inverse_log_returns(ret_benchmark) if is_log_return else ret_benchmark,
                index=dates,
            )

            # Old approach (removed): assumed benchmark rebalances daily for free
            # which created an unfair comparison vs portfolio that rebalances every N days with TC
            # ret_benchmark = np.dot(simple_returns, weights_benchmark)
            # ret_benchmark = np.dot(simple_returns, weights_benchmark)
            # benchmark_series = pd.Series(ret_benchmark, index=dates)

            logger.debug(
                f"Benchmark return  max={np.max(ret_benchmark):.4f}  min={np.min(ret_benchmark):.4f}"
            )
        else:
            benchmark_series = "SPY"

        print("Generating QuantStats Report...")
        if is_saved:
            qs.reports.html(
                portfolio_series,
                benchmark=benchmark_series,
                output=f"{save_path}.html",
                title=title,  # FIX 3: use param, not hardcoded string
                rf=self.risk_free_rate,
            )
            print(f"Report saved to {save_path}")

        metrics = qs.reports.metrics(portfolio_series, mode="full", display=False)
        return metrics

    # ------------------------------------------------------------------
    # Efficient Frontier plot
    # ------------------------------------------------------------------

    def plot_ef(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        n_portfolios: int = 5_000,
        figsize: Tuple[int, int] = (10, 6),
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot the Efficient Frontier with Max-Sharpe tangency portfolio.

        Args:
            mu           : (N,) expected returns.
            sigma        : (N, N) covariance matrix.
            n_portfolios : Random portfolios to scatter.
            figsize      : Figure size.
            show         : If True, calls plt.show().
        Returns:
            fig : matplotlib Figure.
        """
        rf = self.risk_free_rate if self.is_daily else self.risk_free_rate / 252

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#0f0f1a")
        ax.set_facecolor("#0f0f1a")

        # --- Random portfolio cloud ---
        n_assets = len(mu)
        cloud_rets, cloud_vols, cloud_sharpes = [], [], []

        rng = np.random.default_rng(42)
        for _ in range(n_portfolios):
            w = rng.dirichlet(np.ones(n_assets))
            r = w @ mu
            v = np.sqrt(w @ sigma @ w)
            cloud_rets.append(r)
            cloud_vols.append(v)
            cloud_sharpes.append((r - rf) / (v + 1e-9))

        sc = ax.scatter(
            cloud_vols,
            cloud_rets,
            c=cloud_sharpes,
            cmap="plasma",
            s=4,
            alpha=0.5,
            linewidths=0,
            zorder=2,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Sharpe Ratio", color="#cccccc", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="#cccccc")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#cccccc")

        # --- Efficient Frontier curve ---
        ef_curve = EfficientFrontier(mu, sigma, weight_bounds=self.weight_bounds)
        plotting.plot_efficient_frontier(
            ef_curve,
            ax=ax,
            show_assets=True,
            # FIX 4: correct matplotlib kwargs (was line_color / line_width)
            color="#00e5ff",
            linewidth=2,
        )

        # --- Max-Sharpe tangency point ---
        ef_tangent = EfficientFrontier(mu, sigma, weight_bounds=self.weight_bounds)
        ef_tangent.max_sharpe(risk_free_rate=rf)
        ret_t, vol_t, sharpe_t = ef_tangent.portfolio_performance(risk_free_rate=rf)

        ax.scatter(
            vol_t,
            ret_t,
            marker="*",
            s=400,
            color="#ff4081",
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
            label=f"Max Sharpe  ({sharpe_t:.2f})",
        )

        # --- Capital Market Line ---
        cml_vols = np.linspace(0, vol_t * 1.6, 200)
        cml_rets = rf + (ret_t - rf) / vol_t * cml_vols
        ax.plot(
            cml_vols,
            cml_rets,
            "--",
            color="#ffd740",
            linewidth=1.2,
            alpha=0.7,
            label="Capital Market Line",
            zorder=3,
        )

        # --- Risk-free point ---
        ax.scatter(
            0,
            rf,
            marker="D",
            s=80,
            color="#ffd740",
            zorder=5,
            label=f"Risk-free  ({rf:.4f})",
        )

        # --- Styling ---
        ax.set_title(
            "Efficient Frontier", color="white", fontsize=14, fontweight="bold", pad=12
        )
        ax.set_xlabel("Volatility (σ)", color="#aaaaaa", fontsize=11)
        ax.set_ylabel("Expected Return (μ)", color="#aaaaaa", fontsize=11)
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3, color="#555577")
        ax.legend(
            facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white", fontsize=9
        )

        fig.tight_layout()
        if show:
            plt.show()

        return fig
