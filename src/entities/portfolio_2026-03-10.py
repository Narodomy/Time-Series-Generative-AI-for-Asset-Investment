import numpy as np
import pandas as pd
import logging
import os
from datetime import date
from typing import Tuple, Optional, List, Union
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, plotting
from scipy.optimize import minimize, LinearConstraint, Bounds
import quantstats as qs
from utils import inverse_log_returns
from utils.paths import REPORTS_DIR, REPORTS_QS_DIR

logger = logging.getLogger(__name__)


class Portfolio:
    def __init__(
        self,
        risk_free_rate: float,
        weight_bounds: Optional[Tuple[float, float]] = (0, 1),
        start_date: str = str(date.today()),
        save_dir: str = REPORTS_QS_DIR,
        is_daily: bool = False,
    ):
        self.start_date = start_date
        self.is_daily = is_daily
        self._risk_free_rate = risk_free_rate
        self.weight_bounds = weight_bounds
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    @property
    def risk_free_rate(self):
        return self._risk_free_rate  # / 252

    def calc(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Input: (Steps, Assets) -> Output: (Assets,)
        mu = np.mean(returns, axis=0)

        # Input: (Steps, Assets) -> Output: (Assets, Assets)
        sigma = np.cov(returns, rowvar=False)

        return mu, sigma

    def calc_distribution(
        self, all_returns: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Output: (Total_Steps, Assets)
        combined_returns = np.vstack(all_returns)

        mu_total, sigma_total = self.calc(combined_returns)

        return mu_total, sigma_total

    def _optimize_weights_with_scipy(
        self, mu: np.ndarray, sigma: np.ndarray, risk_free_rate: Optional[float] = None
    ) -> np.ndarray:
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        n_assets = len(mu)  # mu.shape[0]

        def negative_sharpe(w):
            ret = np.dot(w, mu)  # expected return
            vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            sharpe = (ret - risk_free_rate) / (vol + 1e-9)
            return -sharpe

        bounds = Bounds(lb=self.weight_bounds[0], ub=self.weight_bounds[1])

        # Linear Constraint (Sum of weights = 1)
        A = np.ones(n_assets)
        constraint = LinearConstraint(A, lb=1.0, ub=1.0)

        init_guess = np.ones(n_assets) / n_assets

        result = minimize(
            negative_sharpe,
            init_guess,
            method="SLSQP",  # or 'trust-constr', 'BFGS'
            bounds=bounds,
            constraints=constraint,
        )

        return result.x

    def optimize_weights(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_free_rate: Optional[float] = None,
        scipy: bool = False,
    ) -> np.ndarray:
        if scipy:
            weights_array = self._optimize_weights_with_scipy(mu, sigma)
        else:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate

            # EfficientFrontier
            ef = EfficientFrontier(mu, sigma, weight_bounds=self.weight_bounds)

            # ef.add_objective(objective_functions.L2_reg, gamma=0.1)

            # Maximize Sharpe Ratio
            ef.max_sharpe(risk_free_rate=risk_free_rate)

            # Clean Weights
            cleaned_weights = ef.clean_weights()

            weights_array = np.array(list(cleaned_weights.values()))

        # Output: weights (N,)
        return weights_array

    def balance(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        rebalance_days: int,
        is_log_return: bool = True,
    ) -> np.ndarray:
        """
        จำลองการเติบโตของพอร์ตโดยมีการ Rebalance ทุกๆ N วัน
        คืนค่ากลับมาเป็น Daily Returns (ประเภทเดียวกับ input is_log_return)
        """
        # 1. แปลงเป็น Simple Return เพื่อคำนวณเงินจริง (Nav)
        if is_log_return:
            # สมมติ returns คือ log return -> แปลงเป็น simple
            simple_returns = np.exp(returns) - 1
        else:
            simple_returns = returns

        n_days, n_assets = simple_returns.shape

        # 2. จำลองเงินในพอร์ต
        portfolio_value = np.zeros(n_days + 1)  # +1 เพื่อเก็บวันเริ่มต้น
        portfolio_value[0] = 1.0  # เริ่มต้นด้วยเงิน 1 หน่วย

        current_holdings = weights * portfolio_value[0]  # กระจายเงินตาม weight

        for i in range(n_days):
            # คำนวณเงินที่โตขึ้นในแต่ละ Asset ของวันนี้
            asset_growth = 1 + simple_returns[i]
            current_holdings = current_holdings * asset_growth

            # รวมเงินทั้งหมด ณ สิ้นวัน
            total_val = np.sum(current_holdings)
            portfolio_value[i + 1] = total_val

            # ตรวจสอบรอบ Rebalance
            # เช่น rebalance_days=5, จะปรับพอร์ตในวันที่ 4, 9, 14... (index)
            # เพื่อให้วันรุ่งขึ้น (5, 10, 15) เริ่มต้นด้วย weight ที่ถูกต้อง
            if (i + 1) % rebalance_days == 0:
                current_holdings = total_val * weights  # ตบกลับเข้า Weight เป้าหมาย

        # 3. คำนวณ Return รายวันจากมูลค่าพอร์ตที่โตขึ้น
        # สูตร: (NAV วันนี้ / NAV เมื่อวาน) - 1
        portfolio_nav_series = pd.Series(portfolio_value)
        portfolio_daily_simple_returns = (
            portfolio_nav_series.pct_change().dropna().values
        )

        # 4. คืนค่าตาม format ที่รับมา (ถ้า input เป็น log ก็คืน log เพื่อให้เข้ากับ code เดิม)
        if is_log_return:
            return np.log(1 + portfolio_daily_simple_returns)

        return portfolio_daily_simple_returns

    def back_test(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        weights_benchmark: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        rebalance_days: int = 1,
        is_saved: bool = False,
        is_log_return: bool = True,
        filename: str = f"Portfolio_Report_{str(date.today())}",
    ) -> pd.DataFrame:
        # weights: [Assets], returns: [Obs, Assets]
        # R_p = w1*r1 + w2*r2 + ...
        save_path = os.path.join(self.save_dir, filename)

        if dates is None:
            N_obs, N_assets = returns.shape
            dates = pd.date_range(start=self.start_date, periods=N_obs, freq="D")

        logger.debug(f"Returns: {returns.shape}, Weights: {weights.shape}")

        # ret_portfolio = inverse_log_returns(np.dot(returns, weights)) if is_log_return else np.dot(returns, weights)
        # Add new for reblance port every n days!
        # -------------------
        ret_portfolio = self.balance(
            weights=weights,
            returns=returns,
            rebalance_days=rebalance_days,
            is_log_return=is_log_return,
        )

        qs_returns = inverse_log_returns(ret_portfolio)  # simple returns for quantstats
        portfolio_series = pd.Series(qs_returns, index=dates)

        logger.debug(f"Max Log Return: {np.max(qs_returns):.4f}")
        logger.debug(f"Min Log Return: {np.min(qs_returns):.4f}")

        if weights_benchmark is not None:
            simple_returns = np.exp(returns) - 1  # inverse_log_returns(returns)
            ret_benchmark = np.dot(simple_returns, weights_benchmark)
            benchmark_series = pd.Series(ret_benchmark, index=dates)

            logger.debug(f"Max Log Return Benchmark: {np.max(ret_benchmark):.4f}")
            logger.debug(f"Min Log Return Benchmark: {np.min(ret_benchmark):.4f}")
        else:
            benchmark_series = "SPY"

        print("Generating QuantStats Report...")
        if is_saved:
            qs.reports.html(
                portfolio_series,
                benchmark=benchmark_series,
                output=f"{save_path}.html",
                title="GenAI Portfolio Backtest",
                rf=self.risk_free_rate,
            )
            print(f"Report saved to {save_path}")

        metrics = qs.reports.metrics(portfolio_series, mode="full", display=False)
        return metrics

    def plot_ef(self, mu: np.ndarray, sigma: np.ndarray):
        ef = EfficientFrontier(mu, sigma, weight_bounds=self.weight_bounds)

        # Setup Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Draw Frontier
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

        # Find Max Sharpe (Red Star)
        ef_max = EfficientFrontier(mu, sigma, weight_bounds=self.weight_bounds)
        ef_max.max_sharpe(
            risk_free_rate=(
                self.risk_free_rate / 252 if self.is_daily else self.risk_free_rate
            )
        )
        ret_tangent, std_tangent, _ = ef_max.portfolio_performance()

        # Draw Red Star
        ax.scatter(
            std_tangent,
            ret_tangent,
            marker="*",
            s=300,
            c="r",
            label="Max Sharpe Portfolio",
        )

        ax.set_title(f"Efficient Frontier (Rf={self.risk_free_rate})")
        ax.set_xlabel("Volatility (Risk)")
        ax.set_ylabel("Expected Return")
        ax.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
