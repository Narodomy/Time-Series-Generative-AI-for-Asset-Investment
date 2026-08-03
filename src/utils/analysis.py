import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from typing import Annotated
from IPython.display import display
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Return Distribution
def plot_return_dist(
    returns: Annotated[np.ndarray, ("n_lengths", "n_assets")],
    assets: list[str],
    is_log: bool = True,
    n_cols: int = 4,
    bins: int = 80,
    title: str = "Log-Return Distribution per Asset",
) -> pd.DataFrame:
    n_length, n_assets = returns.shape
    n_rows = -(-n_assets // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    axes = axes.flatten()

    for i, sym in enumerate(assets):
        ax = axes[i]
        r = returns[:, i]
        r = r[~np.isnan(r)]

        mu, std = r.mean(), r.std()
        skew = pd.Series(r).skew()
        kurt = pd.Series(r).kurtosis()

        # Real distribution
        ax.hist(
            r,
            bins=bins,
            density=True,
            alpha=0.6,
            color="steelblue",
            edgecolor="none",
            label="real",
        )
        ax.axvline(mu, color="red", lw=1.5, linestyle="--", label=f"μ={mu:.4f}")
        ax.axvline(mu + 2 * std, color="orange", lw=1.0, linestyle=":")
        ax.axvline(mu - 2 * std, color="orange", lw=1.0, linestyle=":")

        ax.set_title(f"{sym}\nskew={skew:.2f}  kurt={kurt:.2f}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    kind = "log-return" if is_log else "return"
    fig.suptitle(f"{title} [{kind}] (orange dashed = ±2σ)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()

    return fig


def _kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 200) -> float:
    """Approximate KL divergence KL(p || q) via histogram binning."""
    lo = min(p.min(), q.min())
    hi = max(p.max(), q.max())
    edges = np.linspace(lo, hi, bins + 1)
    eps = 1e-10
    ph, _ = np.histogram(p, bins=edges, density=True)
    qh, _ = np.histogram(q, bins=edges, density=True)
    ph = ph + eps
    ph /= ph.sum()
    qh = qh + eps
    qh /= qh.sum()
    return float(np.sum(rel_entr(ph, qh)))


# Compare return distribution
def plot_compare_dist(
    real_returns: Annotated[np.ndarray, ("n_lengths", "n_assets")],
    gen_returns: Annotated[np.ndarray, ("n_lengths", "n_assets")],
    assets: list[str],
    is_log: bool = True,
    bins: int = 80,
    title: str = "Real vs Generated — Return Distribution",
) -> pd.DataFrame:
    """
    Per-asset, 4-column layout:
      Col 0: Hist + KDE + μ/±2σ lines (image-1 style, overlaid real vs gen)
      Col 1: QQ-plot
      Col 2: Stats grouped bar (mean / std / skew / kurt)
      Col 3: Distance metrics (KL + Wasserstein)
    """
    n_assets = real_returns.shape[1]
    kind = "log-return" if is_log else "return"

    REAL_C = "#4C96D7"  # steel-blue  → real
    GEN_C = "#F07167"  # coral       → generated
    COL_W, ROW_H = 4.2, 3.6

    fig = plt.figure(figsize=(COL_W * 4, ROW_H * n_assets))
    outer = gridspec.GridSpec(n_assets, 4, figure=fig, hspace=0.60, wspace=0.42)

    records = []

    for i, sym in enumerate(assets):
        r = real_returns[:, i]
        r = r[~np.isnan(r)]
        g = gen_returns[:, i]
        g = g[~np.isnan(g)]

        # ── shared stats ──────────────────────────────────────────────────
        def _s(x):
            s = pd.Series(x)
            return dict(
                mean=x.mean(),
                std=x.std(),
                skew=float(s.skew()),
                kurt=float(s.kurtosis()),
            )

        rs, gs = _s(r), _s(g)
        kl = _kl_divergence(r, g)
        wd = wasserstein_distance(r, g)
        records.append(
            {
                "asset": sym,
                **{f"real_{k}": v for k, v in rs.items()},
                **{f"gen_{k}": v for k, v in gs.items()},
                "kl_divergence": kl,
                "wasserstein": wd,
            }
        )

        # ════════════════════════════════════════════════════════════════
        # Col 0 : Histogram + KDE + μ / ±2σ  (image-1 style, overlaid)
        # ════════════════════════════════════════════════════════════════
        ax0 = fig.add_subplot(outer[i, 0])

        # — histograms —
        ax0.hist(
            r,
            bins=bins,
            density=True,
            alpha=0.45,
            color=REAL_C,
            edgecolor="none",
            label="real",
        )
        ax0.hist(
            g,
            bins=bins,
            density=True,
            alpha=0.45,
            color=GEN_C,
            edgecolor="none",
            label="gen",
        )

        # — KDE curves —
        for arr, c in ((r, REAL_C), (g, GEN_C)):
            xs = np.linspace(arr.min(), arr.max(), 500)
            ax0.plot(xs, gaussian_kde(arr)(xs), color=c, lw=1.8)

        # — μ dashed lines (red = real, darkorange = gen) —
        ax0.axvline(
            rs["mean"],
            color="red",
            lw=1.4,
            linestyle="--",
            label=f"μ_r={rs['mean']:.4f}",
        )
        ax0.axvline(
            gs["mean"],
            color="darkorange",
            lw=1.4,
            linestyle="--",
            label=f"μ_g={gs['mean']:.4f}",
        )

        # — ±2σ dotted lines —
        for sign in (-1, 1):
            ax0.axvline(
                rs["mean"] + sign * 2 * rs["std"], color="red", lw=0.9, linestyle=":"
            )
            ax0.axvline(
                gs["mean"] + sign * 2 * gs["std"],
                color="darkorange",
                lw=0.9,
                linestyle=":",
            )

        # — title : skew & kurt for both (like image 1) —
        ax0.set_title(
            f"{sym}\n"
            f"real  skew={rs['skew']:+.2f}  kurt={rs['kurt']:.2f}\n"
            f"gen   skew={gs['skew']:+.2f}  kurt={gs['kurt']:.2f}",
            fontsize=7.5,
            fontweight="bold",
            linespacing=1.4,
        )
        ax0.set_xlabel(kind, fontsize=6.5)
        ax0.legend(fontsize=5.8, ncol=2)
        ax0.tick_params(labelsize=6.5)

        # ════════════════════════════════════════════════════════════════
        # Col 1 : QQ-plot
        # ════════════════════════════════════════════════════════════════
        ax1 = fig.add_subplot(outer[i, 1])
        quants = np.linspace(0.01, 0.99, 200)
        qr, qg = np.quantile(r, quants), np.quantile(g, quants)
        ax1.scatter(qr, qg, s=6, alpha=0.65, color="#7B5EA7")
        lims = [min(qr.min(), qg.min()), max(qr.max(), qg.max())]
        ax1.plot(lims, lims, "k--", lw=1.2, label="y = x  (reference)")
        ax1.set_title("QQ-plot\n(real vs gen)", fontsize=7.5, fontweight="bold")
        ax1.set_xlabel("real quantiles", fontsize=6.5)
        ax1.set_ylabel("gen quantiles", fontsize=6.5)
        ax1.tick_params(labelsize=6.5)
        ax1.legend(fontsize=6)

        # ════════════════════════════════════════════════════════════════
        # Col 2 : Stats grouped bar
        # ════════════════════════════════════════════════════════════════
        ax2 = fig.add_subplot(outer[i, 2])
        stat_keys = ["mean", "std", "skew", "kurt"]
        x = np.arange(len(stat_keys))
        w = 0.35
        rv = [rs[k] for k in stat_keys]
        gv = [gs[k] for k in stat_keys]
        br = ax2.bar(x - w / 2, rv, w, color=REAL_C, alpha=0.85, label="real")
        bg = ax2.bar(x + w / 2, gv, w, color=GEN_C, alpha=0.85, label="gen")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["μ", "σ", "skew", "kurt"], fontsize=7)
        ax2.axhline(0, color="grey", lw=0.6)
        ax2.set_title("Stats comparison", fontsize=7.5, fontweight="bold")
        ax2.legend(fontsize=6.5)
        ax2.tick_params(labelsize=6.5)
        for bar in [*br, *bg]:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.004 if h >= 0 else -0.008),
                f"{h:.3f}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=5.2,
            )

        # ════════════════════════════════════════════════════════════════
        # Col 3 : Distance metrics
        # ════════════════════════════════════════════════════════════════
        ax3 = fig.add_subplot(outer[i, 3])
        names = ["KL div\n(real‖gen)", "Wasserstein"]
        vals = [kl, wd]
        colors = ["#E07A5F", "#3D405B"]
        bars = ax3.bar(names, vals, color=colors, alpha=0.85, width=0.45)
        for bar, v in zip(bars, vals):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{v:.4f}",
                ha="center",
                fontsize=7,
                fontweight="bold",
            )
        ax3.set_title("Distance metrics", fontsize=7.5, fontweight="bold")
        ax3.set_ylabel("distance", fontsize=6.5)
        ax3.tick_params(labelsize=6.5)
        ax3.set_ylim(0, max(vals) * 1.28)

    fig.suptitle(f"{title}  [{kind}]", fontsize=13, fontweight="bold", y=1.003)
    plt.show()

    return fig, pd.DataFrame(records).set_index("asset").round(6)


# Correlation Heatmap
def plot_correlation(
    returns: np.ndarray,  # [T, A]
    assets: list[str],
    threshold: float = 0.85,
) -> pd.DataFrame:
    corr_real = pd.DataFrame(returns, columns=assets).corr()

    n_plots = 1
    fig, axes = plt.subplots(1, n_plots, figsize=(11 * n_plots, 9))
    if n_plots == 1:
        axes = [axes]

    mask = np.triu(np.ones_like(corr_real, dtype=bool), k=1)

    for ax, corr, label in zip(
        axes,
        [corr_real],
        ["Real"],
    ):
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            linewidths=0.4,
            linecolor="white",
            ax=ax,
        )
        ax.set_title(f"Pairwise Correlation — {label}", fontsize=12)

    plt.tight_layout()
    plt.show()

    # High-corr pairs from real
    pairs = []
    for i in range(len(corr_real)):
        for j in range(i + 1, len(corr_real)):
            r = corr_real.iloc[i, j]
            if abs(r) > threshold:
                pairs.append(
                    {"Asset A": assets[i], "Asset B": assets[j], "r": round(r, 3)}
                )
    if pairs:
        print(f"\nHighly correlated pairs (|r| > {threshold}):")
        display(pd.DataFrame(pairs).sort_values("r", ascending=False))
    else:
        print(f"\nNo pairs with |r| > {threshold}")

    return fig, corr_real


# Rolling Volatility
def plot_rolling_vol(
    returns: np.ndarray,  # [T, A]
    assets: list[str],
    window: int = 30,
    top_n: int = 6,
    stats_df: pd.DataFrame = None,  # ถ้ามี stats_df จาก plot_return_dist ก็ส่งมาได้
) -> None:

    df_real = pd.DataFrame(returns, columns=assets)
    roll_real = df_real.rolling(window).std() * np.sqrt(252)

    # Top N by std
    if stats_df is not None:
        top_assets = stats_df.head(top_n).index.tolist()
    else:
        top_assets = (
            df_real.std().sort_values(ascending=False).head(top_n).index.tolist()
        )

    n_rows = -(-top_n // 2)
    fig1, axes = plt.subplots(n_rows, 2, figsize=(16, n_rows * 3), sharex=False)
    axes = axes.flatten()

    for i, sym in enumerate(top_assets):
        ax = axes[i]
        vol = roll_real[sym].dropna()
        med = vol.median()

        ax.fill_between(range(len(vol)), vol.values, alpha=0.3, color="steelblue")
        ax.plot(vol.values, lw=0.8, color="steelblue", label="real")
        ax.axhline(med, color="red", lw=1.2, linestyle="--", label=f"median={med:.2f}")
        ax.set_title(sym, fontsize=10)
        ax.set_ylabel("Ann. Vol")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig1.suptitle(
        f"Rolling {window}-day Annualised Volatility (top-{top_n})", fontsize=13
    )
    plt.tight_layout()
    plt.show()

    # Cross-asset average vol
    avg_vol = roll_real.mean(axis=1).dropna()
    q75, q25 = avg_vol.quantile(0.75), avg_vol.quantile(0.25)

    fig2, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(range(len(avg_vol)), avg_vol.values, alpha=0.3, color="coral")
    ax.plot(avg_vol.values, lw=1, color="coral", label="Portfolio avg vol (real)")
    ax.axhline(q75, color="red", lw=1, linestyle="--", label=f"Q75={q75:.2f}")
    ax.axhline(q25, color="green", lw=1, linestyle="--", label=f"Q25={q25:.2f}")
    ax.set_title("Cross-Asset Average Annualised Volatility")
    ax.set_ylabel("Ann. Vol")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return fig1, fig2


def plot_compare_correlation(
    returns: np.ndarray,  # [T, A]
    gen_returns: np.ndarray,  # [H, A]
    assets: list[str],
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise correlation heatmap เทียบ Real vs Generated side-by-side
    พร้อม diff heatmap ตรงกลาง

    Returns
    -------
    corr_real, corr_gen : pd.DataFrame
    """
    corr_real = pd.DataFrame(returns, columns=assets).corr()
    corr_gen = pd.DataFrame(gen_returns, columns=assets).corr()
    corr_diff = corr_gen - corr_real

    mask = np.triu(np.ones_like(corr_real, dtype=bool), k=1)

    fig, axes = plt.subplots(1, 3, figsize=(33, 9))

    for ax, corr, label, cmap, vmin, vmax in [
        (axes[0], corr_real, "Real", "RdYlGn", -1, 1),
        (axes[1], corr_gen, "Generated", "RdYlGn", -1, 1),
        (axes[2], corr_diff, "Diff (Gen − Real)", "coolwarm", -0.5, 0.5),
    ]:
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.4,
            linecolor="white",
            ax=ax,
        )
        ax.set_title(f"Pairwise Correlation — {label}", fontsize=12)

    plt.tight_layout()
    plt.show()

    # High-corr pair comparison
    rows = []
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            r_real = corr_real.iloc[i, j]
            r_gen = corr_gen.iloc[i, j]
            if abs(r_real) > threshold or abs(r_gen) > threshold:
                rows.append(
                    {
                        "Asset A": assets[i],
                        "Asset B": assets[j],
                        "r_real": round(r_real, 3),
                        "r_gen": round(r_gen, 3),
                        "delta": round(r_gen - r_real, 3),
                    }
                )

    if rows:
        print(f"\nPairs ที่ |r| > {threshold} ในอย่างน้อยหนึ่ง set:")
        display(
            pd.DataFrame(rows)
            .sort_values("r_real", ascending=False)
            .reset_index(drop=True)
        )
    else:
        print(f"\nNo pairs with |r| > {threshold} in either set")

    return fig, corr_real, corr_gen


def plot_compare_rolling_vol(
    returns: np.ndarray,  # [T, A]
    gen_returns: np.ndarray,  # [H, A]
    assets: list[str],
    window: int = 30,
    top_n: int = 6,
    stats_df: pd.DataFrame = None,
) -> None:
    """
    Rolling annualised volatility เทียบ Real vs Generated
    - Per-asset: overlay เส้น gen บน real ในกราฟเดียวกัน
    - Cross-asset: avg vol ของทั้งสอง set บน ax เดียว พร้อม shaded band
    """
    df_real = pd.DataFrame(returns, columns=assets)
    df_gen = pd.DataFrame(gen_returns, columns=assets)

    roll_real = df_real.rolling(window).std() * np.sqrt(252)
    roll_gen = df_gen.rolling(window).std() * np.sqrt(252)

    # Top N by real std
    if stats_df is not None:
        top_assets = stats_df.head(top_n).index.tolist()
    else:
        top_assets = (
            df_real.std().sort_values(ascending=False).head(top_n).index.tolist()
        )

    n_rows = -(-top_n // 2)
    fig1, axes = plt.subplots(n_rows, 2, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    for i, sym in enumerate(top_assets):
        ax = axes[i]

        vol_real = roll_real[sym].dropna().values
        vol_gen = roll_gen[sym].dropna().values
        t_real = np.arange(len(vol_real))
        t_gen = np.arange(len(vol_gen))

        ax.fill_between(t_real, vol_real, alpha=0.15, color="steelblue")
        ax.plot(t_real, vol_real, lw=0.9, color="steelblue", label="Real")

        ax.fill_between(t_gen, vol_gen, alpha=0.15, color="coral")
        ax.plot(
            t_gen, vol_gen, lw=0.9, color="coral", label="Generated", linestyle="--"
        )

        ax.axhline(
            np.median(vol_real),
            color="steelblue",
            lw=1.2,
            linestyle=":",
            label=f"med_real={np.median(vol_real):.2f}",
        )
        ax.axhline(
            np.median(vol_gen),
            color="coral",
            lw=1.2,
            linestyle=":",
            label=f"med_gen={np.median(vol_gen):.2f}",
        )

        ax.set_title(sym, fontsize=10)
        ax.set_ylabel("Ann. Vol")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig1.suptitle(
        f"Rolling {window}-day Ann. Vol — Real vs Generated (top-{top_n})", fontsize=13
    )
    plt.tight_layout()
    plt.show()

    # Cross-asset average vol
    avg_real = roll_real.mean(axis=1).dropna()
    avg_gen = roll_gen.mean(axis=1).dropna()
    t_real = np.arange(len(avg_real))
    t_gen = np.arange(len(avg_gen))

    fig2, ax = plt.subplots(figsize=(14, 4))

    ax.fill_between(t_real, avg_real.values, alpha=0.2, color="steelblue")
    ax.plot(t_real, avg_real.values, lw=1.2, color="steelblue", label="Real")

    ax.fill_between(t_gen, avg_gen.values, alpha=0.2, color="coral")
    ax.plot(
        t_gen, avg_gen.values, lw=1.2, color="coral", linestyle="--", label="Generated"
    )

    for val, color, lbl in [
        (
            avg_real.quantile(0.75),
            "steelblue",
            f"Real Q75={avg_real.quantile(0.75):.2f}",
        ),
        (
            avg_real.quantile(0.25),
            "steelblue",
            f"Real Q25={avg_real.quantile(0.25):.2f}",
        ),
        (avg_gen.quantile(0.75), "coral", f"Gen  Q75={avg_gen.quantile(0.75):.2f}"),
        (avg_gen.quantile(0.25), "coral", f"Gen  Q25={avg_gen.quantile(0.25):.2f}"),
    ]:
        ax.axhline(val, color=color, lw=1, linestyle="--", alpha=0.7, label=lbl)

    ax.set_title("Cross-Asset Average Ann. Vol — Real vs Generated")
    ax.set_ylabel("Ann. Vol")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig1, fig2
