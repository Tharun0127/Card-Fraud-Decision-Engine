"""Static PNG figures. Palette: validated reference categorical slots 1-3
(blue, orange, aqua), single-hue sequential ramps, neutral grid."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "legend.frameon": False,
})


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def profit_curve(val_sweep: pd.DataFrame, test_sweep: pd.DataFrame, marks: dict, path) -> None:
    """Net benefit per 1,000 transactions against share of traffic declined."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for sw, color, label in ((val_sweep, INK2, "Validation (thresholds chosen here)"),
                             (test_sweep, BLUE, "Test (reported)")):
        s = sw[sw["decline_rate"] > 0].sort_values("decline_rate")
        ax.plot(s["decline_rate"] * 100, s["net_benefit_per_1000"], color=color, label=label,
                linestyle="--" if color == INK2 else "-", linewidth=1.5 if color == INK2 else 2)
    for name, (x, y, color) in marks.items():
        ax.plot([x * 100], [y], "o", ms=8, color=color, markeredgecolor="white", markeredgewidth=2, zorder=5)
        ax.annotate(name, (x * 100, y), textcoords="offset points", xytext=(8, -14 if "Youden" in name else 6),
                    color=INK, fontsize=9)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Share of transactions declined (%, log scale)")
    ax.set_ylabel("Net benefit vs approve-all ($ per 1,000 txns)")
    ax.set_title("Profit curve: single decline threshold, full model")
    ax.legend(loc="lower left")
    _save(fig, path)


def roc_with_cutoffs(fpr, tpr, points: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr, tpr, color=BLUE, label="Full model (test)")
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    for name, (x, y, color) in points.items():
        ax.plot([x], [y], "o", ms=8, color=color, markeredgecolor="white", markeredgewidth=2, zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, -12), fontsize=9)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Where each cutoff sits on the ROC curve (test)")
    ax.legend(loc="lower right")
    _save(fig, path)


def pr_curves(curves: dict, rules_point: tuple, path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    for (name, (rec, prec)), color in zip(curves.items(), (BLUE, ORANGE)):
        ax.plot(rec, prec, color=color, label=name)
    ax.plot([rules_point[0]], [rules_point[1]], "s", ms=8, color=AQUA, markeredgecolor="white",
            markeredgewidth=2, label="Rules baseline (single operating point)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on test")
    ax.legend(loc="upper right")
    _save(fig, path)


def three_way_heatmap(nb_matrix: np.ndarray, thresholds: np.ndarray, chosen: tuple, val_n: int,
                      t_min: float, path) -> None:
    """Validation net benefit over (t_decline, t_review), restricted to t >= t_min."""
    keep = np.isfinite(thresholds) & (thresholds >= t_min)
    m = nb_matrix[np.ix_(keep, keep)] / val_n * 1000
    tt = thresholds[keep]
    fig, ax = plt.subplots(figsize=(6.4, 5))
    im = ax.pcolormesh(tt, tt, m, cmap="Blues", shading="nearest", vmin=0)
    ax.plot([chosen[1]], [chosen[0]], "o", ms=9, color=ORANGE, markeredgecolor="white", markeredgewidth=2)
    ax.annotate("chosen", (chosen[1], chosen[0]), textcoords="offset points", xytext=(10, -4), fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("t_decline (score, log scale)")
    ax.set_ylabel("t_review (score, log scale)")
    ax.set_title("Three-way policy search on validation\n"
                 "(blank = infeasible: t_review > t_decline or over capacity)")
    fig.colorbar(im, ax=ax, label="Net benefit, $ per 1,000 transactions (floored at 0)")
    ax.grid(False)
    _save(fig, path)


def swap_figure(sets: dict[str, pd.DataFrame], path) -> None:
    """Amount distribution and product mix for swap set vs shared declines."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = (ORANGE, BLUE, AQUA)
    bins = np.logspace(0, np.log10(max(max(float(d["amount"].max()) for d in sets.values() if len(d)), 10)), 40)
    for (name, d), c in zip(sets.items(), colors):
        if len(d) == 0:
            continue
        axes[0].hist(d["amount"], bins=bins, density=True, histtype="step", color=c, linewidth=2,
                     label=f"{name} (n={len(d):,})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("TransactionAmt ($, log scale)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Amount distribution")
    axes[0].legend(fontsize=8)
    prods = sorted(set().union(*[set(d["ProductCD"].astype(str)) for d in sets.values() if len(d)]))
    x = np.arange(len(prods))
    live = [(n, d) for n, d in sets.items() if len(d)]
    w = 0.8 / max(len(live), 1)
    for i, ((name, d), c) in enumerate(zip(live, colors)):
        share = d["ProductCD"].astype(str).value_counts(normalize=True).reindex(prods).fillna(0)
        axes[1].bar(x + i * w - 0.4 + w / 2, share.to_numpy() * 100, width=w * 0.92, color=c, label=name)
    axes[1].set_xticks(x, prods)
    axes[1].set_ylabel("Share of set (%)")
    axes[1].set_title("Product code mix")
    fig.suptitle("Swap set: transactions the Youden-J and profit-optimal cutoffs decide differently (test)",
                 fontweight="bold")
    _save(fig, path)


def sensitivity_heatmap(pivot: pd.DataFrame, title: str, cbar: str, path, fmt: str = "{:.3f}") -> None:
    fig, ax = plt.subplots(figsize=(6, 4.4))
    im = ax.imshow(pivot.to_numpy(), cmap="Blues", aspect="auto", origin="lower")
    ax.set_xticks(range(pivot.shape[1]), [f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [f"{r:g}" for r in pivot.index])
    ax.set_xlabel(pivot.columns.name)
    ax.set_ylabel(pivot.index.name)
    vmax = np.nanmax(pivot.to_numpy())
    vmin = np.nanmin(pivot.to_numpy())
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            dark = (v - vmin) / (vmax - vmin + 1e-12) > 0.6
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if dark else INK)
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=cbar)
    _save(fig, path)


def psi_bar(psi: pd.DataFrame, warn: float, alert: float, path, top: int = 25) -> None:
    d = psi.dropna(subset=["psi"]).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.26 * len(d) + 1.2))
    colors = [ORANGE if v >= alert else BLUE if v >= warn else "#9c9a94" for v in d["psi"]]
    ax.barh(d["feature"], d["psi"], color=colors, height=0.7)
    ax.axvline(warn, color=INK2, linestyle="--", linewidth=1)
    ax.axvline(alert, color=INK, linestyle="--", linewidth=1)
    ax.text(warn, len(d) - 0.3, f" {warn:g} investigate", fontsize=8, color=INK2)
    ax.text(alert, len(d) - 0.3, f" {alert:g} significant", fontsize=8, color=INK)
    ax.set_xlabel("PSI (training period vs test period)")
    ax.set_title(f"Feature drift: top {len(d)} features by PSI")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def score_stability_fig(stab: pd.DataFrame, warn: float, alert: float, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(stab["days_from_test_start"], stab["score_psi_vs_valid"], "o-", color=BLUE, ms=6)
    axes[0].axhline(warn, color=INK2, linestyle="--", linewidth=1)
    axes[0].axhline(alert, color=INK, linestyle="--", linewidth=1)
    axes[0].set_xlabel("Relative days since start of test period")
    axes[0].set_ylabel("Score PSI vs validation")
    axes[0].set_title("Score distribution drift")
    axes[1].plot(stab["days_from_test_start"], stab["decline_rate"] * 100, "o-", color=ORANGE, ms=6,
                 label="Decline rate at deployed threshold")
    axes[1].plot(stab["days_from_test_start"], stab["fraud_rate"] * 100, "o-", color=AQUA, ms=6,
                 label="Observed fraud rate")
    axes[1].set_xlabel("Relative days since start of test period")
    axes[1].set_ylabel("%")
    axes[1].set_title("Decline rate vs fraud rate per window")
    axes[1].legend(fontsize=8)
    _save(fig, path)


def importance_fig(imp: pd.DataFrame, path, top: int = 25) -> None:
    d = imp.head(top).iloc[::-1]
    fam_color = {"velocity": ORANGE, "entity_risk": AQUA, "linkage": "#4a3aa7"}
    fig, ax = plt.subplots(figsize=(7, 0.26 * len(d) + 1.2))
    ax.barh(d["feature"], d["gain_share"] * 100, color=[fam_color.get(f, BLUE) for f in d["family"]], height=0.7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA, "#4a3aa7")]
    ax.legend(handles, ["raw / vesta", "velocity", "entity risk", "linkage"], loc="lower right", fontsize=8)
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_title(f"Full model: top {len(d)} features by gain")
    ax.grid(axis="y", visible=False)
    _save(fig, path)
