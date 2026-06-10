"""
study3_timeline.py — SPCX Flow Timeline (hardcoded from the S-1).

A single proprietary visual mapping insider unlock tranches against the
index-inclusion window, on a trading-day axis from first trade (~2026-06-12).
The point: distribution is *engineered* — mechanical index demand lands early
(~+15 td), and the lockup releases are scheduled to feed insider supply into the
liquidity that index inclusion + retail create.

All inputs are taken as given from the S-1 (per the spec). Earnings-linked
tranche dates are APPROXIMATE (exact dates are set by SpaceX); fixed-offset
tranches (+70/+90/+105/+120/+135/+180) are exact trading-day offsets.

Run:  python study3_timeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import utils as u

FIRST_TRADE = np.datetime64("2026-06-12")

# NYSE holidays in the IPO+180td window (for approximate calendar-date labels).
NYSE_HOLIDAYS = np.array([
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15",
], dtype="datetime64[D]")

# Index inclusion (S-1): Nasdaq-100 fast entry eligible ~+15 trading days.
NQ_FASTENTRY_TD = 15
NQ_WINDOW = (13, 18)  # shaded "mechanical demand" window around fast entry

# Approximate trading-day offsets of the first two public earnings reports.
Q2_EARN_TD = 38   # ~early Aug 2026  (approx)
Q3_EARN_TD = 103  # ~early Nov 2026  (approx)

# Lockup tranches (% of ELIGIBLE insider shares). Musk excluded (fully restricted).
# kind: 'earnings' (approx date) | 'conditional' | 'fixed' (exact offset).
TRANCHES = [
    dict(td=Q2_EARN_TD, pct=20, kind="earnings",
         label="20% — post Q2'26 earnings\n(first public earnings)"),
    dict(td=Q2_EARN_TD, pct=10, kind="conditional",
         label="+10% conditional\n(stock ≥30% over offer, 5 of 10 days)"),
    dict(td=70, pct=7, kind="fixed", label="7% (+70)"),
    dict(td=90, pct=7, kind="fixed", label="7% (+90)"),
    dict(td=105, pct=7, kind="fixed", label="7% (+105)"),
    dict(td=Q3_EARN_TD, pct=28, kind="earnings",
         label="28% — post Q3'26 earnings"),
    dict(td=120, pct=7, kind="fixed", label="7% (+120)"),
    dict(td=135, pct=7, kind="fixed", label="7% (+135)"),
    dict(td=180, pct=None, kind="fixed", label="Remainder — +180"),
]


def td_to_date(td_offset: int) -> np.datetime64:
    """Calendar date `td_offset` trading days after first trade (NYSE holidays)."""
    return np.busday_offset(FIRST_TRADE, td_offset, roll="forward", holidays=NYSE_HOLIDAYS)


def fmt_date(d: np.datetime64) -> str:
    return pd.Timestamp(d).strftime("%b %d, %Y")


def build_schedule() -> pd.DataFrame:
    """Resolve tranche percentages (remainder = balance to 100%) + cumulative."""
    rows = [dict(t) for t in TRANCHES]
    fixed_sum = sum(r["pct"] for r in rows if r["pct"] is not None and r["kind"] != "conditional")
    for r in rows:
        if r["pct"] is None:
            r["pct"] = max(0, 100 - fixed_sum)  # remainder (base case, ex-conditional)
    order = sorted(rows, key=lambda r: (r["td"], r["kind"] == "conditional"))
    cum = 0
    for r in order:
        if r["kind"] != "conditional":
            cum += r["pct"]
        r["cum_base"] = cum
        r["date"] = fmt_date(td_to_date(r["td"]))
    return pd.DataFrame(order)


def make_chart(sched: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(13, 7))
    ax2 = ax.twinx()

    xmax = 188
    # --- index-inclusion mechanical-demand window ---
    ax.axvspan(NQ_WINDOW[0], NQ_WINDOW[1], color="#2ca02c", alpha=0.16, zorder=0)
    ax.axvline(NQ_FASTENTRY_TD, color="#2ca02c", ls="-", lw=2, zorder=2)
    ax.annotate(f"Nasdaq-100 fast entry  ≈ +{NQ_FASTENTRY_TD} td\n({fmt_date(td_to_date(NQ_FASTENTRY_TD))})\n"
                "→ mechanical index demand",
                xy=(NQ_FASTENTRY_TD, 102), ha="center", va="bottom",
                fontsize=9, color="#1b6b1b", fontweight="bold")

    # --- cumulative insider supply (step) on right axis ---
    base = sched[sched["kind"] != "conditional"]
    xs = [0] + list(base["td"]) + [xmax]
    ys = [0] + list(base["cum_base"]) + [list(base["cum_base"])[-1]]
    ax2.step(xs, ys, where="post", color="#d62728", lw=2.4, zorder=3,
             label="Cumulative insider shares unlocked (% of eligible)")
    # conditional path (to +10% extra early)
    cond = sched[sched["kind"] == "conditional"]
    if len(cond):
        ct = int(cond.iloc[0]["td"])
        ax2.plot([ct, ct], [20, 30], color="#d62728", ls=":", lw=2, zorder=3)
        ax2.annotate("conditional +10%", xy=(ct + 2, 30), fontsize=8, color="#d62728")

    # --- tranche stems ---
    colmap = {"earnings": "#9467bd", "conditional": "#ff7f0e", "fixed": "#1f77b4"}
    for r in sched.itertuples():
        y = r.cum_base
        if r.kind == "conditional":
            continue
        ax.vlines(r.td, 0, 92, color="#cccccc", lw=0.8, zorder=1)
        ax.plot(r.td, 8, "o", color=colmap[r.kind], ms=9, zorder=4)
        lab = r.label.split("\n")[0]
        approx = "  (approx)" if r.kind == "earnings" else ""
        ax.annotate(f"{lab}{approx}\n+{r.td} td · {r.date}",
                    xy=(r.td, 8), xytext=(r.td, 14 + (r.Index % 3) * 13),
                    ha="center", fontsize=7.6, color="#333",
                    arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.7))

    # Musk restriction lane (full bar, no early release)
    ax.barh(-7, xmax, left=0, height=5, color="#444", alpha=0.85, zorder=2)
    ax.annotate("Musk: fully restricted — exempt from early release (no insider selling into the IPO)",
                xy=(xmax / 2, -7), ha="center", va="center", color="white", fontsize=8.5, zorder=5)

    ax.annotate("IPO t=0\nonly SpaceX (the entity) sells;\nno insiders sell into the IPO",
                xy=(0, 8), xytext=(6, 60), fontsize=7.8, color="#333",
                arrowprops=dict(arrowstyle="->", color="#888"))

    ax.set_xlim(-4, xmax)
    ax.set_ylim(-12, 118)
    ax2.set_ylim(-12 / 118 * 100, 100)
    ax.set_yticks([])
    ax.set_xlabel("Trading days since first trade (June 12, 2026)", fontsize=10.5)
    ax2.set_ylabel("Cumulative % of eligible insider shares unlocked", fontsize=10, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title("SPCX Flow Timeline — insider unlocks vs the index-inclusion window\n"
                 "Distribution scheduled to feed mechanical demand (hardcoded from S-1)",
                 fontsize=12.5, fontweight="bold")

    legend_el = [
        Patch(facecolor="#2ca02c", alpha=0.3, label="Index-inclusion / mechanical-demand window"),
        plt.Line2D([0], [0], color="#d62728", lw=2.4, label="Cumulative insider supply unlocked"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#9467bd", ms=9, label="Earnings-linked tranche (approx date)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", ms=9, label="Fixed trading-day tranche (exact)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", ms=9, label="Conditional tranche"),
    ]
    ax.legend(handles=legend_el, loc="center right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("STUDY 3 — SPCX Flow Timeline (from S-1)")
    print("=" * 60)
    sched = build_schedule()
    print(sched[["td", "pct", "kind", "cum_base", "date", "label"]].to_string(index=False))
    sched_out = sched[["td", "pct", "kind", "cum_base", "date", "label"]].copy()
    sched_out["label"] = sched_out["label"].str.replace("\n", " ")
    sched_out.to_csv(u.OUT_TABLES / "spcx_timeline_schedule.csv", index=False)
    make_chart(sched, u.OUT_CHARTS / "spcx_timeline.png")
    print(f"\nNasdaq-100 fast entry ≈ +{NQ_FASTENTRY_TD} td ({fmt_date(td_to_date(NQ_FASTENTRY_TD))})")
    print(f"+180 td lands ≈ {fmt_date(td_to_date(180))}")
    print("\nWrote:")
    print("  outputs/charts/spcx_timeline.png")
    print("  outputs/tables/spcx_timeline_schedule.csv")


if __name__ == "__main__":
    main()
