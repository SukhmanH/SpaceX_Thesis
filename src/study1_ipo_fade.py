"""
study1_ipo_fade.py — The IPO Fade (market-adjusted).

Reproduces and improves the "% change from day-one peak by cohort" chart, but
MARKET-ADJUSTED against QQQ so we strip out the market's beta and isolate the
IPO-specific fade.

Cohorts are fixed BEFORE looking at outcomes (methodology rule):
  * by first-day pop terciles  -> hot / mid / cold
  * a large-cap (>$10B at listing) cohort
  * robustness: a VIX-at-listing regime split

Peak reference = day-1 daily HIGH. This is a *proxy* for the intraday peak and
UNDERSTATES the true peak (the true high tick can exceed the daily High bar only
trivially, but more importantly day-1 High <= the offer-to-open gap means we
measure the fade from a conservative, observable peak). Stated explicitly per
the spec.

Run:  python study1_ipo_fade.py
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

import utils as u

# ----------------------------------------------------------------------------
# CONFIG — fixed in advance (do not edit after seeing outcomes)
# ----------------------------------------------------------------------------
BENCHMARK = "QQQ"
MAX_H = 180  # trading days
HORIZONS = [0, 5, 10, 15, 20, 60, 90, 180]  # 0 = day-1 close
HLABELS = ["Day-1\nclose", "Wk1\n(+5)", "Wk2\n(+10)", "Wk3\n(+15)",
           "Wk4\n(+20)", "+60", "+90", "+180"]

# Universe (spec seed list + every traditional underwritten IPO >$10B at listing,
# 2010–present, that we could source cleanly — review fix #2 to firm up the
# large-cap analog). Direct listings (SPOT/WORK/PLTR/COIN/RBLX) and SPACs are
# EXCLUDED by construction. First-trade dates are VERIFIED from the yfinance row.
# IPO offer prices (USD), from prospectus / IPO-pricing press coverage.
# Used ONLY to compute the first-day pop (offer-to-close) for cohort assignment.
# The pop is computed from the RAW (unadjusted) day-1 close vs this raw offer
# price (see build_ticker_row) so dividend/split adjustments cannot fake a pop.
# Approximate equity valuation at listing (USD billions), widely reported from
# IPO-pricing coverage / prospectus fully-diluted share counts. Used ONLY for
# the >$5B universe filter and the >$10B large-cap cohort. The PRIMARY cohort
# split (pop terciles) does not use this, so the headline result is robust to
# valuation imprecision. Marked approximate in the README.
UNIVERSE_MIN_MCAP_B = 5.0   # spec: market cap > $5B at listing
LARGECAP_MIN_MCAP_B = 10.0  # spec: large-cap cohort
MIN_YEAR = 2010             # spec: ~2010 to present
POP_PLAUSIBLE_MIN = -0.90   # sanity guard for offer-to-close pop
POP_PLAUSIBLE_MAX = 5.00    # +500%; above this is usually a bad split event

# ---------------------------------------------------------------------------
# Universe-completeness set (review pass). These are additional traditional
# underwritten IPOs, 2010–present, with >$10B equity value at listing that are
# NOT in the headline seed list. They are evaluated ONLY as an "expanded
# universe" robustness row — they do NOT enter the headline figures (which stay
# frozen on the seed universe) so the headline conclusion cannot depend on them.
# (offer_price, mcap_listing_b). US-listed underwritten IPOs incl. ADRs.
# Names considered for the >$10B universe but EXCLUDED, with the reason (logged
# for completeness transparency). Threshold held at $10B — not lowered.
# The universe lives in data/ipo_universe.csv so additions/exclusions, offer
# prices, and listing valuations are auditable data changes instead of
# source-code edits.
UNIVERSE_FILE = u.ROOT / "data" / "ipo_universe.csv"


def load_universe_config() -> pd.DataFrame:
    """
    Load the frozen Study-1 IPO universe from data, not source code.

    bucket:
      headline   -> enters the headline Study-1 cohort if it has clean data
      expanded   -> enters only the universe-completeness robustness row
      considered -> logged as considered/dropped with the provided reason
    """
    cfg = pd.read_csv(UNIVERSE_FILE)
    cfg["ticker"] = cfg["ticker"].astype(str).str.strip().str.upper()
    cfg["bucket"] = cfg["bucket"].astype(str).str.strip().str.lower()
    cfg["offer_price"] = pd.to_numeric(cfg["offer_price"], errors="coerce")
    cfg["mcap_listing_b"] = pd.to_numeric(cfg["mcap_listing_b"], errors="coerce")
    cfg["reason"] = cfg["reason"].fillna("").astype(str)
    bad = cfg[~cfg["bucket"].isin(["headline", "expanded", "considered"])]
    if len(bad):
        raise SystemExit(f"invalid bucket(s) in {UNIVERSE_FILE}: {bad['bucket'].unique()}")
    return cfg


UNIVERSE_CFG = load_universe_config()
SEED = list(UNIVERSE_CFG.loc[UNIVERSE_CFG["bucket"] == "headline", "ticker"])
COMPLETENESS_ADDITIONS = {
    r.ticker: (r.offer_price, r.mcap_listing_b)
    for r in UNIVERSE_CFG[UNIVERSE_CFG["bucket"] == "expanded"].itertuples()
}
CONSIDERED_DROPPED = {
    r.ticker: r.reason
    for r in UNIVERSE_CFG[UNIVERSE_CFG["bucket"] == "considered"].itertuples()
}
OFFER_PRICE = {
    r.ticker: float(r.offer_price)
    for r in UNIVERSE_CFG.dropna(subset=["offer_price"]).itertuples()
}
MCAP_AT_LISTING_B = {
    r.ticker: float(r.mcap_listing_b)
    for r in UNIVERSE_CFG.dropna(subset=["mcap_listing_b"]).itertuples()
}
# Expected IPO date per ticker (from build_universe). Used to reject a ticker
# whose yfinance history is actually a DIFFERENT company that later reused the
# symbol (recycled tickers) — build_universe already drops these, but study1
# fetches prices independently, so we guard here too.
EXPECTED_FIRST_TRADE = {
    r.ticker: str(r.first_trade)
    for r in UNIVERSE_CFG.itertuples()
    if "first_trade" in UNIVERSE_CFG.columns and str(getattr(r, "first_trade", "")) not in ("", "nan")
}
RECYCLE_GAP_DAYS = 270


# ----------------------------------------------------------------------------
# Per-ticker metrics
# ----------------------------------------------------------------------------
def resolve_day1_raw_prices(tkr: str, win: pd.DataFrame, ftd, offer: float | None):
    """
    Return original-dollar day-1 close/open for first-day pop calculation.

    Yahoo's unadjusted Close is split-adjusted for normal stock splits, so we
    usually un-apply post-IPO split factors. Some feeds, however, expose
    reclassification/exchange-ratio events as giant "splits" (ALLY's 310.0). Once
    the window is anchored on the audited IPO date, a phantom split dated *on*
    that date is excluded by split_factor_since (strict >), so ALLY's factor is
    1.0 and this guard no longer needs to fire for it. The guard stays as
    defense-in-depth: if some other name ever exposes a factor that is plausible
    when ignored but absurd when applied, keep the uncorrected price and log it.
    """
    close_base = float(win["Close"].iloc[0])
    open_base = float(win["Open"].iloc[0])
    sf = u.split_factor_since(tkr, ftd)
    split_note = ""

    if offer and offer > 0 and sf != 1.0:
        pop_uncorrected = close_base / offer - 1.0
        pop_corrected = (close_base * sf) / offer - 1.0
        uncorrected_ok = POP_PLAUSIBLE_MIN <= pop_uncorrected <= POP_PLAUSIBLE_MAX
        corrected_bad = not (POP_PLAUSIBLE_MIN <= pop_corrected <= POP_PLAUSIBLE_MAX)
        if uncorrected_ok and corrected_bad:
            print(
                f"  [WARN] {tkr}: ignored split factor {sf:g} for IPO pop; "
                f"uncorrected pop={pop_uncorrected:+.1%}, corrected pop={pop_corrected:+.1%}"
            )
            return close_base, open_base, 1.0, f"ignored implausible split factor {sf:g}"

    return close_base * sf, open_base * sf, sf, split_note


def build_ticker_row(tkr: str, df: pd.DataFrame, bench: pd.DataFrame) -> dict | None:
    """
    df / bench are RAW frames (Open, High, Low, Close=unadjusted, AdjClose, Volume).

    First-day pop uses the RAW day-1 close vs the raw offer price. The fade PATH
    uses split/dividend-adjusted prices (AdjClose, and day-1 High scaled by the
    AdjClose/Close factor) so multi-month returns are continuous. Mixing a raw
    offer with an *adjusted* close would invent a fake pop for dividend payers.

    The window is anchored on the audited offer/IPO date (resolve_first_trade),
    not blindly on the earliest cached row, so pre-IPO when-issued/placeholder
    artifacts (ALLY/PECO/VICI) don't corrupt day 1, the fade window, or the split
    factor.
    """
    ftd = u.resolve_first_trade(df, EXPECTED_FIRST_TRADE.get(tkr))
    if ftd.year < MIN_YEAR:
        u.log_drop(tkr, f"first trade {ftd.date()} before {MIN_YEAR}")
        return None

    win = u.window_from(df, ftd, MAX_H)
    if len(win) < 2 or "AdjClose" not in win.columns:
        u.log_drop(tkr, "fewer than 2 trading days / missing adjusted close")
        return None

    # split/dividend factor (== 1.0 when none since IPO) for the adjusted PATH.
    factor = (win["AdjClose"] / win["Close"]).values
    day1_high_adj = float(win["High"].iloc[0] * factor[0])  # adjusted peak proxy
    adj_close = win["AdjClose"]

    # First-day pop = offer-to-close (preferred), from the RAW day-1 close.
    offer = OFFER_PRICE.get(tkr)
    # For the POP we need the ORIGINAL-dollar day-1 close: Yahoo's 'Close' is
    # split-adjusted for normal splits, so un-apply post-IPO splits only when
    # that correction passes the offer-price sanity check.
    day1_close_raw, day1_open_raw, pop_split_factor, pop_split_note = resolve_day1_raw_prices(
        tkr, win, ftd, offer
    )

    if offer and offer > 0:
        pop = day1_close_raw / offer - 1.0
        pop_method = "offer-to-close"
    else:
        pop = day1_close_raw / day1_open_raw - 1.0
        pop_method = "open-to-close(fallback)"
        u.log_drop(tkr, "no offer price -> open-to-close pop fallback")

    # Benchmark adjusted close aligned to this ticker's trading dates.
    bench_adj = bench["AdjClose"].reindex(win.index).ffill()
    bench_day1 = float(bench_adj.iloc[0])

    row = {
        "ticker": tkr,
        "first_trade": ftd.date().isoformat(),
        "year": ftd.year,
        "day1_high_adj": day1_high_adj,
        "day1_open_raw": day1_open_raw,
        "day1_close_raw": day1_close_raw,
        "offer_price": offer,
        "first_day_pop": pop,
        "pop_method": pop_method,
        "pop_split_factor_used": pop_split_factor,
        "pop_split_note": pop_split_note,
        "mcap_listing_b": MCAP_AT_LISTING_B.get(tkr, np.nan),
        "n_days_available": len(win) - 1,
    }
    for h in HORIZONS:
        close_h = float(adj_close.iloc[h]) if h < len(adj_close) else np.nan
        raw = u.cumret(close_h, day1_high_adj)
        b_h = float(bench_adj.iloc[h]) if h < len(bench_adj) else np.nan
        bench_ret = u.cumret(b_h, bench_day1)
        row[f"raw_h{h}"] = raw
        row[f"madj_h{h}"] = u.market_adjust(raw, bench_ret)
    return row


# ----------------------------------------------------------------------------
# Cohort assignment (terciles by pop; large-cap; VIX regime)
# ----------------------------------------------------------------------------
def assign_cohorts(panel: pd.DataFrame, vix_at_listing: dict) -> pd.DataFrame:
    panel = panel.copy()
    # pop terciles -> hot / mid / cold  (objective, set in advance)
    q1, q2 = panel["first_day_pop"].quantile([1 / 3, 2 / 3])
    def pop_bucket(p):
        if p <= q1:
            return "cold"
        if p <= q2:
            return "mid"
        return "hot"
    panel["pop_cohort"] = panel["first_day_pop"].apply(pop_bucket)
    panel["large_cap"] = panel["mcap_listing_b"] > LARGECAP_MIN_MCAP_B

    panel["vix_listing"] = panel["ticker"].map(vix_at_listing)
    vix_med = panel["vix_listing"].median()
    panel["vix_regime"] = np.where(panel["vix_listing"] > vix_med, "high-VIX", "low-VIX")
    panel.attrs["pop_terciles"] = (float(q1), float(q2))
    panel.attrs["vix_median"] = float(vix_med)
    return panel


def cohort_summary(panel: pd.DataFrame, mask, label: str) -> pd.DataFrame:
    sub = panel[mask]
    rows = []
    for h in HORIZONS:
        s = u.tstat(sub[f"madj_h{h}"])
        rows.append({
            "cohort": label, "horizon": h, "n": s["n"],
            "madj_mean": s["mean"], "madj_median": s["median"],
            "madj_se": s["se"], "madj_t": s["t"], "madj_p": s["p"],
            "raw_mean": float(sub[f"raw_h{h}"].mean()),
            "raw_median": float(sub[f"raw_h{h}"].median()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------------
def make_chart(summaries: dict, panel: pd.DataFrame, out_path: Path):
    style = {
        "hot": ("#d62728", "-", "Hot (top pop tercile)"),
        "mid": ("#7f7f7f", "-", "Mid"),
        "cold": ("#1f77b4", "-", "Cold (bottom pop tercile)"),
        "large_cap": ("#000000", "--", "Large-cap (>$10B)"),
    }
    x = np.arange(len(HORIZONS))
    fig, (axm, axmed) = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)

    for key, sm in summaries.items():
        color, ls, base_label = style[key]
        sm = sm.set_index("horizon").loc[HORIZONS]
        mean = sm["madj_mean"].values * 100
        med = sm["madj_median"].values * 100
        se = sm["madj_se"].values * 100
        n = int(sm["n"].max())
        axm.plot(x, mean, ls, color=color, lw=2.2, marker="o", ms=5,
                 label=f"{base_label}  (n={n})")
        if key != "large_cap":
            axm.fill_between(x, mean - se, mean + se, color=color, alpha=0.13)
        axmed.plot(x, med, ls, color=color, lw=2.2, marker="o", ms=5,
                   label=f"{base_label}  (n={n})")

    for ax, sub in ((axm, "Mean path (±1 SE shaded)"), (axmed, "Median path")):
        ax.axhline(0, color="#444", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(HLABELS, fontsize=8.5)
        ax.set_xlabel("Trading days since IPO", fontsize=10)
        ax.set_title(sub, fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    axm.set_ylabel("Market-adjusted cum. return from day-1 high  (%)", fontsize=10)

    n_universe = panel.shape[0]
    fig.suptitle("The IPO Fade — market-adjusted (vs QQQ) drawdown from the day-1 peak\n"
                 "Traditional underwritten IPOs, mkt cap > $5B at listing, 2010–present",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.02,
             f"Peak proxy = day-1 daily High (understates the true intraday peak). "
             f"Universe n={n_universe}. Benchmark = QQQ. "
             f"Mean panel shows the right-tail skew from a few recent runaways; "
             f"the median path is the cleaner central tendency.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_hero_chart(summaries: dict, out_path: Path):
    """
    Figure A (hero, body): single panel, MEDIAN path only, market-adjusted vs QQQ.
    Large-cap (>$10B) and hot cohorts are the prominent solid lines; mid is faint
    context; cold is dropped. No SE bands — clean, like the index chart's hero.
    """
    x = np.arange(len(HORIZONS))
    fig, ax = plt.subplots(figsize=(10, 6.2))

    # draw faint mid first (context), then the two hero lines on top
    series = [
        ("mid", "#9aa0a6", "-", 1.6, 0.55, "Mid pop tercile"),
        ("large_cap", "#111111", "-", 3.0, 1.0, "Large-cap (>$10B at listing)"),
        ("hot", "#d62728", "-", 2.8, 1.0, "Hot (top pop tercile)"),
    ]
    for key, color, ls, lw, alpha, base_label in series:
        sm = summaries[key].set_index("horizon").loc[HORIZONS]
        med = sm["madj_median"].values * 100
        n = int(sm["n"].max())
        ax.plot(x, med, ls, color=color, lw=lw, alpha=alpha, marker="o", ms=5,
                label=f"{base_label}  (n={n})", zorder=3 if key != "mid" else 2)
        if key in ("large_cap", "hot"):  # mark +180 endpoint value
            ax.annotate(f"{med[-1]:+.0f}%", xy=(x[-1], med[-1]),
                        xytext=(x[-1] + 0.12, med[-1]), va="center", ha="left",
                        fontsize=10, fontweight="bold", color=color)

    ax.axhline(0, color="#444", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(HLABELS, fontsize=9)
    ax.set_xlim(-0.3, len(HORIZONS) - 0.3 + 0.6)
    ax.set_ylabel("Market-adjusted median return from day-1 high  (%)", fontsize=10.5)
    ax.set_xlabel("Trading days since IPO", fontsize=10.5)
    ax.set_title("The IPO Fade — large-cap & hot IPOs give back the day-1 peak\n"
                 "Median market-adjusted (vs QQQ) path; traditional underwritten IPOs, 2010–present",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="lower left", framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.25)
    ax.annotate("Peak proxy = day-1 daily High (understates the true intraday peak). "
                "Median path (no SE band). Benchmark = QQQ.",
                xy=(0.5, -0.135), xycoords="axes fraction", ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_decomposition_chart(summaries: dict, out_path: Path):
    """
    Figure B (decomposition): large-cap cohort only. Two MEDIAN lines — raw
    cumulative return from the day-1 high, and market-adjusted (vs QQQ). The gap
    between them is the market's (beta) contribution; the market-adjusted line is
    the IPO-specific fade and is the visually primary (solid, bold) line.
    """
    x = np.arange(len(HORIZONS))
    sm = summaries["large_cap"].set_index("horizon").loc[HORIZONS]
    raw = sm["raw_median"].values * 100
    madj = sm["madj_median"].values * 100
    n = int(sm["n"].max())

    fig, ax = plt.subplots(figsize=(10, 6.2))
    # shaded gap = market (beta) contribution
    ax.fill_between(x, madj, raw, color="#4c78c8", alpha=0.13,
                    label="Market (beta) contribution = gap")
    # raw = secondary (lighter, dashed)
    ax.plot(x, raw, "--", color="#7f7f7f", lw=2.0, marker="o", ms=5,
            label=f"Raw median return from day-1 high  (n={n})", zorder=2)
    # market-adjusted = primary (solid, bold)
    ax.plot(x, madj, "-", color="#c0392b", lw=3.0, marker="o", ms=6,
            label="Market-adjusted (vs QQQ) — IPO-specific fade", zorder=3)

    for y, color in ((raw, "#5f5f5f"), (madj, "#c0392b")):
        ax.annotate(f"{y[-1]:+.0f}%", xy=(x[-1], y[-1]), xytext=(x[-1] + 0.12, y[-1]),
                    va="center", ha="left", fontsize=10, fontweight="bold", color=color)

    ax.axhline(0, color="#444", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(HLABELS, fontsize=9)
    ax.set_xlim(-0.3, len(HORIZONS) - 0.3 + 0.6)
    ax.set_ylabel("Median cumulative return from day-1 high  (%)", fontsize=10.5)
    ax.set_xlabel("Trading days since IPO", fontsize=10.5)
    ax.set_title("Decomposing the large-cap IPO fade — raw vs market-adjusted\n"
                 "The gap is the market's beta; stripping it out reveals the IPO-specific fade",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="lower left", framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.25)
    ax.annotate("Large-cap (>$10B at listing) cohort, median path. Peak proxy = day-1 daily High. "
                "Benchmark = QQQ.",
                xy=(0.5, -0.135), xycoords="axes fraction", ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Robustness battery (review pass) — reads the frozen headline universe; does
# NOT regenerate or alter any headline figure.
# ----------------------------------------------------------------------------
def _fade_value(df, bench, ftd, h, peak="high"):
    """Market-adjusted return at horizon h trading days, from a peak reference."""
    win = u.window_from(df, ftd, h)
    if len(win) <= h or "AdjClose" not in win.columns:
        return np.nan
    f0 = float(win["AdjClose"].iloc[0] / win["Close"].iloc[0])
    if peak == "high":
        peak_level = float(win["High"].iloc[0]) * f0     # adjusted day-1 high
    elif peak == "close":
        peak_level = float(win["AdjClose"].iloc[0])       # adjusted day-1 close
    else:
        return np.nan                                     # VWAP: not in daily OHLCV
    close_h = float(win["AdjClose"].iloc[h])
    badj = bench["AdjClose"].reindex(win.index).ffill()
    bret = float(badj.iloc[h]) / float(badj.iloc[0]) - 1.0
    return (close_h / peak_level - 1.0) - bret


def _agg_fade(frames, bench, h, peak="high"):
    vals = [_fade_value(df, bench, ftd, h, peak) for (df, ftd) in frames.values()]
    vals = [x for x in vals if not pd.isna(x)]
    s = u.tstat(vals)
    return dict(median=s["median"], mean=s["mean"], t=s["t"], n=s["n"])


def run_robustness(panel):
    print("\n" + "=" * 60)
    print("STUDY 1 — ROBUSTNESS BATTERY (large-cap cohort, frozen headline universe)")

    lc = panel[panel["large_cap"]][["ticker", "first_trade"]]
    frames = {}
    for _, r in lc.iterrows():
        df = u.fetch_history(r["ticker"], raw=True)
        if df is not None:
            frames[r["ticker"]] = (df, pd.Timestamp(r["first_trade"]))
    benches = {b: u.fetch_history(b, raw=True) for b in ["QQQ", "SPY", "IWM"]}
    QQQ = benches["QQQ"]

    # ---- Task 1: win rate + full distribution (+180 td, madj vs QQQ) ----
    # The win rate is on the MARKET-ADJUSTED return (stripping out QQQ). We also
    # emit the RAW price decline (close vs day-1 high) so the market-adjusted
    # number isn't misread as a raw price move — e.g. FIG fell raw −84.7% from its
    # day-1 high, and QQQ rose +14.9% over the window, so madj = −99.6% (not a
    # −99.6% price collapse). Both columns are now in study1_winrate_names.csv.
    lc180 = (panel[panel["large_cap"]][["ticker", "first_trade", "raw_h180", "madj_h180"]]
             .dropna(subset=["madj_h180"]).sort_values("madj_h180"))
    n = len(lc180)
    n_neg = int((lc180["madj_h180"] < 0).sum())
    positives = lc180[lc180["madj_h180"] > 0]
    q = lc180["madj_h180"]
    pd.DataFrame({
        "metric": ["n", "n_faded_negative", "pct_faded", "min", "p25", "median", "p75", "max"],
        "value": [n, n_neg, round(n_neg / n, 4), q.min(), q.quantile(.25),
                  q.median(), q.quantile(.75), q.max()],
    }).to_csv(u.OUT_TABLES / "study1_winrate.csv", index=False)
    dist = lc180.rename(columns={"raw_h180": "raw_return_+180",
                                 "madj_h180": "madj_return_+180"}).copy()
    dist["faded"] = dist["madj_return_+180"] < 0
    dist.to_csv(u.OUT_TABLES / "study1_winrate_names.csv", index=False)

    print(f"\nTask 1 — Win rate (large-cap, +180 td, madj vs QQQ):  "
          f"{n_neg} of {n} faded ({n_neg / n:.0%})")
    print(f"  distribution  min={q.min():+.0%}  p25={q.quantile(.25):+.0%}  "
          f"median={q.median():+.0%}  p75={q.quantile(.75):+.0%}  max={q.max():+.0%}")
    print("  positives (did NOT fade): "
          + ", ".join(f"{r.ticker} {r.madj_h180:+.0%}" for r in positives.itertuples()))

    # ---- Tasks 2-4 + completeness: consolidated robustness rows ----
    rows = []
    for label, peak in [("peak: day-1 high (baseline)", "high"),
                        ("peak: day-1 close", "close")]:
        a = _agg_fade(frames, QQQ, 180, peak)
        rows.append(dict(variant=label, **a))
    rows.append(dict(variant="peak: day-1 VWAP (skipped — no intraday data)",
                     median=np.nan, mean=np.nan, t=np.nan, n=0))  # logged in this table
    for b in ["QQQ", "SPY", "IWM"]:
        a = _agg_fade(frames, benches[b], 180, "high")
        rows.append(dict(variant=f"benchmark: {b}" + (" (baseline)" if b == "QQQ" else ""), **a))
    for h in [120, 180, 250]:
        a = _agg_fade(frames, QQQ, h, "high")
        rows.append(dict(variant=f"horizon: +{h} td" + (" (baseline)" if h == 180 else ""), **a))

    # Completeness: expanded universe = seed large-cap + verified >$10B additions.
    add_frames = dict(frames)
    comp_log = []
    for tkr, (offer, mcap) in COMPLETENESS_ADDITIONS.items():
        df = u.fetch_history(tkr, raw=True)
        if df is None:
            comp_log.append(dict(ticker=tkr, status="dropped", reason="no clean data", mcap_b=mcap))
            continue
        ftd = u.resolve_first_trade(df, EXPECTED_FIRST_TRADE.get(tkr))
        if ftd.year < MIN_YEAR:
            comp_log.append(dict(ticker=tkr, status="dropped",
                                 reason=f"first trade {ftd.date()} before {MIN_YEAR}", mcap_b=mcap))
            continue
        add_frames[tkr] = (df, ftd)
        comp_log.append(dict(ticker=tkr, status="added",
                             reason=f"IPO {ftd.date()}, ~${mcap}B at listing", mcap_b=mcap))
    for tkr, reason in CONSIDERED_DROPPED.items():
        comp_log.append(dict(ticker=tkr, status="considered-dropped", reason=reason, mcap_b=np.nan))
    pd.DataFrame(comp_log).to_csv(u.OUT_TABLES / "study1_universe_completeness.csv", index=False)
    n_added = sum(1 for c in comp_log if c["status"] == "added")

    a = _agg_fade(add_frames, QQQ, 180, "high")
    # With the auto-derived universe there is no separate hand-built expansion set:
    # the headline universe IS the complete sourced set, so this row equals the
    # baseline (kept for continuity / to show n explicitly).
    comp_label = (f"expanded universe (+{n_added} more >$10B IPOs)" if n_added
                  else "universe auto-complete (Ritter x Nasdaq x EDGAR; no manual expansion)")
    rows.append(dict(variant=comp_label, **a))

    rob = pd.DataFrame(rows)[["variant", "median", "mean", "t", "n"]]
    rob.to_csv(u.OUT_TABLES / "study1_robustness.csv", index=False)

    print(f"\nUniverse completeness: +{n_added} verified >$10B IPOs added to expanded set; "
          f"{len(CONSIDERED_DROPPED)} names considered-but-dropped (logged).")
    print("Tasks 2-4 + completeness — large-cap median/mean/t (madj +180 unless noted):")
    for r in rob.itertuples():
        if r.n == 0:
            print(f"  {r.variant:<44} skipped")
        else:
            print(f"  {r.variant:<44} median={r.median:+.1%}  mean={r.mean:+.1%}  t={r.t:+.2f}  n={r.n}")
    return rob


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("STUDY 1 — IPO Fade (market-adjusted vs QQQ)")
    print("=" * 60)
    u.begin_study("study1")

    bench = u.fetch_history(BENCHMARK, raw=True)  # need AdjClose + raw for the pop math
    if bench is None:
        raise SystemExit("benchmark QQQ unavailable")

    # VIX at listing for the robustness regime split.
    vix = u.fetch_history("^VIX")

    panel_rows = []
    for tkr in SEED:
        df = u.fetch_history(tkr, raw=True)
        if df is None:
            u.log_drop(tkr, "no clean price data (delisted/unavailable on Yahoo)")
            continue
        # recycled-ticker guard: yfinance history must line up with the IPO date,
        # else the symbol now belongs to a different company.
        exp = EXPECTED_FIRST_TRADE.get(tkr)
        if exp:
            ftd0 = u.first_trade_date(df)
            if abs((ftd0 - pd.Timestamp(exp)).days) > RECYCLE_GAP_DAYS:
                u.log_drop(tkr, f"yfinance first trade {ftd0.date()} != IPO {exp} "
                                f"(recycled/relisted ticker)")
                continue
        # universe filter: market cap > $5B at listing
        mc = MCAP_AT_LISTING_B.get(tkr, np.nan)
        if not (mc > UNIVERSE_MIN_MCAP_B):
            u.log_drop(tkr, f"mcap at listing ~${mc}B <= ${UNIVERSE_MIN_MCAP_B}B threshold")
            continue
        row = build_ticker_row(tkr, df, bench)
        if row is not None:
            panel_rows.append(row)

    panel = pd.DataFrame(panel_rows)

    # VIX close on (or just before) each listing date
    vix_at_listing = {}
    if vix is not None:
        for _, r in panel.iterrows():
            d = pd.Timestamp(r["first_trade"])
            s = vix["Close"].loc[:d]
            vix_at_listing[r["ticker"]] = float(s.iloc[-1]) if len(s) else np.nan

    panel = assign_cohorts(panel, vix_at_listing)

    q1, q2 = panel.attrs["pop_terciles"]
    print(f"\nUniverse after filters: n={len(panel)}")
    print(f"Pop tercile breakpoints: cold <= {q1:+.1%} < mid <= {q2:+.1%} < hot")
    print(f"VIX regime median: {panel.attrs['vix_median']:.1f}\n")
    print(panel[["ticker", "first_trade", "first_day_pop", "pop_method",
                 "mcap_listing_b", "pop_cohort", "large_cap", "vix_regime",
                 "madj_h180"]].to_string(index=False))

    # ---- cohort summaries ----
    summaries = {
        "hot": cohort_summary(panel, panel["pop_cohort"] == "hot", "hot"),
        "mid": cohort_summary(panel, panel["pop_cohort"] == "mid", "mid"),
        "cold": cohort_summary(panel, panel["pop_cohort"] == "cold", "cold"),
        "large_cap": cohort_summary(panel, panel["large_cap"], "large_cap"),
    }
    # robustness: VIX regime
    regime = pd.concat([
        cohort_summary(panel, panel["vix_regime"] == "high-VIX", "high-VIX"),
        cohort_summary(panel, panel["vix_regime"] == "low-VIX", "low-VIX"),
    ], ignore_index=True)

    # ---- write tables ----
    # offer_price + day1_close_raw are emitted so each first_day_pop is directly
    # auditable: pop = day1_close_raw / offer_price - 1 (split-corrected via
    # pop_split_factor_used; a spurious Yahoo factor like ALLY's 310 is logged as
    # ignored in pop_split_note).
    grid_cols = (["ticker", "first_trade", "year", "offer_price", "day1_close_raw",
                  "first_day_pop", "pop_method", "pop_split_factor_used",
                  "pop_split_note", "mcap_listing_b", "pop_cohort", "large_cap",
                  "vix_listing", "vix_regime"]
                 + [f"raw_h{h}" for h in HORIZONS] + [f"madj_h{h}" for h in HORIZONS])
    panel[grid_cols].to_csv(u.OUT_TABLES / "ipo_fade.csv", index=False)

    all_summary = pd.concat(list(summaries.values()) + [regime], ignore_index=True)
    all_summary.to_csv(u.OUT_TABLES / "ipo_fade_cohort_summary.csv", index=False)

    # ---- charts ----
    make_chart(summaries, panel, u.OUT_CHARTS / "ipo_fade.png")            # appendix (full 2x4)
    make_hero_chart(summaries, u.OUT_CHARTS / "ipo_fade_hero.png")         # Figure A (hero)
    make_decomposition_chart(summaries, u.OUT_CHARTS / "ipo_fade_decomposition.png")  # Figure B

    # ---- headline numbers (P1's X-Y) ----
    print("\n" + "=" * 60)
    print("HEADLINE (market-adjusted drawdown by +180 trading days):")
    headline_cohorts = {
        "hot": panel["pop_cohort"] == "hot",
        "large_cap": panel["large_cap"],
        "large+hot": panel["large_cap"] & (panel["pop_cohort"] == "hot"),  # closest SPCX analog
    }
    for key, mask in headline_cohorts.items():
        d = panel.loc[mask, "madj_h180"].dropna()
        s = u.tstat(d)
        print(f"  {key:<10} n={s['n']:<2} mean={s['mean']:+.1%}  median={s['median']:+.1%}  "
              f"range=[{d.min():+.1%}, {d.max():+.1%}]  t={s['t']:+.2f} (p={s['p']:.3f})")

    # Robustness #1: large∩hot with the top-2 positive outliers removed (does the
    # SPCX analog stay negative when the right tail is trimmed?). Median preferred.
    lh = panel.loc[panel["large_cap"] & (panel["pop_cohort"] == "hot"), ["ticker", "madj_h180"]].dropna()
    lh = lh.sort_values("madj_h180", ascending=False)
    drop2 = list(lh["ticker"].head(2))
    lh_trim = lh.iloc[2:]
    sf, stm = u.tstat(lh["madj_h180"]), u.tstat(lh_trim["madj_h180"])
    print(f"\nRobustness #1 — large∩hot (SPCX analog), drop top-2 positive outliers {drop2}:")
    print(f"  full     n={sf['n']:<2} mean={sf['mean']:+.1%}  median={sf['median']:+.1%}")
    print(f"  trimmed  n={stm['n']:<2} mean={stm['mean']:+.1%}  median={stm['median']:+.1%}  "
          f"-> stays {'NEGATIVE' if stm['median'] < 0 else 'non-negative'}")

    # Robustness #2: VIX-at-listing regime.
    print("\nRobustness #2 — fade by VIX-at-listing regime (madj +180):")
    for reg in ["high-VIX", "low-VIX"]:
        d = panel.loc[panel["vix_regime"] == reg, "madj_h180"].dropna()
        s = u.tstat(d)
        print(f"  {reg:<9} n={s['n']:<2} mean={s['mean']:+.1%}  t={s['t']:+.2f}")

    # ---- robustness battery + universe completeness (new tables; figures frozen) ----
    run_robustness(panel)

    print("\nWrote:")
    print("  outputs/charts/ipo_fade.png            (appendix: full 4-cohort, mean|median)")
    print("  outputs/charts/ipo_fade_hero.png       (Figure A: hero, median, large-cap+hot)")
    print("  outputs/charts/ipo_fade_decomposition.png (Figure B: raw vs market-adjusted)")
    print("  outputs/tables/ipo_fade.csv  +  ipo_fade_cohort_summary.csv")
    print("  outputs/tables/study1_winrate.csv  +  study1_winrate_names.csv")
    print("  outputs/tables/study1_robustness.csv")
    print("  outputs/tables/study1_universe_completeness.csv")
    return panel


if __name__ == "__main__":
    main()
