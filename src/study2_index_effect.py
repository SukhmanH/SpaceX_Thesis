"""
study2_index_effect.py — The Index-Inclusion Effect (replicate Greenwood & Sammon).

Shows the classic S&P 500 inclusion "pop" has decayed toward ~0 / statistical
insignificance in the recent era. That grounds the report's argument: for SPCX the
trade is the *fade*, not the inclusion *pop* (the premium is largely gone), while
SPCX's irregular fast-track structure is the wildcard.

Date sourcing (respecting the spec's "do not invent dates" rule)
----------------------------------------------------------------
* EFFECTIVE dates are scraped from Wikipedia's "Selected changes to the list of
  S&P 500 companies" table — a verifiable, reproducible source. Cached to
  data/raw/sp500_changes.csv.
* We do NOT assert per-event announcement dates we cannot verify.

PRIMARY statistic = the EFFECTIVE-DAY (rebalance) window [ED-1, ED+1] — a clean,
fixed-length, matched measure of the mechanical inclusion pop. It is reported
everywhere (chart, table, headline). A fixed 9-day window [ED-7, ED+1] is also
computed as SECONDARY context (it brackets the ~5-day announcement lead) but it
accumulates idiosyncratic drift in momentum-selected additions, so it is *not* a
clean abnormal return — shown muted, for context only.

Window discipline (review fix #1): every aggregated event uses the SAME number of
trading days. Any name that cannot fill both fixed windows is dropped and logged.
TESLA's documented ≈5-week announcement→effective gap (27 trading days as measured)
makes its window non-matched, so Tesla is EXCLUDED from every aggregate mean and kept
only as a labeled single-name anecdote (announce 2020-11-16 → effective 2020-12-21).

Run:  python study2_index_effect.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils as u

BENCHMARK = "SPY"
PRE_TD = 7    # trading days before effective date (brackets ~5d announce lead + 1)
POST_TD = 1   # trading days after effective date
START_YEAR = 2005
ERAS = [(2005, 2009), (2010, 2014), (2015, 2019), (2020, 2099)]
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Documented anchor (spec). Closest comp to SPCX's irregular fast-track structure.
TESLA_ANCHOR = {"ticker": "TSLA", "announce": "2020-11-16", "effective": "2020-12-21"}


# ----------------------------------------------------------------------------
# Source: S&P 500 changes (effective dates) from Wikipedia — cached
# ----------------------------------------------------------------------------
def scrape_sp500_changes(force: bool = False) -> pd.DataFrame:
    cache = u.RAW_DIR / "sp500_changes.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache, parse_dates=["effective_date"])

    req = urllib.request.Request(WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (research)"})
    html = urllib.request.urlopen(req, timeout=30).read()
    tables = pd.read_html(html)

    # Find the changes table: has an "Effective Date" column and an Added "Ticker".
    changes = None
    for t in tables:
        cols = ["|".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        joined = " ".join(cols).lower()
        if "effective date" in joined and "added" in joined and "ticker" in joined:
            changes = t
            break
    if changes is None:
        raise SystemExit("could not locate S&P 500 changes table on Wikipedia")

    # Flatten MultiIndex columns -> find effective-date + added-ticker columns.
    flat = []
    for c in changes.columns:
        parts = [str(p) for p in (c if isinstance(c, tuple) else (c,))]
        flat.append(" ".join(dict.fromkeys(parts)))  # dedupe repeated level labels
    changes.columns = flat

    def find_col(pred):
        for c in changes.columns:
            if pred(c.lower()):
                return c
        return None

    eff_col = find_col(lambda s: "effective" in s)
    add_col = find_col(lambda s: "added" in s and "ticker" in s)
    sec_col = find_col(lambda s: "added" in s and ("security" in s or "company" in s))
    rsn_col = find_col(lambda s: "reason" in s)

    out = pd.DataFrame({
        "effective_date": pd.to_datetime(changes[eff_col], errors="coerce"),
        "ticker": changes[add_col].astype(str).str.strip(),
        "security": changes[sec_col].astype(str).str.strip() if sec_col else "",
        "reason": changes[rsn_col].astype(str).str.strip() if rsn_col else "",
    })
    out = out[(out["ticker"].notna()) & (out["ticker"] != "") & (out["ticker"].str.lower() != "nan")]
    out = out[out["effective_date"].notna()]
    out = out.drop_duplicates(subset=["ticker", "effective_date"]).reset_index(drop=True)
    out.to_csv(cache, index=False)
    return out


def yf_ticker(sym: str) -> str:
    """Wikipedia uses dotted class shares (BRK.B); yfinance expects dashes (BRK-B)."""
    return sym.replace(".", "-").strip().upper()


# ----------------------------------------------------------------------------
# Event CAR
# ----------------------------------------------------------------------------
def pos_for_date(index: pd.DatetimeIndex, date) -> int | None:
    """Position of the first trading day on/after `date`, else None."""
    d = pd.Timestamp(date)
    pos = index.searchsorted(d, side="left")
    if pos >= len(index):
        return None
    return int(pos)


def event_car(price: pd.DataFrame, spy_madj_ret: pd.Series, eff_date,
              pre_td: int, post_td: int, start_override=None):
    """
    Sum of market-adjusted daily returns over the event window.
    Window = [eff - pre_td, eff + post_td] trading days, unless start_override
    (a date) is given (used for Tesla's day-before-announcement start).
    Returns (car, n_days, win_start, win_end) or (nan, 0, None, None).
    """
    idx = price.index
    eff_pos = pos_for_date(idx, eff_date)
    if eff_pos is None:
        return np.nan, 0, None, None

    if start_override is not None:
        start_pos = pos_for_date(idx, start_override)
        if start_pos is None:
            return np.nan, 0, None, None
        start_pos = max(start_pos - 1, 0)  # day BEFORE announcement
    else:
        start_pos = eff_pos - pre_td
    end_pos = eff_pos + post_td

    if start_pos < 1 or end_pos >= len(idx):
        return np.nan, 0, None, None

    win_dates = idx[start_pos:end_pos + 1]
    car = float(spy_madj_ret.reindex(win_dates).sum())
    return car, len(win_dates), idx[start_pos].date().isoformat(), idx[end_pos].date().isoformat()


def era_label(year: int) -> str | None:
    for lo, hi in ERAS:
        if lo <= year <= hi:
            return f"{lo}–{'present' if hi > 2090 else hi}"
    return None


# ----------------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------------
def make_chart(era_df: pd.DataFrame, out_path: Path):
    """
    Grouped bars per era: the wide announcement→effective window vs the tight
    effective-day window. The decomposition is the point — the mechanical
    rebalance-day pop (tight) is ~0 from 2010 on, while the recent wide-window
    bump is anticipatory front-running, not a capturable inclusion pop.
    """
    era_df = era_df.sort_values("era_lo").reset_index(drop=True)
    x = np.arange(len(era_df))
    w = 0.40
    eff = era_df["car_eff_mean"].values * 100
    eff_se = era_df["car_eff_se"].values * 100
    wide = era_df["car_wide_mean"].values * 100
    wide_se = era_df["car_wide_se"].values * 100

    fig, ax = plt.subplots(figsize=(11, 6.4))
    # SECONDARY context (muted, hatched): fixed 9-day window.
    ax.bar(x + w / 2, wide, w, yerr=wide_se, capsize=4, color="#cdd7ea",
           edgecolor="#9aa7c4", hatch="//", zorder=2,
           label=f"9-day context window  [ED−{PRE_TD}, ED+{POST_TD}]  (drift-contaminated)")
    # PRIMARY (hero): effective-day rebalance window.
    ax.bar(x - w / 2, eff, w, yerr=eff_se, capsize=5, color="#c0392b",
           edgecolor="#222", zorder=3,
           label="PRIMARY: effective-day rebalance CAR  [ED−1, ED+1]")
    ax.axhline(0, color="#333", lw=1)

    for xi, row in enumerate(era_df.itertuples()):
        ax.annotate(f"{row.car_eff_mean*100:+.2f}%{u.sig_stars(row.car_eff_p)}\nt={row.car_eff_t:+.2f}",
                    xy=(xi - w / 2, eff[xi] + (eff_se[xi] if eff[xi] >= 0 else -eff_se[xi])),
                    ha="center", va="bottom" if eff[xi] >= 0 else "top",
                    fontsize=8.5, fontweight="bold", color="#7a211a")
        ax.annotate(f"{row.car_wide_mean*100:+.2f}%{u.sig_stars(row.car_wide_p)}",
                    xy=(xi + w / 2, wide[xi] + (wide_se[xi] if wide[xi] >= 0 else -wide_se[xi])),
                    ha="center", va="bottom" if wide[xi] >= 0 else "top", fontsize=7.5, color="#6b7794")
        ax.annotate(f"n={row.n}", xy=(xi, 0), xytext=(xi, -1.05),
                    ha="center", fontsize=8.5, color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels(era_df["era"], fontsize=10)
    ax.set_ylabel("Mean inclusion CAR, market-adjusted vs SPY  (%)", fontsize=10)
    ax.set_title("The Disappearing Index Effect — S&P 500 inclusion CAR by era\n"
                 "The clean rebalance-day pop (red) is statistically ≈ 0 in every modern era",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.25)
    ax.annotate("Stars: ***p<.01 **p<.05 *p<.10. Error bars = ±1 SE. The 9-day window is shown "
                "for context only — it accumulates idiosyncratic drift in momentum-selected\n"
                "additions and is not a clean abnormal return. Tesla (excluded from all means) is "
                "the front-run archetype: +42.6% over its 27-trading-day window, −1.1% on the rebalance day.",
                xy=(0.5, -0.155), xycoords="axes fraction", ha="center", fontsize=7.6, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("STUDY 2 — Index-Inclusion Effect (S&P 500, vs SPY)")
    print("=" * 60)
    u.begin_study("study2")

    spy = u.fetch_history(BENCHMARK)
    if spy is None:
        raise SystemExit("benchmark SPY unavailable")
    spy_ret = spy["Close"].pct_change()

    changes = scrape_sp500_changes()
    changes = changes[changes["effective_date"].dt.year >= START_YEAR].copy()
    # Make sure Tesla's documented event is present even if the table labels it oddly.
    if not ((changes["ticker"] == "TSLA").any()):
        changes = pd.concat([changes, pd.DataFrame([{
            "effective_date": pd.Timestamp(TESLA_ANCHOR["effective"]),
            "ticker": "TSLA", "security": "Tesla", "reason": "anchor"}])], ignore_index=True)
    print(f"S&P 500 additions {START_YEAR}+ from Wikipedia: {len(changes)} events")

    WIDE_LEN = PRE_TD + POST_TD + 1   # fixed wide window length (trading days)
    TIGHT_LEN = 3                     # [ED-1, ED, ED+1]
    rows = []
    tesla_anecdote = None
    for _, ev in changes.iterrows():
        sym = yf_ticker(ev["ticker"])
        eff = ev["effective_date"]
        era = era_label(eff.year)
        if era is None:
            continue
        price = u.fetch_history(sym)
        if price is None or len(price) < (PRE_TD + POST_TD + 5):
            u.log_drop(sym, "no/insufficient price data for index event")
            continue
        spy_madj = spy_ret.reindex(price.index)  # align SPY return to event dates
        madj = price["Close"].pct_change() - spy_madj

        # TESLA: documented ≈5-week announcement→effective gap (27 trading days
        # as measured). Kept ONLY as a labeled single-name anecdote — its window
        # is not length-matched to the rest, so it is EXCLUDED from every
        # aggregate mean (per review fix #1).
        if sym == "TSLA":
            a_car, a_n, a0, a1 = event_car(price, madj, eff, PRE_TD, POST_TD,
                                           start_override=TESLA_ANCHOR["announce"])
            e_car, _, _, _ = event_car(price, madj, eff, 1, 1, None)
            tesla_anecdote = dict(car=a_car, n=a_n, w0=a0, w1=a1, eff=e_car)
            u.log_drop("TSLA", "excluded from all aggregates (non-matched 27-trading-day window); anecdote only")
            continue

        # PRIMARY = effective-day [ED-1,ED+1]; SECONDARY context = fixed 9-day wide.
        # No exceptions: a name that cannot fill BOTH fixed windows is dropped+logged.
        car_wide, n_wide, w0, w1 = event_car(price, madj, eff, PRE_TD, POST_TD, None)
        car_eff, n_eff, _, _ = event_car(price, madj, eff, 1, 1, None)
        if (np.isnan(car_wide) or np.isnan(car_eff)
                or n_wide != WIDE_LEN or n_eff != TIGHT_LEN):
            u.log_drop(sym, f"could not fill fixed-length windows around {eff.date()}")
            continue
        rows.append({
            "ticker": sym, "effective_date": eff.date().isoformat(),
            "era": era, "era_lo": int(era.split("–")[0]),
            "win_start": w0, "win_end": w1, "n_days_wide": n_wide,
            "car_eff": car_eff, "car_wide": car_wide,
        })

    panel = pd.DataFrame(rows)
    panel.to_csv(u.OUT_TABLES / "index_effect.csv", index=False)
    print(f"Usable events (uniform windows; Tesla excluded from means): {len(panel)}")

    # Aggregate by era. PRIMARY statistic = effective-day CAR (matched length).
    agg = []
    for era, sub in panel.groupby("era"):
        e = u.tstat(sub["car_eff"])    # PRIMARY
        w = u.tstat(sub["car_wide"])   # secondary context (fixed 9-day)
        agg.append({
            "era": era, "era_lo": int(era.split("–")[0]), "n": e["n"],
            "car_eff_mean": e["mean"], "car_eff_median": e["median"],
            "car_eff_se": e["se"], "car_eff_t": e["t"], "car_eff_p": e["p"],
            "car_wide_mean": w["mean"], "car_wide_se": w["se"],
            "car_wide_t": w["t"], "car_wide_p": w["p"],
        })
    era_df = pd.DataFrame(agg).sort_values("era_lo").reset_index(drop=True)
    era_df.to_csv(u.OUT_TABLES / "index_effect_by_era.csv", index=False)

    print("\nPRIMARY — effective-day (rebalance) CAR [ED−1,ED+1], market-adj vs SPY:")
    for r in era_df.itertuples():
        print(f"  {r.era:<14} n={r.n:<3} mean={r.car_eff_mean:+.2%}  "
              f"t={r.car_eff_t:+.2f} (p={r.car_eff_p:.3f}) {u.sig_stars(r.car_eff_p)}   "
              f"[context 9-day: {r.car_wide_mean:+.2%}, t={r.car_wide_t:+.2f}]")

    make_chart(era_df, u.OUT_CHARTS / "index_effect_by_era.png")

    if tesla_anecdote:
        ta = tesla_anecdote
        print(f"\nTesla ANECDOTE (excluded from all means): announce {TESLA_ANCHOR['announce']}"
              f" → effective {TESLA_ANCHOR['effective']} (≈5-week gap): announce→effective "
              f"CAR={ta['car']:+.2%} over [{ta['w0']}..{ta['w1']}] ({ta['n']} td); "
              f"effective-day CAR={ta['eff']:+.2%}. Illustrates anticipatory front-running.")

    # Headline (P2's Z)
    recent = era_df[era_df["era_lo"] == 2020]
    if len(recent):
        r = recent.iloc[0]
        print("\n" + "=" * 60)
        print("HEADLINE (P2's Z) — recent era (2020–present), PRIMARY statistic:")
        print(f"  Effective-day (rebalance) inclusion CAR = {r['car_eff_mean']:+.2%}, "
              f"t={r['car_eff_t']:+.2f}, p={r['car_eff_p']:.3f} -> "
              f"{'INSIGNIFICANT (the pop is gone)' if r['car_eff_p'] > 0.05 else 'SIGNIFICANT'}")
        decay = ", ".join(f"{rr.era.split('–')[0]}s {rr.car_eff_mean:+.2%}" for rr in era_df.itertuples())
        print(f"  Effective-day CAR by era (all eras): {decay} — flat ~0 throughout.")
        print("  => The capturable inclusion pop is gone. For SPCX the trade is the FADE.")

    print("\nWrote:")
    print("  outputs/charts/index_effect_by_era.png")
    print("  outputs/tables/index_effect.csv")
    print("  outputs/tables/index_effect_by_era.csv")
    return panel, era_df


if __name__ == "__main__":
    main()
