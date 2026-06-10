"""
build_universe.py — assemble the Study-1 IPO universe from REAL public sources
instead of a hand-curated ticker list.

Why this exists
---------------
The Study-1 universe used to be a hardcoded list of ~40 tickers. That invites
cherry-picking concerns for a public research report. This script *derives* the
universe from public, citeable, cached sources and applies the spec's filters
mechanically. Three sources, each covering the others' gaps:

  * MASTER LIST  — Jay Ritter (University of Florida) "IPO-age" dataset
    (``site.warrington.ufl.edu/ritter/files/IPO-age.xlsx``), the academic
    standard the spec mandates. Every operating-company IPO 1975-2025 with offer
    date, name, IPO-era ticker, ADR flag, and post-issue share count. It EXCLUDES
    SPACs, closed-end funds and spinoffs by construction, so those drop for free
    (verified: GEHC/VLTO/GEV/SOLV spinoffs, SOFI/LCID/DKNG SPAC-mergers absent).

  * OFFER PRICE + IPO-era ticker + exact priced date — the Nasdaq IPO calendar
    (``api.nasdaq.com/api/ipo/calendar?date=YYYY-MM``). Ritter has no offer price;
    Nasdaq has complete monthly priced-deal lists back to 2010 (ticker, price,
    date, exchange) with no truncation. Both Ritter and Nasdaq carry the IPO-era
    ticker; a small TICKER_ALIASES map resolves post-IPO renames that break a
    present-day lookup (Facebook FB -> META).

  * SHARES-AT-LISTING — SEC EDGAR XBRL company facts
    (``data.sec.gov/api/xbrl/companyfacts/CIK##########.json``): the authoritative
    common-shares-outstanding from the first post-IPO filing.

Market cap at listing = offer price x shares-outstanding-at-listing. No single
share source is clean (calibrated against the prior hand-checked values):
  * EDGAR  -> exact for single-class domestic (GM, HCA) and 1:1 ADRs (BABA), but
    reports *ordinary* shares for n:1 ADRs (PDD/TME overstate by the ADR ratio)
    and can pick a partial early period for multi-class names (RIVN/MBLY).
  * yfinance ``get_shares_full`` at the IPO date -> right for most multi-class
    (ABNB/SNOW/RIVN), but undercounts a few to a single class (PINS/AFRM/IQ) and
    is missing for pre-2016 / ADR names.
  * Ritter ``post-issue shares`` -> correct for NON-ADR names (incl. FB/META and
    the multi-class names yfinance misses), but in *ordinary* units for ADRs.

So we RECONCILE: gather the ADR-appropriate estimates, discard obvious
single-class undercounts (< 0.55x the max), take the median when >=2 sources
agree (high confidence), else fall back to a logged hand-verified value, else a
best-effort single estimate. Every source estimate + the chosen value + its
provenance + confidence is written to data/raw/universe_audit.csv.

Filters (fixed in advance, per spec):
  * offer year >= 2010
  * market cap at listing > $5B
  * traditional underwritten IPO  -> Ritter membership handles SPACs/spinoffs;
    direct listings ARE in Ritter, so an explicit spec-mandated exclusion removes
    them (the spec names Spotify/Slack/Palantir/Coinbase/Roblox as direct
    listings to exclude/segment).

Outputs:
  * data/ipo_universe.csv        — ticker,bucket,offer_price,mcap_listing_b,reason
                                    (+ provenance). bucket='headline' for every
                                    qualifying IPO, 'considered' for logged
                                    near-miss / explicit exclusions. study1 reads
                                    this.
  * data/raw/universe_audit.csv  — every matched 2010+ Ritter row, all share
                                    estimates, chosen mcap, source + confidence,
                                    and keep/drop reason (full transparency).
  * cached raw downloads in data/raw/ (ritter_ipo_age.xlsx, nasdaq_ipos/YYYY-MM.json,
    sec_ticker_cik.json, edgar/CIK##########.json, _yf_shares_at_ipo.csv)

Run:  python build_universe.py     (first run is slow: EDGAR/yfinance fetches are
                                     cached, so re-runs are fast and resumable)
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import requests

import utils as u

# ----------------------------------------------------------------------------
# CONFIG — fixed in advance (do not edit after seeing outcomes)
# ----------------------------------------------------------------------------
MIN_YEAR = 2010            # spec: ~2010 to present
MIN_MCAP_B = 5.0           # spec: market cap > $5B at listing
NEAR_MISS_FLOOR_B = 3.0    # log $3-5B names as 'considered'; below: bulk-drop
THIS_YEAR = 2026
# IPOs after this can't yet have a full +180 trading-day (~261 calendar-day)
# outcome as of the report date (2026-06-04): 2026-06-04 - 261d ~= 2025-09-15.
RECENT_CUTOFF = "2025-09-15"
YF_GLITCH_RATIO = 3.0      # reject a yfinance at-IPO share count > this x its recent
                           # count (a known data glitch that overstates market cap)
RECYCLE_GAP_DAYS = 270     # treat the ticker as reused by a DIFFERENT company only
                           # if yfinance's first trade is this far from the IPO date.
                           # Genuine recycling is years apart; a few-month gap is just
                           # a when-issued/data artifact (e.g. ALLY, PECO) -> keep.
# Only run the expensive per-name EDGAR/yfinance share lookups for names whose
# free pre-estimate (offer x Ritter post-issue shares) could plausibly clear $5B.
# Ritter post-issue is a TOTAL count (rarely a >2x undercount), so 2.5B leaves a
# 2x safety margin. Names missing Ritter shares with a real offer also qualify.
CANDIDATE_PRE_FLOOR_B = 2.5
UNDERCOUNT_KEEP = 0.55     # discard share estimates < this x the max (single-class)
CONSENSUS_SPREAD = 0.35    # >=2 estimates within this rel. spread => high confidence

RITTER_URL = "https://site.warrington.ufl.edu/ritter/files/IPO-age.xlsx"
# Nasdaq's IPO calendar (priced deals) — complete monthly coverage back to 2010
# with ticker + offer price + exact priced date + exchange (no truncation, unlike
# stockanalysis which was 2019+ and capped at ~500 rows/year).
NASDAQ_URL = "https://api.nasdaq.com/api/ipo/calendar?date={month}"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
UA = "Mozilla/5.0 (research; SPCX report universe build)"
SEC_UA = "spcx-research sh80business@gmail.com"  # SEC requires a real UA w/ contact

# Post-IPO ticker renames that break a present-day yfinance/EDGAR lookup. Nasdaq
# and Ritter both carry the IPO-era ticker; map it to the symbol that trades now.
# (A ticker-identity fix, not a hand-picked universe.)
TICKER_ALIASES = {"FB": "META", "WISH": "LOGC"}  # Facebook->Meta, Wish->ContextLogic

# Direct listings (no underwriter / no lockup) — present in Ritter but EXCLUDED
# by the spec ("Exclude or segment separately: direct listings"). A documented
# methodological exclusion, not a hand-picked universe.
DIRECT_LISTINGS = {
    "SPOT", "WORK", "PLTR", "ASAN", "COIN", "RBLX", "SQSP", "AMPL", "WRBY", "ZIP",
}

RAW = u.RAW_DIR
EDGAR_DIR = RAW / "edgar"
EDGAR_DIR.mkdir(parents=True, exist_ok=True)
RITTER_CACHE = RAW / "ritter_ipo_age.xlsx"
CIK_CACHE = RAW / "sec_ticker_cik.json"
YF_SHARES_CACHE = RAW / "_yf_shares_at_ipo.csv"   # ticker -> shares at IPO date
AUDIT_OUT = RAW / "universe_audit.csv"
UNIVERSE_OUT = u.ROOT / "data" / "ipo_universe.csv"
CURATED_BACKUP = u.ROOT / "data" / "ipo_universe_curated_backup.csv"

_SESSION = requests.Session()


# ----------------------------------------------------------------------------
# 1. Ritter master list
# ----------------------------------------------------------------------------
def fetch_ritter() -> pd.DataFrame:
    if not RITTER_CACHE.exists():
        print(f"[ritter] downloading {RITTER_URL}")
        r = _SESSION.get(RITTER_URL, headers={"User-Agent": UA}, timeout=120)
        r.raise_for_status()
        RITTER_CACHE.write_bytes(r.content)
    else:
        print(f"[ritter] using cache {RITTER_CACHE.name}")
    df = pd.read_excel(RITTER_CACHE, sheet_name=0).rename(columns={
        "offer date": "offer_date", "IPO name": "name", "Ticker": "ticker",
        "ADR (2=ADR)": "adr", "Post-issue shares": "post_issue_shares"})
    df["offer_date"] = pd.to_numeric(df["offer_date"], errors="coerce")
    df = df.dropna(subset=["offer_date"])
    df["year"] = (df["offer_date"] // 10000).astype(int)
    df["offer_dt"] = pd.to_datetime(df["offer_date"].astype(int).astype(str),
                                    format="%Y%m%d", errors="coerce")
    # Ritter stores some tickers with a stray leading "=" (spreadsheet artifact).
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper().str.lstrip("=")
    df["name"] = df["name"].astype(str).str.strip()
    df["adr"] = pd.to_numeric(df["adr"], errors="coerce")
    df["post_issue_shares"] = pd.to_numeric(df["post_issue_shares"], errors="coerce")
    rec = df[df["year"] >= MIN_YEAR].copy()
    print(f"[ritter] {len(df)} total rows, {len(rec)} since {MIN_YEAR}")
    return rec[["offer_date", "offer_dt", "year", "name", "ticker",
                "adr", "post_issue_shares"]]


# ----------------------------------------------------------------------------
# 2. Nasdaq IPO calendar -> offer price + IPO-era ticker + exact priced date
# ----------------------------------------------------------------------------
NASDAQ_DIR = RAW / "nasdaq_ipos"
NASDAQ_DIR.mkdir(parents=True, exist_ok=True)


def fetch_nasdaq_month(month: str) -> list[dict]:
    """Priced-IPO rows for one YYYY-MM (cached). Empty list on any miss/error."""
    cache = NASDAQ_DIR / f"{month}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        r = _SESSION.get(NASDAQ_URL.format(month=month),
                         headers={"User-Agent": UA, "Accept": "application/json"},
                         timeout=45)
        time.sleep(0.25)
        rows = (((r.json() or {}).get("data") or {}).get("priced") or {}).get("rows") or []
    except Exception:
        rows = []
    cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def build_nasdaq_index(years) -> pd.DataFrame:
    rows = []
    for y in years:
        for mo in range(1, 13):
            month = f"{y}-{mo:02d}"
            for r in fetch_nasdaq_month(month):
                price = pd.to_numeric(str(r.get("proposedSharePrice", "")).replace("$", "").replace(",", ""),
                                      errors="coerce")
                dt = pd.to_datetime(r.get("pricedDate"), errors="coerce")
                rows.append({
                    "src_ticker": str(r.get("proposedTickerSymbol", "")).strip().upper(),
                    "src_name": str(r.get("companyName", "")).strip(),
                    "src_date": dt, "src_year": dt.year if pd.notna(dt) else np.nan,
                    "offer_price": float(price) if pd.notna(price) else np.nan,
                    "exchange": str(r.get("proposedExchange", "")).strip(),
                })
    sa = pd.DataFrame(rows)
    sa = sa[sa["src_ticker"] != ""].drop_duplicates(subset=["src_ticker", "src_date"])
    print(f"[nasdaq] {len(sa)} priced IPO rows {min(years)}-{max(years)} "
          f"({sa['offer_price'].notna().sum()} with an offer price)")
    return sa


# ----------------------------------------------------------------------------
# 3. Match Ritter <-> Nasdaq  (ticker+date, then fuzzy name+date)
# ----------------------------------------------------------------------------
_SUFFIX = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|plc|ltd|limited|holdings?|"
    r"group|sa|nv|ag|the|class\s+[ab]|cl\s+[ab]|common\s+stock|ordinary\s+shares|"
    r"ads|adr|lp|llc|technologies|technology)\b")


def _norm_name(s: str) -> set[str]:
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    s = _SUFFIX.sub(" ", s)
    return {t for t in s.split() if len(t) > 1}


def match_offer_price(rit: pd.DataFrame, sa: pd.DataFrame) -> pd.DataFrame:
    """Attach offer price (+ IPO-era ticker) from Nasdaq to each Ritter row.

    Ritter and Nasdaq both store the IPO-ERA ticker, so a ticker match plus a
    same-window priced date is exact even for recycled tickers (the date
    disambiguates SNOW-2014 Intrawest vs SNOW-2020 Snowflake). Fuzzy name+date
    is the fallback. The yfinance/EDGAR-tradeable ticker is the IPO-era ticker
    mapped through TICKER_ALIASES (e.g. FB -> META)."""
    sa_p = sa[sa["offer_price"].notna() & (sa["offer_price"] > 0)].copy()
    by_ticker: dict[str, list] = {}
    for r in sa_p.itertuples():
        by_ticker.setdefault(r.src_ticker, []).append(r)
    sa_p["norm"] = sa_p["src_name"].map(_norm_name)
    sa_norm = list(sa_p.itertuples())

    out = []
    for r in rit.itertuples():
        chosen, method = None, ""
        for cand in by_ticker.get(r.ticker, []):    # ticker + priced date (+-10d)
            if pd.notna(cand.src_date) and abs((cand.src_date - r.offer_dt).days) <= 10:
                if chosen is None or abs((cand.src_date - r.offer_dt).days) < abs((chosen.src_date - r.offer_dt).days):
                    chosen, method = cand, "ticker+date"
        if chosen is None:                          # fuzzy name+date: rename-safe
            rn = _norm_name(r.name)
            best, best_score = None, 0.0
            for cand in sa_norm:
                if pd.isna(cand.src_date) or abs((cand.src_date - r.offer_dt).days) > 7:
                    continue
                inter = rn & cand.norm
                if not inter:
                    continue
                score = len(inter) / max(1, len(rn | cand.norm))
                if score > best_score:
                    best, best_score = cand, score
            if best is not None and best_score >= 0.5:
                chosen, method = best, f"name+date({best_score:.2f})"
        ipo_era = chosen.src_ticker if chosen is not None else r.ticker
        out.append({
            "ritter_ticker": r.ticker,
            "ticker": TICKER_ALIASES.get(ipo_era, ipo_era),  # tradeable symbol
            "name": r.name, "offer_date": int(r.offer_date), "year": r.year,
            "offer_dt": r.offer_dt, "adr": r.adr,
            "post_issue_shares": r.post_issue_shares,
            "offer_price": chosen.offer_price if chosen is not None else np.nan,
            "match_method": method or "unmatched",
        })
    m = pd.DataFrame(out)
    print(f"[match] {(m['match_method'] != 'unmatched').sum()}/{len(m)} Ritter rows "
          f"matched to an offer price")
    return m


# ----------------------------------------------------------------------------
# 4a. SEC EDGAR shares-at-listing (authoritative)
# ----------------------------------------------------------------------------
def _ticker_cik() -> dict:
    if CIK_CACHE.exists():
        return json.loads(CIK_CACHE.read_text(encoding="utf-8"))
    print(f"[edgar] downloading ticker->CIK map")
    j = _SESSION.get(SEC_TICKERS, headers={"User-Agent": SEC_UA}, timeout=30).json()
    m = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in j.values()}
    CIK_CACHE.write_text(json.dumps(m), encoding="utf-8")
    return m


def _company_facts(cik: str) -> dict | None:
    cache = EDGAR_DIR / f"CIK{cik}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        r = _SESSION.get(SEC_FACTS.format(cik=cik), headers={"User-Agent": SEC_UA}, timeout=60)
        time.sleep(0.12)  # SEC fair-access (<10 req/s)
        if r.status_code != 200:
            cache.write_text("null", encoding="utf-8")
            return None
        j = r.json()
        cache.write_text(json.dumps(j), encoding="utf-8")
        return j
    except Exception:
        return None


def edgar_shares_at_ipo(ticker: str, ipo_dt, cikmap: dict) -> float:
    """First post-IPO common-shares-outstanding from EDGAR XBRL (np.nan if none)."""
    cik = cikmap.get(ticker.upper())
    if not cik:
        return np.nan
    cf = _company_facts(cik)
    if not cf:
        return np.nan
    ipo = pd.Timestamp(ipo_dt)
    best, best_end = np.nan, None
    for ns, con in [("dei", "EntityCommonStockSharesOutstanding"),
                    ("us-gaap", "CommonStockSharesOutstanding"),
                    ("us-gaap", "CommonStockSharesIssued")]:
        node = cf.get("facts", {}).get(ns, {}).get(con)
        if not node:
            continue
        for pts in node.get("units", {}).values():
            for p in pts:
                e, v = p.get("end"), p.get("val")
                if not e or not v:
                    continue
                ed = pd.Timestamp(e)
                # first reported balance on/after the IPO, within ~6 months
                if ipo - pd.Timedelta(days=10) <= ed <= ipo + pd.Timedelta(days=190):
                    if best_end is None or ed < best_end:
                        best, best_end = float(v), ed
        if best_end is not None:
            return best
    return np.nan


# ----------------------------------------------------------------------------
# 4b. yfinance shares at IPO date (get_shares_full) + first-trade + recent shares
# ----------------------------------------------------------------------------
def _load_yf_cache() -> dict:
    if YF_SHARES_CACHE.exists():
        try:
            d = pd.read_csv(YF_SHARES_CACHE)
            return {str(r.ticker).upper(): {"shares": float(r.shares),
                    "recent": float(r.recent), "first_trade": str(r.first_trade)}
                    for r in d.itertuples()}
        except Exception:
            return {}
    return {}


def _save_yf_cache(cache: dict) -> None:
    pd.DataFrame([{"ticker": k, **v} for k, v in cache.items()]) \
        .to_csv(YF_SHARES_CACHE, index=False)


def yf_lookup(ticker: str, cache: dict) -> dict:
    """{shares: at-IPO common shares (glitch-guarded), recent: latest share count,
    first_trade: yfinance's earliest trade date}. Recent + first_trade let callers
    catch yfinance glitches (inflated at-IPO count) and recycled tickers (the
    symbol now belongs to a different, earlier/later-listed company)."""
    key = ticker.upper()
    if key in cache:
        return cache[key]
    out = {"shares": np.nan, "recent": np.nan, "first_trade": ""}
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        h = tk.history(period="max", auto_adjust=False)
        if len(h):
            d1 = h.index.min()
            out["first_trade"] = d1.date().isoformat()
            sf = tk.get_shares_full(start=str(d1.date()), end=None)
            if sf is not None and len(sf):
                first, last = float(sf.iloc[0]), float(sf.iloc[-1])
                out["recent"] = last
                # Guard a yfinance glitch where the earliest share count is wildly
                # inflated (XCHG 20x, Sportradar 3.8x its recent). Shares normally
                # GROW post-IPO, so first >> recent => bad data.
                out["shares"] = np.nan if (last > 0 and first > YF_GLITCH_RATIO * last) else first
    except Exception:
        pass
    cache[key] = out
    return out


# ----------------------------------------------------------------------------
# 4c. Reconcile share estimates -> market cap at listing
# ----------------------------------------------------------------------------
def _load_verified() -> dict:
    """Hand-verified mcap (from the prior curated file) used only as a logged
    fallback where the public sources are missing or disagree."""
    if not CURATED_BACKUP.exists():
        return {}
    c = pd.read_csv(CURATED_BACKUP).dropna(subset=["mcap_listing_b"])
    return {str(r.ticker).upper(): float(r.mcap_listing_b) for r in c.itertuples()}


def reconcile_mcap(offer, adr, edgar_sh, yf_sh, ritter_sh, yf_recent, verified) -> dict:
    """
    Return chosen mcap_b + source + confidence + per-source estimates ($B).

    Logic (calibrated against the prior hand-checked valuations):
      * Estimates are offer x shares. For ADRs (flag==2) EDGAR/Ritter are in
        ORDINARY-share units (wrong vs a per-ADS offer), so only yfinance is used.
      * EDGAR can return a wrong-period/concept share count far above the
        company's actual float (NXP: EDGAR 4.3B vs ~250M real). Drop the EDGAR
        estimate when its shares exceed 3x the company's recent share count.
      * NON-ADR with a Ritter post-issue count -> ANCHOR on Ritter (a vetted total
        across share classes). EDGAR/yfinance within [0.6, 1.66]x the anchor
        corroborate it (high confidence); otherwise keep Ritter alone (med). This
        stops a single source from hijacking it (Sportradar yf=$30B vs Ritter=$8B).
      * Otherwise (ADR, or non-ADR with no Ritter count) -> PREFER yfinance (the
        most reliable at-IPO snapshot; EDGAR alone over/under-counts, e.g. NXP),
        corroborated by EDGAR when they agree (high) else single source (low).
      * 'verified' is used ONLY as a last-resort gap-filler when no public source
        yields any number (e.g. HOOD: EDGAR/yfinance/Ritter all blank).
    """
    # EDGAR sanity guard: a share count >> the company's recent float is bad data.
    if (pd.notna(edgar_sh) and pd.notna(yf_recent) and yf_recent > 0
            and edgar_sh > YF_GLITCH_RATIO * yf_recent):
        edgar_sh = np.nan

    est = {}
    if pd.notna(offer) and offer > 0:
        if adr != 2:
            if pd.notna(edgar_sh) and edgar_sh > 0:
                est["edgar"] = offer * edgar_sh / 1e9
            if pd.notna(ritter_sh) and ritter_sh > 0:
                est["ritter"] = offer * ritter_sh / 1e9
        if pd.notna(yf_sh) and yf_sh > 0:
            est["yf"] = offer * yf_sh / 1e9

    mcap, src, conf = np.nan, "unresolved", "none"
    if "ritter" in est:                       # vetted total-share anchor (non-ADR)
        anchor = est["ritter"]
        corrob = {k: v for k, v in est.items()
                  if k != "ritter" and 0.6 * anchor <= v <= 1.66 * anchor}
        if corrob:
            mcap = float(median([anchor] + list(corrob.values())))
            src, conf = "auto:" + "+".join(sorted(["ritter"] + list(corrob))), "high"
        else:
            mcap, src, conf = anchor, "ritter-anchor", "med"
    elif "yf" in est:                          # ADR / no-Ritter: yfinance leads
        yfv = est["yf"]
        if "edgar" in est and 0.6 * yfv <= est["edgar"] <= 1.66 * yfv:
            mcap = float(median([yfv, est["edgar"]]))
            src, conf = "auto:edgar+yf", "high"
        else:
            mcap, src, conf = yfv, "single:yf", "low"
    elif "edgar" in est:                       # EDGAR-only (guarded above)
        mcap, src, conf = est["edgar"], "single:edgar", "low"

    if not (mcap == mcap) and verified is not None:   # gap-fill only
        mcap, src, conf = float(verified), "verified", "verified"
    return {"mcap_listing_b": mcap, "mcap_src": src, "mcap_conf": conf,
            "est_edgar": est.get("edgar", np.nan), "est_yf": est.get("yf", np.nan),
            "est_ritter": est.get("ritter", np.nan)}


# ----------------------------------------------------------------------------
# 5. Assemble + classify
# ----------------------------------------------------------------------------
def assemble(matched: pd.DataFrame) -> pd.DataFrame:
    verified_tbl = _load_verified()
    cikmap = _ticker_cik()
    yf_cache = _load_yf_cache()

    m = matched.copy()
    m["free_pre_b"] = m["offer_price"] * m["post_issue_shares"] / 1e9
    # Candidate = could plausibly clear $5B (so worth the expensive lookups).
    cand = (m["offer_price"].notna()) & (
        (m["free_pre_b"] > CANDIDATE_PRE_FLOOR_B)
        | (m["post_issue_shares"].isna() & (m["offer_price"] >= 8))
        | (m["ticker"].isin(verified_tbl.keys()))     # always resolve known names
    )
    cands = m[cand].copy()
    print(f"[mcap] {len(cands)} candidate names get EDGAR+yfinance share lookups "
          f"(of {len(m)} matched); cached + resumable")

    rows = []
    for i, r in enumerate(cands.itertuples(), 1):
        edgar_sh = edgar_shares_at_ipo(r.ticker, r.offer_dt, cikmap)
        yf = yf_lookup(r.ticker, yf_cache)
        rec = reconcile_mcap(r.offer_price, r.adr, edgar_sh, yf["shares"],
                             r.post_issue_shares, yf["recent"], verified_tbl.get(r.ticker))
        # Recycled ticker: yfinance has data but its first trade is far from this
        # IPO date => the symbol now belongs to a DIFFERENT company (ET = Energy
        # Transfer not ExactTarget; PATH = UiPath not NuPathe). EDGAR/yfinance
        # shares AND Study-1 prices would be the wrong entity -> drop + log.
        yf_ft = yf["first_trade"]
        recycled = bool(yf_ft) and yf_ft != "nan" and pd.notna(r.offer_dt) and \
            abs((pd.Timestamp(yf_ft) - r.offer_dt).days) > RECYCLE_GAP_DAYS
        rows.append({
            "ticker": r.ticker, "ritter_ticker": r.ritter_ticker, "name": r.name,
            "offer_date": r.offer_date, "year": r.year,
            "first_trade": r.offer_dt.date().isoformat() if pd.notna(r.offer_dt) else "",
            "yf_first_trade": yf_ft, "recycled": recycled,
            "offer_price": r.offer_price, "adr": r.adr,
            "ritter_shares": r.post_issue_shares, "edgar_shares": edgar_sh,
            "yf_shares": yf["shares"], "match_method": r.match_method, **rec,
        })
        if i % 25 == 0:
            _save_yf_cache(yf_cache)
            print(f"  ...{i}/{len(cands)} resolved")
    _save_yf_cache(yf_cache)
    df = pd.DataFrame(rows)

    def classify(r):
        if r["recycled"]:
            return "drop", (f"ticker reused by another company (yfinance first trade "
                            f"{r['yf_first_trade']} != IPO {r['first_trade']})")
        if r["ticker"] in DIRECT_LISTINGS:
            return "considered", "direct listing (no underwriter/lockup) - excluded by spec"
        if r["first_trade"] and r["first_trade"] >= RECENT_CUTOFF:
            return "considered", f"too recent: IPO {r['first_trade']}; +180 td outcome not yet observable"
        if pd.isna(r["mcap_listing_b"]):
            return "drop", "at-listing shares unresolved (EDGAR/yfinance/Ritter all missing)"
        if r["mcap_listing_b"] > MIN_MCAP_B:
            return "headline", ""
        if r["mcap_listing_b"] >= NEAR_MISS_FLOOR_B:
            return "considered", f"~${r['mcap_listing_b']:.1f}B at listing (<${MIN_MCAP_B:g}B)"
        return "drop", f"~${r['mcap_listing_b']:.1f}B at listing (well below ${MIN_MCAP_B:g}B)"

    cls = df.apply(classify, axis=1, result_type="expand")
    df["bucket"], df["reason"] = cls[0], cls[1]
    return df


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("BUILD UNIVERSE — Ritter list x Nasdaq offer price x EDGAR/yfinance shares")
    print("=" * 66)
    rit = fetch_ritter()
    nasdaq = build_nasdaq_index(range(MIN_YEAR, THIS_YEAR + 1))
    matched = match_offer_price(rit, nasdaq)
    df = assemble(matched)

    audit_cols = ["ticker", "ritter_ticker", "name", "first_trade", "yf_first_trade",
                  "year", "offer_price", "adr", "ritter_shares", "edgar_shares",
                  "yf_shares", "est_edgar", "est_yf", "est_ritter", "mcap_listing_b",
                  "mcap_src", "mcap_conf", "match_method", "bucket", "reason"]
    df.sort_values(["bucket", "mcap_listing_b"], ascending=[True, False])[audit_cols] \
      .to_csv(AUDIT_OUT, index=False)

    keep = df[df["bucket"].isin(["headline", "considered"])].copy()
    # On a current-ticker collision keep the headline row over a considered one,
    # then the larger mcap (recycled rows are already dropped above).
    keep["_rank"] = (keep["bucket"] == "headline").astype(int)
    keep = keep.sort_values(["_rank", "mcap_listing_b"], ascending=[False, False]) \
               .drop_duplicates(subset=["ticker"], keep="first").drop(columns="_rank")
    # Log every spec-mandated direct-listing exclusion explicitly, even ones that
    # never became candidates (no offer price -> never priced), for survivorship
    # transparency (COIN/SPOT/RBLX etc. have no underwritten offer to match).
    missing_dl = sorted(set(DIRECT_LISTINGS) - set(keep["ticker"]))
    if missing_dl:
        extra = pd.DataFrame([{
            "ticker": t, "bucket": "considered", "offer_price": np.nan,
            "mcap_listing_b": np.nan,
            "reason": "direct listing (no underwriter/lockup) - excluded by spec",
            "name": "", "first_trade": "", "mcap_src": "n/a", "mcap_conf": "n/a",
            "ritter_ticker": t, "match_method": "n/a"} for t in missing_dl])
        keep = pd.concat([keep, extra], ignore_index=True)
    out_cols = ["ticker", "bucket", "offer_price", "mcap_listing_b", "reason",
                "name", "first_trade", "mcap_src", "mcap_conf", "ritter_ticker",
                "match_method"]
    keep.sort_values(["bucket", "mcap_listing_b"], ascending=[True, False])[out_cols] \
        .to_csv(UNIVERSE_OUT, index=False)

    head = df[df["bucket"] == "headline"].sort_values("mcap_listing_b", ascending=False)
    cons = df[df["bucket"] == "considered"]
    print("\n" + "=" * 66)
    print(f"HEADLINE universe (traditional IPO, >${MIN_MCAP_B:g}B, {MIN_YEAR}+): "
          f"n={len(head)}  ({(head['mcap_listing_b']>10).sum()} large-cap >$10B)")
    print(f"  mcap source mix: " +
          ", ".join(f"{k}={v}" for k, v in head["mcap_conf"].value_counts().items()))
    print(head[["ticker", "name", "first_trade", "offer_price", "mcap_listing_b",
                "mcap_src", "mcap_conf"]].to_string(index=False, max_colwidth=24))
    print(f"\nconsidered/excluded (logged): n={len(cons)}")
    print(f"\nWrote {UNIVERSE_OUT.relative_to(u.ROOT)} and {AUDIT_OUT.relative_to(u.ROOT)}")
    return df


if __name__ == "__main__":
    main()
