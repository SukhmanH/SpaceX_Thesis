"""
survivorship_table.py — by-era kept-vs-dropped survivorship summary for Study 1
(IPO fade) and Study 2 (index effect), so a reader can see whether sample
exclusions bias the recent-vs-older comparison.

Reads only existing artifacts (no network):
  * data/ipo_universe.csv         — Study-1 universe (headline / considered)
  * outputs/tables/ipo_fade.csv   — Study-1 names actually measured (kept)
  * outputs/tables/dropped_study1.csv
  * outputs/tables/index_effect.csv        — Study-2 events kept (filled window)
  * outputs/tables/dropped_study2.csv      — Study-2 events dropped (date in reason)

Writes outputs/tables/survivorship_by_era.csv and prints a Markdown table.

Run:  python survivorship_table.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import utils as u

T = u.OUT_TABLES
ERAS = [("2005–2009", 2005, 2009), ("2010–2014", 2010, 2014),
        ("2015–2019", 2015, 2019), ("2020–present", 2020, 2100)]


def era_of(year) -> str:
    if pd.isna(year):
        return "unknown"
    for label, lo, hi in ERAS:
        if lo <= int(year) <= hi:
            return label
    return "unknown"


# ----------------------------------------------------------------------------
# STUDY 1 — IPO universe survivorship by IPO era
# ----------------------------------------------------------------------------
uni = pd.read_csv(u.ROOT / "data" / "ipo_universe.csv")
uni["ticker"] = uni["ticker"].astype(str).str.strip().str.upper()
uni["bucket"] = uni["bucket"].astype(str).str.strip().str.lower()
uni["ipo_year"] = pd.to_datetime(uni["first_trade"], errors="coerce").dt.year

fade = pd.read_csv(T / "ipo_fade.csv")
measured = set(fade["ticker"].astype(str).str.upper())

head = uni[uni["bucket"] == "headline"].copy()
head["era"] = head["ipo_year"].map(era_of)
head["kept"] = head["ticker"].isin(measured)

# 'considered' = excluded by design (direct listings, $3–5B near-miss, recycled);
# reported separately as context, not as survivorship attrition.
cons = uni[uni["bucket"] == "considered"].copy()
cons["era"] = cons["ipo_year"].map(era_of)

s1_rows = []
for label, _, _ in ERAS:
    if label == "2005–2009":
        continue  # Study 1 is 2010+
    h = head[head["era"] == label]
    n_uni = len(h)
    n_kept = int(h["kept"].sum())
    n_drop = n_uni - n_kept
    s1_rows.append(dict(study="Study 1 (IPO)", era=label, universe=n_uni,
                        kept=n_kept, dropped=n_drop,
                        pct_kept=(round(100 * n_kept / n_uni, 1) if n_uni else np.nan),
                        excluded_by_design=int((cons["era"] == label).sum())))
s1 = pd.DataFrame(s1_rows)

# ----------------------------------------------------------------------------
# STUDY 2 — S&P 500 additions survivorship by effective-date era
# ----------------------------------------------------------------------------
ix = pd.read_csv(T / "index_effect.csv")
ix["year"] = pd.to_datetime(ix["effective_date"], errors="coerce").dt.year
ix["era"] = ix["year"].map(era_of)

drp = pd.read_csv(T / "dropped_study2.csv")
def _year_from_reason(s):
    m = re.search(r"(\d{4})-\d{2}-\d{2}", str(s))
    return int(m.group(1)) if m else np.nan
drp["year"] = drp["reason"].map(_year_from_reason)
drp["era"] = drp["year"].map(era_of)

s2_rows = []
for label, _, _ in ERAS:
    n_kept = int((ix["era"] == label).sum())
    n_drop = int((drp["era"] == label).sum())
    n_tot = n_kept + n_drop
    s2_rows.append(dict(study="Study 2 (index)", era=label, universe=n_tot,
                        kept=n_kept, dropped=n_drop,
                        pct_kept=(round(100 * n_kept / n_tot, 1) if n_tot else np.nan),
                        excluded_by_design=0))
s2 = pd.DataFrame(s2_rows)

out = pd.concat([s1, s2], ignore_index=True)
out.to_csv(T / "survivorship_by_era.csv", index=False)

# ----------------------------------------------------------------------------
# Markdown (for the README)
# ----------------------------------------------------------------------------
def md(df, study):
    sub = df[df["study"] == study]
    lines = ["| Era | Additions/Universe | Kept (measured) | Dropped (no clean data) | % kept |",
             "|-----|--------------------|-----------------|-------------------------|--------|"]
    for r in sub.itertuples():
        lines.append(f"| {r.era} | {r.universe} | {r.kept} | {r.dropped} | {r.pct_kept:.0f}% |")
    tot_u, tot_k, tot_d = sub["universe"].sum(), sub["kept"].sum(), sub["dropped"].sum()
    pk = 100 * tot_k / tot_u if tot_u else float("nan")
    lines.append(f"| **All** | **{tot_u}** | **{tot_k}** | **{tot_d}** | **{pk:.0f}%** |")
    return "\n".join(lines)

print("=" * 78)
print("STUDY 1 — IPO universe (headline >$5B) survivorship by IPO era")
print("(excluded-by-design 'considered' names per era:",
      dict(zip(s1["era"], s1["excluded_by_design"])), ")")
print(md(out, "Study 1 (IPO)"))
print("\n" + "=" * 78)
print("STUDY 2 — S&P 500 additions survivorship by effective-date era")
print(md(out, "Study 2 (index)"))
print("\nWrote outputs/tables/survivorship_by_era.csv")
