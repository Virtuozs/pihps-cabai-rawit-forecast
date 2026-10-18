"""
Turn dataset/bbm_price_events.csv (hand-compiled, cited step-changes) into a daily,
per-province fuel-price feature table and join the selected fuel onto the main chili price panel.

Source of truth: dataset/bbm_price_events.csv: this table is irreplaceable hand-compiled research, not
regenerable by rerunning a fetch. Every row has a `source` and `confidence`
column, see README.md ("BBM / Solar price") for how the events were compiled and what's
still unverified. This script does NOT invent values: gaps in the events table (confidence
in {confirmed_unavailable, unconfirmed_gap}) stay NaN in the output, never forward-filled
across a gap marker.

Mechanism: each fuel/scope has a short list of (effective_date, price) step-changes. For
every (province, date) in the target range we look up the most recent event whose
effective_date <= date (an as-of / backward-fill join). This is leak-safe by construction:
these are public, same-day government/Pertamina price announcements, so using "price as of
t" to predict t+7 leaks nothing no --min-lag needed, unlike the weather features.

Regional scope handling:
  - scope "all"       -> applies to every province
  - scope "jmb"       -> Jawa-Madura-Bali (Jawa Tengah, DI Yogyakarta, Jawa Timur, Bali,
                          DKI Jakarta in this project's province list -- all on the
                          Jawa-Madura-Bali fuel-distribution footprint of MEMR Reg 39/2014)
  - scope "ntb_other" -> everywhere outside Jawa-Madura-Bali (Nusa Tenggara Barat)
A province's timeline is the union of its "all" events and its group-specific events,
sorted by date; a more specific event overrides "all" from its date on.

Only Solar (subsidized diesel) is selected for the main dataset because of 100% coverage vs.
44-71% for Premium/Pertalite, and it's the fuel with the direct transport-cost story
(trucks). Premium and Pertalite are still built into bbm_features_all.csv for reference.

Usage (from the UAS/pihps-cabai-rawit-forecast folder):
  uv run scripts/fetch_bbm.py
  uv run scripts/fetch_bbm.py --join dataset/pihps_red_cayenne_pepper.csv
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Must match the province list in fetch_pihps.py / fetch_weather.py (README table).
PROVINCES = {
    "Jawa Tengah": "jmb",
    "DI Yogyakarta": "jmb",
    "Jawa Timur": "jmb",
    "Bali": "jmb",
    "Nusa Tenggara Barat": "ntb_other",
    "DKI Jakarta": "jmb",
}
ALL_FUELS = ["premium", "pertalite", "solar"]
# Encoding (raw Rp/liter vs. a regime flag, since Solar only has 2 price levels in the
# whole span) is deferred to the EDA / feature-engineering step in the notebook (§6.6).
SELECTED_FUELS = ["solar"]


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["effective_date"])
    df["price_rp_per_l"] = pd.to_numeric(df["price_rp_per_l"], errors="coerce")
    bad_scope = set(df["scope"]) - {"all", "jmb", "ntb_other"}
    if bad_scope:
        raise ValueError(f"unknown scope(s) in events csv: {bad_scope}")
    return df


def province_timeline(events: pd.DataFrame, fuel: str, group: str) -> pd.DataFrame:
    sub = events[(events["fuel"] == fuel) & (events["scope"].isin(["all", group]))]
    sub = sub.sort_values("effective_date")[["effective_date", "price_rp_per_l"]]
    sub = sub.rename(columns={"effective_date": "date", "price_rp_per_l": fuel})
    return sub.reset_index(drop=True)


def build_features(events: pd.DataFrame, start: str, end: str, fuels: list[str] = ALL_FUELS) -> pd.DataFrame:
    calendar = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    frames = []
    for province, group in PROVINCES.items():
        panel = calendar.copy()
        panel["province"] = province
        for fuel in fuels:
            tl = province_timeline(events, fuel, group)
            if tl.empty:
                panel[fuel] = pd.NA
                continue
            panel = pd.merge_asof(
                panel.sort_values("date"), tl, on="date", direction="backward"
            )
        frames.append(panel)
    out = pd.concat(frames, ignore_index=True)
    return out[["date", "province", *fuels]].sort_values(["province", "date"])


def compute_quality(merged: pd.DataFrame) -> pd.DataFrame:
    """Coverage per (province, fuel) against rows that actually have a chili price
    -- i.e. the rows that would matter in the main dataset -- not raw calendar days."""
    rows = []
    panel = merged[(merged["level"] == "province") & merged["price"].notna()]
    for province in PROVINCES:
        prov = panel[panel["region"] == province].sort_values("date")
        n_panel = len(prov)
        panel_first = prov["date"].min() if n_panel else pd.NaT
        panel_last = prov["date"].max() if n_panel else pd.NaT
        for fuel in ALL_FUELS:
            covered = prov[fuel].notna()
            n_covered = int(covered.sum())
            covered_dates = prov.loc[covered, "date"]

            is_na = prov[fuel].isna()
            gap_id = (is_na != is_na.shift()).cumsum()
            gap_groups = prov.loc[is_na, "date"].groupby(gap_id[is_na])
            gap_sizes = gap_groups.size()
            n_gap_segments = int(len(gap_sizes))
            longest_gap_rows = int(gap_sizes.max()) if n_gap_segments else 0

            n_levels = int(prov.loc[covered, fuel].nunique())

            rows.append({
                "province": province,
                "fuel": fuel,
                "n_panel_rows": n_panel,
                "panel_first_date": panel_first.date() if pd.notna(panel_first) else None,
                "panel_last_date": panel_last.date() if pd.notna(panel_last) else None,
                "n_covered": n_covered,
                "coverage_pct": round(100 * n_covered / n_panel, 1) if n_panel else 0.0,
                "first_covered_date": covered_dates.min().date() if n_covered else None,
                "last_covered_date": covered_dates.max().date() if n_covered else None,
                "n_gap_segments": n_gap_segments,
                "longest_gap_rows": longest_gap_rows,
                "n_distinct_price_levels": n_levels,
            })
    return pd.DataFrame(rows)


def write_meta(path: Path, events: pd.DataFrame, start: str, end: str, join_path: Path | None) -> None:
    meta = {
        "description": "Hand-compiled BBM (subsidized fuel) price step-changes, "
                        "NOT scraped -- see README.md ('BBM / Solar price') for why and for "
                        "full source citations.",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fuels": {
            "premium": "RON 88, discontinued nationwide 2023-01-01",
            "pertalite": "RON 90, current subsidized gasoline",
            "solar": "subsidized diesel (Biosolar CN48)",
        },
        "unit": "Rp/liter",
        "scope_definitions": {
            "all": "applies to every province",
            "jmb": "Jawa-Madura-Bali: " + ", ".join(p for p, g in PROVINCES.items() if g == "jmb"),
            "ntb_other": "outside Jawa-Madura-Bali: " + ", ".join(p for p, g in PROVINCES.items() if g == "ntb_other"),
        },
        "join_mechanism": "merge_asof (backward) per province -- most recent event with "
                           "effective_date <= date. Leak-safe: same-day public announcements.",
        "selected_fuels": SELECTED_FUELS,
        "date_range_generated": {"start": start, "end": end},
        "n_events_by_fuel": events.groupby("fuel").size().to_dict(),
        "n_events_by_confidence": events.groupby("confidence").size().to_dict(),
        "events_file": "bbm_price_events.csv",
        "joined_into": str(join_path) if join_path else None,
        "sources": [
            "Xie, Y. & Harjono, M., 'The retail fuels market in Indonesia', "
            "ICCT Working Paper 2020-23 (Oct 2020) -- Table 4 and Table 5",
            "Kementerian ESDM official press release, 19 Jan 2015 Premium/Solar price (esdm.go.id)",
            "Ditjen Migas ESDM official announcement, 5 Jan 2016 BBM price cut (migas.esdm.go.id)",
            "Menteri ESDM Sudirman Said, 1 Apr 2016 Premium/Solar price cut (Republika)",
            "indonesiabaik.id (Kominfo) infographic, 3 Sept 2022 fuel price adjustment",
            "News search confirming Solar/Pertalite flat through 2023 (Bisnis.com), 2024-2025 "
            "(CNBC Indonesia, Sri Mulyani statement), and Apr-Oct 2026 (Bisnis.com, RRI, "
            "CNBC Indonesia, BCA Sekuritas, Antara, RCTI+/iNews)",
            "Kepmen ESDM No. 245.K/MG.01/MEM.M/2022 (Premium discontinuation), via secondary reporting",
        ],
    }
    path.write_text(json.dumps(meta, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "dataset",
                     help="dataset/ directory for all generated files (default: dataset/)")
    ap.add_argument("--events", type=Path, default=None,
                     help="default: <out>/bbm_price_events.csv")
    ap.add_argument("--start", default="2017-01-02")
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    ap.add_argument("--join", type=Path, default=None,
                     help="pihps_red_cayenne_pepper.csv (province-level price+weather panel) "
                          "to left-join the selected BBM feature(s) onto, in place "
                          "(matched on region==province, date). Default: <out>/pihps_red_cayenne_pepper.csv "
                          "if it exists; pass --no-join to skip.")
    ap.add_argument("--no-join", action="store_true", help="skip joining into the main dataset file")
    args = ap.parse_args()

    events_path = args.events or (args.out / "bbm_price_events.csv")
    out_all = args.out / "bbm_features_all.csv"
    out_selected = args.out / "bbm_features.csv"
    meta_out = args.out / "bbm_meta.json"
    quality_out = args.out / "bbm_quality.csv"
    join_path = None if args.no_join else (args.join or (args.out / "pihps_red_cayenne_pepper.csv"))
    end = str(args.end)

    events = load_events(events_path)

    feats_all = build_features(events, args.start, end, ALL_FUELS)
    feats_all.to_csv(out_all, index=False)
    print(f"Saved full reference (all fuels) -> {out_all}")

    feats = feats_all[["date", "province", *SELECTED_FUELS]].rename(
        columns={"solar": "solar_rp_per_l"})
    feats.to_csv(out_selected, index=False)
    print(f"Saved selected deliverable ({', '.join(SELECTED_FUELS)}) -> {out_selected}")

    write_meta(meta_out, events, args.start, end, join_path)
    print(f"Saved metadata -> {meta_out}")

    sel_cols = [c for c in feats.columns if c not in ("date", "province")]
    n_total = len(feats) * len(sel_cols)
    n_missing = feats[sel_cols].isna().sum().sum()
    print(f"Missing (NaN) cells in selected output: {n_missing:,} / {n_total:,} "
          f"({n_missing / n_total:.1%})" if n_total else "")

    if join_path and not join_path.exists():
        print(f"\n{join_path} not found -- run fetch_pihps.py and fetch_weather.py first "
              f"to build it. Skipping join.")
        return
    if not join_path:
        print("\n(pass --join <path> to also get bbm_quality.csv and join solar_rp_per_l "
              "into the main dataset file)")
        return

    panel = pd.read_csv(join_path, parse_dates=["date"])
    # drop a stale BBM join (incl. the redundant "province" dup of "region") if rerun
    stale = [c for c in (*sel_cols, "province", "province_bbm") if c in panel.columns]
    panel = panel[[c for c in panel.columns if c not in stale]]
    merged_all = panel.merge(
        feats_all, left_on=["region", "date"], right_on=["province", "date"], how="left",
    ).drop(columns="province")
    quality = compute_quality(merged_all)
    quality.to_csv(quality_out, index=False)
    print(f"\nSaved coverage report (all fuels, for reference) -> {quality_out}\n")
    print(quality.to_string(index=False))

    merged = panel.merge(
        feats, left_on=["region", "date"], right_on=["province", "date"], how="left",
    ).drop(columns="province")
    merged.to_csv(join_path, index=False)
    print(f"\nJoined solar_rp_per_l into -> {join_path} ({len(merged):,} rows)")


if __name__ == "__main__":
    main()
