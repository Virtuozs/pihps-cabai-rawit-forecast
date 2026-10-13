# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31", "pandas>=2.1"]
# ///
"""Fetch daily traditional-market prices from PIHPS Nasional (Bank Indonesia).

Source : https://www.bi.go.id/hargapangan  (public JSON behind the "Tabel Harga" page)
Output : <out>/pihps_prices.csv   long format, one row per (date, commodity, region)
         <out>/pihps_quality.csv  per-series coverage / variation report
         <out>/pihps_meta.json    access date, parameters, endpoint
         <out>/raw/...            cached raw JSON responses (one per request)

Verified 2026-10-12:
  * history starts 2017-01-02; data is updated to the current working day
  * one request per calendar quarter (~65 working-day columns, ~13 s for all
    provinces); a full year of all provinces times out on the server
  * Cabai Rawit Merah: dense daily data from 2017-04-03 for ~31 provinces (~94-98%
    of working days). Q1-2017 is almost all "-". Not usable: Aceh (stops 2022-03,
    27% coverage), Sumatera Barat (1 value), Kepulauan Riau (starts 2026-01),
    "Dum" (a dummy test row on the server). See pihps_quality.csv.
  * non-trading values are "-"; numbers use "," as thousands separator ("63,900")

Polite scraping: 2 s between requests, raw responses cached and reused,
hard stop on HTTP 403/429/503. robots.txt allows /hargapangan.

Usage (from the UAS folder):
  uv run scripts/fetch_pihps.py --list                      # show commodity + province ids
  uv run scripts/fetch_pihps.py                             # cabai rawit merah, all provinces
  uv run scripts/fetch_pihps.py --commodity 16 14 --level regency --province 12 14 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.bi.go.id/hargapangan/WebSite"
GRID = f"{BASE}/TabelHarga/GetGridDataKomoditas"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 (student research project)",
    "Accept": "application/json",
    "Referer": "https://www.bi.go.id/hargapangan/TabelHarga/PasarTradisionalKomoditas",
}
DELAY_S = 2.0
STOP_CODES = {403, 429, 503}
FIRST_DATE = date(2017, 1, 1)  # nothing earlier exists on the server

session = requests.Session()
session.headers.update(HEADERS)
_last_call = 0.0


class Blocked(RuntimeError):
    pass


def get_json(url: str, params: dict | None = None, retries: int = 3) -> dict:
    """GET with rate limiting, retry on network errors, hard stop on blocking codes."""
    global _last_call
    for attempt in range(1, retries + 1):
        wait = DELAY_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            r = session.get(url, params=params, timeout=180)
        except requests.RequestException as e:
            print(f"  network error ({e}); retry {attempt}/{retries}", file=sys.stderr)
            time.sleep(5 * attempt)
            continue
        if r.status_code in STOP_CODES:
            raise Blocked(f"HTTP {r.status_code} from {r.url} - stopping to be polite")
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"giving up on {url} {params}")


# ---------------------------------------------------------------- metadata
def commodities() -> dict[int, str]:
    tree = get_json(f"{BASE}/Home/GetCommoditiesTree")["data"]
    out = {}
    for n in tree:
        if "_" in n["TreeID"]:  # leaf = "<cat>_<com_id>"
            out[int(n["TreeID"].split("_")[1])] = n["TreeName"].strip()
    return out


def provinces() -> dict[int, str]:
    data = get_json(f"{BASE}/Home/GetProvinceAll")["data"]
    return {p["province_id"]: p["province_name"] for p in data if p["province_id"] != 0}


# ---------------------------------------------------------------- fetching
def fetch_chunk(com_id: int, province_id: int | None, show_kota: bool,
                start: date, end: date, raw_dir: Path, refresh: bool) -> list[dict]:
    """One request (cached). province_id=None -> every province in one call."""
    prov_tag = "all" if province_id is None else f"p{province_id:02d}"
    cache = raw_dir / f"com_{com_id}" / f"{prov_tag}_{'kota' if show_kota else 'prov'}_{start}_{end}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())["data"]

    params = {
        "price_type_id": 1,  # 1 = pasar tradisional
        "comcat_id": f"com_{com_id}",
        "province_id": "" if province_id is None else province_id,
        "regency_id": "",
        "showKota": str(show_kota).lower(),
        "showPasar": "false",
        "tipe_laporan": 1,  # 1 = harian (daily)
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    payload = get_json(GRID, params)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"fetched_at": datetime.now().isoformat(timespec="seconds"),
                                 "params": params, "data": payload.get("data", [])},
                                ensure_ascii=False))
    return payload.get("data", [])


def parse_rows(rows: list[dict], com_id: int, com_name: str) -> list[dict]:
    """Wide grid -> long records. level 0 = national, 1 = province, 2 = city."""
    out, current_prov = [], None
    for row in rows:
        level, name = row["level"], row["name"].strip()
        if level == 1:
            current_prov = name
        province = None if level == 0 else current_prov
        for key, val in row.items():
            if "/" not in key:  # skip no / name / level
                continue
            raw = str(val).strip()
            price = float(raw.replace(",", "")) if raw.replace(",", "").replace(".", "").isdigit() else None
            out.append({
                "date": datetime.strptime(key, "%d/%m/%Y").date(),
                "commodity_id": com_id, "commodity": com_name,
                "level": {0: "national", 1: "province", 2: "regency"}.get(level, str(level)),
                "province": province, "region": name,
                "price": price, "raw_value": raw,
            })
    return out


def quarter_chunks(start: date, end: date):
    """Calendar quarters. A full year of all provinces times out server-side (>60 s)."""
    for y in range(start.year, end.year + 1):
        for m in (1, 4, 7, 10):
            qs, qe = date(y, m, 1), (date(y, m + 3, 1) if m < 10 else date(y + 1, 1, 1))
            qe = date.fromordinal(qe.toordinal() - 1)
            if qe >= start and qs <= end:
                yield max(start, qs), min(end, qe)


# ---------------------------------------------------------------- quality
def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage vs. Mon-Fri business days, zero/'-' share, coefficient of variation."""
    rows = []
    for (com, region), g in df.groupby(["commodity", "region"]):
        bdays = pd.bdate_range(g["date"].min(), g["date"].max())
        valid = g["price"].where(g["price"] > 0)
        rows.append({
            "commodity": com, "region": region, "level": g["level"].iat[0],
            "first_date": g["date"].min().date(), "last_date": g["date"].max().date(),
            "n_valid": int(valid.notna().sum()),
            "coverage_pct": round(100 * valid.notna().sum() / len(bdays), 1),
            "dash_or_zero_pct": round(100 * valid.isna().sum() / len(g), 1),
            "mean": round(valid.mean(), 0), "std": round(valid.std(), 0),
            "cv_pct": round(100 * valid.std() / valid.mean(), 1) if valid.notna().any() else None,
            "min": valid.min(), "max": valid.max(),
        })
    return pd.DataFrame(rows).sort_values(["commodity", "level", "coverage_pct"],
                                          ascending=[True, True, False])


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commodity", type=int, nargs="+", default=[16],
                    help="commodity ids (16 = Cabai Rawit Merah, 14 = Cabai Merah Keriting). Default 16")
    ap.add_argument("--level", choices=["province", "regency"], default="province",
                    help="province: all provinces in one call/year. regency: cities of --province")
    ap.add_argument("--province", type=int, nargs="*", default=None,
                    help="province ids (required for --level regency; filters output for province level)")
    ap.add_argument("--start", type=date.fromisoformat, default=FIRST_DATE)
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "dataset")
    ap.add_argument("--refresh", action="store_true", help="ignore cached raw responses")
    ap.add_argument("--list", action="store_true", help="print commodity and province ids, then exit")
    args = ap.parse_args()

    com_names, prov_names = commodities(), provinces()
    if args.list:
        print("Commodities:"); [print(f"  {k:>3}  {v}") for k, v in sorted(com_names.items())]
        print("Provinces:");   [print(f"  {k:>3}  {v}") for k, v in sorted(prov_names.items())]
        return
    if args.level == "regency" and not args.province:
        ap.error("--level regency needs --province ids (see --list)")

    start = max(args.start, FIRST_DATE)
    raw_dir = args.out / "raw"
    targets = [None] if args.level == "province" else args.province
    today = date.today()

    records = []
    try:
        for com in args.commodity:
            name = com_names.get(com, f"com_{com}")
            for prov in targets:
                label = "all provinces" if prov is None else prov_names.get(prov, prov)
                for s, e in quarter_chunks(start, args.end):
                    # the chunk containing today is still growing -> always refetch it
                    refresh = args.refresh or e >= today
                    print(f"[{name}] {label} {s}..{e}", flush=True)
                    rows = fetch_chunk(com, prov, args.level == "regency", s, e, raw_dir, refresh)
                    records += parse_rows(rows, com, name)
    except Blocked as e:
        print(f"\nSTOPPED: {e}\nPartial results are cached in {raw_dir}; rerun later to resume.",
              file=sys.stderr)
        if not records:
            sys.exit(1)

    df = pd.DataFrame(records)
    if df.empty:
        sys.exit("no data returned")
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["date", "commodity_id", "region"]).sort_values(
        ["commodity_id", "level", "region", "date"])
    if args.level == "province" and args.province:
        keep = {prov_names[p] for p in args.province}
        df = df[(df["level"] == "national") | df["region"].isin(keep)]

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "pihps_prices.csv", index=False)
    q = quality_report(df)
    q.to_csv(args.out / "pihps_quality.csv", index=False)
    (args.out / "pihps_meta.json").write_text(json.dumps({
        "source": "Bank Indonesia - PIHPS Nasional, https://www.bi.go.id/hargapangan",
        "endpoint": GRID, "price_type": "pasar tradisional (price_type_id=1)",
        "report_type": "harian (tipe_laporan=1)", "unit": "Rp/kg",
        "accessed_at": datetime.now().isoformat(timespec="seconds"),
        "commodities": {c: com_names.get(c) for c in args.commodity},
        "level": args.level, "provinces": args.province,
        "start": str(start), "end": str(args.end), "rows": len(df),
    }, indent=2, ensure_ascii=False))

    print(f"\nSaved {len(df):,} rows -> {args.out / 'pihps_prices.csv'}")
    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print(q.drop(columns=["first_date"]).to_string(index=False))


if __name__ == "__main__":
    main()
