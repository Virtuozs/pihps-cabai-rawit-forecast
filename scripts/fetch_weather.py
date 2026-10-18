"""Fetch daily weather from the Open-Meteo archive and build leak-safe lag features.

Source : https://open-meteo.com/en/docs/historical-weather-api  (ERA5 reanalysis, free,
         no key; the last few days are filled from forecast models and may be revised)
Location: one point per province = the provincial capital, where the PIHPS markets are
         (a production-area point would be a defensible alternative; edit COORDS).

Output : <out>/weather_daily_raw.csv   raw daily values, calendar days. EDA ONLY:
                                       do not join this to prices as model features.
         <out>/weather_features.csv    lagged / rolling features, safe to join on date
         <out>/pihps_red_cayenne_pepper.csv  pihps_prices.csv (province level) + features
         <out>/weather_meta.json       source, coordinates, access date
         <out>/raw/weather/*.json      cached raw responses

Leakage rule: a feature on date t uses only weather up to t-1 (--min-lag, default 1).
  * Rolling windows are computed on the full calendar-day series (weekends included)
    and THEN shifted, so "rain_sum_7d" on Monday = rain over the 7 days ending Sunday.
  * Today's weather is a partial day and reanalysis arrives with a delay, so lag 0 is
    never used. With a 7-day-ahead target (price at t+7 predicted at t), these features
    contain no information from t..t+7.

Usage (from the UAS folder):
  uv run scripts/fetch_weather.py
  uv run scripts/fetch_weather.py --min-lag 2      # stricter, if you assume a 2-day data delay
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

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = ["precipitation_sum", "rain_sum", "precipitation_hours",
              "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
              "relative_humidity_2m_mean",
              "shortwave_radiation_sum", "et0_fao_evapotranspiration"]
DELAY_S = 31.0  # ~2 ten-year requests per minute fits the 600 weighted calls/min limit
START = date(2016, 10, 1)  # 6 months before PIHPS data so the 90-day windows are full

# Provincial capital (PIHPS name -> city, lat, lon)
COORDS = {
    "Aceh": ("Banda Aceh", 5.5483, 95.3238),
    "Sumatera Utara": ("Medan", 3.5952, 98.6722),
    "Sumatera Barat": ("Padang", -0.9471, 100.4172),
    "Riau": ("Pekanbaru", 0.5071, 101.4478),
    "Kepulauan Riau": ("Tanjung Pinang", 0.9186, 104.4554),
    "Jambi": ("Jambi", -1.6101, 103.6131),
    "Bengkulu": ("Bengkulu", -3.7928, 102.2608),
    "Sumatera Selatan": ("Palembang", -2.9761, 104.7754),
    "Kepulauan Bangka Belitung": ("Pangkal Pinang", -2.1291, 106.1090),
    "Lampung": ("Bandar Lampung", -5.3971, 105.2668),
    "Banten": ("Serang", -6.1200, 106.1503),
    "Jawa Barat": ("Bandung", -6.9175, 107.6191),
    "DKI Jakarta": ("Jakarta", -6.2088, 106.8456),
    "Jawa Tengah": ("Semarang", -6.9667, 110.4167),
    "DI Yogyakarta": ("Yogyakarta", -7.7956, 110.3695),
    "Jawa Timur": ("Surabaya", -7.2575, 112.7521),
    "Bali": ("Denpasar", -8.6705, 115.2126),
    "Nusa Tenggara Barat": ("Mataram", -8.5833, 116.1167),
    "Nusa Tenggara Timur": ("Kupang", -10.1772, 123.6070),
    "Kalimantan Barat": ("Pontianak", -0.0263, 109.3425),
    "Kalimantan Selatan": ("Banjarmasin", -3.3186, 114.5944),
    "Kalimantan Tengah": ("Palangka Raya", -2.2161, 113.9135),
    "Kalimantan Timur": ("Samarinda", -0.5022, 117.1536),
    "Kalimantan Utara": ("Tanjung Selor", 2.8375, 117.3653),
    "Gorontalo": ("Gorontalo", 0.5435, 123.0568),
    "Sulawesi Selatan": ("Makassar", -5.1477, 119.4327),
    "Sulawesi Tenggara": ("Kendari", -3.9985, 122.5129),
    "Sulawesi Tengah": ("Palu", -0.8917, 119.8707),
    "Sulawesi Utara": ("Manado", 1.4748, 124.8421),
    "Sulawesi Barat": ("Mamuju", -2.6748, 118.8885),
    "Maluku": ("Ambon", -3.6954, 128.1814),
    "Maluku Utara": ("Ternate", 0.7893, 127.3753),
    "Papua": ("Jayapura", -2.5337, 140.7181),
    "Papua Barat": ("Manokwari", -0.8615, 134.0620),
}

_last_call = 0.0


def get_json(params: dict, retries: int = 12) -> dict:
    global _last_call
    for attempt in range(1, retries + 1):
        wait = DELAY_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            r = requests.get(ARCHIVE, params=params, timeout=120)
        except requests.RequestException as e:
            print(f"  network error ({e}); retry {attempt}/{retries}", file=sys.stderr)
            time.sleep(5 * attempt)
            continue
        if r.status_code == 429:
            # Open-Meteo weights calls by size: ~10 years x 1 point ~= 260 calls, against
            # limits of 600/min, 5,000/hour, 10,000/day. Wait out minute/hour limits.
            reason = r.json().get("reason", "")
            if "Minutely" in reason or "Hourly" in reason:
                pause = 65 if "Minutely" in reason else 15 * 60
                print(f"  rate limited ({reason}) - sleeping {pause}s", file=sys.stderr, flush=True)
                time.sleep(pause)
                continue
            sys.exit(f"HTTP 429 from Open-Meteo ({reason}) - rerun tomorrow; cache is kept")
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"giving up on {params}")


def fetch_province(name: str, start: date, end: date, raw_dir: Path, refresh: bool) -> pd.DataFrame:
    city, lat, lon = COORDS[name]
    cache = raw_dir / f"{name.replace(' ', '_')}_{start}_{end}.json"
    if cache.exists() and not refresh:
        payload = json.loads(cache.read_text())
    else:
        payload = get_json({
            "latitude": lat, "longitude": lon,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "daily": ",".join(DAILY_VARS), "timezone": "auto",  # local day boundary
        })
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))
    df = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df.insert(1, "province", name)
    df.insert(2, "city", city)
    return df


def build_features(raw: pd.DataFrame, min_lag: int) -> pd.DataFrame:
    """Per province: rolling windows on calendar days, then shift by min_lag days."""
    out = []
    for prov, g in raw.groupby("province"):
        # reindex to a complete calendar so shift(n) means n days, not n rows
        g = g.set_index("date").asfreq("D")
        rain, tmean = g["precipitation_sum"], g["temperature_2m_mean"]
        f = pd.DataFrame(index=g.index)
        # point lags (lag k = weather of day t-k); k >= min_lag
        for k in sorted({min_lag, 3, 7, 14, 30}):
            if k >= min_lag:
                f[f"rain_lag{k}"] = rain.shift(k)
        # windows ending at t-min_lag
        for w in (7, 14, 30, 60, 90):
            f[f"rain_sum_{w}d"] = rain.rolling(w, min_periods=w).sum()
        for w in (7, 30):
            f[f"rain_days_{w}d"] = (rain >= 1.0).rolling(w, min_periods=w).sum()  # wet days
            f[f"rain_hours_{w}d"] = g["precipitation_hours"].rolling(w, min_periods=w).sum()
            f[f"temp_mean_{w}d"] = tmean.rolling(w, min_periods=w).mean()
            f[f"temp_max_{w}d"] = g["temperature_2m_max"].rolling(w, min_periods=w).max()
            f[f"rh_mean_{w}d"] = g["relative_humidity_2m_mean"].rolling(w, min_periods=w).mean()
            f[f"srad_mean_{w}d"] = g["shortwave_radiation_sum"].rolling(w, min_periods=w).mean()
            f[f"et0_mean_{w}d"] = g["et0_fao_evapotranspiration"].rolling(w, min_periods=w).mean()
        # anomaly: last 30 days of rain vs. the 365-day average 30-day rain
        f["rain_30d_anom"] = f["rain_sum_30d"] - rain.rolling(365, min_periods=180).sum() * 30 / 365
        roll_cols = [c for c in f.columns if not c.startswith("rain_lag")]
        f[roll_cols] = f[roll_cols].shift(min_lag)
        f.insert(0, "province", prov)
        out.append(f.reset_index())
    return pd.concat(out, ignore_index=True)


def check_no_leak(raw: pd.DataFrame, feats: pd.DataFrame, min_lag: int) -> None:
    """Spot-check: rain_sum_7d at t must equal rain summed over [t-min_lag-6, t-min_lag]."""
    prov = feats["province"].iloc[0]
    r = raw[raw.province == prov].set_index("date")["precipitation_sum"]
    f = feats[feats.province == prov].set_index("date")["rain_sum_7d"].dropna()
    t = f.index[len(f) // 2]
    expect = r.loc[t - pd.Timedelta(days=min_lag + 6): t - pd.Timedelta(days=min_lag)].sum()
    assert abs(f.loc[t] - expect) < 1e-6, (prov, t, f.loc[t], expect)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "dataset")
    ap.add_argument("--start", type=date.fromisoformat, default=START)
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    ap.add_argument("--min-lag", type=int, default=1, help="smallest lag in days (>= 1). Default 1")
    ap.add_argument("--province", nargs="*", default=None, help="province names (default: all in COORDS)")
    ap.add_argument("--refresh", action="store_true", help="ignore cached raw responses")
    args = ap.parse_args()
    if args.min_lag < 1:
        ap.error("--min-lag must be >= 1; lag 0 leaks same-day weather")

    names = args.province or list(COORDS)
    raw_dir = args.out / "raw" / "weather"
    frames = []
    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name} ({COORDS[name][0]})", flush=True)
        frames.append(fetch_province(name, args.start, args.end, raw_dir, args.refresh))
    raw = pd.concat(frames, ignore_index=True)
    feats = build_features(raw, args.min_lag)
    check_no_leak(raw, feats, args.min_lag)

    raw.to_csv(args.out / "weather_daily_raw.csv", index=False)
    feats.to_csv(args.out / "weather_features.csv", index=False)

    prices_path = args.out / "pihps_prices.csv"
    if prices_path.exists():
        prices = pd.read_csv(prices_path, parse_dates=["date"])
        prov = prices[prices["level"] == "province"]
        merged = prov.merge(feats, left_on=["region", "date"], right_on=["province", "date"],
                            how="left", suffixes=("", "_w")).drop(columns="province_w")
        merged.to_csv(args.out / "pihps_red_cayenne_pepper.csv", index=False)
        miss = merged.loc[merged["price"].notna(), "rain_sum_30d"].isna().groupby(
            merged["region"]).mean().loc[lambda s: s > 0.01]
        print(f"\nSaved {len(merged):,} rows -> {args.out / 'pihps_red_cayenne_pepper.csv'}")
        if len(miss):
            print("regions with >1% missing weather features:\n" + miss.round(3).to_string())

    (args.out / "weather_meta.json").write_text(json.dumps({
        "source": "Open-Meteo Historical Weather API (ERA5 reanalysis), " + ARCHIVE,
        "license": "CC BY 4.0 - attribute Open-Meteo",
        "accessed_at": datetime.now().isoformat(timespec="seconds"),
        "daily_variables": DAILY_VARS, "timezone": "auto (local)",
        "start": str(args.start), "end": str(args.end), "min_lag_days": args.min_lag,
        "leakage_rule": f"every feature on date t uses weather up to t-{args.min_lag} only",
        "locations": {k: {"city": v[0], "lat": v[1], "lon": v[2]} for k, v in COORDS.items() if k in names},
    }, indent=2, ensure_ascii=False))
    print(f"Saved {len(raw):,} raw rows, {len(feats):,} feature rows, "
          f"{feats.shape[1] - 2} features (min lag {args.min_lag} day)")


if __name__ == "__main__":
    main()
