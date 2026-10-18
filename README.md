# pihps-cabai-rawit-forecast

Predicts the daily traditional-market price of **cabai rawit merah** (red cayenne pepper), 
Rp/kg, 7 working days ahead, for 6 Indonesian provinces,
using price history from [PIHPS](https://www.bi.go.id/hargapangan) Nasional (Bank Indonesia) joined with rainfall
and temperature features from [Open-Meteo](https://open-meteo.com/).

Chili was picked over other staples because it's the most volatile (year-on-
year moves far exceed flat staples like rice), has ~1,100 working days of
history per province (several seasonal cycles), and has plausible external
drivers (rainfall, Ramadan/Lebaran/Christmas demand, cross-region prices) to
join in. See `artifact/Pred2-group-assignment.pdf` for the full assignment
brief.

## Provinces

6 provinces were chosen from all 34 for data coverage (≥ 93% of working days
filled) and price volatility (coefficient of variation, so the series isn't
nearly flat).

| Province | PIHPS ID | Coverage | CV% |
|---|---|---|---|
| Jawa Tengah | 14 | 97.6% | 46.3% |
| DI Yogyakarta | 15 | 96.5% | 47.8% |
| Jawa Timur | 16 | 96.5% | 52.3% |
| Bali | 17 | 93.9% | 51.7% |
| Nusa Tenggara Barat | 18 | 96.4% | 55.6% |
| DKI Jakarta | 13 | 95.9% | 40.7% |

Full history, 2017-01-02 to today. `dataset/pihps_quality.csv` has the
coverage/volatility report for every province, including the ones dropped
(Aceh, Sumatera Barat, Kepulauan Riau are too little or too short history).

## Project design

- **Target:** price 7 working days ahead, per province.
- **Baseline:** To be established. A seasonal-naive forecast will be evaluated as the initial benchmark.
- **Candidates:** We will evaluate some model and approach (SARIMAX, Holt-Winters/ETS, and tree-based models (LightGBM or Random Forest) using lagged, rolling, and rainfall features). 
Model selection and tuning will use time-series-aware validation, such as `TimeSeriesSplit` or expanding-window evaluation.
- **Interpretation:** feature importance or SHAP, framed as predictive, not
  causal.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) (manages the Python version and
dependencies no manual virtualenv needed) :

```bash
uv sync
```

## 1. Fetch prices (PIHPS)

```bash
uv run scripts/fetch_pihps.py --level province --province 13 14 15 16 17 18
```

- Fetches cabai rawit merah (commodity 16, the default) for **all** provinces
  in one request per quarter because that's how the PIHPS API works at province
  level, it can't be narrowed server-side then keeps only the national row
  plus the 6 chosen provinces in the saved CSV.
- Default date range is the full history: 2017-01-02 to today.
- Outputs to `dataset/`: `pihps_prices.csv`, `pihps_quality.csv` (coverage/CV
  report), `pihps_meta.json`, and raw JSON responses cached under
  `dataset/raw/` so reruns don't re-hit the server.
- If the server returns HTTP 403/429/503 it stops politely; just rerun later
  to resume from the cache. Add `--refresh` to force a full re-pull.
- To look up other province or commodity IDs:
  `uv run scripts/fetch_pihps.py --list`

## 2. Fetch weather (Open-Meteo) and join to prices

```bash
uv run scripts/fetch_weather.py --province "Jawa Tengah" "DI Yogyakarta" "Jawa Timur" "Bali" "Nusa Tenggara Barat" "DKI Jakarta"
```

- One request per province (the provincial capital's coordinates), covering
  2016-10-01 to today by default 6 months before the price history so
  rolling 90-day windows are full from the start.
- Builds lagged/rolling rainfall, temperature, humidity, shortwave radiation
  and ET0 (evapotranspiration) features (`weather_features.csv`), leak-safe:
  every feature at date *t* only uses weather up to *t-1* (`--min-lag` to
  make this stricter).
- Requires `dataset/pihps_prices.csv` to already exist (step 1) because it will merges
  province-level prices with the weather features into
  `dataset/pihps_red_cayenne_pepper.csv`, the file to load for modeling.
- Also writes `dataset/weather_daily_raw.csv` (EDA only not lagged, don't
  join this one directly to prices) and `dataset/weather_meta.json`.
- Open-Meteo is rate-limited; the script paces requests automatically
  (~31 s apart) and sleeps out any 429s so a full 5-province pull can takes a few
  minutes.

## 3. Add BBM (Solar/diesel) price

```bash
uv run scripts/fetch_bbm.py --join dataset/pihps_red_cayenne_pepper.csv
```

- Not an API fetch: subsidized-fuel prices in Indonesia are discrete,
  government-announced step-changes, not a scrapeable live feed (see sources
  below), so the source of truth is a **hand-compiled, cited** table,
  `dataset/bbm_price_events.csv` (one row per confirmed price change or
  confirmed/suspected gap, with `source` and `confidence` columns). Nothing
  in it is guessed all unverified periods are left `NaN`, never forward-filled.
  Unlike `dataset/raw/`, this file is **not** gitignored cache: it's
  irreplaceable research, not something a rerun regenerates.
- Turns that events table into a daily, per-province feature via an as-of
  (backward-fill) join that leak-safe by construction, since these are
  same-day public announcements, so no `--min-lag` is needed.
- Only **Solar** (subsidized diesel, Biosolar CN48) is selected for the main
  dataset: 100% coverage vs. 44–71% for Premium/Pertalite, and it carries
  the direct causal story (diesel trucks move chili through the supply
  chain). Over the full history Solar has exactly two price levels: Rp
  5,150/L until 2022-09-03, then Rp 6,800/L after encoding that (raw
  Rp/liter vs. a regime flag) is deferred to the EDA/feature-engineering
  step.
- Outputs to `dataset/`: `bbm_features.csv` (selected: `solar_rp_per_l`),
  `bbm_features_all.csv` (Premium + Pertalite + Solar, for reference/
  documentation only), `bbm_meta.json`, `bbm_quality.csv` (per-province,
  per-fuel coverage against actual chili-price rows). With `--join`, it also
  left-joins `solar_rp_per_l` into `dataset/pihps_red_cayenne_pepper.csv` in
  place (matched on province + date).
- Sources: Xie & Harjono, *The retail fuels market in Indonesia*, ICCT
  Working Paper 2020-23; official Kementerian/Ditjen Migas ESDM
  announcements for the 2015-01-19, 2016-01-05 and 2016-04-01 Solar price
  steps (esdm.go.id / migas.esdm.go.id); indonesiabaik.id (Kominfo)
  infographic on the 3 Sep 2022 price adjustment; news confirming
  Solar/Pertalite unchanged through 2023-2026 (Bisnis.com, CNBC Indonesia,
  RRI, BCA Sekuritas, Antara, RCTI+); Kepmen ESDM No. 245.K/MG.01/MEM.M/2022
  (Premium discontinuation). Full citations and known gaps are documented
  per-row in `dataset/bbm_price_events.csv`.

## Output for modeling

After all three steps, `dataset/pihps_red_cayenne_pepper.csv` has one row
per (province, date) with `price`, all weather features, and
`solar_rp_per_l`, load this for the forecasting notebook. Check
`dataset/pihps_quality.csv` after step 1, the "missing weather features"
warning printed after step 2, and `dataset/bbm_quality.csv` after step 3,
before trusting a given province's series.

## Layout (Temporary)

```
scripts/      fetch_pihps.py, fetch_weather.py, fetch_bbm.py
dataset/      fetched CSVs + raw/ cache (gitignored except small metadata)
artifact/     assignment brief (PDF)
```
