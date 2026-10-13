# pihps-cabai-rawit-forecast

Predicts the daily traditional-market price of **cabai rawit merah** (red cayenne pepper), 
Rp/kg, 7 working days ahead, for 5 Indonesian provinces,
using price history from [PIHPS](https://www.bi.go.id/hargapangan) Nasional (Bank Indonesia) joined with rainfall
and temperature features from [Open-Meteo](https://open-meteo.com/).

Chili was picked over other staples because it's the most volatile (year-on-
year moves far exceed flat staples like rice), has ~1,100 working days of
history per province (several seasonal cycles), and has plausible external
drivers (rainfall, Ramadan/Lebaran/Christmas demand, cross-region prices) to
join in. See `artifact/Pred2-group-assignment.pdf` for the full assignment
brief.

## Provinces

5 provinces, chosen from all 34 for data coverage (≥ 93% of working days
filled) and price volatility (coefficient of variation, so the series isn't
nearly flat):

| Province | PIHPS ID | Coverage | CV% |
|---|---|---|---|
| Jawa Tengah | 14 | 97.6% | 46.3% |
| DI Yogyakarta | 15 | 96.5% | 47.8% |
| Jawa Timur | 16 | 96.5% | 52.3% |
| Bali | 17 | 93.9% | 51.7% |
| Nusa Tenggara Barat | 18 | 96.4% | 55.6% |

Full history, 2017-01-02 to today. `dataset/pihps_quality.csv` has the
coverage/volatility report for every province, including the ones dropped
(Aceh, Sumatera Barat, Kepulauan Riau — too little or too short history).

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
uv run scripts/fetch_pihps.py --level province --province 14 15 16 17 18
```

- Fetches cabai rawit merah (commodity 16, the default) for **all** provinces
  in one request per quarter — that's how the PIHPS API works at province
  level, it can't be narrowed server-side — then keeps only the national row
  plus the 5 chosen provinces in the saved CSV.
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
uv run scripts/fetch_weather.py --province "Jawa Tengah" "DI Yogyakarta" "Jawa Timur" "Bali" "Nusa Tenggara Barat"
```

- One request per province (the provincial capital's coordinates), covering
  2016-10-01 to today by default — 6 months before the price history so
  rolling 90-day windows are full from the start.
- Builds lagged/rolling rainfall and temperature features
  (`weather_features.csv`), leak-safe: every feature at date *t* only uses
  weather up to *t-1* (`--min-lag` to make this stricter).
- Requires `dataset/pihps_prices.csv` to already exist (step 1) — it merges
  province-level prices with the weather features into
  `dataset/pihps_weather.csv`, the file to load for modeling.
- Also writes `dataset/weather_daily_raw.csv` (EDA only — not lagged, don't
  join this one directly to prices) and `dataset/weather_meta.json`.
- Open-Meteo is rate-limited; the script paces requests automatically
  (~31 s apart) and sleeps out any 429s — a full 5-province pull takes a few
  minutes.

## Output for modeling

After both steps, `dataset/pihps_weather.csv` has one row per
(province, date) with `price` and all weather features, load this for the
forecasting notebook. Check `dataset/pihps_quality.csv` after step 1, and the
"missing weather features" warning printed after step 2, before trusting a
given province's series.

## Layout (Temporary)

```
scripts/      fetch_pihps.py, fetch_weather.py
dataset/      fetched CSVs + raw/ cache (gitignored except small metadata)
artifact/     assignment brief (PDF)
```
