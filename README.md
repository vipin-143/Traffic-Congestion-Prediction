# Helsinki Traffic Forecast

An English ML notebook and Streamlit dashboard for 15-minute-ahead average speed forecasting at five Helsinki-region traffic stations. Real Fintraffic vehicle observations are aggregated into five-minute station-direction records and joined to Open-Meteo modelled weather. No fabricated observations or performance claims.

## Run

On the original workspace, right-click run_dashboard.ps1 and run with PowerShell, or:

```powershell
./run_dashboard.ps1
```

On another computer, use Python 3.12 and run from this directory:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m streamlit run app.py
```

The saved data, trained model, measured metrics and executed notebook are included. Historical replay and model evaluation work without API access; map tiles require internet. Live predictions need internet but no API credentials. Click **Live forecast**, select a station and direction, then **Fetch latest traffic and predict**.

## Reproduce

```powershell
python -m traffic.data --start 2026-09-01 --end 2026-11-13
python -m traffic.model
python scripts/build_notebook.py
python -m unittest discover -s tests -v
```

Downloader is resumable and caches daily files. Metadata and the manifest preserve source URLs, checksums and quality counts. Missing station-days are recorded, not synthesized. To expand the date range, rerun the download and train again. Numerical results are in artifacts/summary.json and artifacts/test_metrics.csv. Notebook opens in Jupyter or VS Code and is displayed on the dashboard's Notebook page.

## Method

Target: five-minute average speed ending 15 minutes after the feature observation. Inputs: current five-minute mean speed and hourly-equivalent flow, station, direction, cyclic local hour, weekday, month, weekend; the weather version adds temperature, humidity, precipitation, snowfall and wind. No lag-history warm-up is required. Models: persistence, Random Forest and histogram gradient boosting. Chronological 70/15/15 split; rows whose future labels cross split boundaries are excluded. Each model version is selected by validation MAE; final fitting uses train and validation, never test. MAE, RMSE and R2 are reported on test. A persistence winner is retained honestly if learned models do not improve validation.

## Limitations

The archive covers a limited season and the same stations, not unseen cities. Fintraffic live inputs are rolling five-minute estimates while training uses fixed five-minute raw aggregates. Open-Meteo historical forecasts are stitched retrospective model outputs, not a guaranteed issuance-time backtest. Weather gains must therefore be treated as exploratory until prospective testing. Timestamp joins never use later weather timestamps. Ambiguous daylight-saving observations are dropped. Missing weather triggers the evaluated traffic-only model. Congestion is a project-defined speed-ratio display category, not official labelled ground truth. No prediction confidence percentages are invented.

Live speed and flow must align, use expected units, and be under 10 minutes old. The forecast time is observation time plus 15 minutes, not page-open time plus 15 minutes. No continuous collection happens while the app is closed. The app exposes a manual refresh with 60-second caching.

## Sources and license

- Traffic: Fintraffic / digitraffic.fi, CC BY 4.0: https://www.digitraffic.fi/en/terms-of-service/
- Traffic API: https://www.digitraffic.fi/en/road-traffic/
- Raw format: https://www.digitraffic.fi/en/road-traffic/lam/
- Weather: Open-Meteo, CC BY 4.0: https://open-meteo.com/en/docs/historical-forecast-api
- Free weather API is for non-commercial use within request limits: https://open-meteo.com/en/terms

Data have been filtered, aggregated and joined. Only load this project's trusted joblib model; never load untrusted uploaded pickle/joblib files.
