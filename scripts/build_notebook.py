"""Create and execute the reproducible notebook with the current Python environment."""
import os
import sys
from pathlib import Path
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0,str(root))
cells=[]
def md(text):cells.append(nbf.v4.new_markdown_cell(text))
def code(text):cells.append(nbf.v4.new_code_cell(text))

md('''# Helsinki Traffic Forecasting with Traffic and Weather

## Objective and scope
Predict the five-minute average traffic speed ending **15 minutes after the observation time** at five Helsinki-region stations. Compare traffic-only and traffic-plus-weather models against a persistence baseline. This is regression; R² must not be described as classification accuracy.

Real Fintraffic vehicle data and Open-Meteo archived weather forecasts are used. Results apply to later dates at these stations, not automatically to other roads. Weather is modelled, not a roadside observation. All results below are computed from saved real data.''')
code('''from pathlib import Path
import json, inspect, sys
PROJECT_ROOT = Path.cwd() if (Path.cwd() / 'traffic').exists() else Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from traffic.config import DATA, ARTIFACTS, BASE_FEATURES, WEATHER_FEATURES
from traffic.features import supervised, chronological_split
from traffic.model import train_all, predict_rows, load_bundle, make_model
plt.rcParams.update({'figure.figsize': (11,4), 'axes.spines.top': False, 'axes.spines.right': False})
pd.set_option('display.max_columns', 25)
''')
md('''## Data sources and provenance
Traffic: [Digitraffic](https://www.digitraffic.fi/en/road-traffic/lam/), Fintraffic CC BY 4.0. Weather: [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api), CC BY 4.0; free API is for non-commercial use within its limits.

The manifest records daily source URLs, download timestamps, checksums, missing station-days and quality counts. Fetching can be repeated with `python -m traffic.data --start 2026-09-01 --end 2026-11-13`. Cached daily files avoid unnecessary downloads.''')
code('''df = pd.read_csv(DATA / 'traffic_weather.csv.gz')
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
manifest = json.loads((DATA / 'manifest.json').read_text())
display(pd.Series({k:manifest[k] for k in ['start_date','end_date','raw_rows','valid_raw_rows','aggregated_rows','downloaded_station_days','requested_station_days','weather_missing_fraction']}))
print('Saved table dimensions:', df.shape)
display(df.head())''')
md('''## Cleaning and temporal alignment
Raw files have 16 semicolon-separated fields. Invalid/faulty speed readings are removed. Ambiguous timestamps during the autumn repeated hour are discarded because their UTC offsets cannot be recovered safely. Vehicle observations are aggregated by station, direction and fixed five-minute window; the timestamp marks the window end. Count is converted to an hourly-equivalent flow by multiplying by 12.

Weather joins use the most recent hourly timestamp at or before the traffic timestamp with a two-hour tolerance. No future-time interpolation is used. A stitched archived forecast is not an issuance-time backtest, so prospective weather validation remains necessary. Missing values are imputed using medians fitted on training data only.''')
code('''quality = pd.DataFrame({'missing_count':df.isna().sum(), 'missing_percent':100*df.isna().mean()})
display(quality)
display(df.groupby(['station_id','station_name']).agg(rows=('speed_kmh','size'), mean_speed_kmh=('speed_kmh','mean')))
assert not df.duplicated(['timestamp','station_id','direction']).any()
assert df.speed_kmh.between(2,199).all()
if 'weather_time' in df:
    weather_time = pd.to_datetime(df.weather_time, utc=True)
    assert (weather_time.dropna() <= df.loc[weather_time.notna(),'timestamp']).all()
print('Integrity checks passed.')''')
md('## Exploratory analysis')
code('''fig, ax = plt.subplots()
for sid, part in df.groupby('station_id'):
    daily=part.set_index('timestamp').speed_kmh.resample('D').mean()
    ax.plot(daily.index,daily,label=str(sid))
ax.set(title='Daily average traffic speed',ylabel='Speed (km/h)',xlabel='UTC date')
ax.legend(title='Station'); plt.tight_layout(); plt.show()

local=df.timestamp.dt.tz_convert('Europe/Helsinki')
profile=df.assign(local_hour=local.dt.hour).groupby(['local_hour','direction']).speed_kmh.mean().unstack()
profile.plot(title='Hourly traffic profile in Helsinki local time',ylabel='Speed (km/h)',xlabel='Local hour')
plt.tight_layout(); plt.show()''')
code('''fig, axes = plt.subplots(1,2,figsize=(11,4))
axes[0].hist(df.speed_kmh,bins=60,color='#147d77')
axes[0].set(title='Distribution of five-minute average speed',xlabel='km/h',ylabel='Observations')
temp=df.groupby(df.temperature_2m.round()).speed_kmh.agg(['mean','size'])
temp=temp[temp['size']>=50]
axes[1].plot(temp.index,temp['mean'],marker='.')
axes[1].set(title='Speed by temperature bucket',xlabel='Temperature °C',ylabel='Average speed km/h')
plt.tight_layout(); plt.show()
print('Association is not causation; station, season and time of day are confounders.')''')
md('''## Target and features
The target is joined using the **same station and direction at exactly t + 15 minutes**. Gaps are not treated as adjacent observations. Inputs are the latest five-minute speed and flow, station and direction, sine/cosine of local time, weekday, month and weekend indicator. The weather experiment adds temperature, humidity, precipitation, snowfall and wind. No history warm-up is required for this snapshot model.''')
code('''paired=supervised(df)
train, validation, test=chronological_split(paired)
display(pd.DataFrame([{'partition':name,'rows':len(part),'start':part.timestamp.min(),'end':part.timestamp.max(),'latest_label':part.target_time.max()} for name,part in [('train',train),('validation',validation),('test',test)]]))
assert train.target_time.max() < validation.timestamp.min()
assert validation.target_time.max() < test.timestamp.min()
assert ((paired.target_time-paired.timestamp)==pd.Timedelta('15min')).all()
print('Traffic features:',BASE_FEATURES)
print('Additional weather features:',WEATHER_FEATURES)''')
md('''## Models and experimental design
Persistence predicts that speed remains unchanged. Random Forest and histogram gradient boosting model nonlinear patterns. Categorical station and direction are one-hot encoded. Numeric missing values use training medians. A fixed seed supports reproducibility.

The earliest 70% of observation timestamps form training, the next 15% validation, and the final 15% test. Labels crossing partition boundaries are purged. Choose each feature-set winner by validation MAE, refit it on training plus validation, and then evaluate on test. The automatic dashboard model is also selected on validation. Test scores are never used to tune hyperparameters.''')
code('''# Inspect the actual shared pipeline used in both the notebook and application.
print(inspect.getsource(make_model))
# Set True to rerun every candidate. False uses the already measured artifacts.
RETRAIN = False
if RETRAIN or not (ARTIFACTS/'summary.json').exists():
    summary=train_all()
else:
    summary=json.loads((ARTIFACTS/'summary.json').read_text())
    print('Loaded completed experiment from',summary['created_at'])
validation_metrics=pd.read_csv(ARTIFACTS/'validation_metrics.csv')
test_metrics=pd.read_csv(ARTIFACTS/'test_metrics.csv')
display(validation_metrics)
display(test_metrics)''')
md('''## Test results and weather comparison
MAE is the average absolute error in km/h; RMSE gives greater weight to large errors; R² compares squared error with a constant mean predictor. Lower MAE/RMSE are better. Weather is retained as an explicit comparison even if it does not improve forecasting.''')
code('''baseline=test_metrics.loc[test_metrics['mode']=='baseline','MAE_kmh'].iloc[0]
best_mode=summary['default_mode']
best=test_metrics.loc[test_metrics['mode']==best_mode,'MAE_kmh'].iloc[0]
print(f'Validation-selected default: {best_mode}')
print(f'Test MAE: {best:.3f} km/h; persistence: {baseline:.3f} km/h')
print(f'Test MAE reduction relative to persistence: {100*(baseline-best)/baseline:.1f}%')
weather_mae=test_metrics.loc[test_metrics['mode']=='weather','MAE_kmh'].iloc[0]
traffic_mae=test_metrics.loc[test_metrics['mode']=='traffic','MAE_kmh'].iloc[0]
print('Weather improves test MAE in this run:',weather_mae < traffic_mae)
test_metrics.set_index('mode')[['MAE_kmh','RMSE_kmh']].plot.bar(rot=0,title='Held-out test errors')
plt.ylabel('km/h');plt.tight_layout();plt.show()''')
code('''pred=pd.read_csv(ARTIFACTS/'test_predictions.csv.gz')
pred['target_time']=pd.to_datetime(pred.target_time,utc=True)
example=pred[(pred.station_id==20029)&(pred.direction==1)].tail(288)
example.set_index('target_time')[['target_speed_kmh','predicted_traffic','predicted_weather']].plot(title='Actual and forecast speed at station 20029 direction 1',ylabel='km/h',xlabel='Target UTC time')
plt.tight_layout();plt.show()
display(pred.groupby(['station_id','direction']).apply(lambda g:pd.Series({'rows':len(g),'traffic_MAE':np.abs(g.target_speed_kmh-g.predicted_traffic).mean(),'weather_MAE':np.abs(g.target_speed_kmh-g.predicted_weather).mean()})))''')
code('''importance_path=ARTIFACTS/'feature_importance.csv'
if importance_path.exists():
    imp=pd.read_csv(importance_path)
    imp.sort_values('mae_increase').plot.barh(x='feature',y='mae_increase',legend=False,title='Validation permutation importance',xlabel='Increase in MAE km/h')
    plt.tight_layout();plt.show()
    print('Computed on validation with a model fitted only on training.')''')
md('''## Saved model and dashboard inference
The application loads the trusted saved pipeline and applies the same feature function. Missing live weather triggers the separately evaluated traffic-only model. Live windows are rolling provider estimates, whereas historical windows are fixed raw aggregates. Live units and timestamp freshness are checked; stale predictions are refused.

The congestion display uses speed divided by a training-only reference speed: below 0.70 is Heavy, below 0.90 is Moderate, otherwise Low. These are project definitions, not official congestion labels.''')
code('''bundle=load_bundle()
display(predict_rows(df.tail(5),bundle,mode='auto'))
print('Launch: python -m streamlit run app.py')''')
md('''## Conclusions and limitations
The saved tables above contain the actual measured results. No classification accuracy or prediction confidence is inferred from R². This experiment compares traffic and weather features using chronological holdout data. It does not establish that weather causes congestion or that performance will transfer to other roads.

The deployment should monitor future errors, provider outages, changing road limits and seasonal shifts. Extend collection across a full year and conduct issuance-time weather evaluation before making stronger operational claims. The historical weather comparison is retrospective. Repeat training after expanding the archive; preserve a new final test period.

## Reproduction and sources
1. Install `requirements.txt` in Python 3.12.
2. Run the historical downloader, then `python -m traffic.model`.
3. Execute this notebook with `RETRAIN=True` for a fresh experiment.
4. Run tests with `python -m unittest discover -s tests -v`.
5. Start Streamlit. Historical replay works from included data; live requests require internet.

Traffic source: https://www.digitraffic.fi/en/road-traffic/  
Traffic format: https://www.digitraffic.fi/en/road-traffic/lam/  
Weather source: https://open-meteo.com/en/docs/historical-forecast-api  
Weather use terms: https://open-meteo.com/en/terms

Attribution: Fintraffic / digitraffic.fi and Open-Meteo, CC BY 4.0. Data were filtered, aggregated and joined.''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':sys.version.split()[0]}})
out=root/'notebooks'/'Traffic_Forecasting.ipynb'
out.parent.mkdir(exist_ok=True)
shell=InteractiveShell.instance()
shell.run_cell("import matplotlib; matplotlib.use('module://matplotlib_inline.backend_inline')")
execution=0
for cell in nb.cells:
    if cell.cell_type!='code': continue
    execution+=1
    print('Executing code cell',execution,flush=True)
    with capture_output() as captured:
        result=shell.run_cell(cell.source,store_history=False)
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(f'Notebook cell {execution} failed: {result.error_before_exec or result.error_in_exec}')
    outputs=[]
    if captured.stdout: outputs.append(nbf.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr: outputs.append(nbf.v4.new_output('stream',name='stderr',text=captured.stderr))
    for rich in captured.outputs:
        outputs.append(nbf.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
    cell.outputs=outputs
    cell.execution_count=execution
    nbf.write(nb,out)
nbf.write(nb,out)
print('Executed notebook:',out,flush=True)
