"""Run with python -m streamlit run app.py from the project directory."""
import base64
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from traffic.config import DATA, ARTIFACTS, ROOT, WEATHER_FEATURES
from traffic.model import load_bundle, predict_rows
from traffic.live import fetch_snapshot

st.set_page_config(page_title='Helsinki Traffic Forecast', page_icon='🚦', layout='wide')
st.markdown('''<style>.block-container{padding-top:2rem;max-width:1400px}div[data-testid="stMetric"]{background:#edf5fa;padding:18px;border-radius:12px}h1{letter-spacing:-1.5px} [data-testid="stSidebar"]{background:#eaf1f5}</style>''', unsafe_allow_html=True)
st.title('Helsinki Traffic Forecast')
st.caption('Five Helsinki-region stations · 15-minute speed forecasts · Real traffic and modelled weather')

@st.cache_data
def historical():
    df = pd.read_csv(DATA/'traffic_weather.csv.gz')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df

@st.cache_resource
def model():
    return load_bundle()

@st.cache_data(ttl=60, show_spinner=False)
def live(detail, direction):
    return fetch_snapshot(detail, direction)

if not (ARTIFACTS/'models.joblib').exists():
    st.error('Train the model first: python -m traffic.model')
    st.stop()
details = json.loads((DATA/'stations.json').read_text(encoding='utf-8'))
by_id = {d['id']:d for d in details}
station = st.sidebar.selectbox('Traffic station', list(by_id), format_func=lambda s: by_id[s]['properties']['name'].replace('_',' '))
direction = st.sidebar.radio('Direction', [1,2], horizontal=True)
detail = by_id[station]
destination = detail['properties'].get(f'direction{direction}Municipality', 'Provider direction '+str(direction))
st.sidebar.caption(f'Direction {direction}: toward {destination}')
page = st.sidebar.radio('Explore', ['Historical demo', 'Live forecast', 'Model evaluation', 'Notebook', 'Data and sources'])
choice = st.sidebar.selectbox('Prediction model', ['auto','weather','traffic'], format_func=lambda s:{'auto':'Best validation model','weather':'Traffic plus weather','traffic':'Traffic only'}[s])
st.sidebar.caption('No API keys required. Live access needs internet.')
df = historical()
subset = df[(df.station_id==station)&(df.direction==direction)].copy()
bundle = model()

def show_prediction(row, title):
    pred = predict_rows(row, bundle, choice).iloc[0]
    st.subheader(title)
    a,b,c,d = st.columns(4)
    a.metric('Measured speed', f'{pred.speed_kmh:.1f} km/h')
    b.metric('Predicted speed in 15 minutes',f'{pred.predicted_speed_kmh:.1f} km/h',f'{pred.predicted_speed_kmh-pred.speed_kmh:+.1f} km/h')
    c.metric('Traffic flow',f'{pred.flow_vph:,.0f} vehicles/h')
    d.metric('Predicted congestion',pred.congestion)
    st.caption(f'Observation: {pred.timestamp.tz_convert("Europe/Helsinki")} · Prediction for: {pred.prediction_at.tz_convert("Europe/Helsinki")} · {pred.algorithm} · {pred.model_mode}')
    st.caption('Congestion is a project-defined label relative to the training-period 95th-percentile speed, not an official provider classification.')
    if pred.weather_fallback:
        st.warning('Weather is missing. Using the separately evaluated traffic-only model.')
    present = [x for x in WEATHER_FEATURES if x in row and pd.notna(row.iloc[0][x])]
    if present:
        st.write('Weather inputs')
        st.dataframe(row[present].rename(columns={'temperature_2m':'Temperature °C','relative_humidity_2m':'Humidity %','precipitation':'Precipitation mm','snowfall':'Snowfall cm','wind_speed_10m':'Wind km/h'}),hide_index=True)
    st.download_button('Download this prediction',pd.DataFrame([pred]).to_csv(index=False),'prediction.csv','text/csv')

if page == 'Historical demo':
    st.info('Historical replay using real saved observations. This page is not live.')
    dates = sorted(subset.timestamp.dt.date.unique())
    day = st.selectbox('Observation day', dates,index=len(dates)-1)
    daily = subset[subset.timestamp.dt.date==day]
    pos = st.slider('Five-minute observation',0,len(daily)-1,len(daily)//2)
    row = daily.iloc[[pos]]
    show_prediction(row,'Historical forecast')
    target = row.iloc[0].timestamp+pd.Timedelta('15min')
    actual = subset[subset.timestamp==target]
    if not actual.empty:
        st.write(f'Actual speed at the target time: **{actual.iloc[0].speed_kmh:.1f} km/h**')
    st.plotly_chart(px.line(daily,x='timestamp',y='speed_kmh',labels={'speed_kmh':'Average speed (km/h)','timestamp':'UTC time'},title='Measured speed through the day'),width='stretch')
    lon,lat = detail['geometry']['coordinates'][:2]
    st.map(pd.DataFrame({'latitude':[lat],'longitude':[lon]}),zoom=11)
elif page == 'Live forecast':
    st.write('Fetch the provider’s latest five-minute traffic estimate and weather for this station.')
    st.caption('Traffic responses are cached for 60 seconds. Fetching does not imply continuous background collection.')
    if st.button('Fetch latest traffic and predict',type='primary'):
        try:
            with st.spinner('Requesting traffic and weather…'):
                row,payload,error = live(detail,direction)
            st.session_state['live_result']=(station,direction,row,payload,error)
        except Exception as exc:
            st.error(f'Live request failed: {exc}. Historical replay remains available.')
            st.session_state.pop('live_result',None)
    cached=st.session_state.get('live_result')
    if cached and cached[0:2]==(station,direction):
        _,_,row,payload,error=cached
        age=(pd.Timestamp.now(tz='UTC')-row.iloc[0].timestamp).total_seconds()/60
        if age>10:
            st.warning('The displayed snapshot has expired. Fetch a new reading before predicting.')
        else:
            show_prediction(row,'Latest available forecast')
            if error: st.warning('Weather unavailable: '+error)
            with st.expander('Provider traffic response'):st.json(payload)
elif page == 'Model evaluation':
    summary=json.loads((ARTIFACTS/'summary.json').read_text())
    st.subheader('Measured performance on later unseen dates')
    st.caption('Lower MAE and RMSE are better. Models are selected on validation MAE, not test scores.')
    st.dataframe(pd.read_csv(ARTIFACTS/'test_metrics.csv'),hide_index=True,width='stretch')
    st.write('Validation comparison')
    st.dataframe(pd.read_csv(ARTIFACTS/'validation_metrics.csv'),hide_index=True,width='stretch')
    st.write('Chronological partitions')
    st.dataframe(pd.DataFrame(summary['splits']).T,width='stretch')
    p=pd.read_csv(ARTIFACTS/'test_predictions.csv.gz')
    p=p[(p.station_id==station)&(p.direction==direction)].tail(288)
    st.plotly_chart(px.line(p,x='target_time',y=['target_speed_kmh','predicted_traffic','predicted_weather'],title='Actual and predicted target speed'),width='stretch')
    importance=ARTIFACTS/'feature_importance.csv'
    if importance.exists():
        st.plotly_chart(px.bar(pd.read_csv(importance).sort_values('mae_increase'),x='mae_increase',y='feature',orientation='h',title='Validation permutation importance'),width='stretch')
    st.write('Practical limitations')
    for item in summary['limitations']:st.markdown('- '+item)
elif page == 'Notebook':
    path=ROOT/'notebooks'/'Traffic_Forecasting.ipynb'
    if not path.exists():st.info('Notebook generation is pending.')
    else:
        st.download_button('Download executed notebook',path.read_bytes(),path.name)
        nb=json.loads(path.read_text(encoding='utf-8'))
        for cell in nb['cells']:
            source=''.join(cell['source'])
            if cell['cell_type']=='markdown':st.markdown(source)
            elif cell['cell_type']=='code':
                with st.expander('Python code'):st.code(source,language='python')
                for output in cell.get('outputs',[]):
                    data=output.get('data',{})
                    if 'image/png' in data:st.image(base64.b64decode(''.join(data['image/png'])))
                    elif 'text/html' in data:st.html(''.join(data['text/html']))
                    elif 'text/plain' in data:st.code(''.join(data['text/plain']),language=None)
                    elif output.get('text'):st.code(''.join(output['text']),language=None)
elif page == 'Data and sources':
    manifest=json.loads((DATA/'manifest.json').read_text())
    st.subheader('Real data and reproducible provenance')
    a,b,c=st.columns(3)
    a.metric('Aggregated rows',f'{len(df):,}');b.metric('Columns',len(df.columns));c.metric('Raw vehicle records',f'{manifest["raw_rows"]:,}')
    st.write(f'Period: {manifest["start_date"]} to {manifest["end_date"]}; successful station-days: {manifest["downloaded_station_days"]}/{manifest["requested_station_days"]}.')
    st.dataframe(subset.tail(100),hide_index=True)
    st.download_button('Download selected station and direction CSV',subset.to_csv(index=False),'station_history.csv','text/csv')
    st.write('[Fintraffic traffic documentation](https://www.digitraffic.fi/en/road-traffic/) · [Historical traffic format](https://www.digitraffic.fi/en/road-traffic/lam/) · [Open-Meteo historical forecasts](https://open-meteo.com/en/docs/historical-forecast-api)')
    with st.expander('Unavailable station-days'):st.json(manifest['failures'])
st.divider()
st.caption('Traffic: Fintraffic / digitraffic.fi, CC BY 4.0. Weather: Open-Meteo, CC BY 4.0. Data are filtered, aggregated and modelled. Educational use; weather free API subject to non-commercial limits.')
