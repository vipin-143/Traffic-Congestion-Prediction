import math
import pandas as pd
from .api import live_traffic, weather
from .data import weather_frame


def fetch_snapshot(detail, direction):
    payload = live_traffic(detail['id'])
    sensors = {s['name']: s for s in payload['sensorValues']}
    speed = sensors.get(f'KESKINOPEUS_5MIN_LIUKUVA_SUUNTA{direction}')
    flow = sensors.get(f'OHITUKSET_5MIN_LIUKUVA_SUUNTA{direction}')
    if speed is None or flow is None:
        raise ValueError('Five-minute speed and flow are unavailable for this station and direction.')
    stamp = pd.to_datetime(speed['measuredTime'], utc=True)
    flow_stamp = pd.to_datetime(flow['measuredTime'], utc=True)
    if abs((stamp-flow_stamp).total_seconds()) > 90:
        raise ValueError('Speed and flow timestamps do not align.')
    if not 0 <= float(speed['value']) < 200 or float(flow['value']) < 0:
        raise ValueError('Provider returned invalid traffic values.')
    age = (pd.Timestamp.now(tz='UTC')-stamp).total_seconds()/60
    if age < -2 or age > 10:
        raise ValueError(f'Traffic observation is stale or invalid ({age:.1f} minutes old).')
    lon, lat = detail['geometry']['coordinates'][:2]
    row = {'timestamp': stamp, 'station_id': detail['id'], 'direction': direction, 'speed_kmh': speed['value'], 'flow_vph': flow['value']}
    if flow.get('unit') != 'kpl/h':
        raise ValueError('Unexpected traffic flow unit; refusing an unverified conversion.')
    weather_error = None
    try:
        wp = weather(lat, lon)
        wf = weather_frame(wp)
        previous = wf[wf['weather_time'] <= stamp]
        if previous.empty or (stamp-previous.iloc[-1]['weather_time']).total_seconds()>7200:
            raise ValueError('Weather time is unavailable or stale')
        row.update(previous.iloc[-1].to_dict())
    except Exception as exc:
        weather_error = str(exc)
    return pd.DataFrame([row]), payload, weather_error
