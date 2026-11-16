"""Public HTTPS clients. No API keys; never silently substitute fake data."""
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import TRAFFIC_API, WEATHER_API, WEATHER_HISTORY_API, WEATHER_FEATURES

_lock = threading.Lock()
_last_request = 0.0


def get(url, params=None):
    global _last_request
    with _lock:
        wait = 1.15 - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
    session = requests.Session()
    session.headers.update({'Digitraffic-User': 'StudentHelsinkiTraffic/1.0', 'Accept-Encoding': 'gzip'})
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)
    session.mount('https://', HTTPAdapter(max_retries=retry))
    try:
        r = session.get(url, params=params, timeout=(15, 60))
        r.raise_for_status()
        return r
    finally:
        session.close()


def station_detail(station_id):
    return get(f'{TRAFFIC_API}/stations/{int(station_id)}').json()


def live_traffic(station_id):
    return get(f'{TRAFFIC_API}/stations/{int(station_id)}/data').json()


def weather(latitude, longitude, start=None, end=None):
    params = {'latitude': latitude, 'longitude': longitude, 'hourly': ','.join(WEATHER_FEATURES), 'timezone': 'UTC', 'wind_speed_unit': 'kmh'}
    if start is not None:
        params.update(start_date=str(start), end_date=str(end))
        url = WEATHER_HISTORY_API
    else:
        params.update(past_days=2, forecast_days=1, current=','.join(WEATHER_FEATURES))
        url = WEATHER_API
    return get(url, params=params).json()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    temp.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()
