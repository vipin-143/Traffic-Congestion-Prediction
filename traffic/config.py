from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ARTIFACTS = ROOT / 'artifacts'
STATIONS = [20029, 23117, 23146, 23149, 23154]
TRAFFIC_API = 'https://tie.digitraffic.fi/api/tms/v1'
WEATHER_API = 'https://api.open-meteo.com/v1/forecast'
WEATHER_HISTORY_API = 'https://historical-forecast-api.open-meteo.com/v1/forecast'
WEATHER_FEATURES = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'snowfall', 'wind_speed_10m']
BASE_FEATURES = ['speed_kmh', 'flow_vph', 'station_id', 'direction', 'hour_sin', 'hour_cos', 'weekday', 'month', 'weekend']
HORIZON = 15
