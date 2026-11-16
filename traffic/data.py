"""Download real observations, aggregate fixed five-minute windows and record provenance."""
import argparse
import gzip
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .api import get, station_detail, weather, write_json, utc_now
from .config import DATA, STATIONS, TRAFFIC_API, WEATHER_FEATURES

RAW_COLUMNS = ['tms_number', 'year', 'day', 'hour', 'minute', 'second', 'fraction', 'length', 'lane', 'direction', 'vehicle_class', 'speed', 'faulty', 'technical_total', 'technical_interval', 'queue_start']


def aggregate_raw(content, station_id):
    df = pd.read_csv(io.BytesIO(content), sep=';', header=None, names=RAW_COLUMNS)
    total = len(df)
    for c in RAW_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    valid = df['faulty'].eq(0) & df['speed'].between(2, 198.999) & df['direction'].isin([1, 2])
    valid &= df['hour'].between(0, 23) & df['minute'].between(0, 59) & df['second'].between(0, 59)
    df = df[valid].copy()
    # Avoid assuming a unit for the provider's fractional-second field. Five-minute
    # aggregation does not require subsecond precision.
    day = pd.to_datetime((2000 + df['year']).astype('int').astype(str) + '-01-01')
    naive = day + pd.to_timedelta(df['day'] - 1, unit='D') + pd.to_timedelta(df['hour'], unit='h') + pd.to_timedelta(df['minute'], unit='min') + pd.to_timedelta(df['second'], unit='s')
    # During the autumn repeated hour the raw format has no UTC offset. Drop
    # ambiguous times instead of silently assigning the wrong direction in time.
    df['timestamp'] = naive.dt.tz_localize('Europe/Helsinki', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = df['timestamp'].dt.floor('5min') + pd.Timedelta(minutes=5)
    agg = df.groupby(['timestamp', 'direction']).agg(speed_kmh=('speed', 'mean'), vehicle_count=('speed', 'size')).reset_index()
    agg['flow_vph'] = agg['vehicle_count'] * 12.0
    agg['station_id'] = int(station_id)
    return agg, {'raw_rows': total, 'valid_rows': len(df), 'discarded_rows': total - len(df)}


def weather_frame(payload):
    hourly = payload.get('hourly', {})
    frame = pd.DataFrame(hourly)
    if frame.empty or 'time' not in frame:
        raise ValueError('Weather API returned no hourly records')
    frame['weather_time'] = pd.to_datetime(frame.pop('time'), utc=True)
    return frame.sort_values('weather_time')


def join_weather(traffic, payload):
    w = weather_frame(payload)
    # Use only weather timestamps at or before the traffic window end, never
    # interpolate backwards from a later weather timestamp.
    result = pd.merge_asof(traffic.sort_values('timestamp'), w, left_on='timestamp', right_on='weather_time', direction='backward', tolerance=pd.Timedelta('2h'))
    return result


def fetch_day(detail, day):
    p = detail['properties']
    station_id, tms = int(p['id']), int(p['tmsNumber'])
    path = DATA / 'raw' / str(station_id) / f'{day}.csv.gz'
    agg_path = DATA / 'daily' / str(station_id) / f'{day}.csv.gz'
    meta_path = agg_path.with_suffix('.json')
    url = f'{TRAFFIC_API}/history/raw/lamraw_{tms}_{day.year % 100}_{day.timetuple().tm_yday}.csv'
    if agg_path.exists() and meta_path.exists():
        return pd.read_csv(agg_path, parse_dates=['timestamp']), json.loads(meta_path.read_text(encoding='utf-8'))
    if path.exists():
        content = gzip.decompress(path.read_bytes())
    else:
        content = get(url).content
        if not content.strip() or b';' not in content[:200]:
            raise ValueError(f'No vehicle records for station {station_id} on {day}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(content))
    agg, quality = aggregate_raw(content, station_id)
    record = {'station_id': station_id, 'date': str(day), 'source_url': url, 'downloaded_at': utc_now(), 'sha256_uncompressed': hashlib.sha256(content).hexdigest(), **quality}
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(agg_path, index=False, compression='gzip')
    write_json(meta_path, record)
    return agg, record


def prepare(start, end, stations=STATIONS):
    start, end = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    if end < start:
        raise ValueError('End date must not precede start date')
    DATA.mkdir(parents=True, exist_ok=True)
    details = []
    for station in stations:
        path = DATA / 'metadata' / f'{station}.json'
        detail = json.loads(path.read_text(encoding='utf-8')) if path.exists() else station_detail(station)
        write_json(path, detail)
        details.append(detail)
    write_json(DATA / 'stations.json', details)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    frames, records, failures = [], [], []
    tasks = [(detail, day) for detail in details for day in days]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(fetch_day, detail, day): (detail['id'], day) for detail, day in tasks}
        for i, future in enumerate(as_completed(pending), 1):
            sid, day = pending[future]
            try:
                frame, record = future.result()
                frames.append(frame)
                records.append(record)
            except Exception as exc:
                failures.append({'station_id': sid, 'date': str(day), 'error': str(exc)[:300]})
            if i % 10 == 0 or i == len(tasks):
                print(f'Traffic files {i}/{len(tasks)}; successful {len(records)}; unavailable {len(failures)}', flush=True)
    if not frames:
        raise RuntimeError('No real traffic data could be downloaded. ' + str(failures[:2]))
    traffic = pd.concat(frames, ignore_index=True)
    traffic['timestamp'] = pd.to_datetime(traffic['timestamp'], utc=True)
    joined = []
    for detail in details:
        sid = int(detail['id'])
        subset = traffic[traffic['station_id'].eq(sid)].copy()
        if subset.empty:
            continue
        lon, lat = detail['geometry']['coordinates'][:2]
        weather_path = DATA / 'weather' / f'{sid}_{start}_{end}.json'
        try:
            payload = json.loads(weather_path.read_text(encoding='utf-8')) if weather_path.exists() else weather(lat, lon, start, end)
            write_json(weather_path, payload)
            subset = join_weather(subset, payload)
        except Exception as exc:
            # Training must never silently claim a weather experiment without weather.
            raise RuntimeError(f'Weather download failed for {sid}: {exc}') from exc
        subset['latitude'], subset['longitude'] = lat, lon
        subset['station_name'] = detail['properties']['name']
        joined.append(subset)
    frame = pd.concat(joined, ignore_index=True).drop_duplicates(['station_id', 'direction', 'timestamp'])
    frame = frame.sort_values(['timestamp', 'station_id', 'direction']).reset_index(drop=True)
    frame.to_csv(DATA / 'traffic_weather.csv.gz', index=False, compression='gzip')
    manifest = {'created_at': utc_now(), 'start_date': str(start), 'end_date': str(end), 'station_ids': list(stations), 'raw_rows': sum(r['raw_rows'] for r in records), 'valid_raw_rows': sum(r['valid_rows'] for r in records), 'aggregated_rows': len(frame), 'downloaded_station_days': len(records), 'requested_station_days': len(tasks), 'weather_missing_fraction': float(frame[WEATHER_FEATURES].isna().mean().mean()), 'failures': failures, 'files': records, 'weather_source': 'Open-Meteo Historical Forecast API; model estimates, not roadside observations', 'license': 'Traffic: Fintraffic Digitraffic CC BY 4.0. Weather: Open-Meteo CC BY 4.0; free API non-commercial use.'}
    write_json(DATA / 'manifest.json', manifest)
    print(f'Saved {len(frame):,} five-minute station-direction observations from {manifest["raw_rows"]:,} vehicle records.', flush=True)
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--stations', nargs='+', type=int, default=STATIONS)
    args = parser.parse_args()
    prepare(args.start, args.end, args.stations)
