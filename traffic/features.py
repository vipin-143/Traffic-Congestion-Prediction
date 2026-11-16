"""The same causal snapshot features are used by training and inference."""
import numpy as np
import pandas as pd

from .config import BASE_FEATURES, WEATHER_FEATURES, HORIZON


def add_features(frame):
    df = frame.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    local = df['timestamp'].dt.tz_convert('Europe/Helsinki')
    hour = local.dt.hour + local.dt.minute / 60
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['weekday'] = local.dt.day_of_week
    df['month'] = local.dt.month
    df['weekend'] = local.dt.day_of_week.ge(5).astype(int)
    for col in BASE_FEATURES + WEATHER_FEATURES:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return df


def supervised(frame):
    df = add_features(frame)
    df = df[df['speed_kmh'].between(2, 199) & df['flow_vph'].ge(0)].copy()
    df = df.drop_duplicates(['station_id', 'direction', 'timestamp'])
    future = df[['station_id', 'direction', 'timestamp', 'speed_kmh']].rename(columns={'timestamp': 'target_time', 'speed_kmh': 'target_speed_kmh'})
    df['target_time'] = df['timestamp'] + pd.Timedelta(minutes=HORIZON)
    paired = df.merge(future, on=['station_id', 'direction', 'target_time'], how='inner', validate='one_to_one')
    return paired.sort_values(['timestamp', 'station_id', 'direction']).reset_index(drop=True)


def chronological_split(df):
    times = df['timestamp'].drop_duplicates().sort_values().reset_index(drop=True)
    if len(times) < 300:
        raise ValueError('Need at least 300 observation times for a meaningful split')
    val_start, test_start = times.iloc[int(len(times) * .70)], times.iloc[int(len(times) * .85)]
    train = df[(df['timestamp'] < val_start) & (df['target_time'] < val_start)].copy()
    val = df[(df['timestamp'] >= val_start) & (df['timestamp'] < test_start) & (df['target_time'] < test_start)].copy()
    test = df[df['timestamp'] >= test_start].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError('Chronological split produced an empty partition')
    return train, val, test


def congestion_label(speed, reference):
    """Project-defined display thresholds, not official labelled ground truth."""
    if not np.isfinite(speed) or not np.isfinite(reference) or reference <= 0:
        return 'Unavailable'
    ratio = speed / reference
    return 'Heavy' if ratio < .70 else 'Moderate' if ratio < .90 else 'Low'
