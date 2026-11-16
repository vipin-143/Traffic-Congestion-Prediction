"""Honest temporal evaluation, validation-only selection and trusted local artifacts."""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .api import write_json, utc_now
from .config import ARTIFACTS, DATA, BASE_FEATURES, WEATHER_FEATURES, HORIZON
from .features import supervised, chronological_split, add_features, congestion_label


def metrics(actual, prediction):
    return {'MAE_kmh': float(mean_absolute_error(actual, prediction)), 'RMSE_kmh': float(np.sqrt(mean_squared_error(actual, prediction))), 'R2': float(r2_score(actual, prediction))}


def make_model(algorithm, features):
    categorical = ['station_id', 'direction']
    numeric = [c for c in features if c not in categorical]
    pre = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numeric),
        ('station', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ])
    if algorithm == 'Random forest':
        estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=8, max_depth=18, max_features=.9, n_jobs=-1, random_state=42)
    else:
        estimator = HistGradientBoostingRegressor(max_iter=150, learning_rate=.07, max_leaf_nodes=25, l2_regularization=2, early_stopping=False, random_state=42)
    return Pipeline([('prepare', pre), ('regressor', estimator)])


def train_all():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA / 'traffic_weather.csv.gz')
    df = supervised(data)
    train, val, test = chronological_split(df)
    candidates, validation, fitted = {}, [], {}
    for mode, features in [('traffic', BASE_FEATURES), ('weather', BASE_FEATURES + WEATHER_FEATURES)]:
        if mode == 'weather' and train[WEATHER_FEATURES].notna().sum().eq(0).any():
            raise ValueError('A weather feature has no valid training values')
        validation.append({'mode': mode, 'algorithm': 'Persistence', **metrics(val['target_speed_kmh'], val['speed_kmh'])})
        for algorithm in ['Random forest', 'Gradient boosting']:
            print(f'Training {mode}: {algorithm}; {len(train):,} rows', flush=True)
            model = make_model(algorithm, features)
            model.fit(train[features], train['target_speed_kmh'])
            score = metrics(val['target_speed_kmh'], model.predict(val[features]))
            validation.append({'mode': mode, 'algorithm': algorithm, **score})
            fitted[(mode, algorithm)] = model
        scores = [r for r in validation if r['mode'] == mode]
        winner = min(scores, key=lambda r: r['MAE_kmh'])
        candidates[mode] = winner['algorithm']
    # Final fitting only after selection. The test period is untouched until now.
    fitting = pd.concat([train, val], ignore_index=True)
    selected, test_scores, predictions = {}, [], test[['timestamp', 'target_time', 'station_id', 'direction', 'speed_kmh', 'target_speed_kmh']].copy()
    for mode, features in [('traffic', BASE_FEATURES), ('weather', BASE_FEATURES + WEATHER_FEATURES)]:
        algorithm = candidates[mode]
        final_model = None
        if algorithm != 'Persistence':
            final_model = make_model(algorithm, features)
            final_model.fit(fitting[features], fitting['target_speed_kmh'])
            pred = final_model.predict(test[features])
        else:
            pred = test['speed_kmh'].to_numpy()
        selected[mode] = {'algorithm': algorithm, 'features': features, 'model': final_model}
        test_scores.append({'mode': mode, 'algorithm': algorithm, **metrics(test['target_speed_kmh'], pred)})
        predictions[f'predicted_{mode}'] = pred
    baseline = metrics(test['target_speed_kmh'], test['speed_kmh'])
    test_scores.append({'mode': 'baseline', 'algorithm': 'Persistence', **baseline})
    default_mode = min(['traffic', 'weather'], key=lambda mode: min(r['MAE_kmh'] for r in validation if r['mode'] == mode))
    # Reference speeds come from training only, without test-period information.
    refs = train.groupby(['station_id', 'direction'])['speed_kmh'].quantile(.95)
    references = {f'{int(s)}:{int(d)}': float(v) for (s, d), v in refs.items()}
    bundle = {'schema_version': 1, 'created_at': utc_now(), 'horizon_minutes': HORIZON, 'default_mode': default_mode, 'models': selected, 'reference_speeds': references, 'stations': sorted(df['station_id'].astype(int).unique().tolist()), 'directions': [1, 2]}
    joblib.dump(bundle, ARTIFACTS / 'models.joblib', compress=3)
    predictions.to_csv(ARTIFACTS / 'test_predictions.csv.gz', index=False, compression='gzip')
    pd.DataFrame(validation).to_csv(ARTIFACTS / 'validation_metrics.csv', index=False)
    pd.DataFrame(test_scores).to_csv(ARTIFACTS / 'test_metrics.csv', index=False)
    split = {name: {'rows': len(part), 'start_utc': str(part['timestamp'].min()), 'end_utc': str(part['timestamp'].max()), 'last_target_utc': str(part['target_time'].max())} for name, part in [('train', train), ('validation', val), ('test', test)]}
    summary = {'created_at': utc_now(), 'paired_rows': len(df), 'horizon_minutes': HORIZON, 'default_mode': default_mode, 'selected_algorithms': candidates, 'splits': split, 'validation_metrics': validation, 'test_metrics': test_scores, 'weather_missing_fraction': float(df[WEATHER_FEATURES].isna().mean().mean()), 'limitations': [
        'Retrospective evaluation on the same stations; not evidence of generalization to unseen cities or sensors.',
        'Archived weather forecasts are a stitched retrospective series, not an issuance-time backtest. Weather benefit is exploratory until tested prospectively.',
        'Training windows are fixed five-minute vehicle aggregates. Live inputs are the provider rolling five-minute speed and flow estimates; this small distribution mismatch needs monitoring.',
        'Raw repeated local times at the autumn daylight-saving transition are discarded because offsets cannot be recovered unambiguously.',
        'Congestion labels use project-defined speed-ratio thresholds, not human-labelled official congestion classes.',
        'The prediction horizon is measured from the observation timestamp, not from the time a user opens the dashboard.',
    ]}
    write_json(ARTIFACTS / 'summary.json', summary)
    # Explanatory importances use validation data and a model fitted on training only.
    explaining = fitted.get(('weather', candidates['weather']))
    if explaining is not None:
        sample = val.sample(min(1200, len(val)), random_state=42)
        importance = permutation_importance(explaining, sample[BASE_FEATURES + WEATHER_FEATURES], sample['target_speed_kmh'], n_repeats=3, random_state=42, scoring='neg_mean_absolute_error', n_jobs=1)
        pd.DataFrame({'feature': BASE_FEATURES + WEATHER_FEATURES, 'mae_increase': importance.importances_mean}).sort_values('mae_increase', ascending=False).to_csv(ARTIFACTS / 'feature_importance.csv', index=False)
    print(json.dumps({'selected': candidates, 'test': test_scores, 'splits': split}, indent=2), flush=True)
    return summary


def load_bundle():
    # Only load this project's trusted locally produced artifact. Never accept user pickles.
    bundle = joblib.load(ARTIFACTS / 'models.joblib')
    if bundle.get('schema_version') != 1:
        raise ValueError('Unsupported model schema; retrain with the current code')
    return bundle


def predict_rows(frame, bundle=None, mode='auto'):
    bundle = bundle or load_bundle()
    if mode not in ['auto', 'traffic', 'weather']:
        raise ValueError('Unknown model mode')
    data = add_features(frame)
    required = ['speed_kmh', 'flow_vph', 'station_id', 'direction']
    if data[required].isna().any().any() or not data['speed_kmh'].between(0, 199).all() or not data['flow_vph'].ge(0).all():
        raise ValueError('Invalid speed, flow, station or direction')
    if not data['direction'].isin([1, 2]).all() or not data['station_id'].isin(bundle['stations']).all():
        raise ValueError('Predictions are supported only for trained stations and directions')
    requested = bundle['default_mode'] if mode == 'auto' else mode
    used = requested
    if requested == 'weather' and data[WEATHER_FEATURES].isna().any().any():
        used = 'traffic'
    info = bundle['models'][used]
    pred = data['speed_kmh'].to_numpy() if info['model'] is None else info['model'].predict(data[info['features']])
    result = data[['timestamp', 'station_id', 'direction', 'speed_kmh', 'flow_vph']].copy()
    result['prediction_at'] = result['timestamp'] + pd.Timedelta(minutes=bundle['horizon_minutes'])
    result['predicted_speed_kmh'] = pred
    result['model_mode'] = used
    result['algorithm'] = info['algorithm']
    result['weather_fallback'] = used != requested
    result['reference_speed_kmh'] = [bundle['reference_speeds'].get(f'{int(s)}:{int(d)}', np.nan) for s, d in zip(result['station_id'], result['direction'])]
    result['congestion'] = [congestion_label(s, r) for s, r in zip(pred, result['reference_speed_kmh'])]
    return result


if __name__ == '__main__':
    train_all()
