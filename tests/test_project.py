import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from traffic.data import aggregate_raw, join_weather
from traffic.features import supervised, chronological_split
from traffic.model import load_bundle, predict_rows
from traffic.config import DATA
from traffic.live import fetch_snapshot

class IntegrityTests(unittest.TestCase):
    def test_raw_filter_and_units(self):
        records=[[101,26,300,12,m,0,0,4,1,1,1,s,f,0,0,0] for m,s,f in [(1,60,0),(2,80,0),(3,100,1)]]
        raw=('\n'.join(';'.join(map(str,r)) for r in records)+'\n').encode()
        frame,quality=aggregate_raw(raw,23101)
        self.assertEqual(len(frame),1)
        self.assertEqual(frame.iloc[0].speed_kmh,70)
        self.assertEqual(frame.iloc[0].flow_vph,24)
        self.assertEqual(quality['discarded_rows'],1)

    def test_weather_cannot_use_future(self):
        t=pd.DataFrame({'timestamp':pd.to_datetime(['2026-01-01T10:30Z'])})
        payload={'hourly':{'time':['2026-01-01T10:00','2026-01-01T11:00'],'temperature_2m':[1,99]}}
        self.assertEqual(join_weather(t,payload).iloc[0].temperature_2m,1)

    def test_target_never_jumps_gaps(self):
        frame=pd.DataFrame({'timestamp':pd.to_datetime(['2026-01-01T10:00Z','2026-01-01T10:20Z']), 'station_id':[1,1],'direction':[1,1],'speed_kmh':[20,30],'flow_vph':[12,12]})
        self.assertTrue(supervised(frame).empty)

    def test_real_split_is_purged(self):
        data=pd.read_csv(DATA/'traffic_weather.csv.gz')
        pairs=supervised(data)
        train,val,test=chronological_split(pairs)
        self.assertLess(train.target_time.max(),val.timestamp.min())
        self.assertLess(val.target_time.max(),test.timestamp.min())
        self.assertTrue(((pairs.target_time-pairs.timestamp)==pd.Timedelta('15min')).all())

    def test_weather_failure_uses_traffic_model(self):
        row=pd.read_csv(DATA/'traffic_weather.csv.gz').tail(1).copy()
        row['temperature_2m']=np.nan
        result=predict_rows(row,load_bundle(),mode='weather').iloc[0]
        self.assertEqual(result.model_mode,'traffic')
        self.assertTrue(result.weather_fallback)

    def test_unknown_station_refused(self):
        row=pd.read_csv(DATA/'traffic_weather.csv.gz').tail(1).copy()
        row['station_id']=999999
        with self.assertRaises(ValueError):predict_rows(row,load_bundle())

    def test_stale_live_refused(self):
        sensors=[{'name':n,'value':v,'measuredTime':'2020-01-01T00:00:00Z','unit':'kpl/h'} for n,v in [('KESKINOPEUS_5MIN_LIUKUVA_SUUNTA1',60),('OHITUKSET_5MIN_LIUKUVA_SUUNTA1',100)]]
        with patch('traffic.live.live_traffic',return_value={'sensorValues':sensors}):
            with self.assertRaisesRegex(ValueError,'stale'):fetch_snapshot({'id':1},1)

if __name__=='__main__':unittest.main()
