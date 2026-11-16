"""One real live integration check; not part of offline unit tests."""
import json
from traffic.api import write_json, utc_now
from traffic.config import DATA, ARTIFACTS
from traffic.live import fetch_snapshot
from traffic.model import predict_rows

details=json.loads((DATA/'stations.json').read_text(encoding='utf-8'))
row,payload,error=fetch_snapshot(details[0],1)
prediction=predict_rows(row,mode='weather')
result={'checked_at':utc_now(),'station_id':details[0]['id'],'weather_error':error,'inputs':row.to_dict('records'),'prediction':prediction.to_dict('records')}
write_json(ARTIFACTS/'live_verification.json',result)
print(json.dumps(result,default=str,indent=2))
