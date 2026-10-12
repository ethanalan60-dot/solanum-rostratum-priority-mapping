"""Fix constants from training metadata only, matching original development definitions."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result={'version':'confirmation_training_constants_v1','test_metadata_read':False,'models_affected':False,'datasets':{}}
    for dataset,name,count in [('local','local_data_v2',985),('weeds','weedsgalore_data_v1',104)]:
        path=ROOT/name/'train.json';records=json.loads(path.read_text(encoding='utf-8-sig'))
        assert len(records)==count and all(x['split']=='train' for x in records)
        values=[np.float32(x['coverage']) if dataset=='weeds' else x['coverage'] for x in records]
        mean=float(np.mean(values,dtype=np.float64))
        result['datasets'][dataset]={'zero':0.0,'train_mean':mean,'n_train':count,'training_manifest_path':str(path),'training_manifest_sha256':sha(path),'conversion':'float32 coverage then float64 mean' if dataset=='weeds' else 'original JSON coverage then float64 mean'}
    with (HERE/'scalar_baselines.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
