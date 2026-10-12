"""Bind manifests to cache hashes before any training for a dataset version."""
from pathlib import Path
import argparse, hashlib,json,datetime

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);args=ap.parse_args();root=Path(args.data_dir).resolve()
    lockpath=root/'DATASET_LOCK.json'
    if lockpath.exists():raise FileExistsError('Dataset already sealed; create a new version for modifications.')
    manifests={};all_ids=set();count=0
    for name in ['train.json','val.json','test_sealed.json']:
        p=root/name
        if not p.exists():continue
        rows=json.loads(p.read_text(encoding='utf-8'))
        ids=[r['id'] for r in rows];assert len(ids)==len(set(ids));assert not (all_ids&set(ids));all_ids.update(ids)
        for r in rows:r['cache_sha256']=sha(r['cache'])
        p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        manifests[name]=sha(p);count+=len(rows)
    lock={'sealed_at':datetime.datetime.now().astimezone().isoformat(),'manifest_sha256':manifests,'cache_files':count,
          'note':'Test metadata is hashed only; no model predictions or scores are computed.'}
    lockpath.write_text(json.dumps(lock,indent=2),encoding='utf-8');print(json.dumps(lock))

if __name__=='__main__':main()
