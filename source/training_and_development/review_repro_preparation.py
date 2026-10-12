"""Independent bytes/inventory check of the internal non-final text evidence package."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent
HOME=ROOT/'provenance/repro_preparation_20260916_103344_267623'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    results=[]
    for name,digest,count in [('internal_bundle','4b4a5341438a9c79f9e8de2ca8a801efc0041ddd4121ff2441671c3507ece660',2714),('failure_supplement','06d0f358599b61827fc28e4d53ddf19c2f1c5cad300daed2f88207f5961e93f9',20)]:
        folder=HOME/name;manifest=folder/'package_manifest.json';assert sha(manifest)==digest
        payload=read(manifest);assert payload['status']=='INTERNAL_NONFINAL_PREPARATION' and payload['publication_authorized'] is False
        entries=payload['files'];assert len(entries)==count and len({x['logical_path'] for x in entries})==count and len({x['archive_path'] for x in entries})==count
        total=0
        for row in entries:
            path=(folder/row['archive_path']).resolve();assert path.is_relative_to(folder.resolve())
            assert path.suffix.lower() in ('.py','.md','.json','.jsonl','.csv','.txt','.toml','.yaml','.yml','.log','.bib','.html')
            path.read_text(encoding='utf-8-sig')
            assert row['public_release_approved'] is False and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
            assert Path(row['logical_path']).name.lower() not in ('test.json','test_sealed.json','作者致谢.docx')
            total+=row['bytes']
        seal=read(folder/'package_seal.json');assert seal['manifest_sha256']==digest and seal['release'] is False
        for filename,h in seal['documentation_sha256'].items():assert sha(folder/filename)==h
        results.append({'package':str(folder),'files':count,'bytes':total,'manifest_sha256':digest})
    out=ROOT/'provenance'/('repro_preparation_independent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    record={'passed':True,'scope':'Independent archive bytes, uniqueness, explicit internal status and text-only payload checks',
      'packages':results,'reviewer_sha256':sha(__file__),'source_preparation_review':{'path':str(HOME/'PREPARATION_REVIEW.json'),'sha256':sha(HOME/'PREPARATION_REVIEW.json')},
      'actual_study_execution':False,'dataset_or_weight_payload_read':False,'publication_authorized':False,
      'limitations':['Not a copyright clearance or public release review','This is the dated preparation snapshot, not final confirmation evidence','Scientific results depend on the separately named numerical reviews']}
    (out/'review.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'passed':True,'out':str(out),'files':sum(r['files'] for r in results)}))
if __name__=='__main__':main()
