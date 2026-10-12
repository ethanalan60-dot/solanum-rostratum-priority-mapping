"""Bind reviewed fixed16 export/spatial implementation; no production execution."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent;SEED=ROOT/'local_seed_stability_v1';PA=ROOT/'precision_agriculture'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write_new(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
    readiness=ROOT/'provenance/local_seed_management_readiness_20260916_092248_537429/focused_checks.json'
    export_review=ROOT/'provenance/local_seed_export_independent_20260916_092142_029248/review.json'
    focus=SEED/'export_focuschecks/20260916_092027_457382/focuschecks.json'
    for path in (readiness,export_review,focus):assert read(path)['passed']
    assert sha(export_review)=='ffb20511df9d1f78942c07137814eef7d31c48d72f08d14675e251a9dd30f78f'
    spatial_protocol=PA/'RM_local_seed_protocol_v1.json';spatial_source=ROOT/'evaluate_local_seed_management.py'
    assert read(readiness)['implementation_sha256']==sha(spatial_source)
    assert read(readiness)['protocol_sha256']==sha(spatial_protocol)
    sp=read(spatial_protocol)
    for path,h in sp['pinned_files_sha256'].items():assert sha(path)==h
    spatial_review=SEED/'root_spatial_execution_review.json'
    write_new(spatial_review,{'passed':True,'execution_scope':'local_seed_management_v1_fixed16','reviewer':'root',
      'created_at':datetime.datetime.now().astimezone().isoformat(),'implementation_sha256':sha(spatial_source),
      'protocol_sha256':sha(spatial_protocol),'test_or_R6_authorized':False,'producer_readiness_path':str(readiness),
      'producer_readiness_sha256':sha(readiness),'export_independent_review_path':str(export_review),'export_independent_review_sha256':sha(export_review),
      'root_static_review':['Full spatial source read: fixed16 new seeds only, strict original checkpoints, validated source/maps and unchanged geometry.',
      'Selection child consumes only ID/A/P; raw45 choices and corrected bound seals precede annotation outcomes.',
      'Accepted bound function extracted via AST; original side-effectful audit module not imported.',
      'All prefix/fixed/AUC outcomes and paired policy values retained; shared original references require identical ID/A/T.',
      'Permanent per-run claims, failed initialization receipts and owned selector cleanup are preserved.'],
      'scientific_scope':'Development, not independent fields or confirmation. No test/R6, no original-run retries, no label-oracle solver rerun.',
      'new_output_acceptance_pending':True})
    folder=SEED/'postprocess_v1';folder.mkdir(exist_ok=False)
    files={str(Path(p).resolve()):h for p,h in sp['pinned_files_sha256'].items()}
    extras=[spatial_protocol,spatial_source,spatial_review,focus,export_review,readiness,
            ROOT/'export_local_seed_validation.py',SEED/'export_protocol.json',ROOT/'check_local_seed_export.py',ROOT/'check_local_seed_export_lifecycle.py',
            ROOT/'run_local_seed_postprocess_queue.py',SEED/'plan.json',SEED/'SOURCE_LOCK.json',SEED/'INPUT_LOCK.json',SEED/'protocol.json']
    for path in extras:files[str(path.resolve())]=sha(path)
    protocol={'schema':'local_seed_postprocess_queue_v1','created_at':datetime.datetime.now().astimezone().isoformat(),
      'implementation_sha256':sha(ROOT/'run_local_seed_postprocess_queue.py'),'declared_run_ids':[r['run_id'] for r in read(SEED/'plan.json')['runs']],
      'max_runs':16,'R':[1,2,4],'M':[1,2,4],'budgets':[.1,.2,.3,.5,1.],
      'test_or_R6_authorized':False,'automatic_retry_authorized':False,'wait_poll_seconds':20,'per_run_training_wait_limit_seconds':10800,
      'export_focuschecks_path':str(focus.resolve()),'spatial_root_review_path':str(spatial_review.resolve()),'spatial_protocol_sha256':sha(spatial_protocol),
      'pinned_files_sha256':files,'execution_order':'Fixed plan order: await immutable completed-training receipt; export original best once on CPU; evaluate all spatial cases once; stop on any error.',
      'per_run_export_review':'Mechanical metadata/seal binding under the separate root batch authorization; never represented as a fresh independent review.',
      'resources':'One CPU subprocess at a time, export/integration 2 threads and selector1 thread; may overlap the separately limited existing training queue.',
      'interpretation':'All results retained; additional random seeds describe within-scene training variation, not field replication.',
      'production_started':False}
    write_new(folder/'protocol.json',protocol)
    print(json.dumps({'spatial_review':str(spatial_review),'spatial_review_sha256':sha(spatial_review),'queue_protocol':str(folder/'protocol.json'),
                      'queue_protocol_sha256':sha(folder/'protocol.json'),'driver_sha256':sha(ROOT/'run_local_seed_postprocess_queue.py'),'bound_files':len(files)},ensure_ascii=False))
if __name__=='__main__':main()
