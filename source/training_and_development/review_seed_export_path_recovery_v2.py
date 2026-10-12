"""Independent metadata/source/path recovery audit; no prepare, inference or solver."""
from pathlib import Path
import ast,datetime,hashlib,importlib,json,sys

ROOT=Path(__file__).resolve().parent
SEED=ROOT/'local_seed_stability_v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def nodes(path):
    s=Path(path).read_text(encoding='utf-8-sig')
    return {x.name:ast.get_source_segment(s,x) for x in ast.parse(s).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
def main():
    out=ROOT/'provenance'/('seed_export_v2_independent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    proto=SEED/'export_v2/protocol.json';p=read(proto)
    focus=SEED/'export_v2/focuschecks/20260916_095503_243675/focuschecks.json';f=read(focus)
    src=ROOT/'export_local_seed_validation_v2.py';before=sha(src)
    assert before==p['exporter_sha256']==f['exporter_sha256']
    assert sha(proto)==f['export_protocol_sha256'] and f['passed'] and f['check_count']==128
    for path,digest in f['source_start_end_sha256'].items():assert sha(path)==digest,path
    for path,digest in p['preserved_v1_evidence_sha256'].items():assert sha(path)==digest,path
    a=nodes(ROOT/'export_local_seed_validation.py');b=nodes(src)
    exact=['validate_training_authorization','validate_config','validate_history','validate_completion','validate_run','model_binding','save_probability_map','export_predictions']
    for key in exact:assert a[key]==b[key],key
    recovery=read(SEED/'postprocess_v2/recovery.json')
    for path,digest in recovery['preserved_files_sha256'].items():assert sha(path)==digest,path
    failed=Path(recovery['failed_export_directory'])
    assert recovery['original_failed_stage']=='prepare' and not(failed/'EXECUTION_CLAIM.json').exists()
    exporter=importlib.import_module('export_local_seed_validation_v2')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    plan=read(SEED/'plan.json');all_ids=[x['run_id'] for x in plan['runs']]
    assert p['declared_run_ids']==recovery['remaining_run_ids']==all_ids[2:]
    assert p['excluded_completed_run_ids']==recovery['preserved_completed_run_ids']==all_ids[:2]
    for rid in all_ids[:2]:
        try:exporter.declared_entry(plan,rid)
        except ValueError:pass
        else:raise AssertionError('Completed v1 model accepted')
    paths=[]
    for slot,rid in enumerate(all_ids[2:],3):
        parent=exporter.output_parent(rid)
        assert parent==SEED/'vx2'/f'slot{slot:02d}'
        assert exporter.claim_path(rid)==SEED/'export_v2/claims'/f'slot{slot:02d}.json'
        attempt=parent/'a20260916_235959_999999'
        for name in ('pre_execution_inputs.json.tmp','pre_execution_seal.json.tmp','EXECUTION_CLAIM.json','artifact_manifest.json.tmp','source/18.json','maps/4_8_44_29.npz'):
            path=attempt/name;paths.append({'path':str(path),'length':len(str(path))});assert len(str(path))<240
    assert exporter.source_key(proto)=='export_v2_protocol.json'
    assert exporter.source_key(SEED/'protocol.json')=='protocol.json'
    assert exporter.source_key(ROOT/'train_common.py')=='train_common.py'
    # Only archive naming/roster/claim scope changed; exception handler extent was
    # checked in the complete v1->v2 diff. Do not rerun mocked128 or real prepare.
    assert sha(src)==before
    report={'passed':True,'scope':'independent exporter v2 source/metadata/path review',
            'exporter_sha256':before,'export_protocol_sha256':sha(proto),'focuschecks_path':str(focus),'focuschecks_sha256':sha(focus),
            'reviewer_source_sha256':sha(__file__),'scientific_exact_AST':exact,
            'preserved_recovery_files_verified':len(recovery['preserved_files_sha256']),
            'remaining_run_ids':all_ids[2:],'excluded_completed_run_ids':all_ids[:2],
            'third_failure_before_execution_claim':True,'maximum_independent_path':max(paths,key=lambda x:x['length']),
            'logical_protocol_alias_checked':True,'prepare_and_execution_claim_dump_inside_failure_handler':'static full diff reviewed',
            'preparer128_checks_rerun':False,'real_prepare_calls':0,'model_forward_calls':0,'solver_calls':0,
            'cache_arrays_or_test_read':False,'R6_accessed':False,'production_authorization':'Root only; this is source readiness, not output validation'}
    (out/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'paths.json').write_text(json.dumps(paths,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'output':str(out),'sha256':sha(out/'review.json')}))
if __name__=='__main__':main()
