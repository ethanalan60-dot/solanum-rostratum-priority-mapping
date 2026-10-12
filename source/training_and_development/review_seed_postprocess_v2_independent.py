"""Independent fixed14 recovery chain audit; no actual prepare, process or solver."""
from pathlib import Path
from unittest.mock import patch
import ast,datetime,hashlib,importlib,json,sys

ROOT=Path(__file__).resolve().parent
SEED=ROOT/'local_seed_stability_v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def functions(path):
    text=Path(path).read_text(encoding='utf-8-sig')
    return {x.name:ast.get_source_segment(text,x) for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)}
def main():
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=ROOT/'provenance'/('seed_postprocess_v2_independent_'+stamp);out.mkdir()
    d=importlib.import_module('run_local_seed_postprocess_queue_v2')
    e=importlib.import_module('export_local_seed_validation_v2')
    v=importlib.import_module('evaluate_local_seed_management_v2')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    p=read(d.PROTOCOL);proto_sha=sha(d.PROTOCOL);source_sha=sha(d.__file__)
    assert source_sha==p['implementation_sha256'] and p['schema']=='local_seed_postprocess_queue_v2'
    assert proto_sha=='027543f10bb1b7b0dfd2abde250c88e0f075c4c5bb66bfb0f640d566ebe51713'
    for path,digest in p['pinned_files_sha256'].items():assert sha(path)==digest,path
    recovery=read(p['recovery_manifest_path']);assert sha(p['recovery_manifest_path'])==p['recovery_manifest_sha256']
    for path,digest in recovery['preserved_files_sha256'].items():assert sha(path)==digest,path
    plan=read(SEED/'plan.json');ids=[x['run_id'] for x in plan['runs']]
    assert p['declared_run_ids']==recovery['remaining_run_ids']==ids[2:]
    assert recovery['preserved_completed_run_ids']==ids[:2]
    assert len(ids[2:])==len(set(ids[2:]))==p['max_runs']==14
    assert p['R']==[1,2,4] and p['M']==[1,2,4]
    assert p['test_or_R6_authorized'] is p['automatic_retry_authorized'] is False
    assert recovery['original_failed_stage']=='prepare' and not(Path(recovery['failed_export_directory'])/'EXECUTION_CLAIM.json').exists()
    spatial,watched=v.load_protocol(p['spatial_protocol_sha256'])
    assert spatial['declared_runs']==plan['runs'][2:] and spatial['preserved_completed_run_ids']==ids[:2]
    root_spatial=read(p['spatial_root_review_path'])
    assert root_spatial['passed'] and root_spatial['execution_scope']=='local_seed_management_v2_remaining14'
    assert root_spatial['protocol_sha256']==p['spatial_protocol_sha256'] and root_spatial['implementation_sha256']==sha(v.__file__)
    focus=read(p['export_focuschecks_path']);assert focus['passed'] and focus['check_count']==128
    assert focus['exporter_sha256']==sha(e.__file__) and focus['export_protocol_sha256']==sha(e.PROTOCOL)
    preparation_path=ROOT/'provenance/seed_postprocess_v2_readiness_20260916_095746_895890/review.json'
    preparation=read(preparation_path)
    assert preparation['passed'] and preparation['implementation_sha256']==source_sha and preparation['protocol_sha256']==proto_sha
    old_driver=functions(ROOT/'run_local_seed_postprocess_queue.py');new_driver=functions(d.__file__)
    exact_lifecycle=['child_identity','owned_descendants','stop_recorded_descendants','preserve_failure','wait_training']
    assert all(old_driver[x]==new_driver[x] for x in exact_lifecycle)
    old_spatial=functions(ROOT/'evaluate_local_seed_management.py');new_spatial=functions(v.__file__)
    exact_science=['bound_segments','load_bound_function','build_units','seal_bounds','verify_bound_seal','reference_equivalence','prefix_tables','evaluate_annotations']
    assert all(old_spatial[x]==new_spatial[x] for x in exact_science)
    path_check=read(ROOT/'provenance/seed_path_recovery_v2_checks_20260916_095410_231049/checks.json')
    assert path_check['passed'] and path_check['implementation_sha256']==sha(v.__file__) and path_check['driver_sha256']==source_sha
    assert path_check['maximum_prospective_path']['length']<240 and path_check['path_count']==5235
    # Independent short naming and protocol-alias checks for every remaining slot.
    short_paths=[]
    for slot,rid in enumerate(ids[2:],3):
        assert e.output_parent(rid).name==f'slot{slot:02d}'
        for name in (f'{slot:02d}.export_review.json',f'{slot:02d}.completed.json'):
            q=d.HERE/'run_records'/name;assert len(str(q))<240;short_paths.append(str(q))
    assert e.source_key(e.PROTOCOL)=='export_v2_protocol.json' and e.source_key(SEED/'protocol.json')=='protocol.json'
    # Synthetic parent-claim fault. validate() is mocked only here; no real
    # authorization, protocol mutation, training wait, prepare, Popen or solver.
    mock=Path('./temporary_checks/seed_recovery_independent')/stamp;mock.mkdir(parents=True)
    grant=mock/'synthetic_grant.json';write(grant,{'synthetic':True})
    try:
        with patch.object(d,'HERE',mock),patch.object(d,'validate',return_value=p),patch.object(d.json,'dump',side_effect=RuntimeError('synthetic claim serializer failure')):
            d.execute(grant)
    except RuntimeError:pass
    else:raise AssertionError('Claim fault not propagated')
    assert (mock/'QUEUE_CLAIM.json').exists() and read(mock/'queue_status.json')['state']=='failed'
    try:
        with patch.object(d,'HERE',mock),patch.object(d,'validate',return_value=p):d.execute(grant)
    except AssertionError:pass
    else:raise AssertionError('Failed claim reused')
    assert not(d.HERE/'QUEUE_CLAIM.json').exists(),'Independent review must precede launch'
    assert sha(d.__file__)==source_sha and sha(d.PROTOCOL)==proto_sha
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    report={'passed':True,'execution_scope':'local_seed_postprocess_queue_v2_remaining14',
            'implementation_sha256':source_sha,'protocol_sha256':proto_sha,
            'reviewer_source_sha256':sha(__file__),'review_kind':'Independent fixed-source/readiness review with a bounded synthetic claim-failure check',
            'pinned_files_verified':len(p['pinned_files_sha256']),'preserved_original_files_verified':len(recovery['preserved_files_sha256']),
            'spatial_protocol_sha256':p['spatial_protocol_sha256'],'spatial_source_sha256':sha(v.__file__),
            'export_source_sha256':sha(e.__file__),'export_protocol_sha256':sha(e.PROTOCOL),
            'export_focuschecks_sha256':sha(p['export_focuschecks_path']),
            'preparer_readiness_path':str(preparation_path),'preparer_readiness_sha256':sha(preparation_path),
            'readiness46_rerun':False,'independent_claim_failure_and_no_retry':True,
            'preserved_completed_run_ids':ids[:2],'declared_remaining_run_ids':ids[2:],
            'exact_scientific_AST':exact_science,'exact_lifecycle_AST':exact_lifecycle,
            'source_protocol_alias_review':'Only the new export protocol has explicit logical-key alias; all other basename identities retained',
            'path_readiness_count':5235,'path_readiness_max_chars':path_check['maximum_prospective_path']['length'],
            'old_third_attempt_not_started_inference':True,'real_prepare_calls':0,'real_Popen_calls':0,
            'model_forward_calls':0,'solver_calls':0,'test_or_R6_accessed':False,
            'limitations':['Production output acceptance is separate','No export tensor, geometry integration or solver outcome recomputed',
                           'Parent-owned process cleanup reviewed by identity and unchanged AST; no real process killed'],
            'mock_directory':str(mock)}
    write(out/'review.json',report)
    (out/'review.md').write_text('# Remaining14 path recovery: independent review\n\nPassed. The new driver/spatial/export chain preserves the first two completed runs and the third failed preparation, excludes their replacement, and retains the original scientific calculations. Only the originally declared slots03–16 can run under the new versioned claims and paths.\n\nAll1680 pins and1264 preserved hashes passed; the original8 spatial scientific functions and5 process lifecycle functions are unchanged. The exporter has its separate independent review. Claim-serialization failure and permanent no-retry were independently exercised in a short synthetic directory. No real prepare, subprocess, inference, geometry integration or solver was called.\n\nThis receipt supports the root execution decision. It does not certify future model outputs or authorize test/R6 work.\n',encoding='utf-8')
    print(json.dumps({'passed':True,'output':str(out),'sha256':sha(out/'review.json')}))
if __name__=='__main__':main()
