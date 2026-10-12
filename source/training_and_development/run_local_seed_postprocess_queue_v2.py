"""One fixed CPU export/spatial pass for the fourteen remaining already declared seed runs.

No training, source mutation, model selection, retries, test or R6 evaluation.
The root-reviewed protocol authorizes mechanically bound per-run export receipts.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parent
SEED=ROOT/'local_seed_stability_v1'
HERE=SEED/'postprocess_v2'
PROTOCOL=HERE/'protocol.json'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def verify(mapping):
    for p,h in mapping.items():assert sha(p)==h,('Bound file changed',p)

def child_identity(child):
    import psutil
    process=psutil.Process(child.pid)
    return {'pid':process.pid,'created':process.create_time()}

def owned_descendants(child):
    """Inspect only a still-live owned Popen tree; retain creation identities."""
    if child.poll() is not None:return []
    import psutil
    owner=psutil.Process(child.pid)
    assert owner.create_time()==child.owned_identity['created'],'Owned parent identity changed'
    return [{'pid':p.pid,'created':p.create_time(),'parent_pid':p.ppid()} for p in owner.children(recursive=True)]

def stop_recorded_descendants(records):
    import psutil
    result=[]
    for record in reversed(records):
        row=dict(record)
        try:
            process=psutil.Process(record['pid'])
            if process.create_time()!=record['created']:
                row['cleanup']='identity_changed_not_touched';result.append(row);continue
            process.terminate()
            try:process.wait(timeout=20)
            except psutil.TimeoutExpired:process.kill();process.wait(timeout=20)
            row['cleanup']='terminated_owned_descendant'
        except psutil.NoSuchProcess:row['cleanup']='already_exited'
        except BaseException as e:row['cleanup_error']=repr(e)
        result.append(row)
    return result

def validate(review_path):
    p=read(PROTOCOL);review=read(review_path)
    assert p['schema']=='local_seed_postprocess_queue_v2'
    assert p['implementation_sha256']==sha(__file__)
    assert p['R']==[1,2,4] and p['M']==[1,2,4] and p['max_runs']==14
    assert p['test_or_R6_authorized'] is False and p['automatic_retry_authorized'] is False
    assert p['wait_poll_seconds']==20 and p['per_run_training_wait_limit_seconds']==10800
    assert review['passed'] is True and review['execution_scope']=='local_seed_postprocess_queue_v2_remaining14'
    assert review['protocol_sha256']==sha(PROTOCOL) and review['implementation_sha256']==sha(__file__)
    assert review['test_or_R6_authorized'] is False
    verify(p['pinned_files_sha256'])
    recovery=read(p['recovery_manifest_path']);assert sha(p['recovery_manifest_path'])==p['recovery_manifest_sha256']
    assert recovery['remaining_run_ids']==p['declared_run_ids'] and recovery['original_failed_stage']=='prepare'
    verify(recovery['preserved_files_sha256'])
    assert not (Path(recovery['failed_export_directory'])/'EXECUTION_CLAIM.json').exists()
    plan=read(SEED/'plan.json');assert p['declared_run_ids']==[r['run_id'] for r in plan['runs'][2:]]
    assert len(p['declared_run_ids'])==len(set(p['declared_run_ids']))==14
    assert recovery['preserved_completed_run_ids']==[r['run_id'] for r in plan['runs'][:2]]
    required=[ROOT/'export_local_seed_validation_v2.py',ROOT/'evaluate_local_seed_management_v2.py',SEED/'export_v2/protocol.json',
              ROOT/'precision_agriculture/RM_local_seed_protocol_v2.json',Path(p['export_focuschecks_path']),Path(p['spatial_root_review_path'])]
    for path in required:assert p['pinned_files_sha256'].get(str(path.resolve()))==sha(path),('Missing required protocol binding',path)
    focus=read(p['export_focuschecks_path']);assert focus['passed'] is True
    assert focus['exporter_sha256']==sha(ROOT/'export_local_seed_validation_v2.py')
    assert focus['export_protocol_sha256']==sha(SEED/'export_v2/protocol.json')
    spatial=read(p['spatial_root_review_path']);assert spatial['passed'] is True
    assert spatial['execution_scope']=='local_seed_management_v2_remaining14'
    assert spatial['protocol_sha256']==p['spatial_protocol_sha256']
    assert spatial['implementation_sha256']==sha(ROOT/'evaluate_local_seed_management_v2.py')
    independent=read(review['independent_review_path']);assert independent['passed'] is True
    assert sha(review['independent_review_path'])==review['independent_review_sha256']
    assert independent['execution_scope']=='local_seed_postprocess_queue_v2_remaining14'
    assert independent['implementation_sha256']==sha(__file__) and independent['protocol_sha256']==sha(PROTOCOL)
    return p

def preserve_failure(state,error,child=None,command=None):
    state.update(state='failed',failed_at=now(),error=repr(error),traceback=traceback.format_exc(),no_retry=True,
                 child_pid=None if child is None else child.pid,child_command=command)
    if child is not None:
        cleanup={'owned_pid':child.pid}
        descendants=[]
        try:
            descendants=owned_descendants(child)
            cleanup['recorded_descendants']=descendants
        except BaseException as e:cleanup['descendant_inventory_error']=repr(e)
        try:
            if child.poll() is None:child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=20)
            cleanup['returncode']=child.returncode
        except BaseException as e:cleanup['error']=repr(e)
        cleanup['descendant_cleanup']=stop_recorded_descendants(descendants)
        state['owned_child_cleanup']=cleanup
    try:write(HERE/'queue_status.json',state)
    except BaseException as e:
        state['status_write_error']=repr(e)
        try:
            path=SEED/('postprocess_failure_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
            with path.open('x',encoding='utf-8') as f:json.dump(state,f,ensure_ascii=False,indent=2)
        except BaseException:print(json.dumps(state,ensure_ascii=False),file=sys.stderr,flush=True)

def wait_training(rid,p,state):
    receipt=SEED/'completion_records'/(rid+'.json');start=time.monotonic()
    last=None
    while not receipt.exists():
        q=read(SEED/'queue_status.json')
        assert q['state']=='running',('Training ended without required receipt',rid,q['state'])
        assert time.monotonic()-start<p['per_run_training_wait_limit_seconds'],'Bounded training wait expired'
        snapshot=(q.get('current_run'),len(q.get('completed',[])))
        if snapshot!=last:
            state.update(stage='waiting_for_training',waiting_for=rid,training_snapshot=q,updated_at=now());write(HERE/'queue_status.json',state);last=snapshot
        time.sleep(p['wait_poll_seconds'])
    return receipt

def export_review(output,rid,p,review_path):
    ep=read(SEED/'export_v2/protocol.json')
    return {'passed':True,'execution_scope':'local_seed_validation_export_v2','run_id':rid,
            'exporter_sha256':sha(ROOT/'export_local_seed_validation_v2.py'),'export_protocol_sha256':sha(SEED/'export_v2/protocol.json'),
            'source_lock_sha256':ep['training_binding']['source_lock_sha256'],'plan_sha256':ep['training_binding']['plan_sha256'],
            'pre_execution_seal_sha256':sha(output/'pre_execution_seal.json'),
            'focuschecks_path':p['export_focuschecks_path'],'focuschecks_sha256':sha(p['export_focuschecks_path']),
            'test_or_R6_authorized':False,'review_type':'Mechanically bound receipt under root-reviewed remaining14 recovery protocol, not a new independent review',
            'root_batch_review_path':str(Path(review_path).resolve()),'root_batch_review_sha256':sha(review_path),
            'postprocess_protocol_sha256':sha(PROTOCOL),'driver_sha256':sha(__file__),'created_at':now()}

def execute(review_path):
    p=validate(review_path)
    runtime_watch={**p['pinned_files_sha256'],str(PROTOCOL):sha(PROTOCOL),str(Path(review_path).resolve()):sha(review_path),str(Path(__file__).resolve()):sha(__file__)}
    assert not (HERE/'QUEUE_CLAIM.json').exists(),'Permanent postprocess queue claim already exists'
    state={'state':'running','stage':'initialization','owner_pid':os.getpid(),'started_at':now(),'completed':[],
           'protocol_sha256':sha(PROTOCOL),'root_review_sha256':sha(review_path),'test_accessed':False,'R6_accessed':False}
    child=None;command=None
    claim_stream=(HERE/'QUEUE_CLAIM.json').open('x',encoding='utf-8')
    try:
        with claim_stream as f:json.dump(state,f,ensure_ascii=False,indent=2)
        write(HERE/'queue_status.json',state)
        for name in ('logs','run_records','source'):(HERE/name).mkdir()
        for i,path in enumerate([Path(__file__),PROTOCOL,Path(review_path)]):shutil.copy2(path,HERE/'source'/(f'{i:02d}'+path.suffix))
        import export_local_seed_validation_v2 as exporter
        assert 'torch' not in sys.modules
        for slot,rid in enumerate(p['declared_run_ids'],3):
            state.update(current_run=rid,child_pid=None,stage='waiting_for_training',updated_at=now());write(HERE/'queue_status.json',state)
            wait_training(rid,p,state);verify(runtime_watch)
            # Eligibility and complete history checks happen before the permanent export claim.
            output=Path(exporter.prepare(rid)).resolve()
            per_review=HERE/'run_records'/(f'{slot:02d}.export_review.json');write(per_review,export_review(output,rid,p,review_path))
            exporter.check_execution_review(output,per_review)
            commands=[('cpu_export',[sys.executable,'-X','utf8','-B','-s',str(ROOT/'export_local_seed_validation_v2.py'),'--execute','--output',str(output),'--root-review',str(per_review)]),
                      ('spatial_evaluation',[sys.executable,'-X','utf8','-B','-s',str(ROOT/'evaluate_local_seed_management_v2.py'),'--maps-dir',str(output),'--protocol-sha256',p['spatial_protocol_sha256'],'--execute','--root-review',p['spatial_root_review_path']])]
            for stage,command in commands:
                verify(runtime_watch);state.update(stage=stage,command=command,updated_at=now());write(HERE/'queue_status.json',state)
                stdout=HERE/'logs'/(f'{slot:02d}.'+stage+'.stdout.log');stderr=HERE/'logs'/(f'{slot:02d}.'+stage+'.stderr.log')
                with stdout.open('x',encoding='utf-8') as so,stderr.open('x',encoding='utf-8') as se:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=so,stderr=se,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    child.owned_identity=child_identity(child)
                    state.update(child_pid=child.pid,child_identity=child.owned_identity,updated_at=now());write(HERE/'queue_status.json',state)
                    code=child.wait()
                assert code==0,('Owned subprocess failed; no automatic retry',rid,stage,code)
                child=None
                verify(runtime_watch)
                if stage=='cpu_export':
                    status=read(output/'status.json');assert status['status']=='completed' and status['verification']['all_checks_passed']
                    assert status['metadata_sha256']==sha(output/'metadata.json')
                    assert read(output/'metadata.json')['run_id']==rid
            claim=read(SEED/'management_claims_v2'/(rid+'.json'));spatial=Path(claim['output'])
            status=read(spatial/'status.json');assert status['state']=='completed' and status['run_id']==rid
            for relative,h in status['files_sha256'].items():assert sha(spatial/relative)==h
            assert sha(spatial/'summary.json')==status['summary_sha256'];summary=read(spatial/'summary.json')
            assert summary['run_id']==rid and summary['new_predicted_area_solver_calls']==45
            assert not summary['test_accessed'] and not summary['R6_accessed']
            verify(runtime_watch)
            record={'run_id':rid,'completed_at':now(),'export_directory':str(output),'spatial_directory':str(spatial),
                    'export_status_sha256':sha(output/'status.json'),'export_manifest_sha256':sha(output/'artifact_manifest.json'),
                    'spatial_status_sha256':sha(spatial/'status.json'),'spatial_summary_sha256':sha(spatial/'summary.json'),
                    'root_batch_review_sha256':sha(review_path),'protocol_sha256':sha(PROTOCOL),'original_epoch_preserved':True,
                    'all_declared_cases_retained':True,'test_or_R6_accessed':False}
            write(HERE/'run_records'/(f'{slot:02d}.completed.json'),record)
            state['completed'].append(rid);state.update(stage='run_completed',child_pid=None,updated_at=now());write(HERE/'queue_status.json',state)
        assert len(state['completed'])==14
        state.update(state='completed',stage='complete',completed_at=now(),current_run=None,child_pid=None);write(HERE/'queue_status.json',state)
    except BaseException as e:preserve_failure(state,e,child,command);raise

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root-review',type=Path,required=True)
    group=ap.add_mutually_exclusive_group(required=True);group.add_argument('--validate-only',action='store_true');group.add_argument('--execute',action='store_true');a=ap.parse_args()
    if a.validate_only:validate(a.root_review);print(json.dumps({'passed':True,'metadata_only':True,'runs':14,'new_forward_or_solver_calls':0}))
    else:execute(a.root_review)
if __name__=='__main__':main()
