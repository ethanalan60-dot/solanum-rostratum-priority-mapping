"""One authorized sequential attempt for the remaining five spatial models."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,datetime,hashlib,json,subprocess,sys,time,traceback
import psutil

ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture'
RUNS=[
 'dev_common_v2_lr1e4_convnextv2_tiny_soft_seg_L256_s17',
 'dev_common_v2_lr3e4_resnet18_hard_seg_L256_s17',
 'dev_common_v2_lr3e4_resnet18_soft_seg_L256_s17',
 'dev_common_v2_lr3e4_convnextv2_tiny_hard_seg_L256_s17',
 'dev_common_v2_lr3e4_convnextv2_tiny_soft_seg_L256_s17']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now().astimezone().isoformat()
def write(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def verify(plan,plan_path,plan_sha):
    assert sha(plan_path)==plan_sha
    assert plan['driver_sha256']==sha(__file__)
    assert [x['run_id'] for x in plan['runs']]==RUNS
    for path,digest in plan['pinned_files_sha256'].items():assert sha(path)==digest,path
def main(plan_path,plan_sha):
    plan_path=Path(plan_path).resolve();plan=read(plan_path);verify(plan,plan_path,plan_sha)
    assert Path(sys.executable).resolve()==Path(plan['python']).resolve()
    temp=ROOT/'runtime_temp'/'common_rm_v2';temp.mkdir(parents=True,exist_ok=True)
    for key in ('TEMP','TMP','TMPDIR'):os.environ[key]=str(temp)
    out=PA/('cmserial_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    claim=PA/'common_rm_v2_serial_claim.json'
    with claim.open('x',encoding='utf-8') as stream:
        json.dump({'output':str(out),'pid':os.getpid(),'create_time':psutil.Process().create_time(),
                   'plan':str(plan_path),'plan_sha256':plan_sha,'created_at':stamp()},stream,indent=2)
    out.mkdir(exist_ok=False);(out/'source').mkdir()
    for p in (Path(__file__),plan_path): (out/'source'/p.name).write_bytes(p.read_bytes())
    state={'state':'running','created_at':stamp(),'pid':os.getpid(),'create_time':psutil.Process().create_time(),
           'command_line':psutil.Process().cmdline(),'plan_sha256':plan_sha,'results':[],
           'test_accessed':False,'R6_accessed':False,'automatic_retries':0}
    write(out/'status.json',state);print(json.dumps({'output':str(out),'pid':os.getpid()}),flush=True)
    try:
        for index,record in enumerate(plan['runs'],1):
            verify(plan,plan_path,plan_sha)
            rid=record['run_id'];maps=Path(record['maps_directory']).resolve()
            assert maps.parent==ROOT/'runs'/rid
            # Maps are exported separately; wait only for the declared exact path.
            deadline=time.monotonic()+600
            while read(maps/'status.json').get('status')=='running':
                state.update(stage='waiting_for_completed_maps',current_run=rid,current_index=index)
                write(out/'status.json',state)
                if time.monotonic()>deadline:raise RuntimeError('Declared map export still incomplete after600s; no evaluation started')
                time.sleep(2)
            ms=read(maps/'status.json');meta=read(maps/'metadata.json')
            assert ms['status']=='completed' and ms['verification']['all_checks_passed']
            assert ms['verification']['all_npz_reopened']==357 and ms['verification']['checked_pairs']==1071
            assert meta['run_id']==rid and sha(maps/'metadata.json')==ms['metadata_sha256']
            if record['metadata_sha256_at_plan'] is not None:assert sha(maps/'metadata.json')==record['metadata_sha256_at_plan']
            attempt={'index':index,'run_id':rid,'maps_directory':str(maps),'metadata_sha256':sha(maps/'metadata.json'),
                     'checkpoint_epoch':meta['checkpoint_epoch'],'checkpoint_sha256':meta['checkpoint_sha256'],'started_at':stamp()}
            # Frozen worker performs full original validator and permanent per-run claim guard.
            command=[sys.executable,'-X','utf8','-B','-s',str(ROOT/'evaluate_common_management_v2.py'),
                     '--maps-dir',str(maps),'--protocol-sha256',plan['protocol_sha256']]
            stdout=out/f'{index:02d}_stdout.log';stderr=out/f'{index:02d}_stderr.log'
            attempt.update(command=command,stdout=str(stdout),stderr=str(stderr))
            with stdout.open('w',encoding='utf-8') as output,stderr.open('w',encoding='utf-8') as error:
                child=subprocess.Popen(command,stdout=output,stderr=error,cwd=str(ROOT),
                                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                attempt['pid']=child.pid
                try:attempt['process_create_time']=psutil.Process(child.pid).create_time()
                except psutil.NoSuchProcess:attempt['process_create_time']=None
                state.update(stage='evaluating',current_run=rid,current_index=index,current_attempt=attempt)
                write(out/'status.json',state);print(json.dumps({'started':attempt}),flush=True)
                attempt['returncode']=child.wait()
            attempt['finished_at']=stamp()
            worker_claim=PA/'common_rm_v2_claims'/(rid+'.json')
            if worker_claim.exists():
                result_dir=Path(read(worker_claim)['output']);attempt['output']=str(result_dir)
                if (result_dir/'status.json').exists():
                    ws=read(result_dir/'status.json');attempt['worker_state']=ws['state']
                    attempt['worker_status_sha256']=sha(result_dir/'status.json')
                    if ws['state']=='completed':attempt['policy_summary']=ws['policy_summary']
            attempt['state']='completed' if attempt['returncode']==0 and attempt.get('worker_state')=='completed' else 'failed'
            state['results'].append(attempt);state['current_attempt']=None
            write(out/'status.json',state);write(out/f'{index:02d}_result.json',attempt)
            print(json.dumps({'finished':attempt}),flush=True)
            # Failed attempts remain preserved. Continue to distinct next model; never retry.
        verify(plan,plan_path,plan_sha)
        state.update(state='completed' if all(x['state']=='completed' for x in state['results']) else 'completed_with_failures',
                     stage='complete',completed_at=stamp(),completed=sum(x['state']=='completed' for x in state['results']),
                     failed=sum(x['state']=='failed' for x in state['results']),current_run=None)
        write(out/'status.json',state);write(out/'summary.json',state)
    except BaseException as error:
        state.update(state='failed',stage='driver_failure',error=repr(error),traceback=traceback.format_exc(),failed_at=stamp())
        write(out/'status.json',state);raise
    print(json.dumps({'state':state['state'],'output':str(out),'completed':state['completed'],'failed':state['failed']}),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args();main(args.plan,args.plan_sha256)
