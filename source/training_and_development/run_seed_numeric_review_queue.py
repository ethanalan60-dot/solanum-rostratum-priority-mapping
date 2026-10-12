"""Consume the fixed remaining six saved development outputs with an independent auditor."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parent;HOME=ROOT/'seed_numeric_review_v1';SEED=ROOT/'local_seed_stability_v1'
BASE=ROOT/'provenance/seed_numerical_acceptance_index_20260916_103832_972904/index.json'
AUDITOR=ROOT/'audit_completed_seed_outputs_v2.py';BUILDER=ROOT/'prepare_seed_acceptance_index.py'
OWNER=ROOT/'run_local_seed_postprocess_queue_v2.py';OWNER_SHA='be53702860cd35418c832dd77f5ff7b736566589373ae868c4a523b318f9bfa7'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(p)
def now():return datetime.datetime.now().astimezone().isoformat()
def setup():
    plan=read(SEED/'plan.json')['runs'];base=read(BASE)
    assert sha(SEED/'plan.json')=='b96d83d3f5e68addbf5461758f66674b900c8777e8a9f02c50ebcff7735c1168'
    assert [x['run_id'] for x in base['new_models']]==[x['run_id'] for x in plan[:10]] and base['accepted_count']==10
    assert base['pending_run_ids']==[x['run_id'] for x in plan[10:]]
    assert sha(AUDITOR)=='a4e6367f0a4eaf66c24da13980c630f11b2b144b263b6af4aa074ed84153fb8c' and sha(OWNER)==OWNER_SHA
    watched={str(p):sha(p) for p in [Path(__file__),BASE,AUDITOR,BUILDER,OWNER,SEED/'plan.json']}
    for item in base['source_bindings']:assert sha(item['path'])==item['sha256'];watched[item['path']]=item['sha256']
    return plan,base,watched
def execute():
    plan,base,watched=setup();HOME.mkdir(exist_ok=True)
    with (HOME/'QUEUE_CLAIM.json').open('x',encoding='utf-8') as f:json.dump({'at':now(),'pid':os.getpid(),'source_sha256':sha(__file__),'slots':list(range(11,17))},f)
    state={'state':'starting','owner_pid':os.getpid(),'completed':[],'at':now(),'test_or_R6_accessed':False};child=None;helper=None
    def stable():
        for path,digest in watched.items():assert sha(path)==digest,path
    def run(command,index):
        nonlocal child
        stable();stdout=HOME/'logs'/f'{index:02d}.stdout.log';stderr=HOME/'logs'/f'{index:02d}.stderr.log'
        state.update(state='running',command=command,at=now());write(HOME/'queue_status.json',state)
        with stdout.open('x',encoding='utf-8') as so,stderr.open('x',encoding='utf-8') as se:
            child=subprocess.Popen(command,cwd=ROOT,stdout=so,stderr=se,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        child.owned_identity=helper.child_identity(child);state.update(child_pid=child.pid,child_identity=child.owned_identity);write(HOME/'queue_status.json',state)
        code=child.wait();assert code==0,('Audit failed',code,command);child=None;stable()
        payloads=[json.loads(line) for line in stdout.read_text(encoding='utf-8').splitlines() if line.startswith('{')]
        return payloads[-1]
    try:
        (HOME/'logs').mkdir();write(HOME/'source_bindings.json',watched);write(HOME/'queue_status.json',state)
        spec=importlib.util.spec_from_file_location('numeric_queue_owned_helpers',OWNER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        (ROOT/'runtime_temp').mkdir(exist_ok=True)
        os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',TEMP=str(ROOT/'runtime_temp'),TMP=str(ROOT/'runtime_temp'))
        reviews=list(dict.fromkeys(x['numerical_audit']['path'] for x in base['new_models']))
        for slot,row in enumerate(plan[10:],11):
            receipt=SEED/'postprocess_v2/run_records'/f'{slot:02d}.completed.json'
            state.update(state='waiting_for_fixed_postprocess',slot=slot,run_id=row['run_id'],child_pid=None,at=now());write(HOME/'queue_status.json',state)
            while not receipt.exists():
                stable()
                for path in (SEED/'queue_status.json',SEED/'postprocess_v2/queue_status.json'):
                    assert read(path)['state']!='failed',('Upstream failed; no audit retry',str(path))
                time.sleep(20)
            digest=sha(receipt);assert read(receipt)['run_id']==row['run_id'];watched[str(receipt)]=digest
            result=run([sys.executable,'-X','utf8','-B','-s',str(AUDITOR),'--receipt',str(receipt)],slot)
            assert result['passed'] is True;review=Path(result['output'])/'review.json';audit=read(review)
            assert audit['passed'] and len(audit['models'])==1 and audit['models'][0]['run_id']==row['run_id'] and audit['models'][0]['receipt_sha256']==digest
            reviews.append(str(review));watched[str(review)]=sha(review)
            state['completed'].append({'run_id':row['run_id'],'review':str(review),'review_sha256':sha(review)});write(HOME/'queue_status.json',state)
        command=[sys.executable,'-X','utf8','-B','-s',str(BUILDER),'--require-complete']
        for path in reviews:command+=['--audit',path]
        result=run(command,17);index=Path(result['out'])/'index.json';record=read(index)
        assert record['all16_complete'] and record['accepted_count']==16
        state.update(state='completed',child_pid=None,at=now(),full16_index={'path':str(index),'sha256':sha(index)})
        write(HOME/'final_bindings.json',watched);write(HOME/'queue_status.json',state)
    except BaseException as error:
        cleanup={}
        if child is not None:
            descendants=[]
            try:descendants=helper.owned_descendants(child)
            except BaseException as e:cleanup['inventory_error']=repr(e)
            try:
                if child.poll() is None:child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=20)
            except BaseException as e:cleanup['parent_error']=repr(e)
            try:cleanup['descendants']=helper.stop_recorded_descendants(descendants)
            except BaseException as e:cleanup['descendants_error']=repr(e)
        state.update(state='failed',error=repr(error),traceback=traceback.format_exc(),cleanup=cleanup,no_retry=True,at=now())
        write(HOME/'queue_status.json',state);raise
def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--validate-only',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.validate_only:
        plan,base,watched=setup();print(json.dumps({'passed':True,'already_accepted':10,'remaining':6,'no_model_or_solver':True,'watched':len(watched)}))
    else:execute()
if __name__=='__main__':main()
