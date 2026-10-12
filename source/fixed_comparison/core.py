"""Confirmation metadata gates and immutable one-attempt stages. Standard library only."""
from pathlib import Path
import contextlib,csv,datetime,hashlib,json,math,os,re,sys,traceback

HERE=Path(__file__).resolve().parent
R_VALUES=(1,2,4,6)

def now():return datetime.datetime.now().astimezone().isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def require(condition,message):
    if not condition:raise ValueError(message)
def safe_id(value):
    require(isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_-]+',value) is not None,'Unsafe identifier')
    return value
def bound_file(path,digest):
    require(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) is not None,'Missing/invalid SHA256')
    require(sha(path)==digest,'Hash mismatch: '+str(path));return Path(path)
def binding(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def resolve_under(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    require(path.is_relative_to(root),'Path escapes stage');return path

def validate_source_lock(home=HERE):
    home=Path(home);lock=read(home/'SOURCE_LOCK.json')
    require(lock['version']=='confirmation_source_v1','Wrong source lock')
    for name,digest in lock['files'].items():bound_file(resolve_under(home,name),digest)
    return lock

def validate_contract(contract):
    require(contract['version']=='confirmation_v1' and contract['R']==list(R_VALUES),'Wrong scope/R')
    require(contract['intermediate_sizes']==[256,128,64,42] and contract['output_size']==256,'Wrong support')
    require(contract['CPU_threads']==2 and contract['batch_size']==4 and contract['dtype']=='float32','Wrong resources')
    roster=contract['roster'];require(len(roster)==37 and len({x['run_id'] for x in roster})==37,'Wrong model population')
    for ds,n,seg in [('local',28,24),('weeds',9,6)]:
        rows=[x for x in roster if x['dataset']==ds]
        require(len(rows)==n and sum(x['variant']!='regression' for x in rows)==seg,'Wrong task counts')
    require(not contract['val_R6'] and not contract['norm_diagnostics_included'],'Out of scope')
    for row in roster:safe_id(row['run_id'])

def validate_checkpoint_entry(entry):
    """Only development evidence; no checkpoint tensor loading and no test metadata."""
    for prefix in ('checkpoint','config','status','history','best_validation','completion_receipt','training_audit'):
        bound_file(entry[prefix+'_path'],entry[prefix+'_sha256'])
    config=read(entry['config_path']);status=read(entry['status_path']);best=read(entry['best_validation_path'])
    history=[json.loads(line) for line in Path(entry['history_path']).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    require(status['status']=='completed' and status['executed_epochs']==24,'Incomplete fit')
    require([x['epoch'] for x in history]==list(range(1,25)),'Incomplete/duplicated history')
    for key in ('run_id','encoder','variant','seed','lr'):require(config[key]==entry[key],'Config identity: '+key)
    require(config['epochs']==24,'Wrong epoch budget')
    key='selection_balanced_mae' if entry['dataset']=='local' else 'selection_mae'
    require(all(math.isfinite(x[key]) for x in history),'Nonfinite development selection')
    earliest=min(history,key=lambda x:x[key])
    require(earliest==best and earliest['epoch']==entry['best_epoch']==status['best_epoch'],'Original earliest best mismatch')
    require(not status.get('test_accessed',False),'Original fit used test')
    # Root's frozen checkpoint table explicitly binds the independent audit and completion evidence.
    require(entry.get('original_selection_audited') is True,'Missing original-selection attestation')
    return entry

def validate_freeze(home=HERE,check_models=True):
    """First gate for every real-data stage. Missing freeze fails before dataset files are opened."""
    home=Path(home);freeze=read(home/'FREEZE.json')
    require(freeze.get('passed') is True and freeze.get('scope')=='confirmation_v1' and freeze.get('frozen') is True,'Root freeze required')
    bound_file(home/'SOURCE_LOCK.json',freeze['source_lock_sha256']);validate_source_lock(home)
    bound_file(home/'contract.json',freeze['contract_sha256']);bound_file(home/'checkpoints.json',freeze['checkpoints_sha256'])
    contract=read(home/'contract.json');validate_contract(contract);models=read(home/'checkpoints.json')
    require(models.get('frozen') is True,'Checkpoint table not frozen')
    require(len(models['models'])==37,'Checkpoint table incomplete')
    require([{k:x[k] for k in c} for x,c in zip(models['models'],contract['roster'])]==contract['roster'],'Checkpoint roster/order changed')
    require(freeze['R']==list(R_VALUES) and freeze['run_ids']==[x['run_id'] for x in models['models']],'Freeze scope differs')
    require(freeze.get('truth_after_all_predictions_and_selections') is True,'Truth sequencing not frozen')
    for item in contract['development_plans']:bound_file(item['path'],item['sha256'])
    if check_models:
        for row in models['models']:validate_checkpoint_entry(row)
    return contract,models['models'],freeze

def stage_manifest(folder):
    folder=Path(folder)
    return {'created_at':now(),'files':{p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}}

def completed_stage(folder):
    folder=Path(folder);manifest=read(folder/'artifact_manifest.json')
    require(read(folder/'status.json')['state']=='completed','Stage not completed')
    for name,digest in manifest['files'].items():bound_file(resolve_under(folder,name),digest)
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    require(actual==set(manifest['files']),'Stage has unmanifested files')
    return manifest

def preserve_failure(home,out,name,error,secondary=None):
    record={'state':'failed','stage':name,'error':repr(error),'traceback':traceback.format_exc(),'secondary_error':secondary,'at':now(),'no_retry':True}
    try:
        out.mkdir(parents=True,exist_ok=True);write(out/'failure.json',record);write(out/'status.json',record)
    except BaseException as failure:
        record['recording_error']=repr(failure)
        try:
            fallback=home/'failure_receipts';fallback.mkdir(exist_ok=True)
            write(fallback/(name+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'),record)
        except BaseException as last:print(json.dumps({**record,'fallback_error':repr(last)}),file=sys.stderr,flush=True)

@contextlib.contextmanager
def attempt(name,home=HERE):
    """Claim persists on any failure; existing stage/claim never overwritten or resumed."""
    safe_id(name);home=Path(home);claims=home/'claims';claims.mkdir(exist_ok=True)
    out=home/'stages'/name
    require(not out.exists(),'Existing stage is immutable')
    claim_stream=(claims/(name+'.json')).open('x',encoding='utf-8')
    primary=None
    try:
        with claim_stream as stream:
            json.dump({'stage':name,'at':now(),'pid':os.getpid(),'freeze_sha256':sha(home/'FREEZE.json')},stream)
        out.mkdir(parents=True,exist_ok=False)
        write(out/'status.json',{'state':'running','stage':name,'at':now()})
        write(out/'input_seal.json',{'freeze':binding(home/'FREEZE.json'),'source_lock':binding(home/'SOURCE_LOCK.json'),'contract':binding(home/'contract.json'),'checkpoints':binding(home/'checkpoints.json')})
        yield out
        write(out/'status.json',{'state':'completed','stage':name,'at':now()})
    except BaseException as error:
        primary=error;preserve_failure(home,out,name,error);raise
    finally:
        try:write(out/'artifact_manifest.json',stage_manifest(out))
        except BaseException as secondary:
            preserve_failure(home,out,name,primary or secondary,repr(secondary))
            if primary is None:raise

def configure_cpu(out):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
    temp=Path(out)/'temp';temp.mkdir(exist_ok=True)
    for key in ('TMP','TEMP'):os.environ[key]=str(temp)

def table(path,rows):
    require(bool(rows),'Empty table')
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
