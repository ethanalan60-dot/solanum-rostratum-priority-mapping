"""Separate, one-attempt CPU validation exporter for the fixed16 seed repeats.

validate_run(..., load_model=False) returns the legacy eight-tuple without
constructing a model or decoding any cache. No training/test/R6 entry exists.
The legacy exporter and all original run files remain read-only.
"""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, math, os, shutil, sys, time, traceback
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
ROOT = Path(__file__).resolve().parent
SEED = ROOT/'local_seed_stability_v1'
OUTPUT = SEED/'validation_exports'
PROTOCOL = SEED/'export_protocol.json'
FACTORS = (1, 2, 4)
SOURCE_NAMES = ('train_common.py','models_common.py','conservative_resample.py')
PINS = {
    'SOURCE_LOCK.json': '5707ccdfabca5b79238ee95141346e808c6d97b6cbc4b502db4f6ffe568ad797',
    'plan.json': 'b96d83d3f5e68addbf5461758f66674b900c8777e8a9f02c50ebcff7735c1168',
    'protocol.json': '631a9ff843997746c93c050e78062674910478bf49aa36c0a1e8f295329218e1',
    'INPUT_LOCK.json': 'b9db865aa602f5fe05f386b8115200c27bac12555d3c3f3f1208d6df3e61f11e',
}
LEGACY_SHA = '56319f148fadc69b0cdc88c3c44f043ba020e812f2f4f919480304fe155dbecc'
ORIGINAL_PLAN_SHA = 'faefb03b14e4c44360ded4d065ffbe873d4e8512139a68d56e15ee90a2185a95'
SETTINGS = {'epochs':24, 'batch_size':4, 'accumulation_steps':2, 'weight_decay':1e-4,
    'label_grid':256, 'device':'cuda', 'threads':2, 'gpu_memory_fraction':.16,
    'step_pause':.05, 'scale_augmentation':True, 'amp':True, 'sampler':'balanced', 'freeze_batchnorm':False}

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path, value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)
def ensure(condition, message):
    if not condition: raise ValueError(message)
def close(a, b):
    ensure(math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=1e-12,'Numeric provenance mismatch')
def dependencies():
    """Only strict model loading/execution imports scientific dependencies."""
    global np,torch,DataLoader,legacy,training
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import export_common_validation as legacy
    training=legacy.training
    torch.set_num_threads(2)
def confined(base, relative):
    parts = str(relative).replace('\\','/').split('/')
    ensure(all(p not in ('','.','..') and ':' not in p for p in parts), 'Unsafe relative path')
    result = Path(base).joinpath(*parts).resolve()
    ensure(result.is_relative_to(Path(base).resolve()), 'Path escapes owner')
    return result
def recheck(hashes):
    for path, digest in hashes.items(): ensure(sha(path)==digest, 'Changed bound input: '+str(path))
def watched(path, hashes, expected=None):
    path=Path(path).resolve();digest=sha(path)
    ensure(expected is None or digest==expected, 'Hash mismatch: '+str(path))
    hashes[str(path)]=digest
    return digest
def binding():
    return {'source_lock_sha256':PINS['SOURCE_LOCK.json'], 'plan_sha256':PINS['plan.json'],
            'protocol_sha256':PINS['protocol.json'], 'input_lock_sha256':PINS['INPUT_LOCK.json']}

def validate_static_sources():
    """Source/plan only. Does not inspect runs, checkpoints or pixel arrays."""
    hashes={}
    for name,digest in PINS.items(): watched(SEED/name,hashes,digest)
    lock=read(SEED/'SOURCE_LOCK.json')
    for relative,digest in lock['source_sha256'].items(): watched(confined(SEED,relative),hashes,digest)
    watched(ROOT/'export_common_validation.py',hashes,LEGACY_SHA)
    watched(ROOT/'common_queue_plan_v2.json',hashes,ORIGINAL_PLAN_SHA)
    plan=read(SEED/'plan.json');old=read(ROOT/'common_queue_plan_v2.json')
    ensure(plan['version']=='local_seed_stability_v1' and plan['settings']==SETTINGS, 'Seed plan/settings mismatch')
    ensure(plan['protocol_sha256']==PINS['protocol.json'] and plan['original_common_plan_sha256']==ORIGINAL_PLAN_SHA, 'Seed plan links mismatch')
    runs=plan['runs'];ensure(len(runs)==16 and len({r['run_id'] for r in runs})==16, 'Fixed16 population required')
    expected={(e,l,v,s) for e in ('resnet18','convnextv2_tiny') for l in (1e-4,3e-4) for v in ('hard_seg','soft_seg') for s in (29,43)}
    ensure({(r['encoder'],r['lr'],r['variant'],r['seed']) for r in runs}==expected, 'Fixed16 matrix mismatch')
    for name in SOURCE_NAMES:
        digest=old['source_sha256'][name]
        watched(ROOT/name,hashes,digest);watched(SEED/'original_source'/name,hashes,digest)
    watched(PROTOCOL,hashes)
    protocol=read(PROTOCOL)
    ensure(protocol['version']=='local_seed_validation_export_v1' and protocol['declared_run_ids']==[r['run_id'] for r in runs], 'Export roster mismatch')
    ensure(protocol['training_binding']==binding() and protocol['exporter_sha256']==sha(__file__), 'Export protocol/source mismatch')
    ensure(protocol['legacy_exporter_sha256']==LEGACY_SHA and protocol['R']==list(FACTORS), 'Export dependency/scales mismatch')
    watched(ROOT/'check_local_seed_export.py',hashes,protocol['focuschecker_sha256'])
    watched(ROOT/'check_local_seed_export_lifecycle.py',hashes,protocol['lifecycle_checker_sha256'])
    watched(__file__,hashes)
    return plan,old,hashes

def declared_entry(plan, run_id):
    entries=[x for x in plan['runs'] if x['run_id']==run_id]
    ensure(len(entries)==1, 'Only the declared16 seed29/43 segmentation runs are accepted; seed17 is never re-exported')
    return entries[0]

def validate_training_authorization(hashes):
    claim_path=SEED/'QUEUE_CLAIM.json';watched(claim_path,hashes);claim=read(claim_path)
    ensure(claim['binding']==binding(),'Training queue claim binding mismatch')
    review_path=Path(claim['review_path']).resolve()
    ensure(review_path.is_relative_to(ROOT),'Training authorization outside project')
    watched(review_path,hashes,claim['review_sha256']);review=read(review_path)
    ensure(review.get('passed') is True and review.get('execution_scope')=='local_seed_stability_v1_fixed16','Training authorization scope mismatch')
    ensure(review.get('test_or_R6_authorized') is False,'Training authorization includes reserved data')
    for key,digest in binding().items():ensure(review.get(key)==digest,'Training authorization binding mismatch: '+key)
    ensure(review['runner_sha256']==read(SEED/'SOURCE_LOCK.json')['source_sha256']['run_seed_queue.py'],'Training runner authorization mismatch')
    if 'independent_review' in review:
        independent=Path(review['independent_review']).resolve()
        ensure(independent.is_relative_to(ROOT),'Training independent review outside project')
        watched(independent,hashes,review['independent_review_sha256'])
        ensure(read(independent).get('passed') is True,'Training independent review not passed')
    return claim_path,review_path

def validate_config(entry, config, state):
    ensure(config.get('run_id')==entry['run_id'] and config.get('stage')=='development', 'Run identity/stage mismatch')
    ensure(state.get('status')=='completed' and state.get('stage')=='development', 'Completed development status required')
    expected={**SETTINGS, **{k:entry[k] for k in ('run_id','encoder','variant','lr','seed')}}
    expected['device']='cuda:0'
    for key,value in expected.items(): ensure(config.get(key)==value, 'Config differs from plan: '+key)
    for key in ('smoke','diagnostic','diagnostic_epochs','max_train','max_val','no_pretrained'):
        ensure(not config.get(key), 'Restricted/subset run: '+key)
    ensure(config.get('test_accessed') is False and state.get('test_accessed') is False, 'Unexpected test provenance')
    ensure(not config.get('R6_accessed',False) and not state.get('R6_accessed',False), 'Unexpected R6 provenance')
    ensure(config.get('decision_metrics_used_for_selection') is False, 'Decision-based checkpoint selection rejected')
    ensure(config.get('n_train')==985 and config.get('n_val')==357 and state.get('executed_epochs')==24, 'Full population/24 epochs required')
    ensure(config.get('fixed_output_grid')==config.get('label_grid')==256 and config.get('validation_R')==list(FACTORS), 'Output grid/scales mismatch')
    ensure(type(config.get('torch')) is str, 'Original trainer must record TorchVersion as a plain string')
    ensure(state.get('optimizer_attempts_total')==2976, 'Full update budget required')
    updates=state.get('optimizer_updates_total');skips=state.get('optimizer_skips_total')
    ensure(type(updates) is int and type(skips) is int and updates>0 and skips>=0 and updates+skips==2976, 'Update/skip accounting mismatch')

def validate_history(history, best, state):
    ensure([x['epoch'] for x in history]==list(range(1,25)), 'Incomplete/repeated epoch history')
    for metric in history:
        ensure(set(metric['scales'])=={'1','2','4'}, 'Unexpected history scales')
        for result in metric['scales'].values():
            ensure(result['n']==357 and result['n_positive']==27, 'Incomplete validation history')
            for key in ('mae','bias','mae_zero','mae_positive','balanced_mae'):
                ensure(math.isfinite(result[key]), 'Nonfinite history metric')
            close(result['balanced_mae'],(result['mae_zero']+result['mae_positive'])/2)
        close(metric['selection_balanced_mae'],sum(m['balanced_mae'] for m in metric['scales'].values())/3)
    selected=min(history,key=lambda x:x['selection_balanced_mae'])
    ensure(best==selected and state['best_epoch']==selected['epoch'], 'Best is not original earliest strict MAE minimum')
    close(state['best_validation_balanced_mae'],selected['selection_balanced_mae'])
    ensure(sum(x['optimizer_update_attempts'] for x in history)==state['optimizer_attempts_total'], 'Epoch attempt total mismatch')
    ensure(sum(x['optimizer_updates'] for x in history)==state['optimizer_updates_total'], 'Epoch actual update mismatch')
    ensure(sum(x['optimizer_skipped_overflows'] for x in history)==state['optimizer_skips_total'], 'Epoch skips mismatch')
    for row in history:
        ensure(row['optimizer_update_attempts']==124 and row['optimizer_updates']>0 and row['optimizer_skipped_overflows']>=0 and row['optimizer_updates']+row['optimizer_skipped_overflows']==124, 'Epoch step accounting mismatch')
    return selected

def validate_completion(run, receipt, entry, state, hashes):
    ensure(receipt['run_id']==entry['run_id'] and receipt['binding']==binding(), 'Completion training binding mismatch')
    ensure(receipt['best_epoch']==state['best_epoch'], 'Completion epoch mismatch')
    close(receipt['best_validation_balanced_mae'],state['best_validation_balanced_mae'])
    required={'config.json','status.json','metrics.jsonl','best_validation.json','best.pt',
              'source/train.json','source/val.json',*(f'source/{n}' for n in SOURCE_NAMES)}
    ensure(required<=set(receipt['files_sha256']), 'Incomplete completion file ledger')
    actual={p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file()}
    ensure(actual==set(receipt['files_sha256']), 'Run files added/removed since immutable completion')
    for relative,digest in receipt['files_sha256'].items(): watched(confined(run,relative),hashes,digest)
    ensure(receipt['checkpoint_sha256']==hashes[str((run/'best.pt').resolve())], 'Completion checkpoint mismatch')

def validate_run(run_id, *, load_model=False):
    """Legacy eight-tuple API. Metadata default; CPU strict model load opt-in."""
    plan,old,hashes=validate_static_sources();entry=declared_entry(plan,run_id)
    validate_training_authorization(hashes)
    run=ROOT/'runs'/run_id;receipt_path=SEED/'completion_records'/(run_id+'.json')
    ensure(receipt_path.is_file(), 'No immutable queue completion receipt; running/incomplete runs cannot export')
    watched(receipt_path,hashes);receipt=read(receipt_path)
    config=read(run/'config.json');state=read(run/'status.json');validate_config(entry,config,state)
    ensure(not any((run/n).exists() for n in ('USE_RESTRICTION.json','failure.json')), 'Failed/restricted run rejected')
    validate_completion(run,receipt,entry,state,hashes)
    ensure(set(config['source_sha256'])==set(SOURCE_NAMES), 'Unexpected trainer source set')
    for name in SOURCE_NAMES:
        expected=old['source_sha256'][name]
        ensure(config['source_sha256'][name]==expected, 'Run trainer source mismatch')
        watched(run/'source'/name,hashes,expected)
    inputs=read(SEED/'INPUT_LOCK.json')['external_inputs_sha256']
    # The frozen input ledger contains development sources/caches only. No test manifest is opened.
    for absolute,digest in inputs.items():
        path=Path(absolute).resolve()
        ensure(path.is_relative_to(ROOT), 'Input lock escapes project')
        ensure('test' not in path.name.lower(), 'Unexpected test file in development input lock')
        watched(path,hashes,digest)
    data=ROOT/'local_data_v2';data_lock=read(data/'DATASET_LOCK.json')
    ensure(config['manifest_sha256']==old['manifest_sha256'], 'Run manifest map differs from original plan')
    populations={}
    for split,count in (('train',985),('val',357)):
        path=data/(split+'.json');watched(path,hashes,old['manifest_sha256'][path.name])
        ensure(data_lock['manifest_sha256'][path.name]==sha(path), 'Dataset lock mismatch')
        watched(run/'source'/path.name,hashes,sha(path))
        records=read(path);ids=[r['id'] for r in records]
        ensure(len(ids)==len(set(ids))==count and ids==config['selected_'+split+'_ids'], 'Incomplete/reordered '+split+' population')
        ensure(all(r['split']==split for r in records), 'Split mismatch')
        ensure(len({(r['source_id'],r['x'],r['y']) for r in records})==count, 'Repeated native locations')
        for record in records:
            p=Path(record['cache']).resolve()
            ensure(p.is_relative_to((data/'cache').resolve()), 'Cache outside data version')
            ensure(str(p) in hashes and hashes[str(p)]==record['cache_sha256'], 'Cache differs from locked source')
        populations[split]=records
    train_ids={r['id'] for r in populations['train']};records=populations['val']
    ensure(not train_ids&{r['id'] for r in records}, 'Train/validation ID overlap')
    ensure(not {(r['source_id'],r['x'],r['y']) for r in populations['train']}&{(r['source_id'],r['x'],r['y']) for r in records}, 'Train/validation native overlap')
    ensure(sum(r['coverage']>0 for r in records)==27, 'Original validation stratum changed')
    history=[json.loads(line) for line in (run/'metrics.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    original=validate_history(history,read(run/'best_validation.json'),state)
    checkpoint=run/'best.pt';model=None
    if load_model:
        dependencies()
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        ensure(payload['config']==config and payload['validation']==original and payload['epoch']==original['epoch'], 'Checkpoint payload mismatch')
        model=training.make_model(config['encoder'],config['variant'],pretrained=False).cpu().eval()
        model.load_state_dict(payload['model'],strict=True)
        del payload
    watched(data/'manifest_summary.json',hashes)
    recheck(hashes)
    return run,config,state,records,original,checkpoint,model,hashes

def model_binding(run_id, config, original, checkpoint, hashes):
    entry=declared_entry(read(SEED/'plan.json'),run_id)
    return {'seed':config['seed'], 'lr':config['lr'], 'paired_seed17_run_id':entry['paired_seed17_run_id'],
        'training_binding':binding(), 'completion_receipt':str(SEED/'completion_records'/(run_id+'.json')),
        'completion_receipt_sha256':hashes[str((SEED/'completion_records'/(run_id+'.json')).resolve())],
        'checkpoint_sha256':sha(checkpoint), 'original_selected_epoch':original['epoch'],
        'export_protocol_sha256':sha(PROTOCOL)}

def artifact_manifest(out):
    write(out/'artifact_manifest.json',{'created_utc':now(), 'file_sha256':{
        p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}})

def preserve_failure(out, error, stage, state=None, *, filename='failure.json', primary_error=None):
    """Never replace the triggering exception with a failed failure write.

    If the attempt directory cannot exist/be written, use a separate version-owned
    fallback receipt. If both locations fail, emit the entire record to stderr.
    The permanent attempt claim is never removed.
    """
    record={**(state or {}),'status':'failed','stage':stage,'output_directory':str(out),
        'error':repr(error),'traceback':''.join(traceback.format_exception(type(error),error,error.__traceback__)),
        'failed_utc':now(),'claim_retained':True,'no_retry':True,'primary_error':primary_error}
    recording_failures=[]
    try:
        ensure(Path(out).is_dir(),'Attempt directory unavailable; use separate fallback receipt')
        write(Path(out)/filename,record);write(Path(out)/'status.json',record)
        return str(Path(out)/filename)
    except BaseException as recording_error:
        recording_failures.append({'location':str(out),'error':repr(recording_error)})
    fallback=SEED/'export_failure_receipts'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    try:
        fallback.mkdir(parents=True,exist_ok=False)
        record['recording_failures']=recording_failures
        write(fallback/filename,record)
        return str(fallback/filename)
    except BaseException as recording_error:
        recording_failures.append({'location':str(fallback),'error':repr(recording_error)})
        record['recording_failures']=recording_failures
        print(json.dumps(record,ensure_ascii=False,default=str),file=sys.stderr,flush=True)
        return None

def finish_manifest(out, primary_error=None, state=None):
    """Finalization failure is recorded; an earlier failure remains primary."""
    try:artifact_manifest(out)
    except BaseException as manifest_error:
        preserve_failure(out,manifest_error,'artifact_manifest',state,
                         filename='finalization_failure.json',primary_error=None if primary_error is None else repr(primary_error))
        if primary_error is None:raise

def prepare(run_id):
    run,config,state,records,original,checkpoint,_,hashes=validate_run(run_id)
    out=OUTPUT/run_id/datetime.datetime.now().strftime('attempt_%Y%m%d_%H%M%S_%f')
    claims=SEED/'export_claims';claims.mkdir(exist_ok=True)
    claim=claims/(run_id+'.json')
    with claim.open('x',encoding='utf-8') as stream:
        json.dump({'run_id':run_id,'output':str(out),'created_utc':now(),'pid':os.getpid(),'exporter_sha256':sha(__file__)},stream,indent=2)
    try:
        out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
        copies={};copy_bindings={}
        sources=[ROOT/name for name in SOURCE_NAMES]+[Path(__file__),ROOT/'export_common_validation.py',PROTOCOL]
        sources += [SEED/name for name in PINS]+[run/name for name in ('config.json','status.json','best_validation.json')]
        sources += [ROOT/'local_data_v2'/name for name in ('val.json','DATASET_LOCK.json','manifest_summary.json')]
        sources += [SEED/'completion_records'/(run_id+'.json')]
        sources += [SEED/'QUEUE_CLAIM.json',Path(read(SEED/'QUEUE_CLAIM.json')['review_path'])]
        for index,path in enumerate(sources):
            destination=out/'source'/f'{index:02d}_{path.name}';shutil.copy2(path,destination)
            ensure(sha(destination)==sha(path),'Source archival mismatch')
            ensure(path.name not in copies,'Duplicate source-copy basename')
            copies[path.name]=str(destination)
            copy_bindings[path.name]={'source_path':str(path.resolve()),'archive_path':str(destination.resolve()),'sha256':sha(path)}
        inputs={'run_id':run_id,'run_directory':str(run),'input_hashes':hashes,'source_copies':copies,'source_copy_bindings':copy_bindings,
                'validation_ids':[r['id'] for r in records],'exporter_sha256':sha(__file__),
                'legacy_exporter_sha256':LEGACY_SHA,**model_binding(run_id,config,original,checkpoint,hashes)}
        write(out/'pre_execution_inputs.json',inputs)
        seal={'created_utc':now(),'inference_started':False,'run_id':run_id,'exporter_sha256':sha(__file__),
              'export_protocol_sha256':sha(PROTOCOL),'file_sha256':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}}
        write(out/'pre_execution_seal.json',seal)
        write(out/'status.json',{'status':'prepared','run_id':run_id,'output_directory':str(out),
            'pre_execution_seal_sha256':sha(out/'pre_execution_seal.json'),'test_accessed':False,'R6_accessed':False,
            'factors':list(FACTORS),'total_windows':357,'model_family':'local_seed_stability_v1','seed':config['seed']})
        artifact_manifest(out)
    except BaseException as error:
        preserve_failure(out,error,'prepare',{'run_id':run_id,'test_accessed':False,'R6_accessed':False})
        finish_manifest(out,primary_error=error)
        raise
    print(json.dumps({'status':'prepared','output':str(out),'seal_sha256':sha(out/'pre_execution_seal.json')},ensure_ascii=False))
    return out

def check_execution_review(out, review_path):
    ensure(Path(review_path).resolve().is_relative_to(ROOT),'Execution review must be inside project')
    review=read(review_path)
    ensure(review.get('passed') is True and review.get('execution_scope')=='local_seed_validation_export_v1', 'Missing explicit exporter review')
    ensure(review.get('test_or_R6_authorized') is False, 'Development-only authorization required')
    checks={'exporter_sha256':sha(__file__),'export_protocol_sha256':sha(PROTOCOL),
            'source_lock_sha256':PINS['SOURCE_LOCK.json'],'plan_sha256':PINS['plan.json'],
            'pre_execution_seal_sha256':sha(out/'pre_execution_seal.json')}
    for key,digest in checks.items():ensure(review.get(key)==digest,'Review does not bind '+key)
    ensure(review.get('run_id')==read(out/'pre_execution_inputs.json')['run_id'],'Review run mismatch')
    focus=Path(review['focuschecks_path']).resolve()
    ensure(sha(focus)==review['focuschecks_sha256'],'Focuschecks hash mismatch')
    result=read(focus)
    ensure(result.get('passed') is True and result['exporter_sha256']==sha(__file__) and result['export_protocol_sha256']==sha(PROTOCOL),'Focuschecks not bound to exporter')
    ensure(result['checker_sha256']==read(PROTOCOL)['focuschecker_sha256']==sha(ROOT/'check_local_seed_export.py'),'Focuschecker source changed')
    ensure(result['lifecycle_checker_sha256']==read(PROTOCOL)['lifecycle_checker_sha256']==sha(ROOT/'check_local_seed_export_lifecycle.py'),'Lifecycle checker source changed')
    return {str(Path(review_path).resolve()):sha(review_path),str(focus):sha(focus)}

def table(path, rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def save_probability_map(path, probability, validity):
    """Array-only serialization/integration; usable by synthetic CPU checks."""
    import numpy as np
    p=np.asarray(probability);v=np.asarray(validity)
    ensure(p.shape==(3,256,256) and p.dtype==np.float32 and v.shape==(256,256) and v.dtype==np.float32,'Map shape/dtype mismatch')
    ensure(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid probability map')
    ensure(np.isfinite(v).all() and ((v>=0)&(v<=1)).all() and v.sum(dtype=np.float64)>0,'Invalid/empty validity support')
    ensure(not Path(path).exists(),'Refuse to overwrite probability map')
    np.savez_compressed(path,p=p,valid=v)
    with np.load(path,allow_pickle=False) as saved:
        ensure(set(saved.files)=={'p','valid'} and np.array_equal(saved['p'],p) and np.array_equal(saved['valid'],v),'Map roundtrip mismatch')
        q=(saved['p'].astype(np.float64)*saved['valid'][None]).sum((1,2))/saved['valid'].astype(np.float64).sum()
    return q

def export_predictions(out, config, records, original, checkpoint, model, hashes, inputs):
    """Reviewed legacy arithmetic and geometric metadata; no legacy main call."""
    sources,source_summary=legacy.source_metadata(records)
    ensure(sha(source_summary)==hashes[str(source_summary.resolve())],'Source summary changed')
    manifest=ROOT/'local_data_v2/val.json';mh=sha(manifest);ch=sha(checkpoint)
    (out/'maps').mkdir(exist_ok=False)
    loader=DataLoader(training.DevelopmentTiles(records,augment=False),batch_size=4,shuffle=False,num_workers=0)
    rows=[];windows=[];checks=[];cursor=0
    for image,soft,valid,truth in loader:
        pp=[];qq=[]
        for factor in FACTORS:
            logits=model(training.degrade_rgb(image,factor,valid),valid)
            training.require_finite_forward(logits,model)
            p=logits.float().sigmoid();pp.append(p[:,0].numpy());qq.append(training.coverage(p,valid).numpy())
        combined=np.stack(pp,axis=1)
        for offset in range(len(image)):
            rec=records[cursor+offset];v=valid[offset,0].numpy().astype(np.float32,copy=False)
            window=legacy.window_metadata(rec,sources[rec['source_id']],v,mh,ch)
            path=out/'maps'/(rec['id']+'.npz')
            saved_q=save_probability_map(path,combined[offset].astype(np.float32),v)
            window.update(file=str(path),relative_file='maps/'+path.name,file_sha256=sha(path),probability_shape=[3,256,256]);windows.append(window)
            for index,factor in enumerate(FACTORS):
                pred=float(qq[index][offset]);target=float(truth[offset]);binary=torch.from_numpy(combined[offset,index]>.5).float()[None]
                difference=abs(float(saved_q[index])-pred);ensure(difference<=1e-6,'Saved map/scalar integral mismatch')
                rows.append({'id':rec['id'],'source_id':rec['source_id'],'scene_group':rec['scene_group'],'split':'val',
                    'scale_factor':factor,'effective_input_pixels':256//factor,'coverage_true':target,'coverage_true_manifest':rec['coverage'],
                    'coverage_pred':pred,'absolute_error':abs(pred-target),'signed_error':pred-target,'cache_valid_area':float(v.sum()),
                    'native_valid_pixel_equivalent':float(v.sum())*4,'x':rec['x'],'y':rec['y'],'global_x':rec['global_x'],'global_y':rec['global_y'],
                    'checkpoint_epoch':original['epoch'],'checkpoint_sha256':ch,'manifest_sha256':mh,'cache_sha256':rec['cache_sha256'],
                    'tp_area':float((binary*soft[offset]*valid[offset]).sum()),'fp_area':float((binary*(1-soft[offset])*valid[offset]).sum()),
                    'fn_area':float(((1-binary)*soft[offset]*valid[offset]).sum())})
                checks.append({'id':rec['id'],'scale_factor':factor,'scalar_coverage':pred,'map_coverage_saved_float64_sum':float(saved_q[index]),'absolute_difference_saved':difference})
        cursor+=len(image)
        write(out/'progress.json',{'completed_windows':cursor,'expected_windows':357,'completed_factor_predictions':cursor*3,'updated_utc':now()})
        if cursor%64==0 or cursor==357:print(json.dumps({'windows':cursor,'run_id':config['run_id']}),flush=True)
    ensure(cursor==357 and len(rows)==1071 and {(x['id'],x['scale_factor']) for x in rows}=={(r['id'],f) for r in records for f in FACTORS},'Incomplete population')
    table(out/'predictions.csv',rows);write(out/'predictions.json',rows);table(out/'scalar_validation.csv',checks)
    with (out/'predictions.csv').open(encoding='utf-8-sig',newline='') as stream:reopened=list(csv.DictReader(stream))
    ensure(len(reopened)==1071,'Incomplete saved CSV')
    summaries={};differences={}
    for factor in FACTORS:
        subset=[r for r in reopened if int(r['scale_factor'])==factor]
        values=training.summarize([float(r['coverage_pred']) for r in subset],[float(r['coverage_true']) for r in subset])
        tp,fp,fn=[sum(float(r[k]) for r in subset) for k in ('tp_area','fp_area','fn_area')]
        values.update(target_iou=tp/max(tp+fp+fn,1),dice=2*tp/max(2*tp+fp+fn,1));summaries[str(factor)]=values
        differences[str(factor)]={k:values[k]-original['scales'][str(factor)][k] for k in ('mae','bias','mae_zero','mae_positive','balanced_mae')}
    score=sum(x['balanced_mae'] for x in summaries.values())/3
    summary={'run_id':config['run_id'],'seed':config['seed'],'checkpoint_epoch':original['epoch'],'checkpoint_sha256':ch,'manifest_sha256':mh,
        'device':'cpu','precision':'float32; autocast disabled','threads':2,'scales':summaries,'selection_balanced_mae':score,
        'original_amp_validation':original,'cpu_minus_original_amp_by_R':differences,
        'cpu_minus_original_amp_selection_balanced_mae':score-original['selection_balanced_mae'],
        'original_selected_epoch_preserved':True,'checkpoint_reselected':False,'test_accessed':False,'R6_accessed':False,
        'area_units':'cache effective pixels; native equivalent x4; not square metres','training_configuration':config,'export_script_sha256':sha(__file__)}
    write(out/'summary.json',summary)
    metadata={'schema':'local_validation_probability_maps_v1','run_id':config['run_id'],'model_family':'local_seed_stability_v1',
        'encoder':config['encoder'],'variant':config['variant'],'created_at':now(),'device':'cpu','precision':'float32; autocast disabled','threads':2,'batch_size':4,
        'checkpoint':str(checkpoint),'checkpoint_sha256':ch,'checkpoint_epoch':original['epoch'],
        'config':str(ROOT/'runs'/config['run_id']/'config.json'),'config_sha256':sha(ROOT/'runs'/config['run_id']/'config.json'),
        'manifest':str(manifest),'manifest_sha256':mh,'dataset_summary':str(source_summary),'dataset_summary_sha256':sha(source_summary),
        'model_source_sha256':config['source_sha256']['train_common.py'],'model_dependency_sha256':config['source_sha256'],
        'export_source_sha256':sha(__file__),'legacy_export_source_sha256':LEGACY_SHA,'source_copies':inputs['source_copies'],
        'source_copy_bindings':inputs['source_copy_bindings'],'input_hashes':hashes,
        'execution_bindings':inputs['execution_bindings'],
        'scalar_reference_directory':str(out),'scalar_csv_sha256':sha(out/'predictions.csv'),'scalar_summary_sha256':sha(out/'summary.json'),
        'factors':list(FACTORS),'npz_fields':{'p':'float32[3,256,256],R1/2/4','valid':'float32[256,256]'},
        'pixel_support':'p[f,v,u] supports native[x+2u,x+2u+2) by[y+2v,y+2v+2); affine maps corners',
        'valid_semantics':'fraction of alpha-valid native pixels in each2x2 block from sealed val cache',
        'invalid_probability_values':'retained; downstream must integrate native validity',
        'ground_truth_policy':'Exact area remains original alpha/PNG, not cache hard labels','spatial_prediction_available':True,
        'sources':sources,'windows':windows,'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,
        'argv':sys.argv,'test_accessed':False,'R6_accessed':False,'training_run_invoked':False,'raw_geometry_recomputed':False,
        **model_binding(config['run_id'],config,original,checkpoint,hashes)}
    verification={'all_checks_passed':True,'expected_windows':357,'checked_pairs':1071,'expected_pairs':1071,'all_npz_reopened':357,
        'scalar_tolerance':1e-6,'max_absolute_difference_saved':max(x['absolute_difference_saved'] for x in checks),
        'strict_weights_only_checkpoint_loaded':True,'original_selection_audited_24_epochs':True,'original_epoch_preserved':True,
        'manifest_order_preserved':True,'test_accessed':False,'R6_accessed':False}
    recheck(hashes);write(out/'metadata.json',metadata);write(out/'verification.json',verification)
    return metadata,verification

def execute(output, review_path):
    out=Path(output).resolve();ensure(out.is_relative_to(OUTPUT.resolve()),'Output outside exporter directory')
    state=read(out/'status.json');ensure(state['status']=='prepared','Attempt is not prepared; failed/completed attempts cannot restart')
    seal=read(out/'pre_execution_seal.json');ensure(sha(out/'pre_execution_seal.json')==state['pre_execution_seal_sha256'],'Seal changed')
    for relative,digest in seal['file_sha256'].items():ensure(sha(confined(out,relative))==digest,'Prepared artifact changed')
    inputs=read(out/'pre_execution_inputs.json');run_id=inputs['run_id']
    ensure(out.parent.name==run_id and read(SEED/'export_claims'/(run_id+'.json'))['output']==str(out),'Attempt owner mismatch')
    review_hashes=check_execution_review(out,review_path)
    with (out/'EXECUTION_CLAIM.json').open('x',encoding='utf-8') as stream:
        json.dump({'created_utc':now(),'pid':os.getpid(),'review_path':str(Path(review_path).resolve()),'review_sha256':sha(review_path)},stream,indent=2)
    start=time.time()
    primary_error=None
    try:
        dependencies()
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        state.update(status='running',started_utc=now(),pid=os.getpid(),device='cpu',threads=2)
        write(out/'status.json',state)
        _,config,_,records,original,checkpoint,model,hashes=validate_run(run_id,load_model=True)
        ensure(hashes==inputs['input_hashes'],'Inputs changed after preparation')
        runtime_inputs={**inputs,'execution_bindings':{'root_review_and_focuscheck_sha256':review_hashes,
            'pre_execution_seal_sha256':sha(out/'pre_execution_seal.json')}}
        ensure(seal['exporter_sha256']==sha(__file__) and seal['export_protocol_sha256']==sha(PROTOCOL),'Executable source changed')
        ensure(inputs['validation_ids']==[r['id'] for r in records],'Prepared validation order changed')
        with torch.inference_mode():
            _,verification=export_predictions(out,config,records,original,checkpoint,model,hashes,runtime_inputs)
        recheck(review_hashes)
        state.update(status='completed',completed_at=now(),completed_windows=357,completed_factor_predictions=1071,
                     metadata=str(out/'metadata.json'),metadata_sha256=sha(out/'metadata.json'),verification=verification,elapsed_seconds=time.time()-start)
        write(out/'status.json',state)
    except BaseException as error:
        primary_error=error
        preserve_failure(out,error,'execute',state)
        raise
    finally:finish_manifest(out,primary_error=primary_error,state=state)
    print(json.dumps({'status':'completed','output':str(out),'metadata_sha256':sha(out/'metadata.json')},ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--validate-only',action='store_true');group.add_argument('--prepare',action='store_true');group.add_argument('--execute',action='store_true')
    parser.add_argument('--run-id');parser.add_argument('--output');parser.add_argument('--root-review');args=parser.parse_args()
    if args.execute:
        ensure(args.output and args.root_review and not args.run_id,'Execution requires --output and --root-review only')
        execute(args.output,args.root_review)
    else:
        ensure(args.run_id and not args.output and not args.root_review,'Metadata/prepare requires --run-id only')
        if args.prepare:prepare(args.run_id)
        else:
            _,config,state,records,original,checkpoint,_,hashes=validate_run(args.run_id)
            print(json.dumps({'passed':True,'metadata_only':True,'run_id':args.run_id,'seed':config['seed'],'validation_windows':len(records),'best_epoch':original['epoch'],'checkpoint_sha256':sha(checkpoint),'input_count':len(hashes),'model_loaded':False,'inference':False},ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        # Pre-claim refusals are diagnostic records, never an inference retry.
        if not isinstance(error,SystemExit):
            rejected=OUTPUT/'rejections'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            rejected.mkdir(parents=True,exist_ok=False)
            write(rejected/'failure.json',{'created_utc':now(),'error':repr(error),'traceback':traceback.format_exc(),
                'argv':sys.argv,'exporter_sha256':sha(__file__),'note':'Inspect attempt status if claimed; no automatic retry is performed.'})
        raise
