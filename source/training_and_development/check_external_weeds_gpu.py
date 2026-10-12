"""Prepare, then explicitly execute one bounded WeedsGalore GPU diagnostic.

Default invocation only creates an immutable pre-execution package using hashes
and training metadata. It does not import torch or initialize CUDA. After root
reviews this script/package: --execute --output <prepared-directory>.
The formal external_weeds_v1 source lock, run directories and serial claim are
never changed. A failed diagnostic is preserved and never automatically retried.
"""
from pathlib import Path
import argparse, datetime, gc, hashlib, json, math, os, random, shutil, sys, time, traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
EXTERNAL=ROOT/'external_weeds_v1'
DIAGNOSTICS=ROOT/'external_weeds_gpu_diagnostics'
FIXED_IDS=(
    '2023-06-15_0472','2023-06-15_0763','2023-06-15_0157','2023-06-15_1056',
    '2023-06-15_0454','2023-06-15_0728','2023-06-15_0996','2023-06-06_0197',
)
VARIANTS=('regression','hard_seg','soft_seg')
PINS={
    'SOURCE_LOCK.json':'3ff901160f73f70df2ad601f3a40b49b2e8e76d5b83942c5c03bcf584da1ddfe',
    'serial_plan.json':'5225bde2861c6c046e94b514eadfc398523f776c478b007ca6ae8c64639d0596',
    'RESEARCH_PROTOCOL.json':'5b20a0856761470575538dad305486924c63da2ca343811f37fc63b29a6b962d',
    'cpu_checks/20260916_052057_016479/result.json':'735d8e923bc9c3aa1347869225d78d052803f88c9051751fa104df3cba08b64d',
}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,obj):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)

def locked_inputs():
    """Hash all declared metadata/source inputs; read only the training manifest.

    The source lock includes the val JSON hash, which is checked as bytes only;
    no validation dataset/cache pixels, model, scorer, or test manifest is used.
    """
    for relative,digest in PINS.items():
        if sha(EXTERNAL/relative)!=digest:raise ValueError('Pinned preparation changed: '+relative)
    lock=read(EXTERNAL/'SOURCE_LOCK.json');plan=read(EXTERNAL/'serial_plan.json')
    protocol=read(EXTERNAL/'RESEARCH_PROTOCOL.json')
    cpu=read(EXTERNAL/'cpu_checks/20260916_052057_016479/result.json')
    if lock['version']!='external_weeds_v1' or not lock['before_any_GPU_or_external_training']:raise ValueError('Wrong source lock')
    if not cpu['passed'] or cpu['count']!=59 or cpu['source_lock_sha256']!=PINS['SOURCE_LOCK.json']:raise ValueError('Required 59 CPU checks did not pass')
    if cpu['plan_sha256']!=PINS['serial_plan.json'] or cpu['protocol_sha256']!=PINS['RESEARCH_PROTOCOL.json']:raise ValueError('CPU checks bind another protocol')
    if cpu['GPU_started'] or cpu['experiment_training_started'] or cpu['test_pixels_or_manifest_read']:raise ValueError('Unexpected CPU check scope')
    if protocol['hard_target_rule']!='soft >= 0.5':raise ValueError('Wrong hard target')
    source_hashes={}
    for relative,digest in lock['source_files_sha256'].items():
        path=(EXTERNAL/relative).resolve()
        if not path.is_relative_to(EXTERNAL) or sha(path)!=digest:raise ValueError('Locked source mismatch')
        source_hashes[str(path)]=digest
    for path,digest in lock['external_inputs_sha256'].items():
        if sha(path)!=digest:raise ValueError('Locked external metadata/source mismatch: '+path)
    data=ROOT/'weedsgalore_data_v1'
    manifest=data/'train.json';records=read(manifest)
    if len(records)!=104 or len({x['id'] for x in records})!=104:raise ValueError('Incomplete train source')
    lookup={x['id']:x for x in records}
    selected=[lookup[identifier] for identifier in FIXED_IDS]
    if len({x['id'] for x in selected})!=8 or any(x['split']!='train' for x in selected):raise ValueError('Wrong diagnostic IDs')
    for record in selected:
        cache=Path(record['cache']).resolve()
        if not cache.is_relative_to((data/'cache').resolve()) or sha(cache)!=record['cache_sha256']:raise ValueError('Selected training cache changed')
    return {'pins':dict(PINS),'source_files_sha256':source_hashes,'external_inputs_sha256':lock['external_inputs_sha256'],
            'training_manifest_sha256':sha(manifest),'selected_records':selected,
            'diagnostic_script_sha256':sha(Path(__file__)),'formal_settings':plan['settings']}

def prepare():
    inputs=locked_inputs()
    out=DIAGNOSTICS/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    try:
        source=out/'source';source.mkdir()
        shutil.copy2(__file__,source/'check_external_weeds_gpu.py')
        for absolute,digest in inputs['source_files_sha256'].items():
            origin=Path(absolute);relative=origin.relative_to(EXTERNAL)
            destination=source/'external_weeds_v1'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(origin,destination)
            if sha(destination)!=digest:raise ValueError('Source capture changed')
        for relative in PINS:
            destination=source/'external_weeds_v1'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(EXTERNAL/relative,destination)
        shutil.copy2(ROOT/'weedsgalore_data_v1/train.json',source/'development_train_manifest.json')
        generator=random.Random(17+1000);factors=[generator.choice((1,2,4)) for _ in range(2)]
        protocol={'version':'external_weeds_gpu_diagnostic_v1','created_utc':now(),'state':'prepared_before_any_GPU_execution',
            'script_sha256':sha(Path(__file__)),'source_lock_sha256':PINS['SOURCE_LOCK.json'],
            'plan_sha256':PINS['serial_plan.json'],'formal_protocol_sha256':PINS['RESEARCH_PROTOCOL.json'],
            'CPU_59_result_sha256':PINS['cpu_checks/20260916_052057_016479/result.json'],
            'train_ids':list(FIXED_IDS),'variant_order':list(VARIANTS),'seed':17,
            'sample_order':'the eight explicitly listed train IDs in this fixed order; no shuffle in this bounded diagnostic',
            'augmentations':'same frozen production DevelopmentFrames full-FOV flips/gain/offset with RNG17, reinitialized per variant',
            'batch_size':4,'accumulation_steps':2,'attempts_per_variant':1,'max_total_attempts':3,
            'factor_rng_seed':1017,'factor_per_microbatch':factors,
            'optimizer':'same AdamW lr1e-4 weight_decay1e-4; frozen train_epoch and AMP guard; no scheduler required for one attempt',
            'CUDA_device':'cuda:0','GPU_memory_fraction':.16,'CPU_threads':2,
            'public_initialization':'fixed SHA-verified ResNet18; seed17 reset for each variant; same decoder/hard-soft state',
            'checkpoint_replay':'save complete model state after the one attempt; fresh same-architecture model strict=True load, exact tensor equality, then bitwise identical FP32 eval logits on all8 unaugmented train images at the two fixed R factors',
            'failure_policy':'fail immediately, save events/counts/traceback, do not retry or tune; remaining variants may be unattempted',
            'diagnostic_only':True,'formal_initialization_from_diagnostic_allowed':False,
            'validation_or_test_model_calls':0,'checkpoint_selection':False,'R6':False,
            'outputs':'this separately timestamped directory only; formal runs and serial claim unchanged',
            'requires_explicit_execute':True,'not_an_external_model_performance_result':True}
        write(out/'pre_execution_protocol.json',protocol);write(out/'inputs.json',inputs)
        files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
        write(out/'pre_execution_seal.json',{'created_utc':now(),'state':'sealed_before_GPU','file_sha256':files,
              'diagnostic_script_sha256':inputs['diagnostic_script_sha256'],'GPU_started':False})
        write(out/'status.json',{'state':'prepared','GPU_started':False,'protocol_sha256':sha(out/'pre_execution_protocol.json'),
                                'pre_execution_seal_sha256':sha(out/'pre_execution_seal.json'),'test_accessed':False,'R6_accessed':False})
        print(json.dumps({'state':'prepared_only','output':str(out),'protocol_sha256':sha(out/'pre_execution_protocol.json'),
                          'seal_sha256':sha(out/'pre_execution_seal.json'),'script_sha256':sha(Path(__file__)),
                          'fixed_train_ids':list(FIXED_IDS),'factors':factors,'GPU_started':False},ensure_ascii=False))
    except BaseException as error:
        write(out/'preparation_failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'GPU_started':False});raise

def validate_prepared(out):
    out=Path(out).resolve()
    if not out.is_relative_to(DIAGNOSTICS.resolve()):raise ValueError('Output must be a separately prepared diagnostic directory')
    status=read(out/'status.json');seal=read(out/'pre_execution_seal.json')
    if status['state']!='prepared' or status['GPU_started']:raise ValueError('Prepared diagnostic was already used or is not ready')
    if sha(out/'pre_execution_seal.json')!=status['pre_execution_seal_sha256']:raise ValueError('Pre-execution seal changed')
    if sha(Path(__file__))!=seal['diagnostic_script_sha256']:raise ValueError('Use exactly the root-reviewed diagnostic script')
    for relative,digest in seal['file_sha256'].items():
        path=(out/relative).resolve()
        if not path.is_relative_to(out) or sha(path)!=digest:raise ValueError('Prepared artifact changed: '+relative)
    protocol=read(out/'pre_execution_protocol.json')
    if sha(out/'pre_execution_protocol.json')!=status['protocol_sha256']:raise ValueError('Prepared protocol changed')
    if protocol['train_ids']!=list(FIXED_IDS) or protocol['variant_order']!=list(VARIANTS):raise ValueError('Different sample/variant design')
    current=locked_inputs()
    if current!=read(out/'inputs.json'):raise ValueError('Inputs changed since the pre-execution package')
    return out,protocol,current

def execute(output):
    out,protocol,inputs=validate_prepared(output)
    # An exclusive one-shot claim is persisted before importing GPU-capable modules.
    with (out/'EXECUTION_CLAIM.json').open('x',encoding='utf-8') as f:
        json.dump({'created_utc':now(),'pid':os.getpid(),'explicit_execute':True,'script_sha256':sha(Path(__file__))},f,indent=2)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
    for key in ('TEMP','TMP'):os.environ[key]=str(out/'temp')
    (out/'temp').mkdir()
    sys.path.insert(0,str(EXTERNAL))
    import torch
    from torch.utils.data import DataLoader,Subset
    import train_external_weeds as runner
    import frozen_training_core as core
    started=time.time();results=[];current_variant=None;device=None
    initial_models={};initial_encoders={}
    model=optimizer=scaler=guard=reloaded=None
    write(out/'status.json',{'state':'running','GPU_started':False,'pid':os.getpid(),'started_utc':now(),
                             'test_accessed':False,'R6_accessed':False})
    try:
        runner.validate_plan(read(EXTERNAL/'serial_plan.json'))
        for variant in VARIANTS:
            current_variant=variant;folder=out/variant;folder.mkdir()
            device=runner.configure(17,'cuda:0')
            # configure's per-process cache settings are redirected for diagnostic temp ownership.
            for key in ('TEMP','TMP'):os.environ[key]=str(out/'temp')
            augmented=runner.DevelopmentFrames('train',True,17)
            indices=[next(i for i,row in enumerate(augmented.records) if row['id']==identifier) for identifier in FIXED_IDS]
            if [augmented.records[i]['id'] for i in indices]!=list(FIXED_IDS):raise ValueError('ID mapping drift')
            loader=DataLoader(Subset(augmented,indices),batch_size=4,shuffle=False,num_workers=0,pin_memory=True)
            if len(loader)!=2:raise ValueError('Expected exactly two diagnostic microbatches')
            model=core.make_model('resnet18',variant,True)
            initial_models[variant]=runner.state_digest(model.state_dict())
            initial_encoders[variant]=runner.state_digest(model.encoder.state_dict())
            initial={'variant':variant,'seed':17,'train_ids':list(FIXED_IDS),'model_sha256':initial_models[variant],
                     'encoder_sha256':initial_encoders[variant],'pretrained_report':model.pretrained_report,
                     'torch':str(torch.__version__),'python':sys.version,'GPU':torch.cuda.get_device_name(device)}
            write(folder/'initialization.json',initial)
            if len(set(initial_encoders.values()))!=1:raise AssertionError('Public encoder initializations differ')
            if variant=='soft_seg' and initial_models['soft_seg']!=initial_models['hard_seg']:raise AssertionError('Paired hard/soft initial states differ')
            model=model.to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            scaler=torch.cuda.amp.GradScaler(enabled=True)
            events=[]
            def record_event(event):
                events.append(event)
                with (folder/'optimizer_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event,allow_nan=False)+'\n')
            guard=core.OptimizerStepGuard(optimizer,scaler,record_event)
            args=SimpleNamespace(**dict(inputs['formal_settings'],variant=variant))
            training=core.train_epoch(model,loader,optimizer,scaler,device,args,random.Random(1017),guard,1)
            expected_counts={str(r):4*protocol['factor_per_microbatch'].count(r) for r in (1,2,4)}
            write(folder/'training_attempt.json',{'result':training,'attempted':guard.attempted,'actual':guard.actual,'skipped':guard.skipped,
                  'events':events,'expected_factor_counts':expected_counts,'diagnostic_only':True})
            if guard.attempted!=1 or guard.actual+guard.skipped!=1 or training['train_factor_sample_counts']!=expected_counts:raise AssertionError('Bounded attempt/factor accounting failed')
            if guard.actual!=1 or guard.skipped!=0 or not events[0]['all_unscaled_gradients_finite']:raise AssertionError('The single diagnostic attempt did not complete with finite gradients')
            state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
            if any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in state.values()):raise FloatingPointError('Nonfinite saved parameter or buffer')
            checkpoint={'model':state,'variant':variant,'seed':17,'train_ids':list(FIXED_IDS),'training_attempt':training,
                        'initialization':initial,'source_lock_sha256':PINS['SOURCE_LOCK.json'],
                        'diagnostic_protocol_sha256':sha(out/'pre_execution_protocol.json'),'formal_training_checkpoint':False}
            torch.save(checkpoint,folder/'diagnostic_checkpoint.pt.tmp');(folder/'diagnostic_checkpoint.pt.tmp').replace(folder/'diagnostic_checkpoint.pt')
            canonical=runner.DevelopmentFrames('train',False,17)
            replay_loader=DataLoader(Subset(canonical,indices),batch_size=4,shuffle=False,num_workers=0)
            def logits_all(current_model):
                current_model.eval();outputs=[]
                with torch.inference_mode():
                    for batch,factor in zip(replay_loader,protocol['factor_per_microbatch']):
                        x,soft,valid,q=batch;x=x.to(device);valid=valid.to(device)
                        with torch.autocast(device_type='cuda',enabled=False):logits=current_model(core.degrade_rgb(x,factor,valid),valid)
                        core.require_finite_forward(logits,current_model);outputs.append(logits.detach().cpu())
                output=torch.cat(outputs)
                if len(output)!=8:raise AssertionError('Incomplete training-only checkpoint replay')
                return output
            reference=logits_all(model);torch.save(reference,folder/'reference_logits.pt')
            guard.hook.remove();guard=None;optimizer=None;scaler=None;model=None;gc.collect();torch.cuda.empty_cache()
            payload=torch.load(folder/'diagnostic_checkpoint.pt',map_location='cpu',weights_only=True)
            if payload['train_ids']!=list(FIXED_IDS) or payload['variant']!=variant or payload['formal_training_checkpoint']:raise AssertionError('Checkpoint metadata mismatch')
            reloaded=core.make_model('resnet18',variant,False)
            aligned=reloaded.load_state_dict(payload['model'],strict=True)
            if aligned.missing_keys or aligned.unexpected_keys:raise AssertionError('Strict checkpoint alignment failed')
            if any(not torch.equal(reloaded.state_dict()[name],value) for name,value in state.items()):raise AssertionError('Loaded state differs from the saved state')
            reloaded=reloaded.to(device);replayed=logits_all(reloaded)
            difference=float((reference-replayed).abs().max())
            replay={'strict_loading':True,'state_tensors_equal':True,'train_ids':list(FIXED_IDS),'logit_shape':list(reference.shape),
                    'FP32_logits_bitwise_equal':torch.equal(reference,replayed),'max_absolute_logit_difference':difference,
                    'checkpoint_sha256':sha(folder/'diagnostic_checkpoint.pt'),'reference_logits_sha256':sha(folder/'reference_logits.pt'),
                    'same_batching_factors_and_eval_mode':True,'checkpoint_selection':False}
            write(folder/'checkpoint_replay.json',replay)
            if not replay['FP32_logits_bitwise_equal']:raise AssertionError('Saved/reloaded FP32 logits differ; no tolerance relaxation or retry')
            result={'variant':variant,'state':'passed','attempted':1,'actual':1,'skipped':0,'all_gradients_finite':True,
                    'initial_model_sha256':initial_models[variant],'initial_encoder_sha256':initial_encoders[variant],
                    'checkpoint_replay':replay,'train_factor_counts':training['train_factor_sample_counts']}
            results.append(result);write(folder/'result.json',result)
            reloaded=None;state=None;payload=None;checkpoint=None;reference=None;replayed=None
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize(device)
            write(out/'progress.json',{'completed':results,'total_attempts':len(results),'updated_utc':now(),'validation_calls':0})
        if locked_inputs()!=inputs:raise ValueError('Locked inputs changed during the diagnostic')
        final={'state':'completed','passed':True,'results':results,'total_attempts':3,'actual_updates':3,'skipped_updates':0,
               'initial_encoders_identical':len(set(initial_encoders.values()))==1,
               'hard_soft_initial_models_identical':initial_models['hard_seg']==initial_models['soft_seg'],
               'completed_utc':now(),'elapsed_seconds':time.time()-started,'GPU_started':True,
               'formal_runs_or_serial_claim_modified':False,'validation_model_calls':0,'test_accessed':False,'R6_accessed':False,
               'checkpoint_selection':False,'diagnostic_weights_used_for_formal_initialization':False,
               'interpretation':'bounded plumbing diagnostic; no convergence or predictive-performance claim'}
        write(out/'summary.json',final);write(out/'status.json',final)
    except BaseException as error:
        counts={'attempted':getattr(guard,'attempted',0),'actual':getattr(guard,'actual',0),'skipped':getattr(guard,'skipped',0)}
        failure={'state':'failed','passed':False,'current_variant':current_variant,'completed_results':results,
                 'current_guard_counts':counts,'last_guard_event':getattr(guard,'last_event',None),'error':repr(error),
                 'traceback':traceback.format_exc(),'failed_utc':now(),'no_retry':True,'test_accessed':False,'R6_accessed':False}
        write(out/'failure.json',failure);write(out/'status.json',failure);raise
    finally:
        if guard is not None:guard.hook.remove()
        model=optimizer=scaler=guard=reloaded=None;gc.collect()
        if device is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache();torch.cuda.synchronize(device)
            write(out/'GPU_release.json',{'released_utc':now(),'allocated_bytes_after_release':torch.cuda.memory_allocated(device),
                  'reserved_bytes_after_empty_cache':torch.cuda.memory_reserved(device),'process_PID':os.getpid(),
                  'only_this_process_allocator_touched':True})
        write(out/'artifact_manifest.json',{'created_utc':now(),'script_sha256':sha(Path(__file__)),
              'file_sha256':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'},
              'test_accessed':False,'R6_accessed':False})
    print(json.dumps({'state':'completed','output':str(out),'diagnostic_only':True,'total_attempts':3},ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--output')
    args=parser.parse_args()
    if args.execute:
        if not args.output:parser.error('--output must identify the package prepared before review')
        execute(args.output)
    else:
        if args.output:parser.error('Preparation always creates a new timestamped directory; --output is only for explicit execution')
        prepare()

if __name__=='__main__':main()
