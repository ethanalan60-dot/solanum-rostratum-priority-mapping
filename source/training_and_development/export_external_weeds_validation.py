"""Prepare and execute a one-shot CPU export of all nine original WeedsGalore best epochs.

No model selection or spatial policy. Regression stays scalar. Only 26 development
validation frames at R=1,2,4; production sources/checkpoints are never modified.
"""
from pathlib import Path
import argparse,csv,datetime,gc,hashlib,importlib.util,json,math,os,shutil,sys,time,traceback

ROOT=Path(__file__).resolve().parent
EXT=ROOT/'external_weeds_v1';DATA=ROOT/'weedsgalore_data_v1'
OUTPUT_ROOT=ROOT/'external_exports'
LOADER=ROOT/'safe_load_external_checkpoint.py'
R_VALUES=(1,2,4)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)
def table(path,rows):
    assert rows
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def collect_inputs(review_path):
    review_path=Path(review_path).resolve();review=read(review_path)
    plan=read(EXT/'serial_plan.json');lock=read(EXT/'SOURCE_LOCK.json')
    assert review['passed'] and len(review['runs'])==9
    assert sha(LOADER)==review['safe_loader_sha256'],'Use the independently audited restricted checkpoint loader'
    assert len(plan['runs'])==9 and {(x['variant'],x['seed']) for x in plan['runs']}=={(v,s) for v in ('regression','hard_seg','soft_seg') for s in (17,29,43)}
    hashes={str(Path(__file__).resolve()):sha(__file__),str(LOADER):sha(LOADER),str(review_path):sha(review_path)}
    for filename,key in [('SOURCE_LOCK.json','source_lock_sha256'),('serial_plan.json','plan_sha256'),('RESEARCH_PROTOCOL.json','protocol_sha256')]:
        digest=sha(EXT/filename);assert digest==review[key];hashes[str(EXT/filename)]=digest
    for relative,digest in lock['source_files_sha256'].items():
        source=(EXT/relative).resolve();assert source.is_relative_to(EXT) and sha(source)==digest;hashes[str(source)]=digest
    for absolute,digest in lock['external_inputs_sha256'].items():assert sha(absolute)==digest;hashes[absolute]=digest
    records=read(DATA/'val.json');data_lock=read(DATA/'DATASET_LOCK.json')
    assert len(records)==26 and len({x['id'] for x in records})==26 and [x['id'] for x in records]==review['val_ids']
    assert sha(DATA/'val.json')==data_lock['manifest_sha256']['val.json']
    for row in records:
        cache=Path(row['cache']).resolve();assert row['split']=='val' and cache.is_relative_to(DATA/'cache')
        assert 0<row['coverage']<=1 and sha(cache)==row['cache_sha256'];hashes[str(cache)]=row['cache_sha256']
    by_run={x['run_id']:x for x in review['runs']};assert set(by_run)=={x['run_id'] for x in plan['runs']}
    runs=[]
    for entry in plan['runs']:
        directory=EXT/'runs'/entry['run_id'];status=read(directory/'status.json');cfg=read(directory/'config.json')
        audited=by_run[entry['run_id']]
        assert status['status']=='completed' and status['executed_epochs']==24 and status['optimizer_attempts_total']==312
        assert not status['test_accessed'] and not status['R6_accessed']
        assert (cfg['variant'],cfg['seed'],cfg['encoder'])==(entry['variant'],entry['seed'],'resnet18')
        assert cfg['selected_val_ids']==review['val_ids'] and cfg['n_train']==104 and cfg['n_val']==26
        assert status['best_epoch']==audited['best_epoch']
        for filename,key in [('status.json','status_sha256'),('best.pt','checkpoint_sha256'),('completion_manifest.json','completion_manifest_sha256')]:
            digest=sha(directory/filename);assert digest==audited[key];hashes[str(directory/filename)]=digest
        completion=read(directory/'completion_manifest.json')
        for relative,digest in completion['files_sha256'].items():
            source=(directory/relative).resolve();assert source.is_relative_to(directory) and sha(source)==digest;hashes[str(source)]=digest
        original=read(directory/'best_validation.json')
        assert original['epoch']==status['best_epoch'] and set(original['scales'])=={'1','2','4'}
        for scale in original['scales'].values():assert [x['id'] for x in scale['samples']]==review['val_ids']
        runs.append(dict(entry,directory=str(directory),best_epoch=status['best_epoch'],checkpoint_sha256=status['checkpoint_sha256'],
                         original_GPU_selection_mae=status['best_validation_mae']))
    return {'review_path':str(review_path),'review_sha256':sha(review_path),'input_sha256':hashes,'runs':runs,'val_records':records,
            'source_lock_sha256':sha(EXT/'SOURCE_LOCK.json'),'plan_sha256':sha(EXT/'serial_plan.json'),
            'protocol_sha256':sha(EXT/'RESEARCH_PROTOCOL.json'),'exporter_sha256':sha(__file__),'loader_sha256':sha(LOADER)}

def prepare(review_path):
    inputs=collect_inputs(review_path)
    out=OUTPUT_ROOT/datetime.datetime.now().strftime('exw_%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    protocol={'version':'external_weeds_validation_export_v1','created_utc':now(),'state':'prepared_before_inference',
        'declared_run_ids':[x['run_id'] for x in inputs['runs']],'validation_ids':[x['id'] for x in inputs['val_records']],
        'R':list(R_VALUES),'frames_per_run':26,'runs':9,'expected_scalar_rows':702,'expected_segmentation_maps':156,
        'checkpoint_selection':'original ordinary validation MAE earliest-best epoch; export never reselects',
        'execution':'CPU float32, two threads, no CUDA; 26 full frames in fixed manifest order, batch4',
        'map':'one npz per segmentation/frame; probabilities float32 [3,256,256] in R1,2,4 order; no labels in map archive',
        'regression':'78 per-frame/R scalars only; never expand scalar to a spatial map',
        'normalization':'frozen DevelopmentFrames float32 RGB [0,1]; frozen degrade_rgb and ImageNet normalization',
        'comparison':'original GPU AMP predictions retained beside new CPU predictions; differences descriptive, no checkpoint change',
        'map_scalar_consistency_tolerance':1e-7,'checkpoint_loading':'SHA-verified local checkpoint through separately reviewed restricted CPU loader; TorchVersion metadata compatibility disclosed',
        'failure_policy':'exclusive one-shot claim; retain partial output and failure; no overwrite or automatic restart',
        'test_accessed':False,'R6_accessed':False,'spatial_selection':False,'training':False}
    write(out/'pre_execution_protocol.json',protocol);write(out/'inputs.json',inputs)
    sources=out/'source';sources.mkdir()
    shutil.copy2(__file__,sources/'exporter.py');shutil.copy2(LOADER,sources/'checkpoint_loader.py')
    shutil.copy2(inputs['review_path'],sources/'training_review.json')
    for relative in read(EXT/'SOURCE_LOCK.json')['source_files_sha256']:
        source=EXT/relative;destination=sources/'frozen'/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    seal={'created_utc':now(),'source_and_inputs_sha256':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()},
          'exporter_sha256':sha(__file__),'loader_sha256':sha(LOADER),'inference_started':False}
    write(out/'pre_execution_seal.json',seal);write(out/'status.json',{'state':'prepared','seal_sha256':sha(out/'pre_execution_seal.json'),'CPU_inference_started':False})
    print(json.dumps({'state':'prepared','output':str(out),'seal_sha256':sha(out/'pre_execution_seal.json'),'CPU_inference_started':False},ensure_ascii=False))

def summarize(rows):
    n=len(rows);assert n
    return {'n':n,'mae':math.fsum(abs(x['prediction_cpu']-x['target']) for x in rows)/n,
            'bias':math.fsum(x['prediction_cpu']-x['target'] for x in rows)/n,
            'mean_prediction':math.fsum(x['prediction_cpu'] for x in rows)/n,
            'mean_target':math.fsum(x['target'] for x in rows)/n,
            'maximum_absolute_CPU_GPU_prediction_difference':max(abs(x['delta_cpu_gpu']) for x in rows)}

def execute(output):
    out=Path(output).resolve();assert out.is_relative_to(OUTPUT_ROOT.resolve())
    state=read(out/'status.json');seal=read(out/'pre_execution_seal.json');assert state['state']=='prepared'
    assert sha(out/'pre_execution_seal.json')==state['seal_sha256']
    for relative,digest in seal['source_and_inputs_sha256'].items():assert sha(out/relative)==digest
    inputs=read(out/'inputs.json');assert collect_inputs(inputs['review_path'])==inputs
    assert sha(__file__)==seal['exporter_sha256'] and sha(LOADER)==seal['loader_sha256']
    with (out/'EXECUTION_CLAIM.json').open('x',encoding='utf-8') as stream:json.dump({'pid':os.getpid(),'created_utc':now(),'exporter_sha256':sha(__file__)},stream)
    all_rows=[];all_maps=[];all_runs=[];current=None;started=time.time();model=None
    try:
        write(out/'status.json',{'state':'initializing','pid':os.getpid(),'started_utc':now(),'CPU_inference_started':False,'GPU_started':False})
        os.environ['CUDA_VISIBLE_DEVICES']=''
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
        temp=out/'temp';temp.mkdir()
        for key in ('TEMP','TMP'):os.environ[key]=str(temp)
        sys.path.insert(0,str(EXT))
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        import train_external_weeds as runner
        import frozen_training_core as core
        spec=importlib.util.spec_from_file_location('verified_external_checkpoint_loader',LOADER)
        safe_loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(safe_loader)
        device=runner.configure(17,'cpu')
        for key in ('TEMP','TMP'):os.environ[key]=str(temp)
        dataset=runner.DevelopmentFrames('val');assert [x['id'] for x in dataset.records]==[x['id'] for x in inputs['val_records']]
        loader=DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0)
        write(out/'status.json',{'state':'running','pid':os.getpid(),'started_utc':now(),'CPU_inference_started':True,'GPU_started':False})
        for entry in inputs['runs']:
            current=entry['run_id'];folder=out/current;folder.mkdir()
            original=read(Path(entry['directory'])/'best_validation.json')
            payload=safe_loader.load_checkpoint(Path(entry['directory'])/'best.pt',entry['checkpoint_sha256'])
            assert payload['epoch']==entry['best_epoch'] and payload['config']['run_id']==current
            model=core.make_model('resnet18',entry['variant'],False)
            aligned=model.load_state_dict(payload['model'],strict=True);assert not aligned.missing_keys and not aligned.unexpected_keys
            model=model.to(device).eval();del payload
            rows=[];maps={row['id']:[] for row in dataset.records} if entry['variant']!='regression' else None
            for factor in R_VALUES:
                offset=0;reference={x['id']:x for x in original['scales'][str(factor)]['samples']}
                with torch.inference_mode():
                    for x,soft,valid,q in loader:
                        assert x.shape[1:]==(3,256,256) and torch.isfinite(x).all() and float(x.min())>=0 and float(x.max())<=1
                        assert torch.all(valid==1) and soft.shape[1:]==(1,256,256)
                        logits=model(core.degrade_rgb(x,factor,valid),valid);core.require_finite_forward(logits,model)
                        probability=logits.float().sigmoid()
                        if maps is None:
                            assert probability.shape==(len(x),);predicted=probability
                        else:
                            assert probability.shape==(len(x),1,256,256);predicted=core.coverage(probability,valid)
                        assert torch.isfinite(predicted).all() and torch.all((predicted>=0)&(predicted<=1))
                        for i in range(len(x)):
                            record=dataset.records[offset+i];identifier=record['id'];target=float(q[i]);estimate=float(predicted[i])
                            assert target==float(np.float32(record['coverage']))==reference[identifier]['truth']
                            if maps is not None:
                                array=probability[i,0].cpu().numpy().copy();assert array.dtype==np.float32 and np.isfinite(array).all()
                                assert abs(float(array.mean(dtype=np.float64))-estimate)<=1e-7;maps[identifier].append(array)
                            rows.append({'run_id':current,'variant':entry['variant'],'seed':entry['seed'],'best_epoch':entry['best_epoch'],
                                'id':identifier,'acquisition_date':record['acquisition_date'],'R':factor,'target':target,'prediction_cpu':estimate,
                                'original_GPU_prediction':reference[identifier]['prediction'],'delta_cpu_gpu':estimate-reference[identifier]['prediction']})
                        offset+=len(x)
                assert offset==26
            map_records=[]
            if maps is not None:
                map_dir=folder/'maps';map_dir.mkdir()
                for record in dataset.records:
                    identifier=record['id'];array=np.stack(maps[identifier]);assert array.shape==(3,256,256)
                    path=map_dir/(identifier+'.npz');np.savez_compressed(path,probabilities=array,R=np.array(R_VALUES,dtype=np.int64),id=np.array(identifier))
                    with np.load(path,allow_pickle=False) as check:assert np.array_equal(check['probabilities'],array) and str(check['id'])==identifier
                    item={'run_id':current,'id':identifier,'path':path.relative_to(out).as_posix(),'sha256':sha(path),'R':[1,2,4],
                          'shape':[3,256,256],'dtype':'float32','source_cache_sha256':record['cache_sha256']}
                    map_records.append(item);all_maps.append(item)
            assert len(rows)==78
            table(folder/'all78_scalars.csv',rows)
            metrics={str(r):summarize([x for x in rows if x['R']==r]) for r in R_VALUES}
            metadata=dict(entry,scalar_rows=78,map_count=len(map_records),maps=map_records,scales=metrics,
                CPU_mean_MAE=math.fsum(x['mae'] for x in metrics.values())/3,
                CPU_GPU_selection_MAE_difference=math.fsum(x['mae'] for x in metrics.values())/3-entry['original_GPU_selection_mae'],
                CPU_dtype='float32',torch=str(torch.__version__),checkpoint_loader_sha256=inputs['loader_sha256'],
                test_accessed=False,R6_accessed=False,checkpoint_reselected=False)
            write(folder/'metadata.json',metadata);all_rows.extend(rows);all_runs.append(metadata)
            model=None;maps=None;gc.collect()
            write(out/'progress.json',{'completed_run_ids':[x['run_id'] for x in all_runs],'runs_completed':len(all_runs),'updated_utc':now()})
            print(json.dumps({'run_id':current,'state':'exported','scalar_rows':78,'maps':len(map_records)}),flush=True)
        assert len(all_rows)==702 and len(all_maps)==156 and len(all_runs)==9
        assert collect_inputs(inputs['review_path'])==inputs
        table(out/'all702_scalars.csv',all_rows);write(out/'map_index.json',all_maps)
        summary={'state':'completed','runs':all_runs,'run_count':9,'scalar_rows':702,'map_count':156,'elapsed_seconds':time.time()-started,
                 'completed_utc':now(),'GPU_started':False,'checkpoint_reselected':False,'test_accessed':False,'R6_accessed':False,
                 'scope':'Full26 development validation frames; CPU float32 predictions, original GPU AMP predictions retained; no spatial evaluation or regression maps'}
        write(out/'summary.json',summary);write(out/'status.json',{'state':'completed','run_count':9,'scalar_rows':702,'map_count':156,
                  'completed_utc':summary['completed_utc'],'GPU_started':False,'test_accessed':False,'R6_accessed':False})
    except BaseException as error:
        failure={'state':'failed','current_run':current,'completed_run_ids':[x['run_id'] for x in all_runs],'error':repr(error),
                 'traceback':traceback.format_exc(),'failed_utc':now(),'test_accessed':False,'R6_accessed':False,'no_retry':True}
        write(out/'failure.json',failure);write(out/'status.json',failure);raise
    finally:
        model=None;gc.collect()
        write(out/'artifact_manifest.json',{'created_utc':now(),'source_input_seal_sha256':sha(out/'pre_execution_seal.json'),
              'file_sha256':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'},
              'test_accessed':False,'R6_accessed':False})
    print(json.dumps({'state':'completed','output':str(out),'scalars':702,'maps':156},ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training-review');parser.add_argument('--execute',action='store_true');parser.add_argument('--output')
    args=parser.parse_args()
    if args.execute:
        if not args.output:parser.error('--execute requires the previously prepared --output')
        execute(args.output)
    else:
        if not args.training_review or args.output:parser.error('Preparation requires --training-review and no --output')
        prepare(args.training_review)
if __name__=='__main__':main()
