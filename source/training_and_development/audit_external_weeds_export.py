"""Independent artifact-only audit; no training, inference, or model imports."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import csv,datetime,hashlib,json,math,shutil,struct,traceback
import numpy as np

ROOT=Path(__file__).resolve().parent
EXPORT=ROOT/'external_exports/exw_20260916_063648_275359'
EXT=ROOT/'external_weeds_v1'
DATA=ROOT/'weedsgalore_data_v1'
TRAIN_AUDIT=ROOT/'external_weeds_training_reports/20260916_062524_429008/audit.json'
PINS={'training_audit':'172bb72f0b31c761ccb584cba29b8817b4f58b71e9bcff33a6665d88cd2d75fc',
      'seal':'a7605004624192248e4bade0ad7176d10eb5fa3035a42794921298e6183c2fec',
      'manifest':'2e75b9d7259646ea0c3ed8fbc69f11ad316962a74a7949c91363635696105440',
      'exporter':'bcb973609e6d8c2e1bfa2a6afb0f721b92db2efd5d3d6fe906be421b0aa591df'}
WATCH={}

def rawsha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def sha(path):
    path=Path(path).resolve();digest=rawsha(path);WATCH[str(path)]=digest;return digest

def read(path):
    sha(path);return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def rows(path):
    sha(path)
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def write(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def savecsv(path,records):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

def close(a,b):
    a=float(a);b=float(b)
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14),(a,b)

def f32(x):return struct.unpack('f',struct.pack('f',x))[0]

def metrics(group):
    pred=[float(x['prediction_cpu']) for x in group];target=[float(x['target']) for x in group]
    gpu=[float(x['original_GPU_prediction']) for x in group];n=len(group)
    diff=[p-g for p,g in zip(pred,gpu)]
    return {'n':n,'mae':math.fsum(abs(p-t) for p,t in zip(pred,target))/n,
            'bias':math.fsum(p-t for p,t in zip(pred,target))/n,
            'mean_prediction':math.fsum(pred)/n,'mean_target':math.fsum(target)/n,
            'maximum_absolute_CPU_GPU_prediction_difference':max(map(abs,diff)),
            'mean_absolute_CPU_GPU_prediction_difference':math.fsum(map(abs,diff))/n,
            'mean_CPU_GPU_prediction_difference':math.fsum(diff)/n,
            'nonzero_CPU_GPU_prediction_differences':sum(x!=0 for x in diff),
            'original_GPU_MAE':math.fsum(abs(p-t) for p,t in zip(gpu,target))/n}

def main():
    out=ROOT/'provenance'/datetime.datetime.now().strftime('external_export_independent_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
    verified_runs=[];map_checks=[];scale_checks=[]
    try:
        assert sha(TRAIN_AUDIT)==PINS['training_audit']
        assert sha(EXPORT/'pre_execution_seal.json')==PINS['seal']
        assert sha(EXPORT/'artifact_manifest.json')==PINS['manifest']
        assert sha(ROOT/'export_external_weeds_validation.py')==PINS['exporter']
        audit=read(TRAIN_AUDIT);seal=read(EXPORT/'pre_execution_seal.json')
        artifact=read(EXPORT/'artifact_manifest.json');inputs=read(EXPORT/'inputs.json')
        protocol=read(EXPORT/'pre_execution_protocol.json');summary=read(EXPORT/'summary.json')
        status=read(EXPORT/'status.json');claim=read(EXPORT/'EXECUTION_CLAIM.json');progress=read(EXPORT/'progress.json')
        assert audit['passed'] and audit['run_count']==9
        assert artifact['source_input_seal_sha256']==PINS['seal']
        assert not seal['inference_started'] and claim['exporter_sha256']==seal['exporter_sha256']==PINS['exporter']
        assert datetime.datetime.fromisoformat(protocol['created_utc'])<=datetime.datetime.fromisoformat(seal['created_utc'])<datetime.datetime.fromisoformat(claim['created_utc'])<datetime.datetime.fromisoformat(status['completed_utc'])
        for rel,digest in seal['source_and_inputs_sha256'].items():
            path=(EXPORT/rel).resolve();assert path.is_relative_to(EXPORT.resolve()) and sha(path)==digest
        actual_files={p.relative_to(EXPORT).as_posix() for p in EXPORT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
        assert set(artifact['file_sha256'])==actual_files
        for rel,digest in artifact['file_sha256'].items():
            path=(EXPORT/rel).resolve();assert path.is_relative_to(EXPORT.resolve()) and sha(path)==digest
        assert not (EXPORT/'failure.json').exists()
        assert status['state']==summary['state']=='completed'
        assert status['run_count']==summary['run_count']==9 and status['scalar_rows']==summary['scalar_rows']==702
        assert status['map_count']==summary['map_count']==156
        for obj in (status,summary,protocol,artifact):
            assert obj['test_accessed'] is False and obj['R6_accessed'] is False
        assert not status['GPU_started'] and not summary['GPU_started'] and not summary['checkpoint_reselected']
        assert not protocol['training'] and not protocol['spatial_selection'] and protocol['R']==[1,2,4]
        assert protocol['frames_per_run']==26 and protocol['expected_scalar_rows']==702 and protocol['expected_segmentation_maps']==156
        assert protocol['map_scalar_consistency_tolerance']==1e-7
        for path,digest in inputs['input_sha256'].items():assert sha(path)==digest,path
        assert inputs['review_sha256']==PINS['training_audit'] and Path(inputs['review_path']).resolve()==TRAIN_AUDIT.resolve()
        assert sha(EXPORT/'source/training_review.json')==PINS['training_audit']
        assert sha(EXPORT/'source/exporter.py')==inputs['exporter_sha256']==PINS['exporter']
        assert sha(EXPORT/'source/checkpoint_loader.py')==sha(ROOT/'safe_load_external_checkpoint.py')==audit['safe_loader_sha256']==inputs['loader_sha256']
        for name,key in [('SOURCE_LOCK.json','source_lock_sha256'),('serial_plan.json','plan_sha256'),('RESEARCH_PROTOCOL.json','protocol_sha256')]:
            assert sha(EXT/name)==audit[key]==inputs[key]
        lock=read(EXT/'SOURCE_LOCK.json');plan=read(EXT/'serial_plan.json')
        for rel,digest in lock['source_files_sha256'].items():assert sha(EXT/rel)==sha(EXPORT/'source/frozen'/rel)==digest
        data_lock=read(DATA/'DATASET_LOCK.json');val=read(DATA/'val.json')
        assert sha(DATA/'val.json')==data_lock['manifest_sha256']['val.json']
        ids=[x['id'] for x in val];assert len(ids)==len(set(ids))==26
        assert inputs['val_records']==val and audit['val_ids']==protocol['validation_ids']==ids
        assert all(x['split']=='val' for x in val)
        for record in val:
            assert sha(record['cache'])==record['cache_sha256']
            with np.load(record['cache'],allow_pickle=False) as cache:
                valid=cache['valid']
                assert valid.shape==(256,256) and valid.dtype==np.dtype('float32')
                assert np.isfinite(valid).all() and bool((valid==1).all())
        records={x['id']:x for x in val}
        run_ids=[x['run_id'] for x in plan['runs']]
        assert protocol['declared_run_ids']==[x['run_id'] for x in inputs['runs']]==[x['run_id'] for x in summary['runs']]==run_ids
        assert progress['runs_completed']==9 and progress['completed_run_ids']==run_ids
        audited={x['run_id']:x for x in audit['runs']}
        csvrows=rows(EXPORT/'all702_scalars.csv')
        expected_keys=[(rid,r,i) for rid in run_ids for r in (1,2,4) for i in ids]
        keys=[(x['run_id'],int(x['R']),x['id']) for x in csvrows]
        assert keys==expected_keys and len(set(keys))==702
        scalar_by_key=dict(zip(keys,csvrows));map_index=read(EXPORT/'map_index.json')
        segmentation=[x['run_id'] for x in plan['runs'] if x['variant']!='regression']
        assert [(x['run_id'],x['id']) for x in map_index]==[(rid,i) for rid in segmentation for i in ids]
        assert len({x['path'] for x in map_index})==156
        assert {p.relative_to(EXPORT).as_posix() for p in EXPORT.rglob('*.npz')}=={x['path'] for x in map_index}
        original_samples={}
        for entry,meta,exported in zip(inputs['runs'],summary['runs'],plan['runs']):
            rid=entry['run_id'];folder=EXPORT/rid;run=EXT/'runs'/rid
            assert Path(entry['directory']).resolve()==run.resolve()
            config=read(run/'config.json');run_status=read(run/'status.json');completion=read(run/'completion_manifest.json')
            best=read(run/'best_validation.json');metadata=read(folder/'metadata.json')
            assert metadata==meta and all(entry[k]==v for k,v in exported.items())
            for name,key in [('status.json','status_sha256'),('best.pt','checkpoint_sha256'),('completion_manifest.json','completion_manifest_sha256'),('config.json','config_sha256'),('best_validation.json','best_validation_sha256')]:
                assert sha(run/name)==audited[rid][key]
            assert entry['checkpoint_sha256']==metadata['checkpoint_sha256']==audited[rid]['checkpoint_sha256']
            assert entry['best_epoch']==metadata['best_epoch']==run_status['best_epoch']==best['epoch']==audited[rid]['best_epoch']
            assert config['input_bindings']==completion['input_bindings']
            assert config['selected_val_ids']==ids and config['n_val']==26 and config['n_train']==104
            assert config['variant']==entry['variant'] and config['seed']==entry['seed'] and config['encoder']=='resnet18'
            for name,digest in completion['files_sha256'].items():
                path=(run/name).resolve();assert path.is_relative_to(run.resolve()) and sha(path)==digest
            for k in ('test_accessed','R6_accessed','checkpoint_reselected'):assert metadata[k] is False
            assert metadata['CPU_dtype']=='float32' and metadata['checkpoint_loader_sha256']==inputs['loader_sha256']
            mine=[x for x in csvrows if x['run_id']==rid];assert mine==rows(folder/'all78_scalars.csv') and len(mine)==78
            for r in (1,2,4):
                original=best['scales'][str(r)]['samples'];assert [x['id'] for x in original]==ids
                original_samples.update({(rid,r,x['id']):x for x in original})
            for x in mine:
                r=int(x['R']);identifier=x['id'];record=records[identifier];original=original_samples[rid,r,identifier]
                assert x['variant']==entry['variant'] and int(x['seed'])==entry['seed'] and int(x['best_epoch'])==entry['best_epoch']
                assert x['acquisition_date']==record['acquisition_date']==original['acquisition_date']
                target=float(x['target']);cpu=float(x['prediction_cpu']);gpu=float(x['original_GPU_prediction']);delta=float(x['delta_cpu_gpu'])
                assert target==f32(record['coverage'])==original['truth']
                assert gpu==original['prediction'] and delta==cpu-gpu
                assert all(math.isfinite(v) and 0<=v<=1 for v in (target,cpu,gpu))
            for r in (1,2,4):
                calculated=metrics([x for x in mine if int(x['R'])==r])
                for k,v in metadata['scales'][str(r)].items():close(v,calculated[k])
                scale_checks.append({'run_id':rid,'variant':entry['variant'],'seed':entry['seed'],'R':r,**calculated})
            aggregate=metrics(mine)
            close(metadata['CPU_mean_MAE'],aggregate['mae']);close(entry['original_GPU_selection_mae'],aggregate['original_GPU_MAE'])
            close(metadata['CPU_GPU_selection_MAE_difference'],aggregate['mae']-aggregate['original_GPU_MAE'])
            expected_maps=[x for x in map_index if x['run_id']==rid]
            assert metadata['maps']==expected_maps and metadata['map_count']==len(expected_maps)==(0 if entry['variant']=='regression' else 26)
            if entry['variant']=='regression':assert not (folder/'maps').exists()
            verified_runs.append({'run_id':rid,'variant':entry['variant'],'seed':entry['seed'],'best_epoch':entry['best_epoch'],
                'checkpoint_sha256':entry['checkpoint_sha256'],'metadata_sha256':sha(folder/'metadata.json'),
                'config_sha256':sha(run/'config.json'),'status_sha256':sha(run/'status.json'),
                'completion_manifest_sha256':sha(run/'completion_manifest.json'),
                'scalar_rows':78,'maps':len(expected_maps),'CPU_MAE_pp':100*aggregate['mae'],
                'original_GPU_MAE_pp':100*aggregate['original_GPU_MAE'],
                'CPU_minus_GPU_MAE_pp':100*(aggregate['mae']-aggregate['original_GPU_MAE']),
                'CPU_bias_pp':100*aggregate['bias'],
                'max_abs_CPU_GPU_prediction_difference_pp':100*aggregate['maximum_absolute_CPU_GPU_prediction_difference'],
                'mean_abs_CPU_GPU_prediction_difference_pp':100*aggregate['mean_absolute_CPU_GPU_prediction_difference'],
                'nonzero_CPU_GPU_prediction_differences':aggregate['nonzero_CPU_GPU_prediction_differences']})
        global_min=1.;global_max=0.
        for item in map_index:
            rid=item['run_id'];identifier=item['id'];path=(EXPORT/item['path']).resolve()
            assert path.is_relative_to((EXPORT/rid/'maps').resolve()) and path.name==identifier+'.npz'
            assert sha(path)==item['sha256'] and item['source_cache_sha256']==records[identifier]['cache_sha256']
            assert item['R']==[1,2,4] and item['shape']==[3,256,256] and item['dtype']=='float32'
            with np.load(path,allow_pickle=False) as archive:
                assert len(archive.files)==3 and set(archive.files)=={'probabilities','R','id'}
                probability=archive['probabilities'];factors=archive['R'];name=archive['id']
                assert name.shape==() and name.dtype.kind=='U' and str(name)==identifier
                assert factors.shape==(3,) and factors.dtype==np.dtype('int64') and np.array_equal(factors,np.array([1,2,4]))
                assert probability.shape==(3,256,256) and probability.dtype==np.dtype('float32')
                assert np.isfinite(probability).all() and bool((probability>=0).all()) and bool((probability<=1).all())
                global_min=min(global_min,float(probability.min()));global_max=max(global_max,float(probability.max()))
                for index,r in enumerate((1,2,4)):
                    mean=float(probability[index].mean(dtype=np.float64))
                    scalar=float(scalar_by_key[rid,r,identifier]['prediction_cpu']);difference=abs(mean-scalar)
                    assert difference<=1e-7,(rid,r,identifier,difference)
                    map_checks.append({'run_id':rid,'id':identifier,'R':r,'map_mean_float64':mean,
                                       'recorded_CPU_scalar':scalar,'absolute_difference':difference,'npz_sha256':item['sha256']})
        assert len(map_checks)==468 and len(scale_checks)==27 and len(verified_runs)==9
        aggregate=metrics(csvrows);max_map=max(x['absolute_difference'] for x in map_checks)
        for path,digest in WATCH.items():assert rawsha(path)==digest,('Input changed during audit',path)
        savecsv(out/'all9_CPU_GPU_comparison.csv',verified_runs)
        savecsv(out/'all27_scale_checks.csv',scale_checks);savecsv(out/'all468_map_mean_checks.csv',map_checks)
        sanitized={'passed':True,'export_root':str(EXPORT),'artifact_manifest_sha256':PINS['manifest'],
            'map_index_sha256':sha(EXPORT/'map_index.json'),'exporter_sha256':PINS['exporter'],
            'training_audit':{'path':str(TRAIN_AUDIT),'sha256':PINS['training_audit']},
            'valid_all_ones_attested':True,'val_manifest_sha256':sha(DATA/'val.json'),'val_ids':ids,
            'segmentation_models':[{'run_id':x['run_id'],'variant':x['variant'],'seed':x['seed'],'encoder':'resnet18',
                'best_epoch':x['best_epoch'],'checkpoint_sha256':x['checkpoint_sha256'],
                'config_path':str(EXT/'runs'/x['run_id']/'config.json'),'config_sha256':x['config_sha256'],
                'checkpoint_path':str(EXT/'runs'/x['run_id']/'best.pt'),'completed24':True}
                for x in verified_runs if x['variant']!='regression']}
        assert len(sanitized['segmentation_models'])==6
        write(out/'spatial_sanitized_acceptance.json',sanitized)
        result={'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'export_directory':str(EXPORT),'training_audit_sha256':PINS['training_audit'],
            'spatial_sanitized_acceptance_path':str(out/'spatial_sanitized_acceptance.json'),
            'spatial_sanitized_acceptance_sha256':rawsha(out/'spatial_sanitized_acceptance.json'),
            'valid_all_ones_attested':True,'valid_cache_arrays_checked':26,
            'export_seal_sha256':PINS['seal'],'export_artifact_manifest_sha256':PINS['manifest'],
            'exporter_sha256':PINS['exporter'],'export_protocol_sha256':sha(EXPORT/'pre_execution_protocol.json'),
            'export_inputs_sha256':sha(EXPORT/'inputs.json'),'export_status_sha256':sha(EXPORT/'status.json'),
            'export_summary_sha256':sha(EXPORT/'summary.json'),'all702_scalars_sha256':sha(EXPORT/'all702_scalars.csv'),
            'map_index_sha256':sha(EXPORT/'map_index.json'),'source_code_sha256':rawsha(__file__),
            'source_lock_sha256':inputs['source_lock_sha256'],'plan_sha256':inputs['plan_sha256'],'protocol_sha256':inputs['protocol_sha256'],
            'runs':verified_runs,'val_ids':ids,'map_index':map_index,'run_count':9,'scalar_count':702,
            'NPZ_count':156,'R_maps':468,'R':[1,2,4],'full_val_order_passed':True,
            'source_binding_discrepancies':0,'scalar_truth_or_original_GPU_discrepancies':0,
            'scalar_delta_arithmetic_discrepancies':0,'metadata_metric_discrepancies':0,
            'map_schema_range_or_ID_discrepancies':0,'map_mean_tolerance_violations':0,
            'maximum_map_mean_absolute_difference':max_map,'map_mean_tolerance':1e-7,
            'map_probability_minimum':global_min,'map_probability_maximum':global_max,
            'CPU_GPU_prediction_difference_summary':aggregate,'checkpoint_reselected':False,
            'verification_scope':'Artifact and frozen-source checks, not independent replay of model inference or OS trace.',
            'test_accessed':False,'R6_accessed':False,'new_inference_calls':0,'new_training_calls':0,
            'GPU_used':False,'input_sha256':WATCH}
        write(out/'audit.json',result)
        lines=['# WeedsGalore CPU 导出独立验收','',
            '全部9组通过；仅检查已存在的导出、原GPU样本和来源绑定，没有运行模型或改选checkpoint。','',
            f'- 702条标量、156个NPZ（每个含R=1/2/4，合计468张256×256概率图）、26个val ID与官方顺序完整。',
            '- regression仅78标量/组，无虚构空间图；分割NPZ仅含 probabilities/R/id，不含真值或其他标签。',
            f'- 概率图float32、有限且在[0,1]；468个float64图均值与原CPU标量最大绝对差 {max_map:.12g}，阈值1e-7，超限0。',
            '- 来源哈希、原best/config/status/completion、全部真值/原GPU预测、CPU−GPU差值、导出汇总算术的不一致计数均为0。',
            f'- CPU与GPU AMP数值并非逐位相同：702条中{aggregate["nonzero_CPU_GPU_prediction_differences"]}条非零，最大单图差 {100*aggregate["maximum_absolute_CPU_GPU_prediction_difference"]:.8f} pp。这是已选epoch的执行精度差异描述，不以CPU结果重新选择epoch。','',
            '| variant | seed | original best | GPU MAE pp | CPU MAE pp | CPU−GPU pp |',
            '|---|---:|---:|---:|---:|---:|']
        for x in verified_runs:lines.append(f"|{x['variant']}|{x['seed']}|{x['best_epoch']}|{x['original_GPU_MAE_pp']:.8f}|{x['CPU_MAE_pp']:.8f}|{x['CPU_minus_GPU_MAE_pp']:+.8f}|")
        lines+=['','## 边界','',
            '26幅同一开发验证集参与原best选模，仍是单田块/四日期的方法复现；种子不是独立田块。CPU导出支持下一阶段相对空间分析，不创造米制地理参考，也不是独立测试结果。',
            'test/R6排除依据为锁定源码、输入清单、完整val ID/R匹配和导出文件白名单；本验收没有读取test像素，没有再运行推理或构造OS级访问证明。',
            '原GPU AMP标量及其来源完整保留；图均值使用float64检查，标量truth保持原CPU float32口径。',
            '',f'输入目录：`{EXPORT}`',
            '复算：在publication_campaign使用 `python -X utf8 -B -s audit_external_weeds_export.py`，输出新的时间目录。']
        (out/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        write(out/'artifact_manifest.json',{'passed':True,'source_code_sha256':rawsha(__file__),
            'files_sha256':{p.name:rawsha(p) for p in sorted(out.iterdir()) if p.is_file()}})
        print(json.dumps({'passed':True,'directory':str(out),'audit_sha256':rawsha(out/'audit.json'),
                          'maximum_map_mean_absolute_difference':max_map,'CPU_GPU_summary':aggregate},ensure_ascii=False))
    except BaseException as error:
        write(out/'failure.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc(),
                                  'verified_runs':verified_runs,'map_checks_completed':len(map_checks),
                                  'source_code_sha256':rawsha(__file__),'input_sha256':WATCH,
                                  'new_inference_calls':0,'new_training_calls':0,'test_accessed':False,'R6_accessed':False})
        raise

if __name__=='__main__':main()
