"""Independent arithmetic/provenance audit of nine completed WeedsGalore runs.
No runner/summarize import, model construction, inference, optimizer or GPU call.
"""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
ROOT=Path(__file__).resolve().parent
TEMP=ROOT/'external_weeds_v1/temp'
TEMP.mkdir(exist_ok=True)
for key in ('TEMP','TMP'):os.environ[key]=str(TEMP)
import csv, datetime, gc, hashlib, json, math, pickle, random, shutil, statistics, struct, traceback, zipfile
import torch
from safe_load_external_checkpoint import load_checkpoint,inspect_checkpoint,sha256_file
torch.set_num_threads(2)
EXTERNAL=ROOT/'external_weeds_v1'
QUEUE=EXTERNAL/'queue_records/20260916_053741_352995/status.json'
DATA=ROOT/'weedsgalore_data_v1'
PINS={'SOURCE_LOCK.json':'3ff901160f73f70df2ad601f3a40b49b2e8e76d5b83942c5c03bcf584da1ddfe',
      'serial_plan.json':'5225bde2861c6c046e94b514eadfc398523f776c478b007ca6ae8c64639d0596',
      'RESEARCH_PROTOCOL.json':'5b20a0856761470575538dad305486924c63da2ca343811f37fc63b29a6b962d'}
INPUTS={}

def sha(path):
    path=Path(path).resolve();digest=sha256_file(path);INPUTS[str(path)]=digest;return digest
def read(path):
    sha(path);return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def lines(path):
    sha(path);return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines()]
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def close(a,b):
    assert math.isfinite(float(a)) and math.isfinite(float(b)),(a,b)
    assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=2e-12),(a,b)
def f32(value):return struct.unpack('f',struct.pack('f',value))[0]
def mean(values):return math.fsum(values)/len(values)

def metric(prediction,truth):
    assert len(prediction)==len(truth)>0
    assert all(math.isfinite(x) and 0<=x<=1 for x in prediction+truth)
    errors=[p-t for p,t in zip(prediction,truth)]
    return {'n':len(truth),'mae':mean([abs(x) for x in errors]),'bias':mean(errors),
            'mean_prediction':mean(prediction),'mean_target':mean(truth)}

def compare_metric(record,expected):
    assert record['n']==expected['n']
    for key in ('mae','bias','mean_prediction','mean_target'):close(record[key],expected[key])

def summarize_epochs(history,records,seed,variant,run_id):
    result=[];scale_rows=[];date_rows=[];samples=[]
    truth=[f32(x['coverage']) for x in records];ids=[x['id'] for x in records]
    dates=sorted({x['acquisition_date'] for x in records})
    assert [x['epoch'] for x in history]==list(range(1,25))
    factor_rng=random.Random(seed+1000)
    for epoch in history:
        assert set(epoch['scales'])=={'1','2','4'}
        computed=[]
        for r in (1,2,4):
            measured=epoch['scales'][str(r)];original=measured['samples']
            assert len(original)==26 and [x['id'] for x in original]==ids
            assert [x['truth'] for x in original]==truth
            assert [x['acquisition_date'] for x in original]==[x['acquisition_date'] for x in records]
            predicted=[x['prediction'] for x in original]
            calculated=metric(predicted,truth);compare_metric(measured,calculated);computed.append(calculated)
            assert set(measured['by_date'])==set(dates)
            for date in dates:
                idx=[i for i,row in enumerate(records) if row['acquisition_date']==date]
                cm=metric([predicted[i] for i in idx],[truth[i] for i in idx])
                compare_metric(measured['by_date'][date],cm)
                date_rows.append({'run_id':run_id,'variant':variant,'seed':seed,'epoch':epoch['epoch'],'R':r,
                                  'date':date,'n':len(idx),'mae_pp':100*cm['mae'],'bias_pp':100*cm['bias']})
            scale_rows.append({'run_id':run_id,'variant':variant,'seed':seed,'epoch':epoch['epoch'],'R':r,
                'n':26,'mae_pp':100*calculated['mae'],'bias_pp':100*calculated['bias'],
                'mean_prediction':calculated['mean_prediction'],'mean_target':calculated['mean_target']})
            samples.extend({'run_id':run_id,'variant':variant,'seed':seed,'epoch':epoch['epoch'],'R':r,
                            'id':x['id'],'acquisition_date':x['acquisition_date'],'prediction':x['prediction'],'truth':x['truth']} for x in original)
        score=mean([x['mae'] for x in computed]);bias=mean([x['bias'] for x in computed])
        close(score,epoch['selection_mae'])
        expected_factors={'1':0,'2':0,'4':0}
        for _ in range(26):expected_factors[str(factor_rng.choice((1,2,4)))]+=4
        assert epoch['train_factor_sample_counts']==expected_factors
        assert epoch['optimizer_update_attempts']==13
        assert epoch['optimizer_updates']+epoch['optimizer_skipped_overflows']==13
        close(epoch['lr'],1e-4*(1+math.cos(math.pi*(epoch['epoch']-1)/24))/2)
        result.append({'run_id':run_id,'variant':variant,'seed':seed,'epoch':epoch['epoch'],'mae_pp':100*score,
                       'bias_pp':100*bias,'train_loss':epoch['train_loss'],'attempted':13,
                       'actual':epoch['optimizer_updates'],'skipped':epoch['optimizer_skipped_overflows'],
                       'amp_scale_end':epoch['amp_scale_end'],'lr':epoch['lr'],
                       'train_R1':expected_factors['1'],'train_R2':expected_factors['2'],'train_R4':expected_factors['4']})
    return result,scale_rows,date_rows,samples

def check_events(events,history,variant):
    assert len(events)==312 and [x['attempt'] for x in events]==list(range(1,313))
    actual=skipped=consecutive=0;previous_scale=65536.
    for index,event in enumerate(events):
        assert event['epoch']==index//13+1 and event['group']==index%13+1
        assert event['group_samples']==8 and event['group_denominator']==(8 if variant=='regression' else 8*256*256)
        assert event['amp_enabled'] and event['status'] in ('updated','overflow_skipped')
        close(event['scale_before'],previous_scale)
        if event['optimizer_step_executed']:
            assert event['all_unscaled_gradients_finite'] and not event['skipped'] and event['status']=='updated'
            assert math.isfinite(event['unscaled_gradient_norm']);actual+=1;consecutive=0
        else:
            assert not event['all_unscaled_gradients_finite'] and event['skipped'] and event['status']=='overflow_skipped'
            assert event['scale_after']<event['scale_before'];skipped+=1;consecutive+=1
        assert event['actual_updates_total']==actual and event['skipped_updates_total']==skipped and event['consecutive_overflows']==consecutive
        previous_scale=event['scale_after']
    for h in history:
        group=events[(h['epoch']-1)*13:h['epoch']*13]
        assert sum(x['optimizer_step_executed'] for x in group)==h['optimizer_updates']
        assert sum(x['skipped'] for x in group)==h['optimizer_skipped_overflows']
        close(mean([x['group_loss'] for x in group]),h['train_loss'])
        close(group[-1]['scale_after'],h['amp_scale_end'])
        assert group[-1]['consecutive_overflows']==h['consecutive_overflows']
    return actual,skipped

def main():
    out=ROOT/'external_weeds_training_reports'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    shutil.copy2(__file__,out/'audit_external_weeds_training.py')
    shutil.copy2(ROOT/'safe_load_external_checkpoint.py',out/'safe_load_external_checkpoint.py')
    native_failures=[];runs=[];all_configs=[];epoch_rows=[];all_scales=[];all_dates=[];sample_rows=[];initializations=[]
    try:
        for name,digest in PINS.items():assert sha(EXTERNAL/name)==digest
        lock=read(EXTERNAL/'SOURCE_LOCK.json');plan=read(EXTERNAL/'serial_plan.json');protocol=read(EXTERNAL/'RESEARCH_PROTOCOL.json')
        assert len(plan['runs'])==9 and {(x['variant'],x['seed']) for x in plan['runs']}=={(v,s) for v in ('regression','hard_seg','soft_seg') for s in (17,29,43)}
        for relative,digest in lock['source_files_sha256'].items():assert sha(EXTERNAL/relative)==digest
        for path,digest in lock['external_inputs_sha256'].items():assert sha(path)==digest
        source_lock_digest=sha(EXTERNAL/'SOURCE_LOCK.json')
        data_lock=read(DATA/'DATASET_LOCK.json');train=read(DATA/'train.json');val=read(DATA/'val.json')
        assert len(train)==104 and len(val)==26 and not({x['id'] for x in train}&{x['id'] for x in val})
        assert all(x['coverage']>0 for x in train+val)
        assert sha(DATA/'train.json')==data_lock['manifest_sha256']['train.json']
        assert sha(DATA/'val.json')==data_lock['manifest_sha256']['val.json']
        cache_hashes={}
        for row in train+val:
            path=Path(row['cache']).resolve();assert path.is_relative_to((DATA/'cache').resolve())
            assert sha(path)==row['cache_sha256'];cache_hashes[str(path)]=row['cache_sha256']
        assert len(cache_hashes)==130
        expected_bindings={'source_lock_sha256':source_lock_digest,'source_files_sha256':lock['source_files_sha256'],
            'external_inputs_sha256':lock['external_inputs_sha256'],'development_cache_sha256':cache_hashes,
            'plan_sha256':PINS['serial_plan.json'],'protocol_sha256':PINS['RESEARCH_PROTOCOL.json'],
            'pretrained_sha256':'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'}
        queue=read(QUEUE);assert queue['state']=='completed' and len(queue['results'])==9 and not queue['test_accessed'] and not queue['R6_accessed']
        assert [x['run_id'] for x in queue['results']]==[x['run_id'] for x in plan['runs']]
        queue_runs={x['run_id']:x for x in queue['results']}
        truth=[f32(x['coverage']) for x in val];train_mean=mean([f32(x['coverage']) for x in train])
        baseline={'zero':metric([0.]*26,truth),'train_mean':metric([train_mean]*26,truth),'train_mean_value':train_mean}
        for entry in plan['runs']:
            rid=entry['run_id'];variant=entry['variant'];seed=entry['seed'];directory=EXTERNAL/'runs'/rid
            write(out/'progress.json',{'current_run':rid,'completed_runs':len(runs),'new_inference_calls':0})
            cfg=read(directory/'config.json');status=read(directory/'status.json');completion=read(directory/'completion_manifest.json')
            bindings=read(directory/'input_hashes.json');best=read(directory/'best_validation.json');history=lines(directory/'metrics.jsonl')
            events=lines(directory/'optimizer_events.jsonl');constants=read(directory/'constant_baselines.json')
            receipt=read(directory/'independent_review.json')
            assert receipt['receipt']['passed'] and receipt['receipt']['execution_scope']=='external_weeds_v1_gpu'
            assert receipt['receipt']['source_lock_sha256']==source_lock_digest and sha(receipt['path'])==receipt['sha256']
            assert bindings==completion['input_bindings']==cfg['input_bindings']==expected_bindings
            assert completion['state']==status['status']=='completed' and status['executed_epochs']==24
            assert not status['test_accessed'] and not status['R6_accessed'] and not cfg['test_accessed'] and not cfg['R6_accessed']
            assert cfg['version']=='external_weeds_v1' and cfg['selected_train_ids']==[x['id'] for x in train] and cfg['selected_val_ids']==[x['id'] for x in val]
            assert cfg['n_train']==104 and cfg['n_val']==26 and cfg['training_attempts_expected']==312
            for key,value in {**plan['settings'],**entry}.items():assert cfg[key]==value,(rid,key)
            sealed=completion['files_sha256']
            required={'best.pt','best_validation.json','config.json','status.json','metrics.jsonl','optimizer_events.jsonl','input_hashes.json','constant_baselines.json','independent_review.json','source/SOURCE_LOCK.json'}
            assert required.issubset(sealed)
            for relative,digest in sealed.items():
                path=(directory/relative).resolve();assert path.is_relative_to(directory.resolve()) and sha(path)==digest
            for relative,digest in lock['source_files_sha256'].items():assert sha(directory/'source'/relative)==digest
            assert sha(directory/'source/SOURCE_LOCK.json')==source_lock_digest
            for kind in ('zero','train_mean'):compare_metric(constants[kind],baseline[kind])
            close(constants['train_mean_value'],train_mean)
            rows,scales,dates,samples=summarize_epochs(history,val,seed,variant,rid)
            actual,skips=check_events(events,history,variant)
            assert status['optimizer_attempts_total']==actual+skips==312
            assert status['optimizer_updates_total']==actual and status['optimizer_skips_total']==skips
            earliest=min(history,key=lambda x:(x['selection_mae'],x['epoch']))
            assert best==earliest and best['epoch']==status['best_epoch']==queue_runs[rid]['best_epoch']
            close(best['selection_mae'],status['best_validation_mae']);close(best['selection_mae'],queue_runs[rid]['best_validation_mae'])
            assert sha(directory/'status.json')==queue_runs[rid]['status_sha256']
            checkpoint_sha=sha(directory/'best.pt');assert checkpoint_sha==status['checkpoint_sha256']==queue_runs[rid]['checkpoint_sha256']==sealed['best.pt']
            # Retain the actual native-loader failure separately from scientific audit status.
            try:
                test_payload=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
            except Exception as error:
                if 'Unsupported class torch.torch_version.TorchVersion' not in str(error):raise
                native_failures.append({'run_id':rid,'operation':'torch.load(map_location=cpu,weights_only=True)',
                                       'error':str(error),'checkpoint_sha256':checkpoint_sha,'original_file_unchanged':True})
            else:
                del test_payload
            write(out/'native_weights_only_failures.json',native_failures)
            inspected=inspect_checkpoint(directory/'best.pt',checkpoint_sha)
            payload=load_checkpoint(directory/'best.pt',checkpoint_sha)
            assert payload['epoch']==best['epoch'] and payload['validation']==best and payload['config']==cfg
            expected_counts=events[best['epoch']*13-1]
            assert payload['optimizer_step_counts']=={'attempted':best['epoch']*13,'actual':expected_counts['actual_updates_total'],'skipped':expected_counts['skipped_updates_total']}
            close(payload['grad_scaler']['scale'],best['amp_scale_end'])
            tensors=payload['model'];assert tensors and all(x.device.type=='cpu' for x in tensors.values())
            assert all(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()) for tensor in tensors.values())
            parameter_count=sum(tensor.numel() for name,tensor in tensors.items() if not name.endswith(('running_mean','running_var','num_batches_tracked')))
            assert parameter_count==cfg['parameter_count']
            shape=tuple(tensors['head.weight' if variant=='regression' else 'decoder.head.weight'].shape)
            assert shape==((1,512) if variant=='regression' else (1,64,1,1))
            assert cfg['pretrained_report']['sha256']==expected_bindings['pretrained_sha256'] and cfg['pretrained_report']['strict']
            best_row=next(x for x in rows if x['epoch']==best['epoch'])
            run={'run_id':rid,'variant':variant,'seed':seed,'best_epoch':best['epoch'],
                 'mae_pp':100*best['selection_mae'],'bias_pp':best_row['bias_pp'],'actual_updates':actual,'skipped_updates':skips,'attempted_updates':312,
                 'status_sha256':sha(directory/'status.json'),'checkpoint_sha256':checkpoint_sha,
                 'completion_manifest_sha256':sha(directory/'completion_manifest.json'),
                 'config_sha256':sha(directory/'config.json'),'best_validation_sha256':sha(directory/'best_validation.json'),
                 'metrics_sha256':sha(directory/'metrics.jsonl'),'optimizer_events_sha256':sha(directory/'optimizer_events.jsonl'),
                 'initial_encoder_sha256':cfg['initial_encoder_sha256'],'initial_model_sha256':cfg['initial_model_sha256'],
                 'parameter_count':parameter_count,'epoch24_mae_pp':rows[-1]['mae_pp'],'checkpoint_inspection':inspected,
                 'best_metadata_and_counts_match':True,'native_weights_only_compatible':not inspected['TorchVersion_to_plain_str']}
            runs.append(run);all_configs.append(cfg);epoch_rows.extend(rows);all_scales.extend(scales);all_dates.extend(dates);sample_rows.extend(samples)
            del payload,tensors;gc.collect()
        assert len(epoch_rows)==216 and len(sample_rows)==16848
        best_lookup={x['run_id']:x['best_epoch'] for x in runs}
        best_scales=[x for x in all_scales if x['epoch']==best_lookup[x['run_id']]]
        best_dates=[x for x in all_dates if x['epoch']==best_lookup[x['run_id']]]
        assert len(best_scales)==27 and len(best_dates)==108
        pairs=[]
        for seed in (17,29,43):
            selected={x['variant']:x for x in runs if x['seed']==seed}
            assert len({x['initial_encoder_sha256'] for x in selected.values()})==1
            assert selected['hard_seg']['initial_model_sha256']==selected['soft_seg']['initial_model_sha256']
            hard,soft=selected['hard_seg'],selected['soft_seg']
            pairs.append({'seed':seed,'hard_best_epoch':hard['best_epoch'],'soft_best_epoch':soft['best_epoch'],
                          'hard_mae_pp':hard['mae_pp'],'soft_mae_pp':soft['mae_pp'],'soft_minus_hard_mae_pp':soft['mae_pp']-hard['mae_pp'],
                          'hard_bias_pp':hard['bias_pp'],'soft_bias_pp':soft['bias_pp'],'soft_minus_hard_bias_pp':soft['bias_pp']-hard['bias_pp'],
                          'initial_models_identical':True,'initial_encoder_sha256':hard['initial_encoder_sha256']})
        assert len({x['initial_encoder_sha256'] for x in runs})==1
        stats=[]
        for variant in ('regression','hard_seg','soft_seg'):
            rr=[x for x in runs if x['variant']==variant];maes=[x['mae_pp'] for x in rr];biases=[x['bias_pp'] for x in rr]
            assert len({x['initial_model_sha256'] for x in rr})==3
            stats.append({'variant':variant,'n_seeds':3,'mean_mae_pp':mean(maes),'sample_sd_mae_pp':statistics.stdev(maes),
                          'min_mae_pp':min(maes),'max_mae_pp':max(maes),'mean_bias_pp':mean(biases),'sample_sd_bias_pp':statistics.stdev(biases),
                          'attempted_updates_per_run':312,'total_actual_updates':sum(x['actual_updates'] for x in rr),'total_skips':sum(x['skipped_updates'] for x in rr)})
        # Reject wrong hashes and unknown GLOBAL before tensor reconstruction.
        loader_tests=[]
        first=EXTERNAL/'runs'/runs[0]['run_id']/'best.pt'
        try:load_checkpoint(first,'0'*64)
        except ValueError:loader_tests.append({'case':'wrong_checksum','rejected':True})
        else:raise AssertionError('Wrong checksum accepted')
        toy=out/'unknown_GLOBAL_fixture.pt'
        with zipfile.ZipFile(toy,'w') as z:z.writestr('archive/data.pkl',b'cos\nsystem\n.')
        try:load_checkpoint(toy,sha256_file(toy))
        except pickle.UnpicklingError:loader_tests.append({'case':'unknown_GLOBAL_os_system','rejected':True})
        else:raise AssertionError('Unknown GLOBAL accepted')
        stack=out/'STACK_GLOBAL_fixture.pt'
        with zipfile.ZipFile(stack,'w') as z:z.writestr('archive/data.pkl',b'\x80\x04\x8c\x02os\x8c\x06system\x93.')
        try:load_checkpoint(stack,sha256_file(stack))
        except pickle.UnpicklingError:loader_tests.append({'case':'STACK_GLOBAL','rejected':True})
        else:raise AssertionError('STACK_GLOBAL accepted')
        write(out/'loader_rejection_checks.json',loader_tests)
        columns=('run_id','variant','seed','best_epoch','mae_pp','bias_pp','attempted_updates','actual_updates','skipped_updates','parameter_count','epoch24_mae_pp','initial_encoder_sha256','initial_model_sha256','checkpoint_sha256')
        csvsave(out/'all9_configs.csv',[{k:x[k] for k in columns} for x in runs])
        write(out/'all9_full_configs.json',all_configs)
        csvsave(out/'all27_best_scale_metrics.csv',best_scales);csvsave(out/'all216_epochs.csv',epoch_rows)
        csvsave(out/'all3_hard_soft_seed_pairs.csv',pairs);csvsave(out/'all3_variant_seed_summary.csv',stats)
        csvsave(out/'all108_best_date_descriptions.csv',best_dates)
        csvsave(out/'all16848_recorded_validation_samples.csv',sample_rows)
        csvsave(out/'constant_baselines.csv',[{'baseline':k,'n':baseline[k]['n'],'mae_pp':100*baseline[k]['mae'],'bias_pp':100*baseline[k]['bias'],
                                              'constant_prediction':baseline[k]['mean_prediction'],'mean_truth':baseline[k]['mean_target']} for k in ('zero','train_mean')])
        for path,digest in list(INPUTS.items()):assert sha256_file(path)==digest,('Input changed during audit',path)
        result={'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_lock_sha256':PINS['SOURCE_LOCK.json'],'plan_sha256':PINS['serial_plan.json'],'protocol_sha256':PINS['RESEARCH_PROTOCOL.json'],
                'runs':runs,'val_ids':[x['id'] for x in val],'train_ids':[x['id'] for x in train],
                'run_count':9,'epochs':216,'best_scale_rows':27,'best_date_rows':108,'recorded_sample_rows':16848,'optimizer_event_rows':2808,
                'paired_seed_results':pairs,'variant_seed_summaries':stats,'constant_baselines':baseline,
                'all_recorded_initial_encoders_match':True,'all3_hard_soft_initial_model_pairs_match':True,
                'initial_state_limit':'These digests were recorded before training; equality is checked, with locked source/CPU initialization evidence. This audit does not recreate initialization or run a model.',
                'coverage_truth':'CPU IEEE754 float32 scalar conversion from native-count-derived train/val metadata; arithmetic audited with Python float64/math.fsum',
                'native_weights_only_failure_count':len(native_failures),'native_weights_only_compatibility':False,
                'loader':'safe_load_external_checkpoint.load_checkpoint(path,expected_sha256), CPU-only explicit five-GLOBAL allowlist; TorchVersion->str',
                'safe_loader_sha256':sha(ROOT/'safe_load_external_checkpoint.py'),'safe_loader_rejection_checks':loader_tests,
                'source_code_sha256':sha(Path(__file__)),'queue_status_sha256':sha(QUEUE),'input_sha256':dict(INPUTS),
                'seed_count_is_field_replication':False,'independent_field_count':1,'dates':4,'validation_is_development':True,
                'new_inference_calls':0,'new_optimizer_calls':0,'GPU_used':False,'test_accessed':False,'R6_accessed':False,
                'original_run_or_checkpoint_modified':False}
        write(out/'audit.json',result)
        report=['# WeedsGalore 九组正式开发训练：独立只读审计','',
            '全部9组通过来源、算术和检查点绑定审计。保留原GPU AMP开发分数；没有重新推理或改选检查点。',
            '原checkpoint包含TorchVersion元数据，9个文件的原生weights_only=True读取均不兼容。此为序列化兼容问题，不代表训练/权重损坏；经原SHA核验后，受限CPU读取器仅将该元数据类解释为str，完整状态与配置/原best记录一致。原包与配置不修改。','',
            '| 形式 | seed | best epoch | 普通MAE/pp | bias/pp | 实际更新/跳步 |','|---|---:|---:|---:|---:|---:|']
        for x in runs:report.append(f"|{x['variant']}|{x['seed']}|{x['best_epoch']}|{x['mae_pp']:.6f}|{x['bias_pp']:+.6f}|{x['actual_updates']}/{x['skipped_updates']}|")
        report+=['','## 三种子描述性汇总','', '| 形式 | MAE均值±样本SD/pp | bias均值±样本SD/pp |','|---|---:|---:|']
        for x in stats:report.append(f"|{x['variant']}|{x['mean_mae_pp']:.6f} ± {x['sample_sd_mae_pp']:.6f}|{x['mean_bias_pp']:+.6f} ± {x['sample_sd_bias_pp']:.6f}|")
        report+=['','## hard/soft 全部配对','', '| seed | soft−hard MAE/pp |','|---|---:|']
        for x in pairs:report.append(f"|{x['seed']}|{x['soft_minus_hard_mae_pp']:+.6f}|")
        report+=['',f"全零常量MAE={100*baseline['zero']['mae']:.6f} pp；训练均值常量MAE={100*baseline['train_mean']['mae']:.6f} pp。三配对的soft MAE均略高于hard；不能据此声称软监督普遍更优，也不将3个种子当3个田块。",'',
            '## 验收范围与边界','',
            '- 每组24完整epoch、312尝试更新；2808条优化器事件逐项核对。每epoch104图、R抽样序列和13个累积组、自然无替换抽样源码均锁定；样本逐epoch顺序没有单独日志，因此不声称由事件表独立重建了每幅图访问顺序。',
            '- 216epoch×26图×3R的16848条已有验证标量逐条核对ID/日期/float32 truth，并用独立math.fsum复算MAE、bias及日期汇总；9个best均为最早严格最优，checkpoint内容、status和completion哈希链匹配。',
            '- 同一公共encoder初始哈希在9组一致；每seed的hard/soft初始整模哈希一致，3种子的head/decoder初始整模哈希不同。此次审计未重新初始化或前向模型。',
            '- 普通MAE使用全部26图自然权重；全部为正覆盖，不构造零覆盖层。四日期n=8/8/8/2，只作描述，仍为一个田块。',
            '- 这是WeedsGalore内部重新训练的方法复现，不是刺萼龙葵checkpoint跨域迁移。验证参与best选择，分数是开发结果；312尝试更新不等于本地2976，不能宣称完全收敛或跨田块统计泛化。',
            '- 本轮没有空间选区/作物代理计算，不据MAE推导防效、节药、损伤或产量。不计算伪重复显著性。所有9组及不利结果保留。','',
            '复算：在publication_campaign运行 `python -X utf8 -B -s audit_external_weeds_training.py`。将新建时间目录，不改原run。安全加载接口为 `safe_load_external_checkpoint.load_checkpoint(path, expected_sha256)`；checksum不符、未知GLOBAL或STACK_GLOBAL均实测拒绝，不改共享torch允许名单。',
            '本次没有读test/R6或解码源图像；原130开发缓存只核字节哈希，CPU读取9个模型checkpoint但不实例化模型、不推理。']
        (out/'review.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
        write(out/'artifact_manifest.json',{'passed':True,'audit_sha256':sha256_file(out/'audit.json'),
            'files_sha256':{p.relative_to(out).as_posix():sha256_file(p) for p in out.iterdir() if p.is_file()},
            'source_code_sha256':sha256_file(Path(__file__)),'loader_sha256':sha256_file(ROOT/'safe_load_external_checkpoint.py')})
        print(json.dumps({'directory':str(out),'passed':True,'run_count':9,'native_weights_only_failures':len(native_failures),
                          'stats':stats,'pairs':pairs,'audit_sha256':sha256_file(out/'audit.json')},ensure_ascii=False))
    except BaseException as error:
        write(out/'failure.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc(),'completed_runs':runs,
                                 'native_weights_only_failures':native_failures,'source_code_sha256':sha256_file(Path(__file__)),
                                 'GPU_used':False,'new_inference_calls':0,'test_accessed':False,'R6_accessed':False});raise

if __name__=='__main__':main()
