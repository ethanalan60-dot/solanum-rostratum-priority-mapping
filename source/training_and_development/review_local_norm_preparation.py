"""Independent static/source-binding review; does not import training/model code."""
from pathlib import Path
import ast,datetime,hashlib,json,shutil,traceback
ROOT=Path(__file__).resolve().parent
VERSION=ROOT/'local_norm_regression_v1'
PINS={'SOURCE_LOCK.json':'0521b8d72d4d156e6ca3dd3f12cd25a3d03f1b2b3460e52ba614a0410c4e49b5',
      'serial_plan.json':'76547e7481cc756964dd8e6d593d6d39f744f6754d6b3dc47e4ce80292c29e10',
      'RESEARCH_PROTOCOL.json':'bb1c8911dfed80e557ffc7bf33680760250e1c852b54e153a90970b37280ba63',
      'CPU_READINESS.json':'46577fc29a7935bfc6b95e1fd534eb9981f7d095343a6d01316c6b51de016fcd',
      'DIAGNOSTIC_DESIGN.json':'9873ddd21d975d42eaeb99d7d13fac16334dc7502711933a7aa575079fd90bf6'}
WATCH={}
def rawsha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def sha(p):p=Path(p).resolve();h=rawsha(p);WATCH[str(p)]=h;return h
def read(p):sha(p);return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def code(p):sha(p);return Path(p).read_text(encoding='utf-8-sig')
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    out=ROOT/'provenance'/datetime.datetime.now().strftime('local_norm_static_review_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True);shutil.copy2(__file__,out/Path(__file__).name)
    try:
        for name,digest in PINS.items():assert sha(VERSION/name)==digest,name
        lock=read(VERSION/'SOURCE_LOCK.json');cpu=read(VERSION/'CPU_READINESS.json')
        plan=read(VERSION/'serial_plan.json');protocol=read(VERSION/'RESEARCH_PROTOCOL.json');design=read(VERSION/'DIAGNOSTIC_DESIGN.json')
        assert not lock['GPU_authorized'] and lock['future_GPU_requires_separate_bound_root_review']
        for rel,h in lock['source_files_sha256'].items():
            p=(VERSION/rel).resolve();assert p.is_relative_to(VERSION.resolve()) and sha(p)==h,rel
        for p,h in lock['external_inputs_sha256'].items():assert sha(p)==h
        assert cpu['passed'] and cpu['check_count']==len(cpu['checks'])==60 and all(x['passed'] for x in cpu['checks'])
        for key in ('GPU_used','test_accessed','R6_accessed','formal_runs_started','val_pixels_read'):assert cpu[key] is False
        assert lock['CPU_readiness_sha256']==PINS['CPU_READINESS.json']
        for name,digest in cpu['source_sha256'].items():assert digest==lock['source_files_sha256'][name]==sha(VERSION/name)
        assert sha(VERSION/'expected_trace.json')==cpu['expected_trace_sha256']
        assert cpu['development_manifest_sha256']==lock['development_manifest_sha256']
        first,last=cpu['initializations']['identity'],cpu['initializations']['public_norm']
        for key in ('encoder_sha256','head_sha256','rng_before_intervention','rng_after_intervention'):assert first[key]==last[key]
        assert first['rng_before_intervention']==first['rng_after_intervention'] and last['rng_before_intervention']==last['rng_after_intervention']
        assert last['parameter_count']-first['parameter_count']==1536
        assert len(cpu['actual_train_cache_ids_read'])==len(set(cpu['actual_train_cache_ids_read']))==cpu['actual_train_cache_count']==8
        train=read(ROOT/'local_data_v2/train.json');val=read(ROOT/'local_data_v2/val.json')
        assert len(train)==985 and len(val)==357
        assert set(cpu['actual_train_cache_ids_read']).issubset({x['id'] for x in train})
        assert not (set(cpu['actual_train_cache_ids_read'])&{x['id'] for x in val})
        trace=read(VERSION/'expected_trace.json')
        assert len(trace['epochs'])==24 and [x['epoch'] for x in trace['epochs']]==list(range(1,25))
        assert all(x['samples']==985 and x['microbatches']==247 for x in trace['epochs'])
        assert len(plan['runs'])==4 and plan['attempts_per_epoch']==124 and plan['attempts_per_run']==2976 and plan['maximum_optimizer_attempts']==11904
        assert [(x['lr'],x['condition']) for x in plan['runs']]==[(lr,c) for lr in (1e-4,3e-4) for c in ('identity','public_norm')]
        assert plan['settings']['seed']==17 and plan['settings']['epochs']==24 and plan['settings']['batch_size']==4 and plan['settings']['accumulation_steps']==2
        assert plan['design_sha256']==protocol['design_sha256']==PINS['DIAGNOSTIC_DESIGN.json']
        original=ast.parse(code(ROOT/'train_common.py'));frozen=ast.parse(code(VERSION/'frozen_training_core.py'))
        extraction=read(VERSION/'extraction.json');same=[]
        assert len(extraction['excerpts'])==11
        for name in extraction['excerpts']:
            a=next(x for x in original.body if getattr(x,'name',None)==name)
            b=next(x for x in frozen.body if getattr(x,'name',None)==name)
            assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
            same.append(name)
        source={name:code(VERSION/name) for name in ('diagnostic_core.py','train_diagnostic.py','run_serial.py','check_norm_gpu.py','check_readiness.py','seal_version.py')}
        for name,value in source.items():ast.parse(value)
        assert "last_microbatch_ids" in source['diagnostic_core.py'] and 'accumulated_group_ids' in source['diagnostic_core.py']
        assert "'check_norm_gpu.py'" in source['run_serial.py']
        assert "nn.LayerNorm(768,eps=1e-6)" in source['diagnostic_core.py']
        assert "super().__init__('convnextv2_tiny',pretrained=pretrained)" in source['diagnostic_core.py']
        assert "frozen.train_epoch(" in source['train_diagnostic.py'] and "frozen.evaluate(" in source['train_diagnostic.py']
        assert "if metrics['selection_balanced_mae']<best:" in source['train_diagnostic.py']
        assert "local_norm_regression_v1_gpu_diagnostic" in source['check_norm_gpu.py']
        assert "review.get('pre_execution_seal_sha256')" in source['check_norm_gpu.py']
        assert "weights_only=True" in source['check_norm_gpu.py'] and "torch.equal(reference,repeated)" in source['check_norm_gpu.py']
        for p,h in WATCH.items():assert rawsha(p)==h,('Input changed during review',p)
        result={'passed':True,'execution_scope':'local_norm_regression_v1_static_preparation_review',
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_lock_sha256':PINS['SOURCE_LOCK.json'],'plan_sha256':PINS['serial_plan.json'],
            'protocol_sha256':PINS['RESEARCH_PROTOCOL.json'],'CPU_readiness_sha256':PINS['CPU_READINESS.json'],
            'design_sha256':PINS['DIAGNOSTIC_DESIGN.json'],'source_code_sha256':rawsha(__file__),
            'reviewed_source_sha256':lock['source_files_sha256'],'unchanged_original_AST_names':same,
            'CPU_readiness_checks_reviewed':60,'CPU_readiness_independently_rerun':False,
            'initialization_pair_checks':{'encoder_equal':True,'head_equal':True,'RNG_equal':True,'additional_norm_parameters':1536},
            'planned_runs':4,'planned_epochs_per_run':24,'planned_max_attempts':11904,
            'resolved_review_points':['Gradient log distinguishes last_microbatch_ids from accumulated_group_ids/samples.',
                'Serial conflict list includes separate check_norm_gpu.py diagnostic entry.'],
            'static_findings':['Only public affine pooled LayerNorm is introduced; unchanged valid pooling and scalar Linear/loss remain.',
                'CPU readiness verifies identity state/logits/gradients and explicit norm formula; no claim of GPU/AMP equivalence yet.',
                'Shared sampler/DataLoader generator and independent augmentation/factor RNG remain; all24 expected trace hashes are enforced during production.',
                'Detached observers and scaler proxy add no random draw or training forward; whole accumulated gradient group scope is explicit.',
                'Earliest strictly smallest equal-stratum/equal-R validation MAE, AdamW/cosine and original AMP guard are unchanged.',
                'Run directories and queue/GPU claims refuse overwrite; errors after claimed execution are preserved and stop without automatic retry.',
                'Separate fixed-train8 GPU diagnostic requires another bound scope/seal and cannot authorize the formal queue.',
                'No environment install/upgrade, test/R6 loader or score-dependent extension is present.'],
            'limitations':['This is a source/readiness review, not a completed GPU workflow or scientific result.',
                'GPU diagnostic success and root authorization are still separate execution gates.',
                'Norm operation and its pretrained affine parameters form one intervention; outcomes cannot isolate their individual contributions.',
                'Same local scene/single seed remains mechanism diagnosis, not independent-field causal evidence.'],
            'source_sha256':WATCH,'new_model_inference':False,'training':False,'GPU_used':False,'test_accessed':False,'R6_accessed':False,
            'production_execution_authorized':False,'diagnostic_GPU_execution_authorized':False,'modified_version_sources':False}
        write(out/'review.json',result)
        (out/'review.md').write_text('# Local pooled-normalization preparation: independent static review\n\n'
            '**PASS for frozen preparation; this receipt does not authorize GPU execution.**\n\n'
            'The accepted four-cell design and final source/plan/protocol/60-check CPU receipt hashes match. Eleven original training definitions are independently AST-identical. '
            'Paired encoder/head initialization and postconstruction RNG hashes agree; treatment adds only the 1,536 public affine LayerNorm parameters after unchanged valid-area pooling and before the original scalar head. '
            'The sampler/DataLoader generator arrangement, augmentation/R sequence, original SmoothL1, stratified selector, AdamW/cosine and AMP guards are preserved. '
            'All 24 trace hashes are checked during each production run.\n\n'
            'Two review points were resolved before freeze: gradient records explicitly name last-microbatch IDs and complete accumulated-group IDs, and the serial queue recognizes the new GPU diagnostic as a conflicting process. '
            'The fixed8-train two-arm one-update diagnostic has its own authorization scope and pre-execution seal, strict CPU state/logit reload, and retained failure outputs. '
            'Readiness was read and hash-checked; it was not rerun. No model imports, inference, optimizer, GPU, test/R6 access or source edits occurred in this review.\n\n'
            'Pending: root-reviewed bounded GPU diagnostic, then separate formal four-run authorization. No training outcome or improvement is claimed.\n',encoding='utf-8')
        write(out/'artifact_manifest.json',{'passed':True,'file_sha256':{p.name:rawsha(p) for p in out.iterdir() if p.is_file()}})
        print(json.dumps({'passed':True,'directory':str(out),'review_sha256':rawsha(out/'review.json'),
                          'source_lock_sha256':PINS['SOURCE_LOCK.json'],'CPU_readiness_sha256':PINS['CPU_READINESS.json']},ensure_ascii=False))
    except BaseException as error:
        write(out/'failure.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc(),'source_sha256':WATCH});raise

if __name__=='__main__':main()
