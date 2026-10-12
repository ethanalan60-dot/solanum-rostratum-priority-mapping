"""Artifact-only CPU audit of the single bounded normalization GPU diagnostic."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
import datetime,gc,hashlib,json,math,shutil,traceback
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parent
VERSION=ROOT/'local_norm_regression_v1'
DIAG=VERSION/'gpu_diagnostics/20260916_065646_321019'
SEAL_SHA='a347c2f9bcc68b3e4dcce2986d90dfcd156c9c74bb0abd978bedbed77165e64a'
AUTH_SHA='fdbf6c54d4853ca4504924a3ab026745c47907a42f525653c1a19d7d65e94e7e'
LOCK_SHA='0521b8d72d4d156e6ca3dd3f12cd25a3d03f1b2b3460e52ba614a0410c4e49b5'
WATCH={}
def rawsha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def sha(p):p=Path(p).resolve();h=rawsha(p);WATCH[str(p)]=h;return h
def read(p):sha(p);return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):sha(p);return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines()]
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def finite_tree(x):
    if isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values():finite_tree(v)
    elif isinstance(x,list):
        for v in x:finite_tree(v)

def main():
    out=ROOT/'provenance'/datetime.datetime.now().strftime('local_norm_GPU_artifact_review_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True);shutil.copy2(__file__,out/Path(__file__).name)
    arms=[]
    try:
        assert sha(DIAG/'pre_execution_seal.json')==SEAL_SHA
        assert sha(DIAG/'root_diagnostic_review.json')==AUTH_SHA
        assert sha(VERSION/'SOURCE_LOCK.json')==LOCK_SHA
        seal=read(DIAG/'pre_execution_seal.json');protocol=read(DIAG/'pre_execution_protocol.json')
        auth=read(DIAG/'root_diagnostic_review.json');lock=read(VERSION/'SOURCE_LOCK.json')
        status=read(DIAG/'status.json');summary=read(DIAG/'summary.json');claim=read(DIAG/'EXECUTION_CLAIM.json')
        manifest=read(DIAG/'artifact_manifest.json');cpu=read(VERSION/'CPU_READINESS.json')
        assert status['state']==summary['state']=='completed' and summary['passed'] is True
        assert not (DIAG/'failure.json').exists()
        assert status['summary_sha256']==sha(DIAG/'summary.json')
        assert auth['passed'] and auth['execution_scope']=='local_norm_regression_v1_gpu_diagnostic'
        assert auth['pre_execution_seal_sha256']==SEAL_SHA and not auth.get('test_or_R6_authorized',False)
        assert claim['root_review_sha256']==summary['root_review_sha256']==AUTH_SHA
        assert summary['binding']==protocol['binding']==seal['binding']
        for name,key in [('SOURCE_LOCK.json','source_lock_sha256'),('serial_plan.json','plan_sha256'),('RESEARCH_PROTOCOL.json','protocol_sha256'),('CPU_READINESS.json','CPU_readiness_sha256')]:
            assert sha(VERSION/name)==summary['binding'][key]==auth[key]
        assert seal['script_sha256']==sha(VERSION/'check_norm_gpu.py')
        for rel,digest in seal['files_sha256'].items():assert sha(DIAG/rel)==digest
        for rel,digest in lock['source_files_sha256'].items():assert sha(VERSION/rel)==sha(DIAG/'source'/rel)==digest
        for p,digest in lock['external_inputs_sha256'].items():assert sha(p)==digest
        assert manifest['pre_execution_seal_sha256']==SEAL_SHA
        paths={p.relative_to(DIAG).as_posix() for p in DIAG.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
        assert paths==set(manifest['files_sha256'])
        for rel,digest in manifest['files_sha256'].items():assert sha(DIAG/rel)==digest
        assert datetime.datetime.fromisoformat(seal['created_utc'])<datetime.datetime.fromisoformat(claim['created_utc'])<datetime.datetime.fromisoformat(summary['completed_utc'])
        assert protocol['conditions']==['identity','public_norm'] and protocol['R_per_microbatch']==[2,4]
        assert protocol['optimizer_attempts_per_condition']==protocol['expected_actual_updates_per_condition']==1
        assert protocol['seed']==17 and protocol['LR']==1e-4 and protocol['microbatches']==2 and protocol['microbatch_size']==4
        train=read(ROOT/'local_data_v2/train.json')
        expected=[r for r in train if r['coverage']>0][:4]+[r for r in train if r['coverage']==0][:4]
        assert protocol['train_records']==expected and protocol['train_ids']==[r['id'] for r in expected]
        for r in expected:assert sha(r['cache'])==r['cache_sha256']
        assert summary['conditions']==2 and summary['total_attempts']==summary['total_actual_updates']==2
        assert summary['paired_encoder_and_head_equal']
        for key in ('formal_runs_started','val_pixels_read','test_accessed','R6_accessed'):assert summary[key] is False
        assert [x['condition'] for x in summary['outcomes']]==['identity','public_norm']
        for item in summary['outcomes']:
            condition=item['condition'];folder=DIAG/condition
            result=read(folder/'result.json');initial=read(folder/'initialization.json')
            assert result==item and initial==item['initialization']==cpu['initializations'][condition]
            assert item['attempts']==item['actual']==1 and item['skips']==0
            assert item['strict_state_equal'] and item['CPU_FP32_logits_bitwise_equal'] and item['max_absolute_reload_difference']==0
            assert item['train_ids']==protocol['train_ids'] and item['train_only']
            assert item['val_pixels']==item['test_accessed']==item['R6_accessed']==False
            events=lines(folder/'optimizer_events.jsonl');assert events==item['events'] and len(events)==1
            event=events[0];finite_tree(event)
            assert event['epoch']==0 and event['group']==1 and event['attempt']==1 and event['group_samples']==event['group_denominator']==8
            assert event['status']=='updated' and event['amp_enabled'] and event['optimizer_step_executed'] and event['all_unscaled_gradients_finite'] and not event['skipped']
            assert event['actual_updates_total']==1 and event['skipped_updates_total']==0 and event['consecutive_overflows']==0
            obs=lines(folder/'observations.jsonl');assert len(obs)==2 and [x['R'] for x in obs]==[2,4]
            assert [x['ids'] for x in obs]==[protocol['train_ids'][:4],protocol['train_ids'][4:]]
            for record in obs:
                finite_tree(record)
                assert record['phase']=='train' and record['epoch']==0 and record['group']==1 and len(record['logits'])==4
                assert set(record['features'])=={'pre_pool','post_pool','post_norm'}
            grad=lines(folder/'gradient_observations.jsonl');assert len(grad)==1
            record=grad[0];finite_tree(record)
            assert record['last_microbatch_ids']==protocol['train_ids'][4:] and 'ids' not in record
            assert record['accumulated_group_ids']==protocol['train_ids'] and record['accumulated_group_samples']==8
            assert record['R']==4 and 'entire accumulated' in record['gradient_scope']
            assert all(x['all_finite'] for x in record['gradients'].values())
            assert record['gradients']['pool_norm']['tensors']==(0 if condition=='identity' else 2)
            checkpoint=folder/'diagnostic.pt';assert sha(checkpoint)==item['checkpoint_sha256']
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert set(payload)=={'model','condition','seed','diagnostic_only','train_ids'}
            assert payload['condition']==condition and payload['seed']==17 and payload['diagnostic_only'] and payload['train_ids']==protocol['train_ids']
            tensors=payload['model'];assert tensors and all(v.device.type=='cpu' and torch.isfinite(v).all() for v in tensors.values())
            assert tuple(tensors['head.weight'].shape)==(1,768)
            norm_keys=[k for k in tensors if k.startswith('pool_norm.')]
            assert set(norm_keys)==(set() if condition=='identity' else {'pool_norm.weight','pool_norm.bias'})
            parameter_count=sum(v.numel() for v in tensors.values());assert parameter_count==initial['parameter_count']
            path=folder/'train8_CPU_FP32_logits.npy';sha(path);logits=np.load(path,allow_pickle=False)
            assert logits.shape==(8,) and logits.dtype==np.float32 and np.isfinite(logits).all()
            assert sha(checkpoint)==item['checkpoint_sha256']
            arms.append({'condition':condition,'attempts':1,'actual_updates':1,'AMP_skips':0,
                'checkpoint_sha256':item['checkpoint_sha256'],'parameter_count':parameter_count,
                'result_sha256':sha(folder/'result.json'),'optimizer_events_sha256':sha(folder/'optimizer_events.jsonl'),
                'CPU_reference_logits_sha256':sha(path),'strict_reload_recorded_equal':True,
                'checkpoint_CPU_finite_tensors':True,'new_forward_performed_by_auditor':False})
            del payload,tensors,logits;gc.collect()
        assert not torch.cuda.is_initialized()
        for p,digest in WATCH.items():assert rawsha(p)==digest,('Input changed',p)
        receipt={'passed':True,'execution_scope':'local_norm_regression_v1_production_readiness_review',
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'diagnostic_directory':str(DIAG),
            'diagnostic_summary_sha256':sha(DIAG/'summary.json'),'diagnostic_artifact_manifest_sha256':sha(DIAG/'artifact_manifest.json'),
            'pre_execution_seal_sha256':SEAL_SHA,'root_diagnostic_review_sha256':AUTH_SHA,
            **{k:summary['binding'][k] for k in ('source_lock_sha256','plan_sha256','protocol_sha256','CPU_readiness_sha256')},
            'conditions':arms,'train_ids':protocol['train_ids'],'total_attempts':2,'total_actual_updates':2,'total_skips':0,
            'production_readiness_recommendation':'Bounded GPU workflow passed; root may separately authorize the frozen four-run queue.',
            'production_execution_authorized':False,'new_forward_or_training_by_auditor':False,'GPU_used_by_auditor':False,
            'val_pixels_read_by_auditor':False,'test_accessed':False,'R6_accessed':False,
            'reload_evidence_scope':'Sealed diagnostic recorded strict full state equality and CPU FP32 logits equality; auditor independently verified bindings and finite saved tensors/reference logits but did not rerun forward.',
            'gradient_R_scope':'Gradient record R=4 is the final microbatch only; accumulated group had R2 then R4. Join complete input traces in future mechanism analysis.',
            'source_code_sha256':rawsha(__file__),'input_sha256':WATCH}
        write(out/'review.json',receipt)
        (out/'review.md').write_text('# Bounded normalization GPU diagnostic: independent artifact audit\n\n'
            '**PASS for production readiness; formal production still needs separate root authorization.**\n\n'
            'Both identity and public-norm arms performed exactly one finite AMP/AdamW update on the same eight declared training windows; two attempts, two actual updates, zero skips. '
            'Source/protocol/pre-execution seal/root authorization/input/checkpoint bindings match. Initial encoder/head/RNG hashes match the frozen CPU receipt. '
            'The sealed diagnostic recorded strict CPU checkpoint-state and all-eight FP32-logit reload equality. This audit independently loaded the two SHA-verified checkpoints on CPU, checked named finite tensor shapes, and decoded only their saved reference-logit arrays; it ran no model forward.\n\n'
            'Gradient observations refer to the accumulated eight-window group; their R=4 field belongs only to the final microbatch. The group was R2 then R4, so future mechanism interpretation must join the full input trace. '
            'No validation/test pixels, new training or GPU operation occurred in this audit. The bounded smoke establishes workflow viability, not a scientific treatment effect or full-run convergence.\n',encoding='utf-8')
        write(out/'artifact_manifest.json',{'passed':True,'file_sha256':{p.name:rawsha(p) for p in out.iterdir() if p.is_file()}})
        print(json.dumps({'passed':True,'directory':str(out),'review_sha256':rawsha(out/'review.json'),'conditions':arms},ensure_ascii=False))
    except BaseException as error:
        write(out/'failure.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc(),'completed_conditions':arms,'input_sha256':WATCH});raise

if __name__=='__main__':main()
