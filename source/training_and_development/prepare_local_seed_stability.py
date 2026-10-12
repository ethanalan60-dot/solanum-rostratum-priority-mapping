"""Freeze a bounded seed-only extension using the unchanged, previously checked trainer."""
from pathlib import Path
import datetime,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'local_seed_stability_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    old=read(ROOT/'common_queue_plan_v2.json')
    assert sha(ROOT/'common_queue_plan_v2.json')=='faefb03b14e4c44360ded4d065ffbe873d4e8512139a68d56e15ee90a2185a95'
    for name,digest in old['source_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in old['manifest_sha256'].items():assert sha(ROOT/'local_data_v2'/name)==digest
    oldseg=[x for x in old['runs'] if x['variant'] in ('hard_seg','soft_seg')]
    assert len(oldseg)==8
    runs=[]
    for seed in (29,43):
        for oldrun in oldseg:
            lr=oldrun['lr'];encoder=oldrun['encoder'];variant=oldrun['variant'];label='lr1e4' if lr==1e-4 else 'lr3e4'
            rid=f'dev_seed_v1_{label}_{encoder}_{variant}_L256_s{seed}'
            assert not (ROOT/'runs'/rid).exists()
            status=read(ROOT/'runs'/oldrun['run_id']/'status.json');assert status['status']=='completed' and status['executed_epochs']==24
            runs.append({'run_id':rid,'encoder':encoder,'variant':variant,'lr':lr,'seed':seed,'paired_seed17_run_id':oldrun['run_id']})
    assert len(runs)==16 and len({x['run_id'] for x in runs})==16
    protocol={'version':'local_seed_stability_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'purpose':'Complete seed29/43 repeats of every existing eight-cell segmentation configuration for the local R-by-M decision analysis. Descriptive development extension after inspecting seed17 and external development outcomes, not retrospective preregistration.',
        'fixed_matrix':{'encoder':['resnet18','convnextv2_tiny'],'lr':[1e-4,3e-4],'variant':['hard_seg','soft_seg'],'new_seeds':[29,43],'existing_seed':17},
        'new_runs':16,'combined_segmentation_runs':24,'epochs_per_run':24,'optimizer_attempts_per_run':2976,'maximum_new_optimizer_attempts':47616,
        'fixed_training':'Use original train_common.py byte-for-byte; batch4 accumulation2, AdamW WD1e-4, cosine24, balanced replacement985/epoch, original scale/colour/flips and guarded AMP. No new architecture, loss, sampler or learning-rate tuning.',
        'selection':'Original validation equal-zero/positive-stratum MAE averaged R1/2/4; strict improvement, earliest exact tie; no spatial metric selection.',
        'spatial_after_training':'Every completed segmentation model exported at original best and evaluated with the existing fixed domain/geographic geometry/R1,2,4/M1,2,4/budgets/both policies/fullprefix AUC and uniform numerical bound audit. Record CPU-GPU differences; never reselect epoch.',
        'scope':'Only segmentation maps enter spatial seed-stability analysis. Scalar regression and normalization diagnostics retain their original single-seed scope; do not imply new regression replication.',
        'stopping':'Complete fixed16 or stop at first execution failure with preserved evidence. No optional continuation based on favorable or unfavorable scientific scores. Subsequent work is audit, prespecified aggregation and frozen confirmation, not open-ended tuning.',
        'reporting':'Report all24 configurations and each seed; pair hard/soft and R/M within identical seed/configuration. Seed variation is not independent-field uncertainty; no field-significance claim.',
        'test_accessed':False,'R6_accessed':False,'confirmation_before_freeze':False,
        'resource':'CPU2 threads; GPU fraction0.16; microbatch4 accumulation2; pause0.05; cachesTEMP onG; preserve unrelated processes.',
        'failure':'No overwrite/resume/retry, permanent queue claim, source/cache mismatch blocks. Any justified execution repair gets a new version and preserved failed artifacts.'}
    settings={'epochs':24,'batch_size':4,'accumulation_steps':2,'weight_decay':1e-4,'label_grid':256,'device':'cuda','threads':2,'gpu_memory_fraction':.16,'step_pause':.05,'scale_augmentation':True,'amp':True,'sampler':'balanced','freeze_batchnorm':False}
    OUT.mkdir(exist_ok=False);write(OUT/'protocol.json',protocol)
    write(OUT/'plan.json',{'version':'local_seed_stability_v1','runs':runs,'settings':settings,'protocol_sha256':sha(OUT/'protocol.json'),'original_common_plan_sha256':sha(ROOT/'common_queue_plan_v2.json')})
    inputs={str(ROOT/name):sha(ROOT/name) for name in old['source_sha256']}
    inputs.update({str(ROOT/'local_data_v2'/name):digest for name,digest in old['manifest_sha256'].items()})
    for p in [ROOT/'cache/torch/hub/checkpoints/resnet18-f37072fd.pth',ROOT/'vendor/convnextv2/weights/convnextv2_tiny_1k_224_ema.pt']:
        inputs[str(p)]=sha(p)
    for split,n in [('train',985),('val',357)]:
        rows=read(ROOT/'local_data_v2'/(split+'.json'));assert len(rows)==n
        for row in rows:
            assert row['split']==split and sha(row['cache'])==row['cache_sha256']
            inputs[row['cache']]=row['cache_sha256']
    snapshot=OUT/'original_source';snapshot.mkdir()
    for name in old['source_sha256']:shutil.copy2(ROOT/name,snapshot/name)
    shutil.copy2(ROOT/'common_queue_plan_v2.json',snapshot/'common_queue_plan_v2.json')
    shutil.copy2(__file__,OUT/'preparation_source.py')
    write(OUT/'INPUT_LOCK.json',{'version':'local_seed_stability_v1','external_inputs_sha256':inputs,'source_training_unchanged':True,'formal_training_started':False,'local_seed17_runs':[x['run_id'] for x in oldseg]})
    print(json.dumps({'prepared':str(OUT),'runs':16,'protocol_sha256':sha(OUT/'protocol.json'),'plan_sha256':sha(OUT/'plan.json'),'input_lock_sha256':sha(OUT/'INPUT_LOCK.json'),'training_started':False},ensure_ascii=False))
if __name__=='__main__':main()
