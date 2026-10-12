"""Record root's completed source review and exact evidence before one external queue."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess

ROOT=Path(__file__).resolve().parent
EXT=ROOT/'external_weeds_v1'
DIAG=ROOT/'external_weeds_gpu_diagnostics/20260916_053201_551221'
STATIC=EXT/'independent_review_20260916_052858_559152/review.json'
CPU=EXT/'cpu_checks/20260916_052057_016479/result.json'
PINS={'source_lock_sha256':('SOURCE_LOCK.json','3ff901160f73f70df2ad601f3a40b49b2e8e76d5b83942c5c03bcf584da1ddfe'),
      'plan_sha256':('serial_plan.json','5225bde2861c6c046e94b514eadfc398523f776c478b007ca6ae8c64639d0596'),
      'protocol_sha256':('RESEARCH_PROTOCOL.json','5b20a0856761470575538dad305486924c63da2ca343811f37fc63b29a6b962d')}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu-review',required=True)
    args=parser.parse_args();review_path=Path(args.gpu_review).resolve()
    static=read(STATIC);cpu=read(CPU);gpu_review=read(review_path)
    assert static['passed'] and cpu['passed'] and cpu['count']==59 and gpu_review['passed']
    for key,(filename,digest) in PINS.items():
        assert sha(EXT/filename)==digest and static[key]==digest
    source=read(EXT/'SOURCE_LOCK.json')
    for path,digest in source['source_files_sha256'].items():assert sha(EXT/path)==digest
    for path,digest in source['external_inputs_sha256'].items():assert sha(path)==digest
    assert sha(CPU)==static['cpu_result_sha256']
    summary=read(DIAG/'summary.json');manifest=read(DIAG/'artifact_manifest.json')
    assert Path(gpu_review['audited_directory']).resolve()==DIAG.resolve()
    for filename,key in [('summary.json','summary_sha256'),('artifact_manifest.json','artifact_manifest_sha256'),
                         ('pre_execution_seal.json','pre_execution_seal_sha256')]:
        assert sha(DIAG/filename)==gpu_review[key]
    assert summary['passed'] and summary['total_attempts']==summary['actual_updates']==3
    assert summary['skipped_updates']==0 and summary['validation_model_calls']==0
    assert not summary['test_accessed'] and not summary['R6_accessed']
    for name,digest in manifest['file_sha256'].items():assert sha(DIAG/name)==digest
    for row in summary['results']:
        assert row['actual']==1 and row['skipped']==0 and row['all_gradients_finite']
        assert row['checkpoint_replay']['strict_loading'] and row['checkpoint_replay']['FP32_logits_bitwise_equal']
    assert summary['initial_encoders_identical'] and summary['hard_soft_initial_models_identical']
    assert not (EXT/'runs').exists() and not (EXT/'SERIAL_EXECUTION_CLAIM.json').exists()
    snapshot=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],
        check=True,capture_output=True,text=True).stdout
    assert not any('python' in line.lower() for line in snapshot.splitlines()),'Inspect existing GPU Python processes before queue launch'
    result={'passed':True,'execution_scope':'external_weeds_v1_gpu','created_local':datetime.datetime.now().astimezone().isoformat(),
        'reviewer':'root','basis':'Root read the complete locked training loop/core/queue and bounded GPU diagnostic, then checked independent source and numerical receipts.',
        **{key:value[1] for key,value in PINS.items()},'test_or_R6_authorized':False,
        'static_review':str(STATIC),'static_review_sha256':sha(STATIC),
        'cpu_result':str(CPU),'cpu_result_sha256':sha(CPU),'gpu_diagnostic_directory':str(DIAG),
        'gpu_summary_sha256':sha(DIAG/'summary.json'),'gpu_artifact_manifest_sha256':sha(DIAG/'artifact_manifest.json'),
        'gpu_independent_review':str(review_path),'gpu_independent_review_sha256':sha(review_path),
        'gpu_process_preflight':snapshot.splitlines(),'root_receipt_source_sha256':sha(__file__),
        'scope':'One serial execution of all nine frozen 24-epoch development configurations through run_serial.py only; fresh pinned public initialization per run.',
        'limits':['The GPU diagnostic is three one-update train-only checks, not convergence or predictive performance.',
                  'Remaining allocator memory before diagnostic process exit was recorded; no allocator-zero claim.',
                  'No formal initialization from diagnostic checkpoints; no test/R6, decision-based checkpoint choice, automatic retry or source edits.',
                  'Single external field/four dates; ordinary MAE on all-positive frames; no field-replication or actual agronomic efficacy claim.']}
    out=EXT/'root_gpu_review.json'
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'passed':True,'review_receipt':str(out),'sha256':sha(out),'training_started':False},ensure_ascii=False))
if __name__=='__main__':main()
