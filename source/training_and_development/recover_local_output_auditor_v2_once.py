"""One explicit validation-only recovery; preserves the failed v1 attempt."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,subprocess
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def dump(p,j):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2)
def require(v,m):
    if not v:raise ValueError(m)
def main():
    a=argparse.ArgumentParser();a.add_argument('--independent-review',type=Path,required=True);a.add_argument('--independent-review-sha256',required=True);a=a.parse_args()
    conf=ROOT/'confirmation_v1';rt=ROOT/'confirmation_runtime'
    old=ROOT/'review_confirmation_local_outputs.py';new=ROOT/'review_confirmation_local_outputs_v2.py'
    require(sha(old)=='2dbf4e09430aae4ecef8ef60d1ba65d70da9080369039777ec8cf630cc38f410','Original reviewer changed')
    require(sha(new)=='b6178cb0cb28e9d1d2fbc4c8f4c8436fdab660492e7047008bca3c4cfb8b7a3f','New reviewer changed')
    require(sha(conf/'FREEZE.json')=='24f8d1605c58df6bd6169b8434a451975abca0c3f8176e9c55c252abf8f3e732','Freeze changed')
    require(sha(conf/'SOURCE_LOCK.json')=='37edeb2586568cb46837f4259dc28a4477a64ed569661480cf07d35900fdf11f','Lock changed')
    require(sha(a.independent_review)==a.independent_review_sha256 and read(a.independent_review)['passed'] is True,'Independent reviewer receipt')
    stage=(conf/'stages/audit_local_outcomes').resolve();claim=(conf/'claims/audit_local_outcomes.json').resolve()
    require(read(stage/'status.json')['state']=='failed' and read(stage/'status.json')['error']=="ValueError('ASCII input order')",'Only diagnosed failed attempt is eligible')
    watched={str(p.resolve()):sha(p) for p in [old,new,conf/'FREEZE.json',conf/'SOURCE_LOCK.json',conf/'contract.json',conf/'checkpoints.json',rt/'all_choices_root_acceptance.json',rt/'audit_local_outcomes_root_review.json',a.independent_review]+[conf/'stages'/n/'artifact_manifest.json' for n in ['local_outcomes','external_outcomes','scalar_outcomes','truth_local','truth_weeds','local_geometry','local_selections_seal','predictions_seal']]}
    lineage=ROOT/'provenance'/datetime.datetime.now().strftime('local_output_audit_v2_recovery_%Y%m%d_%H%M%S_%f');lineage.mkdir(exist_ok=False)
    deststage=lineage/'original_failed_audit_local_outcomes';destclaim=lineage/'original_claim.json'
    for p in (stage,claim,deststage,destclaim):require(p.resolve().is_relative_to(ROOT.resolve()),'Move outside explicit workspace')
    oldfiles={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()};oldclaim=sha(claim)
    record={'scope':'explicit_validation_only_recovery','at':datetime.datetime.now().astimezone().isoformat(),'original_reviewer':pin(old),'new_reviewer':pin(new),'independent_review':pin(a.independent_review),'watched_sha256':watched,'original_failed_stage':str(stage),'original_failed_files_sha256':oldfiles,'original_claim':pin(claim),'archive_stage':str(deststage),'archive_claim':str(destclaim),'reason':'Original reviewer incorrectly required numeric-grid CSV serialization to be ASCII sorted. New reviewer verifies unique complete IDs and sorts an in-memory row view to match the frozen ASCII solver columns.','scientific_sources_outputs_choices_and_tolerances_changed':False,'manual_root_review_completed':True,'automatic_retry':False,'one_new_validation_attempt_authorized':True}
    dump(lineage/'root_recovery_authorization.json',record)
    # Native Python filesystem moves stay inside the checked workspace.
    shutil.move(str(stage),str(deststage));shutil.move(str(claim),str(destclaim))
    require({str(p.relative_to(deststage)):sha(p) for p in deststage.rglob('*') if p.is_file()}==oldfiles and sha(destclaim)==oldclaim,'Archived failure differs')
    for p,h in watched.items():require(sha(p)==h,'Watched source changed')
    grant=read(rt/'audit_local_outcomes_root_review.json');grant['implementation_sha256']=sha(new);grant['root_approved_at']=datetime.datetime.now().astimezone().isoformat();grant['validation_correction_authorization']=pin(lineage/'root_recovery_authorization.json');grant['independent_v2_implementation_review']=pin(a.independent_review);grant['supersedes_validation_grant']=pin(rt/'audit_local_outcomes_root_review.json')
    gp=rt/'audit_local_outcomes_v2_root_review.json';dump(gp,grant)
    dump(rt/'LOCAL_OUTPUT_AUDIT_V2_RECOVERY_CLAIM.json',{'at':grant['root_approved_at'],'root_authorization':pin(lineage/'root_recovery_authorization.json'),'grant':pin(gp),'no_automatic_retry':True})
    argv=[__import__('sys').executable,'-X','utf8','-B','-s',str(new),'--root-review',str(gp),'--choice-review',str(rt/'all_choices_root_acceptance.json')]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTORCH_NVML_BASED_CUDA_CHECK='1',TEMP=str(rt/'temp'),TMP=str(rt/'temp'))
    with (rt/'audit_local_outcomes_v2.stdout.log').open('xb') as out,(rt/'audit_local_outcomes_v2.stderr.log').open('xb') as err:
        proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    launch={'pid':proc.pid,'launched_at':datetime.datetime.now().astimezone().isoformat(),'argv':argv,'root_review':pin(gp),'recovery_lineage':str(lineage),'automatic_retry':False}
    dump(rt/'audit_local_outcomes_v2_launch.json',launch);print(json.dumps(launch,ensure_ascii=False))
if __name__=='__main__':main()
