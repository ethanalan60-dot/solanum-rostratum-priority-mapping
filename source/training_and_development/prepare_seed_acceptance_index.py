"""Explicit numerical-receipt join, preserving pending runs. No inference or optimization."""
from pathlib import Path
import argparse,datetime,hashlib,json
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',action='append',required=True);p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    plan=ROOT/'local_seed_stability_v1/plan.json';assert sha(plan)=='b96d83d3f5e68addbf5461758f66674b900c8777e8a9f02c50ebcff7735c1168'
    runs=[x['run_id'] for x in read(plan)['runs']];watched=[pin(plan),pin(__file__)];accepted={}
    for path in a.audit:
        path=Path(path).resolve();assert path.is_relative_to(ROOT/'provenance')
        evidence=pin(path);watched.append(evidence);audit=read(path)
        assert audit['passed'] is True and audit['test_or_R6_accessed'] is False
        for row in audit['models']:
            rid=row['run_id'];assert rid in runs and rid not in accepted
            for key,n in {'maps':357,'scalar_pairs':1071,'management_units':2183,'prefix_curve_points':6558,'fixed_prefix_rows':45,'P_policy_rows':45,'auc_rows':9}.items():assert row[key]==n
            assert row['test_or_R6_accessed'] is False and row['no_new_model_forward'] is True and row['no_new_solver'] is True
            receipt=pin(row['receipt']);assert receipt['sha256']==row['receipt_sha256'];watched.append(receipt)
            rec=read(receipt['path']);assert rec['run_id']==rid and rec['all_declared_cases_retained'] and rec['original_epoch_preserved'] and rec['test_or_R6_accessed'] is False
            for directory,name,key in [('export_directory','status.json','export_status_sha256'),('export_directory','artifact_manifest.json','export_manifest_sha256'),('spatial_directory','status.json','spatial_status_sha256'),('spatial_directory','summary.json','spatial_summary_sha256')]:
                current=pin(Path(rec[directory])/name);assert current['sha256']==rec[key];watched.append(current)
            accepted[rid]={'run_id':rid,'completion_receipt':receipt,'numerical_audit':evidence}
    pending=[x for x in runs if x not in accepted]
    if a.require_complete:assert not pending,'All16 independent numerical receipts required'
    for item in watched:assert sha(item['path'])==item['sha256']
    out=ROOT/'provenance'/('seed_numerical_acceptance_index_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    ordered=[accepted[rid] for rid in runs if rid in accepted]
    record={'version':'seed_numerical_acceptance_index_v1','passed_for_registered_models':True,'all16_complete':not pending,
      'accepted_count':len(ordered),'pending_run_ids':pending,'new_models':ordered,
      'readiness_seed_numerical_reviews':[dict(run_id=r['run_id'],**r['numerical_audit']) for r in ordered],
      'source_bindings':watched,'test_or_R6_accessed':False,'new_model_or_solver_calls':0,'execution_authorized':False}
    (out/'index.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(out),'accepted':len(ordered),'pending':len(pending)},ensure_ascii=False))
if __name__=='__main__':main()
