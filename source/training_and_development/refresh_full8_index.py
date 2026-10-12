"""Refresh human-facing result index after independent full8 acceptance."""
from pathlib import Path
import argparse,datetime,hashlib,json
ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture'
FULL=PA/'common_management_reports/full8_20260916_050927_560091'
BOUND=PA/'bca_20260916_050803_135549'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--central-review',required=True)
    args=parser.parse_args();review_path=Path(args.central_review).resolve();review=read(review_path)
    assert review['passed'],'Central review must explicitly pass before indexing'
    assert Path(review['audited_report_directory']).resolve()==FULL.resolve()
    assert review['table_rows_verified']==720 and review['paired_rows_verified']==360
    state=read(FULL/'status.json');figures=read(FULL/'final_artifact_manifest.json')
    assert state['state']=='tables_completed' and state['rows']==720 and state['pairs']==360
    assert figures['table_snapshot_preserved']
    for key,h in state['file_sha256'].items():assert sha(FULL/key)==h
    for key,h in figures['generated_file_sha256'].items():assert sha(FULL/key)==h
    summary=read(BOUND/'summary.json');assert summary['cases']==360 and summary['all_existing_final_ids_unchanged']
    train=read(ROOT/'latest_common_development_report.json');assert train['state_counts']=={'completed':12}
    queue=read(ROOT/'common_queue_status.json');assert queue['status']=='completed'
    auc=PA/'common_prefix_auc_full8_20260916_052212_839933'
    auc_review_path=PA/'prefix_auc_independent_20260916_052458_065751/review.json'
    auc_review=read(auc_review_path)
    assert auc_review['passed'] and auc_review['model_curves_checked']==72 and auc_review['hard_soft_pairs_checked']==36
    assert sha(auc/'all72_model_prefix_auc.csv')==auc_review['all72_model_prefix_auc_sha256']
    assert sha(auc/'all36_hard_soft_auc_pairs.csv')==auc_review['all36_hard_soft_auc_pairs_sha256']
    index=read(PA/'latest_results.json')
    index.update(schema_version=3,updated_local=datetime.datetime.now().astimezone().isoformat(),
       scope='All12 common development training runs and all8 spatial configurations complete; one scene/seed17; external training and sealed confirmation incomplete',
       current_training_status='../common_queue_status.json',
       current_common_report_directory=FULL.relative_to(PA).as_posix(),
       current_common_training_report='../latest_common_development_report.json',
       common_completed_comparison_configurations=8,common_declared_spatial_configurations=8,
       common_completed_spatial_models_including_recovery=8,common_central_report_models=8,
       common_independent_review=review_path.relative_to(PA).as_posix(),
       common_uniform_bound_output=BOUND.relative_to(PA).as_posix(),
       common_uniform_bound_protocol='RM_bound_consistency_protocol_v1.json',
       common_uniform_bound_review='bound_closure_independent_20260916_050846_426215/audit.json',
       common_convnext_hard_status='All45 outcomes complete; updated31 recorded bound-consistent flags; historical32 flag count superseded by uniform audit',
       bound_consistent_recorded_flags=sum(x['updated_proven_count'] for x in summary['by_model']),
       bound_anomaly_count=sum(x['bound_anomalies'] for x in summary['by_model']),
       feasibility_fallback_count=sum(x['fallback_count'] for x in summary['by_model']),
       training_snapshot={'completed':12,'planned':12,'active_run_id':None,'snapshot_time':queue['updated_utc'],'dynamic_status':'../common_queue_status.json'},
       next_spatial_step='Completed local8 configuration matrix; freeze external9-run development preparation and assess baseline/seed stability; no test/R6 selection',
       test_accessed=False,R6_accessed=False,final_confirmation_complete=False,
       weedsgalore_development_acceptance='../external_data/weedsgalore/adapter_v1/acceptance.json')
    index['common_complete_prefix_auc_directory']=auc.relative_to(PA).as_posix()
    index['common_complete_prefix_auc_review']=auc_review_path.relative_to(PA).as_posix()
    index['weedsgalore_training_source_lock']='../external_weeds_v1/SOURCE_LOCK.json'
    index['weedsgalore_training_plan']='../external_weeds_v1/serial_plan.json'
    index['weedsgalore_training_protocol']='../external_weeds_v1/RESEARCH_PROTOCOL.json'
    external_pointer=ROOT/'external_weeds_v1/current_queue.json'
    if external_pointer.exists():
        index['weedsgalore_training_queue']='../external_weeds_v1/current_queue.json'
        index['weedsgalore_training_snapshot']=read(external_pointer)
        index['next_spatial_step']='Follow fixed external9 queue; independently audit all outcomes before map export/relative-grid analysis; no test/R6 selection'
    full_draft=PA/'PA_results_full_common_draft_v1.md'
    if full_draft.exists():index['current_common_results_draft']=full_draft.name
    archives=sorted((ROOT/'code_versions').glob('manifest_*.json'))
    assert archives;index['code_archive_manifest']='../'+archives[-1].relative_to(ROOT).as_posix()
    additions=[FULL/'full_common_management_report_with_figures.md',FULL/'status.json',FULL/'final_artifact_manifest.json',
        FULL/'root_png_visual_review.md',FULL/'all_720_model_policy_budgets.csv',FULL/'all_360_soft_minus_hard_pairs.csv',review_path,
        BOUND/'summary.json',BOUND/'all360_policy_outcomes.csv',BOUND/'binding_seal.json',
        PA/'RM_bound_consistency_protocol_v1.json',PA/'bound_closure_independent_20260916_050846_426215/audit.json',
        ROOT/'latest_common_development_report.json',ROOT/'common_development_reports/appraisal_20260916_044615_102760/critical_appraisal.md',
        ROOT/'external_data/weedsgalore/adapter_v1/acceptance.json',ROOT/'external_data/weedsgalore/adapter_v1/ADAPTER_REVIEW.md',
        ROOT/'provenance/remaining_common_exports_20260916_0432/remaining_common_exports_report.json',archives[-1]]
    additions +=[auc/'report.md',auc/'all72_model_prefix_auc.csv',auc/'all36_hard_soft_auc_pairs.csv',auc/'manifest.json',auc_review_path,
        ROOT/'external_weeds_v1/SOURCE_LOCK.json',ROOT/'external_weeds_v1/serial_plan.json',ROOT/'external_weeds_v1/RESEARCH_PROTOCOL.json',
        ROOT/'external_weeds_v1/PREPARATION_HANDOFF.md',ROOT/'external_weeds_v1/preparation_manifest.json',
        ROOT/'external_weeds_v1/independent_review_20260916_052858_559152/review.json']
    # The live queue pointer is intentionally outside the immutable evidence file hashes.
    for candidate in ('root_gpu_review.json','root_gpu_launch.json'):
        path=ROOT/'external_weeds_v1'/candidate
        if path.exists():additions.append(path)
    execution_review=ROOT/'external_weeds_v1/root_gpu_review.json'
    if execution_review.exists():
        execution=read(execution_review)
        additions +=[Path(execution['gpu_independent_review']),Path(execution['gpu_diagnostic_directory'])/'summary.json',
                     Path(execution['gpu_diagnostic_directory'])/'artifact_manifest.json']
    if full_draft.exists():additions +=[full_draft,full_draft.with_name(full_draft.stem+'_evidence.json')]
    files={x['path']:x for x in index['files']}
    import os
    for path in additions:
        key=os.path.relpath(path,PA).replace('\\','/');files[key]={'path':key,'sha256':sha(path)}
    for key,item in files.items():item['sha256']=sha((PA/key).resolve())
    index['files']=list(files.values());save(PA/'latest_results.json',index)
    checks={'passed':True,'updated_local':index['updated_local'],'indexed_files':len(files),
        'all_indexed_hashes_match':all(sha((PA/x['path']).resolve())==x['sha256'] for x in index['files']),
        'central_tables':720,'paired_comparisons':360,'recorded_bound_consistent_flags':index['bound_consistent_recorded_flags'],
        'bound_anomalies':index['bound_anomaly_count'],'fallbacks':index['feasibility_fallback_count'],
        'central_review_sha256':sha(review_path),'latest_results_sha256':sha(PA/'latest_results.json'),
        'test_accessed':False,'R6_accessed':False,'publication_objective_complete':False}
    save(PA/'full8_index_verification.json',checks);print(json.dumps(checks,ensure_ascii=False))
if __name__=='__main__':main()
