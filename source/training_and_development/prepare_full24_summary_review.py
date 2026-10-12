"""Join exact accepted inputs for the fixed24 summary; incomplete16 never authorizes it."""
from pathlib import Path
import argparse,csv,datetime,importlib.util,json
ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture'
spec=importlib.util.spec_from_file_location('fixed_summary',ROOT/'seed_stability_summary_v1.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
EXPECTED='5fb80182e6c05c33a2157a213de5e65999fa15a4c5ef3c4c75fff941a1ca259e'
def pin(p):return {'path':str(Path(p).resolve()),'sha256':s.sha(p)}
def original_inputs():
    budget=PA/'common_management_reports/full8_20260916_050927_560091/all_720_model_policy_budgets.csv'
    assert s.sha(budget)==s.OLD_PINS['budget_csv']
    unique={}
    for row in s.read_csv(budget):unique.setdefault(row['run_id'],row)
    assert len(unique)==8;exports=[];lineage=[]
    for rid,row in unique.items():
        fixed=Path(row['fixed_budget_csv']);assert s.sha(fixed)==row['fixed_budget_csv_sha256']
        inputs=fixed.parent/'inputs.json';bindings=s.load(inputs)['file_sha256']
        names=[Path(p) for p in bindings if Path(p).name=='metadata.json' and Path(p).parent.parent.name==rid and Path(p).parent.name.startswith('common_validation_export_')]
        assert len(names)==1,(rid,names)
        metadata=names[0];assert s.sha(metadata)==bindings[str(metadata)]
        folder=metadata.parent;summary=s.load(folder/'summary.json')
        assert summary['run_id']==rid and summary['checkpoint_sha256']==row['checkpoint_sha256'] and summary['checkpoint_epoch']==int(row['checkpoint_epoch'])
        status=s.load(folder/'status.json');assert status['metadata_sha256']==s.sha(metadata)
        exports.append({'run_id':rid,**{k:pin(folder/(k+'.json')) for k in ('status','summary','metadata')}})
        lineage.append({'run_id':rid,'accepted_fixed_budget':pin(fixed),'original_spatial_input_binding':pin(inputs)})
    return exports,lineage
def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare-original-only',action='store_true');g.add_argument('--full16-index',type=Path);a=p.parse_args()
    assert s.sha(ROOT/'seed_stability_summary_v1.py')==EXPECTED
    exports,lineage=original_inputs()
    if a.prepare_original_only:
        out=ROOT/'provenance'/('full24_original_inputs_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
        s.dump(out/'review.json',{'passed':True,'scope':'metadata-only exact original8 CPU export joins; no full24 execution grant','exports':exports,'lineage':lineage,'implementation_sha256':s.sha(__file__),'test_or_R6_accessed':False})
        print(json.dumps({'passed':True,'original_exports':len(exports),'out':str(out)}));return
    index=s.load(a.full16_index);assert index['all16_complete'] and index['accepted_count']==16 and not index['pending_run_ids']
    independent=ROOT/'provenance/seed_summary_independent_20260916_102709_016960/review.json'
    audit=s.load(independent);assert audit['passed'] and audit['implementation_sha256']==EXPECTED and audit['numeric_comparisons']==221475
    old={
      'budget_csv':PA/'common_management_reports/full8_20260916_050927_560091/all_720_model_policy_budgets.csv',
      'auc_csv':PA/'common_prefix_auc_full8_20260916_052212_839933/all72_model_prefix_auc.csv',
      'budget_audit':PA/'full8_central_independent_20260916_051750_163534/central_review.json',
      'auc_audit':PA/'prefix_auc_independent_20260916_052458_065751/review.json'}
    for key,path in old.items():assert s.sha(path)==s.OLD_PINS[key]
    review={'passed':True,'execution_scope':s.SCOPE,'implementation_sha256':EXPECTED,'all24_required':True,'test_or_R6_accessed':False,
      'plan':pin(ROOT/'local_seed_stability_v1/plan.json'),'seed17':{k:pin(v) for k,v in old.items()},'seed17_exports':exports,
      'new_models':index['new_models'],'numerical_acceptance_index':pin(a.full16_index),'implementation_independent_review':pin(independent),
      'original_export_lineage':lineage,'review_builder_sha256':s.sha(__file__),'created_at':datetime.datetime.now().astimezone().isoformat()}
    out=ROOT/'seed_stability_summary_v1';out.mkdir(exist_ok=True)
    path=out/('root_execution_review_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    with path.open('x',encoding='utf-8') as f:json.dump(review,f,ensure_ascii=False,indent=2)
    # Full consumer gates, including every source/checkpoint/grid/CPU summary,
    # are reviewed before an eventual distinct execute command can claim work.
    budget,auc,scalar,models,receipts,sources=s.load_inputs(path)
    s.dump(path.with_name(path.stem+'_validated.json'),{'passed':True,'root_review':pin(path),'rows':[len(budget),len(auc),len(scalar)],'models':len(models),'new_completion_receipts':len(receipts),'test_or_R6_accessed':False,'production_executed':False})
    print(json.dumps({'passed':True,'root_review':str(path),'rows':[len(budget),len(auc),len(scalar)]}))
if __name__=='__main__':main()
