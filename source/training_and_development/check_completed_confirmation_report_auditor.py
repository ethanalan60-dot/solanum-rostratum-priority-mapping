"""Small artificial arithmetic/gate smoke only; never runs an actual report audit."""
from pathlib import Path
import ast,csv,datetime,hashlib,importlib.util,io,json,math,shutil,sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'audit_completed_confirmation_report.py'
def main():
    out=ROOT/'provenance'/('confirmation_report_output_auditor_preparation_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    spec=importlib.util.spec_from_file_location('independent_report_auditor',SRC);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    shutil.copy2(SRC,out/'auditor_source.py');shutil.copy2(__file__,out/'smoke_source.py')
    checks=[]
    def yes(name,ok=True):assert ok,name;checks.append(name)
    def bad(name,fn):
        try:fn()
        except (ValueError,KeyError,FileNotFoundError):yes(name);return
        raise AssertionError(name)
    yes('stdlib_no_producer_import',not any(x in sys.modules for x in ('report_tables','core','numpy','torch','scipy','rasterio')))
    tree=ast.parse(SRC.read_text(encoding='utf-8'));imports=[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    yes('no_executable_subprocess_import',not any(x in imports for x in ('subprocess','pickle')))
    yes('manual_n_minus_one_sd',q.mean_sd([1.,2.,6.])==(3.,math.sqrt(7.)))
    yes('single_seed_has_no_SD',q.mean_sd([2.])==(2.,None))
    yes('any_NA_keeps_both_stats_NA',q.mean_sd([1.,None,6.])==(None,None))
    yes('NA_difference_propagates',q.subtract(None,0.) is None and q.subtract(0.,None) is None)
    yes('exact_tied_difference_retained',q.subtract(2.,2.)==0.)
    models={};values=[]
    for variant in ('hard_seg','soft_seg'):
        for seed,base,change in ((17,1.,-1.),(29,2.,0.),(43,6.,2.)):
            rid=f'{variant}_{seed}';models[rid]=dict(run_id=rid,dataset='weeds',encoder='resnet18',variant=variant,lr=.0001,seed=seed,best_epoch=2,checkpoint_sha256='1'*64)
            for R in q.RS:values.append(dict(models[rid],R=R,value=base+R/10+(change if variant=='soft_seg' else 0)))
    result={name:(rows,keys) for name,rows,keys in q.families(values,('R',),('value',),models)}
    yes('complete_artificial_family_counts',{k:len(v[0]) for k,v in result.items()}=={'all':24,'hard_soft_pairs':12,'seed_mean_sample_sd':8,'hard_soft_seed_mean_sample_sd':4,'R_contrasts':30,'R_contrast_seed_mean_sample_sd':10})
    for row in result['hard_soft_pairs'][0]:yes('paired_'+str(row['seed'])+'_'+str(row['R']),abs(row['soft_minus_hard_value']-{17:-1,29:0,43:2}[row['seed']])<1e-14)
    for row in result['hard_soft_seed_mean_sample_sd'][0]:
        yes('paired_mean_'+str(row['R']),abs(row['soft_minus_hard_value_mean']-1/3)<1e-14)
        yes('paired_sd_'+str(row['R']),abs(row['soft_minus_hard_value_sample_sd']-math.sqrt(7/3))<1e-14)
    yes('five_R_directions',set(x['contrast'] for x in result['R_contrasts'][0])=={'R4_minus_R1','R2_minus_R1','R4_minus_R2','R6_minus_R1','R6_minus_R4'})
    bad('missing_fitting_seed_rejected',lambda:q.statistics_rows([x for x in values if x['seed']!=43],('dataset','encoder','variant','lr','R'),('value',),models))
    bad('duplicate_output_key_rejected',lambda:q.indexed(values+[values[0]],('run_id','R')))
    def csv_roundtrip(rows):
        buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return list(csv.DictReader(io.StringIO(buf.getvalue())))
    compare=q.Compare()
    for name,(rows,keys) in result.items():compare.table(csv_roundtrip(rows),rows,keys,name)
    yes('all_saved_family_cells_compared',compare.numeric>300 and compare.exact_zero_differences>0)
    rows,keys=result['hard_soft_pairs'];broken=csv_roundtrip(rows);broken[0]['soft_minus_hard_value']='99'
    bad('corrupt_difference_rejected',lambda:q.Compare().table(broken,rows,keys,'bad'))
    broken=csv_roundtrip(rows);broken[0]['soft_run_id']='other'
    bad('mixed_pair_ID_rejected',lambda:q.Compare().table(broken,rows,keys,'bad'))
    bad('missing_output_row_rejected',lambda:q.Compare().table(csv_roundtrip(rows)[:-1],rows,keys,'bad'))
    raw=[dict(models['hard_seg_17'],R='1',n='26',n_zero='0',n_positive='26',mae='.2',bias='-.1',mae_zero='',mae_positive='.2',balanced_mae='')]
    z=q.normalize(raw,'scalar',models)[0]
    yes('scalar_pp_conversion_and_NA',z['mae_pp']==20 and z['bias_pp']==-10 and z['balanced_mae_pp'] is None)
    bad('scalar_wrong_stratum_NA_rejected',lambda:q.normalize([{**raw[0],'mae_zero':'.1'}],'scalar',models))
    toy=out/'toy';toy.mkdir();r=q.Reader();f=toy/'source.json';q.dump(f,{'x':1});r.json(r.allow(f));q.dump(f,{'x':2})
    bad('begin_end_source_drift_rejected',r.finish)
    bad('NPZ_not_allowlisted',lambda:r.allow(toy/'cache.npz'))
    bad('CSV_unlisted_rejected',lambda:r.csv(toy/'truth_units.csv'))
    bad('path_escape_rejected',lambda:q.beneath(toy,'../outside.json'))
    home=toy/'confirmation_v1';report=home/'stages/report_tables';report.mkdir(parents=True)
    q.dump(report/'status.json',{'state':'running'});q.dump(report/'artifact_manifest.json',{'files':{'status.json':q.digest(report/'status.json')}})
    r=q.Reader()
    with patch.object(q,'ROOT',toy):bad('incomplete_report_refused_before_freeze_or_grant',lambda:q.audit(report,toy/'nonexistent_grant.json','0'*64,r,q.Compare()))
    yes('incomplete_gate_did_not_open_freeze',all(Path(p).name in ('audit_completed_confirmation_report.py','artifact_manifest.json','status.json') for p in r.watched))
    yes('exact_static119_report_files',len(q.report_file_names())==119)
    q.dump(report/'status.json',{'state':'completed'});(report/'unit_truth.csv').write_text('do_not_open\n',encoding='utf-8')
    q.dump(report/'artifact_manifest.json',{'files':{'status.json':q.digest(report/'status.json'),'unit_truth.csv':'0'*64}})
    r=q.Reader();bad('unexpected_truth_named_report_file_rejected_before_hash',lambda:q.stage_metadata(r,report,True))
    yes('unknown_payload_never_allowlisted',all(Path(p).name!='unit_truth.csv' for p in r.allowed))
    local_audit=home/'stages/audit_local_outcomes';local_audit.mkdir();q.dump(local_audit/'review.json',{'passed':True,'stage':'local_outcomes'});q.dump(local_audit/'status.json',{'state':'completed'})
    ah=q.digest(local_audit/'review.json');q.dump(local_audit/'artifact_manifest.json',{'files':{'review.json':ah,'status.json':q.digest(local_audit/'status.json')}})
    with patch.object(q,'ROOT',toy):
        yes('canonical_completed_local_acceptance_allowed',q.acceptance_file(q.Reader(),'local_outcomes',{'path':str(local_audit/'review.json'),'sha256':ah})['passed'] is True)
        bad('wrong_local_audit_stage_rejected',lambda:q.acceptance_file(q.Reader(),'local_outcomes',{'path':str(home/'stages/audit_other/review.json'),'sha256':ah}))
        bad('escaped_local_acceptance_rejected',lambda:q.acceptance_file(q.Reader(),'local_outcomes',{'path':str(local_audit/'../../review.json'),'sha256':ah}))
        bad('local_acceptance_wrong_manifest_hash_rejected',lambda:q.acceptance_file(q.Reader(),'local_outcomes',{'path':str(local_audit/'review.json'),'sha256':'0'*64}))
        bad('external_acceptance_cannot_reuse_local_stage',lambda:q.acceptance_file(q.Reader(),'external_outcomes',{'path':str(local_audit/'review.json'),'sha256':ah}))
    note={'passed':True,'scope':'small_synthetic_report_output_auditor_preparation_not_actual_report_acceptance','implementation_sha256':q.digest(SRC),'source':str(SRC),'checker_sha256':q.digest(__file__),'checks':checks,'check_count':len(checks),'synthetic_numeric_comparisons':compare.numeric,'actual_report_or_outcome_reads':0,'actual_model_solver_subprocess_calls':0,'full_actual_lifecycle_not_executed':True,'old_preparation_reviewer_unchanged_sha256':q.digest(ROOT/'review_confirmation_report_tables.py'),'limitations':['Small synthetic checks target formulas, NA, identities and entry rejection; they do not fabricate a full actual outcome-stage acceptance.','Root must review the final source before invoking the actual audit on completed outputs.']}
    q.dump(out/'focused_checks.json',note);print(json.dumps({'out':str(out),'passed':True,'checks':len(checks),'implementation_sha256':q.digest(SRC),'receipt_sha256':q.digest(out/'focused_checks.json')},ensure_ascii=False))
if __name__=='__main__':main()
