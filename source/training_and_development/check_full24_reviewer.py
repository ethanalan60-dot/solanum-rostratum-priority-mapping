"""Complete artificial25-table/source fixture and corruption checks; no real outputs."""
from pathlib import Path
import copy,csv,datetime,hashlib,json,shutil
import seed_stability_summary_v1 as producer
import review_full24_seed_results as reviewer

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(p,rows):producer.write_csv(p,rows)
def loadcsv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def build_fixture(home):
    pa=home/'precision_agriculture';pa.mkdir(parents=True);(home/'provenance').mkdir();folder=pa/'synthetic_full24';folder.mkdir()
    inputs=home/'synthetic_sources';inputs.mkdir();sourcefiles=[]
    shutil.copy2(ROOT/'seed_stability_summary_v1.py',home/'seed_stability_summary_v1.py');sourcefiles.append(home/'seed_stability_summary_v1.py')
    def save(p,d,csv=False):
        (csvsave if csv else dump)(p,d);sourcefiles.append(p);return {'path':str(p),'sha256':sha(p)}
    budgets,auc,scalar=producer.synthetic_tables();models=list({r['run_id']:{k:r[k] for k in producer.MODEL_KEYS} for r in budgets}.values())
    # Enrich original source schema. No operation calls producer load_inputs/execute.
    for row in budgets:
        row['selected_ids_json']='["synthetic_unit"]';row.update(predicted_upper_m2=None,predicted_relative_gap=None,original_retained_upper_m2=None,original_retained_gap=None,raw_solver_status=None,raw_solver_gap=None,bound_audit_json='')
    oldb=[r for r in budgets if r['seed']==17];olda=[r for r in auc if r['seed']==17]
    bp=inputs/'original_budget.csv';ap=inputs/'original_auc.csv'
    root_review={'passed':True,'all24_required':True,'execution_scope':'seed_stability_summary_v1_full24','implementation_sha256':sha(home/'seed_stability_summary_v1.py'),'test_or_R6_accessed':False,
        'seed17':{'budget_csv':save(bp,oldb,True),'auc_csv':save(ap,olda,True)},'seed17_exports':[],'new_models':[]}
    finalb=[producer.normalize_original(r) for r in loadcsv(bp)];finala=[];finals=[]
    for row in olda:finala.append({**row,'source_csv':str(ap),'source_csv_sha256':sha(ap)})
    for i,m in enumerate(models):
        home_model=inputs/('m%02d'%i);home_model.mkdir();run=m['run_id']
        valrows=[r for r in scalar if r['run_id']==run];scales={str(r['R']):{k[:-3]:r[k]/100 for k in reviewer.SM} for r in valrows}
        for v in scales.values():v.update(n=357,n_positive=27)
        score=sum(v['balanced_mae'] for v in scales.values())/3
        cpu={'run_id':run,'checkpoint_epoch':m['checkpoint_epoch'],'checkpoint_sha256':m['checkpoint_sha256'],'scales':scales,'selection_balanced_mae':score,'original_amp_validation':{'selection_balanced_mae':score+.00001}}
        cpusummary=home_model/'cpu_summary.json';cb=save(cpusummary,cpu)
        for row in valrows:
            final={**row,'n':357,'n_positive':27,'device':'cpu','precision':'float32','summary_path':str(cpusummary),'summary_sha256':cb['sha256'],
                   'cpu_three_R_balanced_mae_pp':100*score,'original_amp_selection_mae_pp':100*(score+.00001),'cpu_minus_original_amp_selection_pp':-.001,'epoch_reselected':False}
            finals.append(final)
        if m['seed']==17:root_review['seed17_exports'].append({'run_id':run,'summary':cb});continue
        spatial=home_model/'spatial';spatial.mkdir();export=home_model/'export';export.mkdir()
        # Same original CPU summary bytes, with the actual new-receipt filename.
        save(export/'summary.json',cpu)
        for row in finals:
            if row['run_id']==run:row['summary_path']=str(export/'summary.json');row['summary_sha256']=sha(export/'summary.json')
        sm={**m,'domain':{'area_m2':100.,'target_m2':10.}};smb=save(spatial/'summary.json',sm)
        rr=[r for r in budgets if r['run_id']==run];wide=[]
        for M,R,budget in __import__('itertools').product((1,2,4),(1,2,4),(.1,.2,.3,.5,1.)):
            p=next(r for r in rr if (r['M'],r['R'],r['budget'],r['policy'])==(M,R,budget,'prefix'));q=next(r for r in rr if (r['M'],r['R'],r['budget'],r['policy'])==(M,R,budget,'predicted_area'))
            record={'run_id':run,'M':M,'R':R,'budget':budget,'budget_area_m2':p['budget_area_m2'],'prefix_ids_json':'["synthetic_unit"]','selected_ids_json':'["synthetic_unit"]'}
            for prefix,r in [('prefix',p),('pmilp',q)]:
                record[prefix+'_selected_count']=1;record[prefix+'_capture_fraction']=r['capture_pct']/100
                for metric in reviewer.BM:
                    if metric not in ('capture_pct','unused_budget_m2'):record[prefix+'_'+metric]=r[metric]
            record.update(fallback_used=False,optimality_proven=False,original_proven=False,bound_anomaly=False,certification_downgraded=False,
                predicted_upper_m2=q['predicted_objective_m2']+1,predicted_relative_gap=.1,raw_solver_gap=.1,raw_solver_status='0',bound_audit_json=json.dumps({'retained_upper_m2':q['predicted_objective_m2']+1}))
            wide.append(record)
        wp=spatial/'all_45_policy_outcomes.csv';save(wp,wide,True)
        for row in loadcsv(wp):finalb.extend(producer.normalize_new(row,sm,wp,sha(wp)))
        aa=[{**r,'method':'model_prefix'} for r in auc if r['run_id']==run];acp=spatial/'prefix_auc.csv';save(acp,aa,True)
        finala.extend({**r,'source_csv':str(acp),'source_csv_sha256':sha(acp)} for r in aa)
        rec={'run_id':run,'spatial_directory':str(spatial),'export_directory':str(export),'spatial_summary_sha256':smb['sha256']}
        recbind=save(home_model/'receipt.json',rec);root_review['new_models'].append({'run_id':run,'completion_receipt':recbind})
    rb=save(inputs/'root_review.json',root_review);tables=producer.make_tables(finalb,finala,finals)
    for name,rows in tables.items():csvsave(folder/(name+'.csv'),rows)
    dump(folder/'models.json',models)
    summary={'models':24,'seeds':[17,29,43],'table_rows':{k:len(v) for k,v in tables.items()},'recorded_flags':{k:sum(r[k] is True for r in finalb if r['policy']=='predicted_area') for k in reviewer.FLAGS},
             'test_or_R6_accessed':False,'new_model_calls':0,'new_solver_calls':0,'field_replicates':1}
    dump(folder/'summary.json',summary);archive=folder/'source';archive.mkdir();ledger=[]
    for i,p in enumerate(sourcefiles):
        dst=archive/('%03d%s'%(i,p.suffix));shutil.copy2(p,dst);ledger.append({'path':str(p),'sha256':sha(p),'archived':dst.relative_to(folder).as_posix()})
    dump(folder/'source_ledger.json',ledger)
    status={'state':'completed','source_sha256':sha(home/'seed_stability_summary_v1.py'),'root_review_sha256':rb['sha256']}
    reseal(folder,status);return folder,tables,models

def reseal(folder,status=None):
    if status is None:status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
    manifest={'file_sha256':{p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('status.json','artifact_manifest.json')}}
    dump(folder/'artifact_manifest.json',manifest);status['artifact_manifest_sha256']=sha(folder/'artifact_manifest.json');dump(folder/'status.json',status)

def main():
    out=ROOT/'provenance'/('full24_reviewer_checks_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();home=out/'fixture';home.mkdir()
    for name in ('check_full24_reviewer.py','review_full24_seed_results.py','seed_stability_summary_v1.py'):shutil.copy2(ROOT/name,out/name)
    checks=[];original_root=reviewer.ROOT;folder=None
    try:
        folder,tables,models=build_fixture(home);reviewer.ROOT=home
        result=reviewer.audit(folder,write_review=False);assert result['passed'] and result['tables_audited']==25
        checks.append({'name':'full25_tables_complete_source_join_and_arithmetic','passed':True,'comparisons':result['numeric_comparisons']})
        def reject(name,fn):
            try:fn()
            except (AssertionError,KeyError,ValueError,TypeError):checks.append({'name':name,'passed':True});return
            raise AssertionError(name+' was accepted')
        # Every25 output table rejects a duplicate replacing an otherwise valid row,
        # even when both the output manifest and declared counts were re-signed.
        for name in tables:
            bad=copy.deepcopy(tables);bad[name][-1]=copy.deepcopy(bad[name][0])
            reject('unique_grid_'+name,lambda bad=bad:reviewer.verify_table_inventory(bad,models))
        def corrupt(name,field,value):
            path=folder/(name+'.csv');original=path.read_bytes();rows=loadcsv(path);rows[0][field]=value;csvsave(path,rows);reseal(folder)
            try:reject('tamper_'+name+'_'+field,lambda:reviewer.audit(folder,write_review=False))
            finally:path.write_bytes(original);reseal(folder)
        for case in [('all2160_model_policy_budgets','selected_ids_json','["wrong"]'),('all2160_model_policy_budgets','capture_pct',99),('all216_model_prefix_auc','auc_0_1',.999),
            ('all72_cpu_scalar_by_R','bias_pp',99),('all24_cpu_scalar_model_summary','cpu_minus_original_amp_selection_pp',1),('all12_hard_soft_model_pairs','soft_run_id','wrong'),
            ('all1080_hard_soft_budget_pairs','capture_pct_direction','lower'),('all108_hard_soft_auc_pairs','soft_checkpoint_epoch',23),('all36_hard_soft_cpu_scalar_pairs','soft_minus_hard_mae_pp',5),
            ('all_budget_R_contrasts','difference_capture_pct',2),('all_auc_M_contrasts','contrast','wrong'),('cpu_scalar_R_contrasts','difference_bias_pp',2),
            ('budget_three_seed_mean_sample_sd','capture_pct_sample_sd',55),('hard_soft_auc_three_seed_mean_sample_sd','soft_minus_hard_auc_0_1_n_valid',2),
            ('cpu_scalar_three_seed_mean_sample_sd','independent_fields',3),('budget_hard_soft_direction_counts','maximum',12),('auc_hard_soft_direction_counts','tolerance',.01)]:corrupt(*case)
        checks.append({'name':'NA_difference_and_direction_not_zero_imputed','passed':reviewer.difference(None,1) is None and reviewer.direction(None,1e-8)=='NA'})
        assert checks[-1]['passed']
        assert reviewer.direction(1e-8,1e-8)=='tied' and reviewer.direction(-1e-8,1e-8)=='tied'
        checks.append({'name':'exact_direction_tolerance_ties','passed':True})
        # Confirm artifact tampering is rejected before arithmetic.
        p=folder/'models.json';raw=p.read_bytes();p.write_bytes(raw+b' ')
        try:reject('unsealed_artifact_changed',lambda:reviewer.audit(folder,write_review=False))
        finally:p.write_bytes(raw)
        report={'passed':True,'scope':'synthetic_full24_reviewer_preparation_only','checks':checks,'count':len(checks),'complete_synthetic_result':result,
                'reviewer_sha256':sha(ROOT/'review_full24_seed_results.py'),'producer_sha256':sha(ROOT/'seed_stability_summary_v1.py'),'checker_sha256':sha(__file__),
                'actual_full24_output_audited':False,'reserved_data_or_R6_accessed':False,'model_solver_or_subprocess_calls':0}
        dump(out/'focused_checks.json',report);print(json.dumps({'out':str(out),'passed':True,'checks':len(checks),'sha256':sha(out/'focused_checks.json')}))
    except BaseException as error:
        import traceback
        dump(out/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'actual_data_accessed':False});raise
    finally:reviewer.ROOT=original_root
if __name__=='__main__':main()
