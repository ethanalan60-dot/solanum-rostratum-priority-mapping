"""Independent synthetic arithmetic review of full24 table consumer; no model/data reads."""
from pathlib import Path
import datetime,importlib.util,json,math,shutil
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('candidate_summary',ROOT/'seed_stability_summary_v1.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

def main():
    checks=[];comparisons=0
    def yes(name,ok):
        assert ok,name
        checks.append(name)
    def close(x,y):
        nonlocal comparisons
        assert (x is None and y is None) or (x is not None and y is not None and math.isclose(x,y,rel_tol=2e-10,abs_tol=1e-10)),(x,y)
        comparisons+=1
    def rejects(name,fn):
        try:fn()
        except (ValueError,KeyError,FileNotFoundError):yes(name,True);return
        raise AssertionError(name)
    budgets,auc,scalars=q.synthetic_tables()
    for row in budgets:
        b=row['budget'];R=row['R'];M=row['M'];s=row['seed']
        v=1 if row['variant']=='soft_seg' else -1
        e=1 if row['encoder']=='resnet18' else 2
        l=1 if row['lr']==.0001 else 3
        gain=.3+.02*e-.01*l+.007*(s%7)+.02*v*R-.018*M
        capture=b+b*(1-b)*gain
        area=100*b*(1 if b==1 else .98-.001*e-.001*R)
        row.update(capture_pct=100*capture,captured_target_m2=10*capture,actual_area_m2=area,
                   actual_area_fraction=area/100,missed_target_m2=10*(1-capture),
                   zero_annotation_area_m2=area-10*capture,selected_zero_target_unit_area_m2=area/4,
                   predicted_objective_m2=8*capture,unused_budget_m2=100*b-area)
    for row in auc:
        v=1 if row['variant']=='soft_seg' else -1
        row.update(auc_0_1=.55+.02*v*row['R']-.005*row['M']+.0001*row['seed'],
                   auc_0_0p5_normalized=.31+.01*v*row['M']-.003*row['R']+.0003*row['seed'])
    for row in scalars:
        e=1 if row['encoder']=='resnet18' else 2
        v=1 if row['variant']=='soft_seg' else -1
        for index,metric in enumerate(q.SCALAR_METRICS):row[metric]=.8+index*.1+.01*v*row['R']+.001*row['seed']+.02*e
    tables=q.make_tables(budgets,auc,scalars)
    yes('full table inventory',len(tables)==25)
    for name,rows,dims,metrics in [('budget',budgets,('M','R','budget','policy'),q.BUDGET_METRICS),('auc',auc,('M','R'),q.AUC_METRICS),('cpu_scalar',scalars,('R',),q.SCALAR_METRICS)]:
        index={(r['encoder'],r['lr'],r['seed'],r['variant'])+tuple(r[k] for k in dims):r for r in rows}
        pname={'budget':'all1080_hard_soft_budget_pairs','auc':'all108_hard_soft_auc_pairs','cpu_scalar':'all36_hard_soft_cpu_scalar_pairs'}[name]
        pairs=tables[pname]
        for row in pairs:
            shared=(row['encoder'],row['lr'],row['seed']);grid=tuple(row[k] for k in dims)
            h=index[shared+('hard_seg',)+grid];s=index[shared+('soft_seg',)+grid]
            yes_id=row['hard_run_id']==h['run_id'] and row['soft_run_id']==s['run_id']
            assert yes_id,'model pairing'
            for metric in metrics:
                close(row['hard_'+metric],h[metric]);close(row['soft_'+metric],s[metric]);close(row['soft_minus_hard_'+metric],s[metric]-h[metric])
        yes(name+' independently paired all rows',True)
        axes=('R','M') if name!='cpu_scalar' else ('R',)
        for axis in axes:
            others=tuple(k for k in dims if k!=axis)
            lookup={(r['run_id'],)+tuple(r[k] for k in others)+(r[axis],):r for r in rows}
            cname='cpu_scalar_R_contrasts' if name=='cpu_scalar' else f'all_{name}_{axis}_contrasts'
            for row in tables[cname]:
                shared=(row['run_id'],)+tuple(row[k] for k in others)
                lo=lookup[shared+(row['reference'],)];hi=lookup[shared+(row['comparison'],)]
                assert (row['comparison'],row['reference']) in ((4,1),(2,1),(4,2))
                for metric in metrics:
                    close(row['comparison_'+metric],hi[metric]);close(row['reference_'+metric],lo[metric]);close(row['difference_'+metric],hi[metric]-lo[metric])
            yes(name+' all '+axis+' contrasts independent',True)
    # Infer each statistics source by the documented output contract, then group
    # independently: no candidate mean, stdev, grouping or difference helper used.
    bases={
      'budget_three_seed_mean_sample_sd':budgets,'auc_three_seed_mean_sample_sd':auc,
      'cpu_scalar_three_seed_mean_sample_sd':scalars,
      'hard_soft_budget_three_seed_mean_sample_sd':tables['all1080_hard_soft_budget_pairs'],
      'hard_soft_auc_three_seed_mean_sample_sd':tables['all108_hard_soft_auc_pairs'],
      'hard_soft_cpu_scalar_three_seed_mean_sample_sd':tables['all36_hard_soft_cpu_scalar_pairs']}
    for name in ('budget','auc'):
        for axis in ('R','M'):bases[f'{name}_{axis}_contrast_three_seed_mean_sample_sd']=tables[f'all_{name}_{axis}_contrasts']
    for table,source in bases.items():
        for row in tables[table]:
            keys=[k for k in row if k not in ('seeds','n_expected','independent_fields') and not k.endswith(('_mean','_sample_sd','_n_valid'))]
            group=[r for r in source if all(r[k]==row[k] for k in keys)]
            assert len(group)==3 and {r['seed'] for r in group}=={17,29,43}
            for field in row:
                if not field.endswith('_mean'):continue
                metric=field[:-5];values=[r[metric] for r in group if r[metric] is not None]
                mean=math.fsum(values)/len(values) if values else None
                sd=math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1)) if len(values)>1 else None
                close(row[field],mean);close(row[metric+'_sample_sd'],sd);close(row[metric+'_n_valid'],len(values))
        yes(table+' independent grouping mean sampleSD',True)
    for table in ('budget_hard_soft_direction_counts','auc_hard_soft_direction_counts'):
        source=tables['all1080_hard_soft_budget_pairs' if table.startswith('budget') else 'all108_hard_soft_auc_pairs']
        for row in tables[table]:
            keys=('encoder','lr','seed','M')+(('policy',) if table.startswith('budget') else ())
            values=[r[row['metric']] for r in source if all(r[k]==row[k] for k in keys)]
            ep=row['tolerance'];close(row['higher'],sum(v>ep for v in values));close(row['lower'],sum(v < -ep for v in values));close(row['tied'],sum(abs(v)<=ep for v in values));close(row['NA'],0);close(row['rows'],len(values))
        yes(table+' all groups independent',True)
    rejects('actual absent root grant rejected',lambda:q.load_inputs(ROOT/'nonexistent_full24_review.json'))
    rejects('missing full24 grid rejected',lambda:q.validate_grids(budgets[:-1],auc,scalars))
    rejects('duplicate scalar replacing missing rejected',lambda:q.validate_grids(budgets,auc,scalars[:-1]+[scalars[0]]))
    rejects('different checkpoint in scalar rejected',lambda:q.validate_grids(budgets,auc,[dict(scalars[0],checkpoint_epoch=9)]+scalars[1:]))
    na=[dict(seed=s,g=1,x=v) for s,v in zip((17,29,43),(None,2.,4.))]
    stat=q.seed_statistics(na,('g',),('x',))[0]
    close(stat['x_n_valid'],2);close(stat['x_mean'],3);close(stat['x_sample_sd'],math.sqrt(2))
    yes('NA stays missing and two observed values use nminus1',True)
    out=ROOT/'provenance'/('seed_summary_independent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    shutil.copyfile(__file__,out/'reviewer.py')
    q.dump(out/'review.json',{'passed':True,'implementation_sha256':q.sha(ROOT/'seed_stability_summary_v1.py'),'reviewer_sha256':q.sha(__file__),
       'checks':checks,'numeric_comparisons':comparisons,'table_counts':{k:len(v) for k,v in tables.items()},
       'empirical_outputs_read':False,'test_or_R6_accessed':False,'model_or_solver_calls':0,'production_execution_authorized':False,
       'scope':'independent full nondegenerate synthetic arithmetic and static input/output contract review; final empirical full24 output review still required'})
    print(json.dumps({'passed':True,'out':str(out),'checks':len(checks),'numeric_comparisons':comparisons}))
if __name__=='__main__':main()
