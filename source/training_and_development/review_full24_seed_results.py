"""Independent source joins and derived arithmetic of the completed24 development summary."""
from pathlib import Path
import argparse,csv,datetime,hashlib,itertools,json,math,shutil,traceback
ROOT=Path(__file__).resolve().parent
BM=('capture_pct','captured_target_m2','actual_area_m2','actual_area_fraction','missed_target_m2','zero_annotation_area_m2','selected_zero_target_unit_area_m2','predicted_objective_m2','unused_budget_m2')
AM=('auc_0_1','auc_0_0p5_normalized');SM=('balanced_mae_pp','mae_pp','mae_zero_pp','mae_positive_pp','bias_pp')
MK=('run_id','encoder','variant','lr','seed','checkpoint_epoch','checkpoint_sha256')
FLAGS=('fallback_used','optimality_proven','original_proven_flag','bound_anomaly','certification_downgraded')
PP_EPS=1e-8;AUC_EPS=1e-12
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csvrows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def typed(row):
    result={}
    for k,v in row.items():
        if v=='':result[k]=None
        elif v in ('True','False'):result[k]=v=='True'
        else:
            try:result[k]=float(v)
            except (ValueError,TypeError):result[k]=v
    return result
def difference(a,b):return None if a is None or b is None else a-b
def direction(value,eps):return 'NA' if value is None else 'higher' if value>eps else 'lower' if value < -eps else 'tied'
def scoped(base,path):
    p=Path(path).resolve();assert p.is_relative_to(Path(base).resolve()),'Path escapes audited source root';return p
def same(actual,expected):
    if isinstance(expected,bool) or expected is None or isinstance(expected,str):assert actual==expected and (not isinstance(expected,bool) or type(actual) is bool),(actual,expected)
    else:assert actual is not None and math.isfinite(float(actual)) and math.isfinite(float(expected)) and math.isclose(float(actual),float(expected),rel_tol=2e-10,abs_tol=1e-10),(actual,expected)
def exact_keys(rows,keys,expected):
    actual=[tuple(r[k] for k in keys) for r in rows]
    assert len(actual)==len(set(actual)) and set(actual)==set(expected),'Duplicate/missing keys: '+str(keys)
def expected_models():
    result={}
    for seed,enc,lr,variant in itertools.product((17,29,43),('resnet18','convnextv2_tiny'),(.0001,.0003),('hard_seg','soft_seg')):
        rid=f"{'dev_common_v2' if seed==17 else 'dev_seed_v1'}_lr{'1e4' if lr==.0001 else '3e4'}_{enc}_{variant}_L256_s{seed}"
        result[rid]={'run_id':rid,'encoder':enc,'variant':variant,'lr':lr,'seed':seed}
    return result

def verify_table_inventory(tables,models):
    """Every one of25 tables has independently derived complete/unique keys."""
    declared=expected_models();assert len(models)==24 and {r['run_id'] for r in models}==set(declared)
    modelmap={r['run_id']:r for r in models};assert len(modelmap)==24
    for rid,m in modelmap.items():
        for k,v in declared[rid].items():assert m[k]==v
        assert type(m['checkpoint_epoch']) is int and 1<=m['checkpoint_epoch']<=24
        assert isinstance(m['checkpoint_sha256'],str) and len(m['checkpoint_sha256'])==64
    checked=set()
    def grid(name,keys,expected):exact_keys(tables[name],keys,expected);checked.add(name)
    b=tables['all2160_model_policy_budgets'];a=tables['all216_model_prefix_auc'];s=tables['all72_cpu_scalar_by_R']
    grid('all2160_model_policy_budgets',('run_id','M','R','budget','policy'),itertools.product(modelmap,(1,2,4),(1,2,4),(.1,.2,.3,.5,1.),('prefix','predicted_area')))
    grid('all216_model_prefix_auc',('run_id','M','R'),itertools.product(modelmap,(1,2,4),(1,2,4)))
    grid('all72_cpu_scalar_by_R',('run_id','R'),itertools.product(modelmap,(1,2,4)))
    grid('all24_cpu_scalar_model_summary',('run_id',),((rid,) for rid in modelmap))
    grid('all12_hard_soft_model_pairs',('encoder','lr','seed'),((r['encoder'],r['lr'],r['seed']) for r in models))
    for row in tables['all12_hard_soft_model_pairs']:
        for variant,prefix in [('hard_seg','hard'),('soft_seg','soft')]:
            m=next(r for r in models if (r['encoder'],r['lr'],r['seed'],r['variant'])==(row['encoder'],row['lr'],row['seed'],variant))
            assert row[prefix+'_run_id']==m['run_id']
    for rows in (b,a,s,tables['all24_cpu_scalar_model_summary']):
        for row in rows:
            for k in MK:assert row[k]==modelmap[row['run_id']][k],'Model/checkpoint identity drift '+k
    stats={}
    for name,source,dims,pairname in [('budget',b,('M','R','budget','policy'),'all1080_hard_soft_budget_pairs'),('auc',a,('M','R'),'all108_hard_soft_auc_pairs'),('cpu_scalar',s,('R',),'all36_hard_soft_cpu_scalar_pairs')]:
        pairkeys=('encoder','lr','seed')+dims
        grid(pairname,pairkeys,{tuple(r[k] for k in pairkeys) for r in source})
        basekeys=('encoder','variant','lr')+dims
        stats[name+'_three_seed_mean_sample_sd']=(source,basekeys)
        stats['hard_soft_'+name+'_three_seed_mean_sample_sd']=(tables[pairname],('encoder','lr')+dims)
        for axis in (('R',) if name=='cpu_scalar' else ('R','M')):
            others=tuple(k for k in dims if k!=axis);key=('run_id',)+others+('comparison','reference')
            ct='cpu_scalar_R_contrasts' if name=='cpu_scalar' else f'all_{name}_{axis}_contrasts'
            bases={tuple(r[k] for k in ('run_id',)+others) for r in source}
            grid(ct,key,{group+(hi,lo) for group in bases for hi,lo in ((4,1),(2,1),(4,2))})
            for row in tables[ct]:
                assert row['contrast']==f'{axis}{int(row["comparison"])}_minus_{axis}{int(row["reference"])}'
                for k in MK:assert row[k]==modelmap[row['run_id']][k]
            if name!='cpu_scalar':stats[f'{name}_{axis}_contrast_three_seed_mean_sample_sd']=(tables[ct],('encoder','variant','lr')+others+('contrast',))
    for name,(source,keys) in stats.items():
        grid(name,keys,{tuple(r[k] for k in keys) for r in source})
        for row in tables[name]:assert row['seeds']=='17;29;43' and row['n_expected']==3 and row['independent_fields']==1
    for name,pairs,keys,metrics in [('budget_hard_soft_direction_counts',tables['all1080_hard_soft_budget_pairs'],('encoder','lr','seed','policy','M'),('soft_minus_hard_capture_pct',)),('auc_hard_soft_direction_counts',tables['all108_hard_soft_auc_pairs'],('encoder','lr','seed','M'),tuple('soft_minus_hard_'+k for k in AM))]:
        groups={tuple(r[k] for k in keys) for r in pairs};grid(name,keys+('metric',),{group+(metric,) for group in groups for metric in metrics})
    assert checked==set(tables) and len(checked)==25,'Unreviewed/missing/extra table'
    return stats

def audit(folder,write_review=True):
    folder=Path(folder).resolve();assert folder.is_relative_to(ROOT/'precision_agriculture')
    watched={str(p):sha(p) for p in (Path(__file__),ROOT/'seed_stability_summary_v1.py',folder/'status.json',folder/'artifact_manifest.json')}
    status=read(folder/'status.json');manifest=read(folder/'artifact_manifest.json');summary=read(folder/'summary.json')
    assert status['state']=='completed' and status['artifact_manifest_sha256']==sha(folder/'artifact_manifest.json') and summary['models']==24 and summary['seeds']==[17,29,43]
    for name,h in manifest['file_sha256'].items():assert sha(scoped(folder,folder/name))==h
    assert {'summary.json','models.json','source_ledger.json'}<=set(manifest['file_sha256'])
    assert status['source_sha256']==sha(ROOT/'seed_stability_summary_v1.py')
    ledger=read(folder/'source_ledger.json')
    bound={}
    for item in ledger:
        p=scoped(ROOT,item['path']);assert p.suffix.lower() in ('.json','.csv','.py','.md') and 'test' not in p.name.lower() and 'r6' not in p.name.lower()
        assert str(p) not in bound;bound[str(p)]=item['sha256']
        assert sha(p)==item['sha256']==sha(scoped(folder,folder/item['archived']))
    def source(path,expected=None):
        path=scoped(ROOT,path);assert str(path) in bound,'Unsealed source reference: '+str(path)
        assert sha(path)==bound[str(path)]
        if expected is not None:assert expected==bound[str(path)]
        return path
    matches=[x for x in ledger if x['sha256']==status['root_review_sha256']];assert len(matches)==1
    root_review=read(source(matches[0]['path']));assert root_review['passed'] and root_review['all24_required'] and root_review['test_or_R6_accessed'] is False
    assert root_review['execution_scope']=='seed_stability_summary_v1_full24' and root_review['implementation_sha256']==status['source_sha256']
    tables={p.stem:[typed(r) for r in csvrows(p)] for p in folder.glob('*.csv')}
    assert {p.name for p in folder.glob('*.csv')}<={Path(k).as_posix() for k in manifest['file_sha256']},'Unsealed CSV'
    assert {k:len(v) for k,v in tables.items()}==summary['table_rows'] and len(tables)==25
    b=tables['all2160_model_policy_budgets'];a=tables['all216_model_prefix_auc'];sc=tables['all72_cpu_scalar_by_R']
    assert (len(b),len(a),len(sc))==(2160,216,72)
    models=read(folder/'models.json');modelids=[x['run_id'] for x in models];assert len(modelids)==len(set(modelids))==24
    verify_table_inventory(tables,models)
    expected=set(itertools.product(modelids,(1,2,4),(1,2,4),(.1,.2,.3,.5,1.),('prefix','predicted_area')))
    assert {(r['run_id'],r['M'],r['R'],r['budget'],r['policy']) for r in b}==expected
    assert {(r['run_id'],r['M'],r['R']) for r in a}==set(itertools.product(modelids,(1,2,4),(1,2,4)))
    assert {(r['run_id'],r['R']) for r in sc}==set(itertools.product(modelids,(1,2,4)))
    comparisons=0
    def close(x,y):
        nonlocal comparisons
        assert (x is None and y is None) or (x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y)) and math.isclose(float(x),float(y),rel_tol=2e-10,abs_tol=1e-10)),(x,y)
        comparisons+=1
    oldb={(r['run_id'],float(r['M']),float(r['R']),float(r['budget']),r['policy']):typed(r) for r in csvrows(source(**{ 'path':root_review['seed17']['budget_csv']['path'],'expected':root_review['seed17']['budget_csv']['sha256']}))}
    olda={(r['run_id'],float(r['M']),float(r['R'])):typed(r) for r in csvrows(source(root_review['seed17']['auc_csv']['path'],root_review['seed17']['auc_csv']['sha256']))}
    news={};newauc={};cpus={};newhomes={}
    assert len(root_review['seed17_exports'])==8 and {e['run_id'] for e in root_review['seed17_exports']}=={m['run_id'] for m in models if m['seed']==17}
    assert len(root_review['new_models'])==16 and {e['run_id'] for e in root_review['new_models']}=={m['run_id'] for m in models if m['seed']!=17}
    for entry in root_review['seed17_exports']:cpus[entry['run_id']]=read(source(entry['summary']['path'],entry['summary']['sha256']))
    for entry in root_review['new_models']:
        rec=read(source(entry['completion_receipt']['path'],entry['completion_receipt']['sha256']));home=Path(rec['spatial_directory']);meta=read(source(home/'summary.json',rec['spatial_summary_sha256']))
        newhomes[entry['run_id']]=home
        assert rec['run_id']==entry['run_id'] and meta['run_id']==entry['run_id']
        for row in csvrows(source(home/'all_45_policy_outcomes.csv')):news[entry['run_id'],float(row['M']),float(row['R']),float(row['budget'])]=(typed(row),meta)
        for row in csvrows(source(home/'prefix_auc.csv')):newauc[entry['run_id'],float(row['M']),float(row['R'])]=typed(row)
        cpus[entry['run_id']]=read(source(Path(rec['export_directory'])/'summary.json'))
    assert len(news)==16*45 and len(newauc)==16*9 and len(cpus)==24
    for row in b:
        key=row['run_id'],row['M'],row['R'],row['budget'],row['policy']
        if row['seed']==17:
            source=oldb[key]
            for metric in BM+('budget_area_m2','domain_area_m2','domain_target_m2','selected_count'):close(row[metric],source[metric])
            for field in ('fallback_used','optimality_proven','original_proven_flag','bound_anomaly','certification_downgraded'):assert row[field]==source[field]
            for field,value in source.items():
                if field!='original_gpu_mae_pp':same(row[field],value)
        else:
            source,meta=news[key[:4]];prefix='prefix' if row['policy']=='prefix' else 'pmilp'
            for metric in BM:
                expected_value=100*source[prefix+'_capture_fraction'] if metric=='capture_pct' else (source['budget_area_m2']-source[prefix+'_actual_area_m2'] if metric=='unused_budget_m2' else source[prefix+'_'+metric])
                close(row[metric],expected_value)
            close(row['domain_area_m2'],meta['domain']['area_m2']);close(row['domain_target_m2'],meta['domain']['target_m2'])
            close(row['budget_area_m2'],source['budget_area_m2']);close(row['selected_count'],source[prefix+'_selected_count'])
            for field in MK:same(row[field],meta[field])
            assert row['selected_ids_json']==source['prefix_ids_json' if prefix=='prefix' else 'selected_ids_json']
            assert Path(row['source_csv']).resolve()==(newhomes[row['run_id']]/'all_45_policy_outcomes.csv').resolve()
            assert row['source_csv_sha256']==bound[str(Path(row['source_csv']).resolve())]
            if prefix=='pmilp':
                for dst,src in [('fallback_used','fallback_used'),('optimality_proven','optimality_proven'),('original_proven_flag','original_proven'),('bound_anomaly','bound_anomaly'),('certification_downgraded','certification_downgraded')]:assert row[dst]==source[src]
                for field in ('predicted_upper_m2','predicted_relative_gap','raw_solver_gap'):close(row[field],source[field])
                same(row['raw_solver_status'],source['raw_solver_status']);assert row['bound_audit_json']==source['bound_audit_json']
                close(row['original_retained_upper_m2'],json.loads(source['bound_audit_json'])['retained_upper_m2']);assert row['original_retained_gap'] is None
            else:
                assert row['fallback_used'] is False
                assert all(row[field] is None for field in FLAGS[1:]+('predicted_upper_m2','predicted_relative_gap','original_retained_upper_m2','original_retained_gap','raw_solver_status','raw_solver_gap','bound_audit_json'))
        assert row['actual_area_m2']<=row['budget_area_m2']
        close(row['budget_area_m2'],row['budget']*row['domain_area_m2']);close(row['actual_area_fraction'],row['actual_area_m2']/row['domain_area_m2'])
        close(row['unused_budget_m2'],row['budget_area_m2']-row['actual_area_m2'])
        if row['capture_pct'] is not None:close(row['capture_pct'],100*row['captured_target_m2']/row['domain_target_m2'])
        assert not (row['fallback_used'] and row['optimality_proven'])
    assert len({(r['domain_area_m2'],r['domain_target_m2']) for r in b})==1
    for row in a:
        key=row['run_id'],row['M'],row['R'];source=olda[key] if row['seed']==17 else newauc[key]
        for metric in AM:close(row[metric],source[metric])
        close(row['curve_points'],source['curve_points'])
        assert row['source_csv_sha256']==bound[str(Path(row['source_csv']).resolve())]
    for row in sc:
        source=cpus[row['run_id']];scale=source['scales'][str(int(row['R']))]
        for metric in SM:close(row[metric],100*scale[metric[:-3]])
        assert source['checkpoint_sha256']==row['checkpoint_sha256'] and source['checkpoint_epoch']==row['checkpoint_epoch']
        assert row['n']==357 and row['n_positive']==27 and row['device']=='cpu' and row['precision']=='float32'
        assert row['summary_sha256']==bound[str(Path(row['summary_path']).resolve())]
        close(row['cpu_three_R_balanced_mae_pp'],100*source['selection_balanced_mae']);close(row['original_amp_selection_mae_pp'],100*source['original_amp_validation']['selection_balanced_mae'])
        close(row['cpu_minus_original_amp_selection_pp'],row['cpu_three_R_balanced_mae_pp']-row['original_amp_selection_mae_pp']);assert row['epoch_reselected'] is False
    for row in tables['all24_cpu_scalar_model_summary']:
        source=cpus[row['run_id']];mean=math.fsum(source['scales'][str(r)]['balanced_mae']*100 for r in (1,2,4))/3
        amp=source['original_amp_validation']['selection_balanced_mae']*100
        close(row['cpu_three_R_balanced_mae_pp'],mean);close(row['original_amp_selection_mae_pp'],amp);close(row['cpu_minus_original_amp_selection_pp'],mean-amp)
        assert row['epoch_reselected'] is False
    for name,source,dims,metrics,pairname in [('budget',b,('M','R','budget','policy'),BM,'all1080_hard_soft_budget_pairs'),('auc',a,('M','R'),AM,'all108_hard_soft_auc_pairs'),('cpu_scalar',sc,('R',),SM,'all36_hard_soft_cpu_scalar_pairs')]:
        lookup={(r['encoder'],r['lr'],r['seed'],r['variant'])+tuple(r[k] for k in dims):r for r in source}
        for row in tables[pairname]:
            shared=row['encoder'],row['lr'],row['seed'];grid=tuple(row[k] for k in dims);hard=lookup[shared+('hard_seg',)+grid];soft=lookup[shared+('soft_seg',)+grid]
            assert row['hard_run_id']==hard['run_id'] and row['soft_run_id']==soft['run_id']
            assert row['hard_checkpoint_epoch']==hard['checkpoint_epoch'] and row['soft_checkpoint_epoch']==soft['checkpoint_epoch']
            for metric in metrics:
                delta=difference(soft[metric],hard[metric]);close(row['hard_'+metric],hard[metric]);close(row['soft_'+metric],soft[metric]);close(row['soft_minus_hard_'+metric],delta)
                if metric in ('capture_pct','balanced_mae_pp')+AM:assert row[metric+'_direction']==direction(delta,AUC_EPS if metric in AM else PP_EPS)
        for axis in (('R',) if name=='cpu_scalar' else ('R','M')):
            others=tuple(k for k in dims if k!=axis);look={(r['run_id'],)+tuple(r[k] for k in others)+(r[axis],):r for r in source}
            for row in tables['cpu_scalar_R_contrasts' if name=='cpu_scalar' else f'all_{name}_{axis}_contrasts']:
                shared=(row['run_id'],)+tuple(row[k] for k in others);lo=look[shared+(row['reference'],)];hi=look[shared+(row['comparison'],)]
                for metric in metrics:close(row['reference_'+metric],lo[metric]);close(row['comparison_'+metric],hi[metric]);close(row['difference_'+metric],difference(hi[metric],lo[metric]))
    bases={'budget_three_seed_mean_sample_sd':b,'auc_three_seed_mean_sample_sd':a,'cpu_scalar_three_seed_mean_sample_sd':sc,
      'hard_soft_budget_three_seed_mean_sample_sd':tables['all1080_hard_soft_budget_pairs'],'hard_soft_auc_three_seed_mean_sample_sd':tables['all108_hard_soft_auc_pairs'],
      'hard_soft_cpu_scalar_three_seed_mean_sample_sd':tables['all36_hard_soft_cpu_scalar_pairs']}
    for name in ('budget','auc'):
        for axis in ('R','M'):bases[f'{name}_{axis}_contrast_three_seed_mean_sample_sd']=tables[f'all_{name}_{axis}_contrasts']
    for table,source in bases.items():
        for row in tables[table]:
            keys=[k for k in row if k not in ('seeds','n_expected','independent_fields') and not k.endswith(('_mean','_sample_sd','_n_valid'))]
            group=[r for r in source if all(r[k]==row[k] for k in keys)];assert len(group)==3 and {r['seed'] for r in group}=={17,29,43}
            for field in row:
                if not field.endswith('_mean'):continue
                metric=field[:-5];values=[r[metric] for r in group if r[metric] is not None];mean=math.fsum(values)/len(values) if values else None
                sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/(len(values)-1)) if len(values)>1 else None
                close(row[field],mean);close(row[metric+'_sample_sd'],sd);close(row[metric+'_n_valid'],len(values))
    for table in ('budget_hard_soft_direction_counts','auc_hard_soft_direction_counts'):
        source=tables['all1080_hard_soft_budget_pairs' if table.startswith('budget') else 'all108_hard_soft_auc_pairs']
        for row in tables[table]:
            keys=('encoder','lr','seed','M')+(('policy',) if table.startswith('budget') else ())
            values=[r[row['metric']] for r in source if all(r[k]==row[k] for k in keys)];eps=row['tolerance']
            assert eps==(PP_EPS if table.startswith('budget') else AUC_EPS)
            close(row['rows'],len(values))
            for label in ('higher','lower','tied','NA'):close(row[label],sum(direction(v,eps)==label for v in values))
            finite=[v for v in values if v is not None];close(row['minimum'],min(finite) if finite else None);close(row['maximum'],max(finite) if finite else None)
            assert row['interpretation']=='descriptive cell counts; not independent field replicates'
    for flag in FLAGS:assert summary['recorded_flags'][flag]==sum(r[flag] is True for r in b if r['policy']=='predicted_area')
    assert summary['test_or_R6_accessed'] is False and summary['new_model_calls']==summary['new_solver_calls']==0 and summary['field_replicates']==1
    for item in ledger:assert sha(item['path'])==item['sha256']
    for name,h in manifest['file_sha256'].items():assert sha(folder/name)==h
    for path,digest in watched.items():assert sha(path)==digest,'Reviewer/producer/status changed during audit'
    review={'passed':True,'scope':'full24_seed_summary_independent','audited_artifact_manifest_sha256':sha(folder/'artifact_manifest.json'),
      'summary_sha256':sha(folder/'summary.json'),'run_ids':modelids,'models':24,'budget_rows':2160,'auc_rows':216,'CPU_rows':72,
      'numeric_comparisons':comparisons,'reviewer_sha256':sha(__file__),'output_directory':str(folder),'source_ledger_sha256':sha(folder/'source_ledger.json'),
      'test_or_R6_accessed':False,'new_model_or_solver_calls':0,'method':'Independent original/wide source joins and arithmetic; does not repeat prior map and solver-evidence numerical audits.'}
    review['tables_audited']=25;review['table_rows']=summary['table_rows'];review['all_tables_have_complete_unique_keys']=True
    if not write_review:return review
    out=ROOT/'provenance'/('full24_seed_summary_independent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    shutil.copy2(__file__,out/'review_full24_seed_results.py')
    (out/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'passed':True,'review':str(out/'review.json'),'numeric_comparisons':comparisons}))
    return review
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-directory',required=True);args=parser.parse_args();audit(args.output_directory)
