"""Confirmation report tables from accepted saved outcomes only; no array/model/solver imports.

Dataset accounting is kept separate. This module never opens sample truths,
input manifests, raster files, NPZ caches, or checkpoint tensors.
"""
from pathlib import Path
import argparse, csv, itertools, math, shutil, statistics
import core

HERE = Path(__file__).resolve().parent
RS = (1, 2, 4, 6)
MS = (1, 2, 4)
GRIDS = (4, 8, 16)
BS = (.1, .2, .3, .5, 1.)
CODES = ('b10', 'b20', 'b30', 'b50', 'b100')
BASES = ('nominal_primary', 'matched_actual_secondary')
R_PAIRS = ((4, 1), (2, 1), (4, 2), (6, 1), (6, 4))
M_PAIRS = ((4, 1), (2, 1), (4, 2))
G_PAIRS = ((16, 4), (8, 4), (16, 8))
STAGES = ('scalar_outcomes', 'local_outcomes', 'external_outcomes')
META = ('run_id', 'dataset', 'encoder', 'variant', 'lr', 'seed', 'best_epoch', 'checkpoint_sha256')
SCALAR_METRICS = ('mae_pp', 'bias_pp', 'mae_zero_pp', 'mae_positive_pp', 'balanced_mae_pp')
LOCAL_METRICS = ('capture_pct', 'captured_target_m2', 'actual_area_m2', 'unused_budget_m2',
                 'missed_target_m2', 'selected_non_target_area_m2')
EXTERNAL_METRICS = ('weed_capture_pct', 'crop_capture_pct', 'crop_fraction_pct', 'selected_area_FAE',
                    'selected_weed_FAE', 'selected_crop_FAE', 'unselected_weed_FAE',
                    'selected_background_FAE', 'unused_effective_budget_area_FAE')
AUC_METRICS = ('auc_0_1', 'auc_0_0p5_normalized')


def number(value, optional=False):
    if value is None or value == '':
        core.require(optional, 'Missing required number')
        return None
    x = float(value); core.require(math.isfinite(x), 'Nonfinite metric'); return x


def boolean(value, optional=False):
    if value is None or value == '':
        core.require(optional, 'Missing required boolean'); return None
    core.require(value in (True, False, 'True', 'False', 'true', 'false'), 'Invalid boolean')
    return value in (True, 'True', 'true')


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path, values):
    core.require(values, 'Empty declared table')
    keys = list(dict.fromkeys(k for r in values for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(values)


def delta(a, b):
    return None if a is None or b is None else a - b


def close(a, b, tolerance=1e-9):
    core.require(math.isclose(a, b, rel_tol=1e-10, abs_tol=tolerance), f'Different saved quantities: {a}, {b}')


def metadata(row, models):
    model = models[row['run_id']]
    result = {k: model[k] for k in META}
    for k in ('dataset', 'encoder', 'variant'):
        if k in row: core.require(row[k] == model[k], 'Saved model identity changed: ' + k)
    for k in ('seed', 'lr'):
        if k in row: core.require(number(row[k]) == model[k], 'Saved model setting changed: ' + k)
    for k in ('best_epoch', 'original_selected_epoch'):
        if k in row: core.require(int(row[k]) == model['best_epoch'], 'Original selected epoch changed')
    if 'checkpoint_sha256' in row: core.require(row['checkpoint_sha256'] == model['checkpoint_sha256'], 'Saved checkpoint changed')
    return result


def normalize_scalar(source, models):
    result = []
    for row in source:
        r = dict(row); r.update(metadata(row, models)); r['R'] = int(row['R'])
        for k in ('n', 'n_zero', 'n_positive'): r[k] = int(row[k])
        core.require(r['n_zero'] + r['n_positive'] == r['n'], 'Stratum counts do not sum')
        core.require(r['n'] == (338 if r['dataset']=='local' else 26), 'Wrong confirmation scalar membership count')
        for metric in SCALAR_METRICS:
            x = number(row[metric[:-3]], optional=True); r[metric] = None if x is None else 100*x
        core.require(r['mae_pp'] is not None and r['bias_pp'] is not None, 'Ordinary scalar error undefined')
        core.require((r['mae_zero_pp'] is None) == (r['n_zero']==0), 'Zero-stratum NA mismatch')
        core.require((r['mae_positive_pp'] is None) == (r['n_positive']==0), 'Positive-stratum NA mismatch')
        if r['mae_zero_pp'] is None or r['mae_positive_pp'] is None:
            core.require(r['balanced_mae_pp'] is None, 'Undefined balanced metric filled')
        else: close(r['balanced_mae_pp'],(r['mae_zero_pp']+r['mae_positive_pp'])/2)
        result.append(r)
    return result


def normalize_local(source, models, auc=False):
    result = []
    for row in source:
        r = dict(row); r.update(metadata(row, models)); r.update(M=int(row['M']),R=int(row['R']),dataset='local')
        core.require(models[r['run_id']]['dataset']=='local' and models[r['run_id']]['variant']!='regression', 'Wrong local spatial model')
        if auc:
            r.update(policy='prefix',budget_basis='complete_prefix_once')
            r['auc_0_1']=number(row['auc_0_1'],True)
            r['auc_0_0p5_normalized']=number(row['auc_0_half_normalized'],True)
        else:
            r['budget']=number(row['budget']);r['original_policy']=row['policy']
            core.require(row['policy'] in ('model_prefix','predicted_area'), 'Wrong local policy')
            r['policy']='prefix' if row['policy']=='model_prefix' else 'predicted_area'
            for k in LOCAL_METRICS:
                source_key='capture_fraction' if k=='capture_pct' else 'unused_area_m2' if k=='unused_budget_m2' else k
                value=number(row[source_key],optional=k=='capture_pct')
                r[k]=None if value is None else value*(100 if k=='capture_pct' else 1)
            r['budget_area_m2']=r['actual_area_m2']+r['unused_budget_m2']
            r['domain_area_m2']=r['budget_area_m2']/r['budget']
            r['domain_target_m2']=r['captured_target_m2']+r['missed_target_m2']
            r['actual_area_fraction']=r['actual_area_m2']/r['domain_area_m2']
            core.require(r['unused_budget_m2']>=0 and r['actual_area_m2']>=0, 'Invalid saved actual/budget area')
            if r['domain_target_m2']==0: core.require(r['capture_pct'] is None,'Zero target capture fabricated')
            else: close(r['capture_pct'],100*r['captured_target_m2']/r['domain_target_m2'])
            r['capture_na_reason']='zero_total_target' if r['capture_pct'] is None else ''
            for k in ('fallback_used','bound_anomaly','updated_proven_flag','certification_downgraded'):
                r[k]=boolean(row.get(k),optional=True)
            for k in ('updated_upper_m2','updated_relative_gap'):r[k]=number(row.get(k),True)
            core.require(not r['fallback_used'] or r['updated_proven_flag'] is False,'Fallback cannot be certified')
        result.append(r)
    return result


def normalize_external(source, models, auc=False):
    result=[]
    for row in source:
        r=dict(row);r.update(metadata(row,models));r.update(dataset='weeds',R=int(row['R']),grid_g=int(row['grid_g']))
        core.require(models[r['run_id']]['dataset']=='weeds' and models[r['run_id']]['variant']!='regression','Wrong external spatial model')
        r['frame_n']=int(row['frame_n'])
        if auc:
            r['budget_basis']='complete_prefix_once'
            for k in AUC_METRICS:r[k]=number(row[k],True)
            total=number(row['total_weed_FAE'])
            core.require(all((r[k] is None)==(total==0) for k in AUC_METRICS),'External AUC zero-target NA mismatch')
        else:
            for k in ('k','N'):r[k]=int(row[k])
            for k in ('effective_budget_fraction','parent_nominal_budget_fraction','actual_fraction'):r[k]=number(row[k])
            for metric in EXTERNAL_METRICS:
                key={'weed_capture_pct':'weed_capture_ratio','crop_capture_pct':'crop_capture_ratio','crop_fraction_pct':'crop_fraction_of_selected_area'}.get(metric,metric)
                value=number(row[key],optional=metric.endswith('_pct'))
                r[metric]=None if value is None else value*(100 if metric.endswith('_pct') else 1)
            core.require(r['N']==r['grid_g']**2 and r['k']/r['N']==r['actual_fraction'],'Grid area accounting mismatch')
            close(r['selected_area_FAE'],r['actual_fraction']*r['frame_n'],1e-12)
            core.require(r['budget_basis'] in BASES and r['parent_nominal_budget_code'] in CODES,'Unexpected external budget')
            b=BS[CODES.index(r['parent_nominal_budget_code'])]
            expected_k=(r['N']*({.1:1,.2:1,.3:3,.5:1,1.:1}[b])//({.1:10,.2:5,.3:10,.5:2,1.:1}[b])) if r['budget_basis']==BASES[0] else r['N']*int(math.floor(16*b))/16
            core.require(r['k']==expected_k,'Saved per-frame whole-unit allocation changed')
            # Preserve original alias identifiers. No new independent selection is inferred.
            r['duplicate_selection_alias']=row.get('duplicate_selection_alias') or ''
            for metric,numerator,denominator in (('weed_capture_pct','selected_weed_FAE','total_weed_FAE'),
                    ('crop_capture_pct','selected_crop_FAE','total_crop_FAE'),('crop_fraction_pct','selected_crop_FAE','selected_area_FAE')):
                den=number(row[denominator]);num=number(row[numerator])
                core.require((r[metric] is None)==(den==0),'External zero-denominator NA mismatch')
                if den:close(r[metric],100*num/den,1e-10)
        result.append(r)
    return result


def exact_grid(values, keys, expected):
    actual=[tuple(r[k] for k in keys) for r in values]
    core.require(len(actual)==len(set(actual)) and set(actual)==set(expected),'Missing/duplicate declared table cell: '+','.join(keys))


def validate_population(scalar, local, local_auc, external, external_auc, models):
    local_ids=[k for k,v in models.items() if v['dataset']=='local' and v['variant']!='regression']
    external_ids=[k for k,v in models.items() if v['dataset']=='weeds' and v['variant']!='regression']
    core.require(len(models)==37 and len(local_ids)==24 and len(external_ids)==6,'Wrong frozen report model population')
    exact_grid(scalar,('run_id','R'),itertools.product(models,RS))
    exact_grid(local,('run_id','M','R','budget','policy'),itertools.product(local_ids,MS,RS,BS,('prefix','predicted_area')))
    exact_grid(local_auc,('run_id','M','R'),itertools.product(local_ids,MS,RS))
    local_total={(r['run_id'],r['M'],r['R']):r['domain_target_m2'] for r in local}
    for r in local_auc:
        core.require(all((r[k] is None)==(local_total[r['run_id'],r['M'],r['R']]==0) for k in AUC_METRICS),'Local AUC zero-target NA mismatch')
    frame=[r for r in external if r['scope']=='frame']
    frame_meta={}
    for r in frame:
        core.require(r['scope_id']==r['frame_id'] and r['frame_n']==1,'Frame accounting mismatch')
        date=r['acquisition_date'];core.require(bool(date),'Missing acquisition date')
        core.require(r['frame_id'] not in frame_meta or frame_meta[r['frame_id']]==date,'Inconsistent frame date')
        frame_meta[r['frame_id']]=date
    core.require(len(frame_meta)==26,'Full26 external frames required')
    scopes={'frame':sorted(frame_meta),'date':sorted(set(frame_meta.values())),'pooled':['all26']}
    for scope, ids in scopes.items():
        subset=[r for r in external if r['scope']==scope];subset_auc=[r for r in external_auc if r['scope']==scope]
        exact_grid(subset,('run_id','scope_id','grid_g','R','budget_basis','parent_nominal_budget_code'),
                   itertools.product(external_ids,ids,GRIDS,RS,BASES,CODES))
        exact_grid(subset_auc,('run_id','scope_id','grid_g','R'),itertools.product(external_ids,ids,GRIDS,RS))
        for r in subset+subset_auc:
            expected=1 if scope=='frame' else 26 if scope=='pooled' else sum(d==r['scope_id'] for d in frame_meta.values())
            core.require(r['frame_n']==expected,'Incorrect grouped-frame denominator')
    core.require({r['scope'] for r in external}==set(scopes) and {r['scope'] for r in external_auc}==set(scopes),'Unknown/missing external scope')
    return frame_meta


def pair_rows(values, dimensions, metrics):
    keys=('dataset','encoder','lr','seed')+dimensions;groups={}
    for r in values:
        if r['variant']=='regression':continue
        group=groups.setdefault(tuple(r[k] for k in keys),{})
        core.require(r['variant'] not in group,'Duplicate pair member');group[r['variant']]=r
    output=[]
    for key,group in sorted(groups.items()):
        core.require(set(group)=={'hard_seg','soft_seg'},'Incomplete hard/soft match')
        h,s=group['hard_seg'],group['soft_seg'];r=dict(zip(keys,key),hard_run_id=h['run_id'],soft_run_id=s['run_id'])
        for metric in metrics:
            r['hard_'+metric]=h[metric];r['soft_'+metric]=s[metric];r['soft_minus_hard_'+metric]=delta(s[metric],h[metric])
        output.append(r)
    return output


def contrast_rows(values, axis, dimensions, pairs, metrics):
    keys=META+dimensions;groups={}
    for r in values:
        group=groups.setdefault(tuple(r[k] for k in keys),{})
        core.require(r[axis] not in group,'Duplicate scale member');group[r[axis]]=r
    output=[]
    needed={n for pair in pairs for n in pair}
    for key,group in sorted(groups.items()):
        core.require(set(group)==needed,'Incomplete scale contrast population')
        for high,low in pairs:
            r=dict(zip(keys,key),contrast=f'{axis}{high}_minus_{axis}{low}',comparison=high,reference=low)
            for metric in metrics:
                r['comparison_'+metric]=group[high][metric];r['reference_'+metric]=group[low][metric]
                r['difference_'+metric]=delta(group[high][metric],group[low][metric])
            output.append(r)
    return output


def seed_stats(values, group_keys, metrics, models):
    groups={}
    for r in values:
        group=groups.setdefault(tuple(r[k] for k in group_keys),{})
        core.require(r['seed'] not in group,'Duplicate fitting seed');group[r['seed']]=r
    output=[]
    for key,group in sorted(groups.items()):
        r=dict(zip(group_keys,key));first=next(iter(group.values()))
        expected={v['seed'] for v in models.values() if v['dataset']==first['dataset'] and v['encoder']==first['encoder'] and v['lr']==first['lr'] and (v['variant']==first['variant'] if 'variant' in first else v['variant']!='regression')}
        core.require(set(group)==expected,'Incomplete declared fitting-seed group')
        r.update(seeds=';'.join(map(str,sorted(group))),n_models=len(group),independent_field_replicates=1)
        for metric in metrics:
            vals=[x[metric] for x in group.values()];defined=[x for x in vals if x is not None]
            r[metric+'_defined_models']=len(defined)
            # Preserve undefined denominator across the complete seed population.
            r[metric+'_mean']=statistics.fmean(defined) if len(defined)==len(vals) else None
            r[metric+'_sample_sd']=statistics.stdev(defined) if len(vals)>1 and len(defined)==len(vals) else None
        output.append(r)
    return output


def family_tables(values, dimensions, metrics, models, extra_axis=None):
    """One family at a time, preserving dataset/scope/budget-basis in every key."""
    result={'all':values}
    paired=pair_rows(values,dimensions,metrics);result['hard_soft_pairs']=paired
    group=('dataset','encoder','variant','lr')+dimensions
    result['seed_mean_sample_sd']=seed_stats(values,group,metrics,models)
    result['hard_soft_seed_mean_sample_sd']=seed_stats(paired,('dataset','encoder','lr')+dimensions,tuple('soft_minus_hard_'+m for m in metrics),models)
    for axis,pairs in [('R',R_PAIRS)]+([extra_axis] if extra_axis else []):
        other=tuple(k for k in dimensions if k!=axis)
        contrast=contrast_rows(values,axis,other,pairs,metrics);result[axis+'_contrasts']=contrast
        result[axis+'_contrast_seed_mean_sample_sd']=seed_stats(contrast,('dataset','encoder','variant','lr')+other+('contrast',),tuple('difference_'+m for m in metrics),models)
    return result


def gate(root_review):
    initial=core.sha(__file__)
    contract,roster,_=core.validate_freeze(check_models=False)
    lock=core.validate_source_lock();core.require(lock['files'].get('report_tables.py')==initial,'Complete freeze must bind report consumer')
    review=core.read(root_review)
    core.require(review.get('passed') is True and review.get('scope')=='confirmation_report_tables','Explicit report execution review required')
    core.require(review['implementation_sha256']==initial and review['freeze_sha256']==core.sha(HERE/'FREEZE.json'),'Report review identity differs')
    core.require(review.get('all_three_outcomes_accepted') is True and [r['stage'] for r in review['stages']]==list(STAGES),'All three outcome stages must be accepted')
    watched={str(Path(__file__).resolve()):initial,str(Path(root_review).resolve()):core.sha(root_review)}
    for name in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json'):watched[str(HERE/name)]=core.sha(HERE/name)
    folders={};audits=[]
    for entry in review['stages']:
        folder=HERE/'stages'/entry['stage'];core.completed_stage(folder)
        core.bound_file(folder/'artifact_manifest.json',entry['artifact_manifest_sha256'])
        core.bound_file(folder/'summary.json',entry['summary_sha256'])
        pin=entry['independent_review'];core.bound_file(pin['path'],pin['sha256']);audit=core.read(pin['path'])
        core.require(audit.get('passed') is True and audit.get('stage')==entry['stage'],'Independent stage audit required')
        core.require(audit.get('audited_artifact_manifest_sha256')==entry['artifact_manifest_sha256'] and audit.get('freeze_sha256')==review['freeze_sha256'],'Audit must bind actual completed stage and freeze')
        for path,h in ((folder/'artifact_manifest.json',entry['artifact_manifest_sha256']),(folder/'summary.json',entry['summary_sha256']),(Path(pin['path']),pin['sha256'])):watched[str(path.resolve())]=h
        folders[entry['stage']]=folder;audits.append(pin)
    return contract,{r['run_id']:r for r in roster},folders,watched,audits


def read_bound_table(folder,name,watched):
    # Only explicitly named aggregate outcome CSVs enter analysis, never truths/samples.
    path=folder/name;manifest=core.read(folder/'artifact_manifest.json')
    core.bound_file(path,manifest['files'][name]);watched[str(path.resolve())]=manifest['files'][name]
    return rows(path)


def load_tables(folders,models,watched):
    scalar=normalize_scalar(read_bound_table(folders['scalar_outcomes'],'all_model_R_metrics.csv',watched),models)
    local=normalize_local(read_bound_table(folders['local_outcomes'],'models/all_fixed_budgets.csv',watched),models)
    local_auc=normalize_local(read_bound_table(folders['local_outcomes'],'models/all_prefix_auc.csv',watched),models,True)
    external=[];external_auc=[]
    for scope in ('frame','date','pooled'):
        external+=normalize_external(read_bound_table(folders['external_outcomes'],f'learned_{scope}_budget_metrics.csv',watched),models)
        external_auc+=normalize_external(read_bound_table(folders['external_outcomes'],f'learned_{scope}_auc.csv',watched),models,True)
    frame_meta=validate_population(scalar,local,local_auc,external,external_auc,models)
    return scalar,local,local_auc,external,external_auc,frame_meta


def copy_saved_table(folder,name,out,name_out,watched):
    manifest=core.read(folder/'artifact_manifest.json');path=folder/name
    core.require(path.suffix.lower() in ('.csv','.json'),'Only saved table/summary copies allowed')
    core.bound_file(path,manifest['files'][name]);watched[str(path.resolve())]=manifest['files'][name]
    dest=out/name_out;shutil.copyfile(path,dest);core.bound_file(dest,manifest['files'][name])
    return {'file':dest.name,'source':str(path),'sha256':manifest['files'][name],'format':'unaltered saved producer table'}


def execute(root_review):
    contract,models,folders,watched,audits=gate(root_review)
    scalar,local,local_auc,external,external_auc,frame_meta=load_tables(folders,models,watched)
    # No claim until every producer, acceptance, identity and full table grid is available.
    with core.attempt('report_tables') as out:
        core.write(out/'begin_binding.json',{'watched_sha256':watched,'root_review':core.binding(root_review),'independent_reviews':audits})
        for dataset in ('local','weeds'):(out/dataset).mkdir()
        ledger=[];counts={}
        families=[('local','scalar',[r for r in scalar if r['dataset']=='local'],('R',),SCALAR_METRICS,None),
                  ('weeds','scalar',[r for r in scalar if r['dataset']=='weeds'],('R',),SCALAR_METRICS,None),
                  ('local','spatial_budget',local,('M','R','budget','policy'),LOCAL_METRICS,('M',M_PAIRS)),
                  ('local','prefix_auc',local_auc,('M','R'),AUC_METRICS,('M',M_PAIRS))]
        for scope in ('frame','date','pooled'):
            families.append(('weeds',scope+'_spatial_budget',[r for r in external if r['scope']==scope],('scope','scope_id','grid_g','R','budget_basis','parent_nominal_budget_code'),EXTERNAL_METRICS,('grid_g',G_PAIRS)))
            families.append(('weeds',scope+'_prefix_auc',[r for r in external_auc if r['scope']==scope],('scope','scope_id','grid_g','R'),AUC_METRICS,('grid_g',G_PAIRS)))
        for dataset,stem,values,dims,metrics,extra in families:
            tables=family_tables(values,dims,metrics,models,extra)
            for suffix,table in tables.items():
                name=stem+'_'+suffix+'.csv';write_rows(out/dataset/name,table);counts[dataset+'/'+name]=len(table)
            del tables
        primary=[r for r in local if r['M']==2 and r['budget']==.1 and r['policy']=='prefix']
        write_rows(out/'local/primary_M2_budget10_prefix.csv',primary)
        for basis in BASES:
            view=[r for r in external if r['scope']=='pooled' and r['grid_g']==8 and r['budget_basis']==basis]
            write_rows(out/'weeds'/('primary_g8_pooled_'+basis+'.csv'),view)
        # Complete curves and reference tables are exact copies, indexed by source.
        ledger.append(copy_saved_table(folders['local_outcomes'],'models/all_complete_prefix_curves.csv',out/'local','complete_prefix_curves.csv',watched))
        for name in ('label_prefix_budgets','label_MILP_budgets','label_prefix_curves','label_prefix_auc','random_budgets','random_auc','random_summary','summary','reference_seal'):
            ledger.append(copy_saved_table(folders['local_outcomes'],'shared_reference/'+name+'.json',out/'local','shared_'+name+'.json',watched))
        for stem,scope,kind in itertools.product(('learned','label_oracle','random_expectation'),('frame','date','pooled'),('complete_prefix_curves','budget_metrics','auc')):
            if stem=='learned' and kind!='complete_prefix_curves':continue
            name=f'{stem}_{scope}_{kind}.csv';ledger.append(copy_saved_table(folders['external_outcomes'],name,out/'weeds',name,watched))
        # Preserve the four scalar constants once per dataset; no duplicated R evidence.
        baseline=read_bound_table(folders['scalar_outcomes'],'constant_baselines.csv',watched)
        for dataset in ('local','weeds'):
            current=[r for r in baseline if r['dataset']==dataset]
            core.require({r['baseline'] for r in current}=={'zero','train_mean'} and len(current)==2,'Baseline population differs')
            write_rows(out/dataset/'constant_scalar_baselines.csv',current)
        core.write(out/'model_metadata.json',list(models.values()))
        core.write(out/'saved_curve_reference_index.json',ledger)
        core.write(out/'summary.json',{'version':'confirmation_report_tables_v1','table_rows':counts,'models':{'local':28,'weeds':9},
            'local_spatial_models':24,'external_spatial_models':6,'R':list(RS),'R_pairs':R_PAIRS,'local_M_pairs':M_PAIRS,'external_grid_pairs':G_PAIRS,
            'external_frames':26,'external_dates':sorted(set(frame_meta.values())),
            'primary_local':{'M':2,'budget':.1,'policy':'prefix'},'primary_external':{'grid_g':8,'scope':'pooled','budget_basis':list(BASES)},
            'scalar_unit':'percentage points in *_pp columns; source fractional columns preserved',
            'capture_unit':'percentage points in *_pct columns; AUC remains a fraction',
            'datasets_pooled':False,'new_model_calls':0,'new_solver_calls':0,'raw_labels_or_arrays_loaded':0,
            'seed_statistics':'Complete declared seed groups; sample SD only for >=2 defined values. Any NA keeps group mean/SD NA; defined count retained.',
            'replication':'Fitting-seed variability within each dataset scene; frames/dates/grids/budgets are not independent fields.',
            'checkpoint_reselected':False,'independent_output_review':'pending'})
        for p,h in watched.items():core.bound_file(p,h)
        core.validate_source_lock()
        for folder in folders.values():core.completed_stage(folder)
        core.write(out/'end_binding.json',{'watched_sha256':watched,'all_begin_and_consumed_sources_unchanged':True})
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root-review',required=True)
    a=parser.parse_args();print(execute(a.root_review))
