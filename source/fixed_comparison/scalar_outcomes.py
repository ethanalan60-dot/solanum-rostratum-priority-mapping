"""All declared scalar outcomes after global predictions and choices, never model selection."""
from pathlib import Path
import argparse,csv,math,statistics
from core import HERE,R_VALUES,attempt,binding,bound_file,completed_stage,configure_cpu,read,require,sha,table,validate_freeze,validate_source_lock,write
METRICS=('mae','bias','mae_zero','mae_positive','balanced_mae')
def summarize(prediction,truth):
    require(len(prediction)==len(truth)>0,'Empty/mismatched scalar population')
    require(all(math.isfinite(x) and 0<=x<=1 for x in prediction+truth),'Invalid scalar probabilities')
    errors=[p-t for p,t in zip(prediction,truth)]
    zero=[abs(e) for e,t in zip(errors,truth) if t==0]
    positive=[abs(e) for e,t in zip(errors,truth) if t>0]
    zm=math.fsum(zero)/len(zero) if zero else None
    pm=math.fsum(positive)/len(positive) if positive else None
    balanced=(zm+pm)/2 if zm is not None and pm is not None else None
    return {'n':len(truth),'n_zero':len(zero),'n_positive':len(positive),
            'mae':math.fsum(abs(e) for e in errors)/len(errors),'bias':math.fsum(errors)/len(errors),
            'mae_zero':zm,'mae_positive':pm,'balanced_mae':balanced,
            'balanced_mae_na_reason':'missing_zero_stratum' if not zero else ('missing_positive_stratum' if not positive else '')}

def paired(rows):
    index={}
    for row in rows:
        if row['variant']=='regression':continue
        key=tuple(row[k] for k in ('dataset','encoder','lr','seed','R'))
        group=index.setdefault(key,{})
        require(row['variant'] not in group,'Duplicate hard/soft row');group[row['variant']]=row
    result=[]
    for key,group in sorted(index.items()):
        require(set(group)=={'hard_seg','soft_seg'},'Missing paired supervision')
        hard,soft=group['hard_seg'],group['soft_seg']
        row={k:hard[k] for k in ('dataset','encoder','lr','seed','R','n','n_zero','n_positive')}
        row.update(hard_run_id=hard['run_id'],soft_run_id=soft['run_id'])
        for metric in METRICS:
            row['hard_'+metric]=hard[metric];row['soft_'+metric]=soft[metric]
            row['soft_minus_hard_'+metric+'_pp']=None if hard[metric] is None or soft[metric] is None else 100*(soft[metric]-hard[metric])
        result.append(row)
    return result

def seed_summaries(rows):
    groups={}
    for row in rows:groups.setdefault(tuple(row[k] for k in ('dataset','encoder','lr','variant','R')),[]).append(row)
    result=[]
    for key,group in sorted(groups.items()):
        require(len({r['seed'] for r in group})==len(group),'Duplicate seed')
        first=group[0];row={k:first[k] for k in ('dataset','encoder','lr','variant','R')}
        row.update(n_models=len(group),seeds=','.join(str(x['seed']) for x in sorted(group,key=lambda r:r['seed'])),
                   sample_sd_interpretation='descriptive fitting-seed variability within the same scene; not independent fields')
        for metric in METRICS:
            vals=[r[metric] for r in group]
            row[metric+'_defined_models']=sum(x is not None for x in vals)
            row[metric+'_mean']=math.fsum(vals)/len(vals) if all(x is not None for x in vals) else None
            row[metric+'_sample_sd']=statistics.stdev(vals) if len(vals)>1 and all(x is not None for x in vals) else None
        result.append(row)
    return result

def evaluate(choice_review,execution_review):
    from data_io import validate_input_manifest
    from pipeline import validate_global_predictions,validate_choices
    contract,models,freeze=validate_freeze();predictions=validate_global_predictions();validate_choices(choice_review)
    review=read(execution_review)
    review_digest=sha(execution_review)
    require(review.get('passed') is True and review['scope']=='confirmation_scalar_outcomes','Wrong scalar grant')
    require(review['implementation_sha256']==sha(__file__) and review['freeze_sha256']==sha(HERE/'FREEZE.json'),'Scalar grant differs')
    baselines=read(HERE/'scalar_baselines.json')
    require(review['baselines_sha256']==sha(HERE/'scalar_baselines.json'),'Baseline constants changed')
    truths={};truth_paths={}
    for dataset in ('local','weeds'):
        inp=validate_input_manifest(dataset);folder=HERE/'stages'/('truth_'+dataset);completed_stage(folder)
        tm=read(folder/'truth_manifest.json')
        require(tm['dataset']==dataset and tm['choice_review']==binding(choice_review) and tm['all_predictions_seal']==binding(HERE/'stages/predictions_seal/all_predictions.json'),'Truth gate binding differs')
        require(tm['source_manifest_sha256']==inp['source_manifest']['sha256'],'Truth source membership differs')
        require([x['id'] for x in tm['records']]==[x['id'] for x in inp['records']],'Truth/input order differs')
        truth_paths[dataset]=(folder,tm)
    with attempt('scalar_outcomes') as out:
        configure_cpu(out)
        import numpy as np
        for dataset,(folder,tm) in truth_paths.items():
            values={}
            for item in tm['records']:
                bound_file(item['truth_cache'],item['truth_cache_sha256'])
                with np.load(item['truth_cache'],allow_pickle=False) as z:
                    coverage=z['coverage']
                require(coverage.dtype==np.float32 and coverage.shape==(),'Scalar reference representation differs')
                value=float(coverage);require(math.isfinite(value) and 0<=value<=1,'Invalid scalar reference')
                values[item['id']]=value
            truths[dataset]=values
        summaries=[];samples=[];model_means=[]
        for model in models:
            export=next(x for x in predictions['models'] if x['run_id']==model['run_id'])
            bound_file(export['predicted_scalars']['path'],export['predicted_scalars']['sha256'])
            with Path(export['predicted_scalars']['path']).open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
            for row in rows:
                for key in ('run_id','dataset','encoder','variant'):require(row[key]==model[key],'Scalar model identity differs: '+key)
                for key in ('seed','best_epoch'):require(int(row[key])==model[key],'Scalar model integer identity differs: '+key)
            truth=truths[model['dataset']];expected=[(i,r) for r in R_VALUES for i in truth]
            require([(x['id'],int(x['R'])) for x in rows]==expected,'Scalar complete population/order differs')
            own=[]
            for r in R_VALUES:
                group=[x for x in rows if int(x['R'])==r]
                p=[float(x['prediction_cpu']) for x in group];t=[truth[x['id']] for x in group]
                meta={k:model[k] for k in ('run_id','dataset','encoder','variant','lr','seed','best_epoch')}
                result={**meta,'R':r,**summarize(p,t)};summaries.append(result);own.append(result)
                for x,pv,tv in zip(group,p,t):
                    samples.append({**meta,'R':r,'id':x['id'],'prediction':pv,'truth':tv,'signed_error':pv-tv,'absolute_error':abs(pv-tv)})
            mean={k:model[k] for k in ('run_id','dataset','encoder','variant','lr','seed','best_epoch')}
            mean.update(R_levels='1,2,4,6',checkpoint_reselected=False)
            for metric in METRICS:
                values=[x[metric] for x in own]
                mean['mean_four_R_'+metric]=math.fsum(values)/4 if all(x is not None for x in values) else None
            model_means.append(mean)
        require(len(samples)==38792 and len(summaries)==148 and len(model_means)==37,'Scalar inventory incomplete')
        baseline_rows=[]
        for dataset,values in truths.items():
            constants=baselines['datasets'][dataset];bound_file(constants['training_manifest_path'],constants['training_manifest_sha256'])
            for name in ('zero','train_mean'):
                value=constants[name]
                baseline_rows.append({'dataset':dataset,'baseline':name,'constant_prediction':value,'R_applicability':'identical at R1,2,4,6; counted once',**summarize([value]*len(values),list(values.values()))})
        pairs=paired(summaries);require(len(pairs)==60,'Incomplete hard-soft scalar pairs')
        table(out/'all_scalar_samples.csv',samples);table(out/'all_model_R_metrics.csv',summaries)
        table(out/'all_model_four_R_means.csv',model_means);table(out/'hard_soft_pairs.csv',pairs)
        table(out/'descriptive_seed_summaries.csv',seed_summaries(summaries));table(out/'constant_baselines.csv',baseline_rows)
        validate_source_lock()
        bound_file(execution_review,review_digest)
        bound_file(__file__,review['implementation_sha256']);bound_file(HERE/'scalar_baselines.json',review['baselines_sha256'])
        for folder,_ in truth_paths.values():completed_stage(folder)
        write(out/'summary.json',{'models':37,'R':list(R_VALUES),'sample_rows':38792,'model_R_rows':148,'paired_hard_soft_rows':60,
              'constant_baseline_rows':4,'metric_unit':'coverage fraction; paired differences pp',
              'missing_stratum_rule':'balanced MAE undefined if either stratum absent; ordinary MAE always retained',
              'checkpoint_reselected':False,'truth_used_for_tuning':False,'freeze':binding(HERE/'FREEZE.json'),
              'choice_review':binding(choice_review),'execution_review':binding(execution_review),'baselines':binding(HERE/'scalar_baselines.json')})
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--choice-review',required=True);parser.add_argument('--root-review',required=True)
    a=parser.parse_args();print(evaluate(a.choice_review,a.root_review))
if __name__=='__main__':main()
