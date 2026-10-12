"""Independent saved-result audit, gated before any reserved arrays are decoded.

No evaluator, model, Torch, optimizer, MILP or legacy execution module is imported.
Preparation tests use artificial inputs. Production requires a source-bound grant,
the complete freeze/all-prediction/all-choice population, and all three outcomes.
"""
from pathlib import Path
import argparse, collections, csv, datetime, itertools, json, math, os, sys, traceback

ROOT=Path(__file__).resolve().parent;HOME=ROOT/'confirmation_v1'
sys.path.insert(0,str(HOME))
import core
RS=(1,2,4,6);GRIDS=(4,8,16)
BUDGETS=(('b10',1,10),('b20',1,5),('b30',3,10),('b50',1,2),('b100',1,1))
METRICS=('mae','bias','mae_zero','mae_positive','balanced_mae')
WATCH={};ERRORS=collections.defaultdict(float);COUNTS={};NUMERIC=0

def need(v,msg):
    if not v:raise ValueError(msg)
def pin(path,digest=None):
    path=Path(path).resolve();value=core.sha(path)
    need(digest is None or value==digest,'Input hash mismatch: '+str(path))
    need(str(path) not in WATCH or WATCH[str(path)]==value,'Input changed during audit: '+str(path))
    WATCH[str(path)]=value;return path
def read(path,digest=None):return core.read(pin(path,digest))
def verify_watched():
    for path,digest in WATCH.items():core.bound_file(path,digest)
def csvrows(path):
    pin(path)
    with Path(path).open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def jsonlines(path):
    pin(path)
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def check(actual,wanted,key='number',atol=2e-11):
    global NUMERIC
    NUMERIC+=1
    if wanted is None:need(actual is None or actual=='',key+': expected NA');return
    if isinstance(wanted,(str,bool)):
        need(str(actual)==str(wanted),key+': identity differs');return
    need(actual is not None and actual!='',key+': unexpected NA')
    a,b=float(actual),float(wanted);need(math.isfinite(a) and math.isfinite(b),key+': nonfinite')
    error=abs(a-b);ERRORS[key]=max(ERRORS[key],error)
    need(error<=atol+1e-12*abs(b),key+': numerical mismatch '+str((a,b,error)))
def fields(row,expected,tag):
    for k,v in expected.items():need(k in row,tag+': missing '+k);check(row[k],v,tag+'/'+k)
def complete_table(path,keys,expected,tag):
    seen=set()
    for row in csvrows(path):
        key=tuple(str(row[k]) for k in keys);need(key in expected and key not in seen,tag+': unexpected/duplicate row');seen.add(key)
        fields(row,expected[key],tag)
    need(seen==set(expected),tag+': incomplete table');COUNTS[tag]=len(seen)
def ratio(a,b):return None if b==0 else a/b
def mean(values):return math.fsum(values)/len(values) if all(x is not None for x in values) else None
def sd(values):
    avg=mean(values)
    return None if len(values)<2 or avg is None else math.sqrt(math.fsum((x-avg)**2 for x in values)/(len(values)-1))
def record_stage(folder):
    pin(folder/'artifact_manifest.json');core.completed_stage(folder)
    pin(folder/'summary.json') if (folder/'summary.json').exists() else None
    return core.binding(folder/'artifact_manifest.json')

def gate(root_review,choice_review):
    """No NumPy import before all actual producer populations are complete."""
    initial=core.sha(__file__)
    pin(__file__,initial);pin(root_review);pin(choice_review)
    for name in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json','scalar_baselines.json'):pin(HOME/name)
    contract,models,freeze=core.validate_freeze(check_models=False);core.validate_source_lock()
    verify_watched()
    review=read(root_review);choice_digest=core.sha(choice_review)
    need(review.get('passed') is True and review.get('scope')=='confirmation_external_scalar_independent','Source-bound audit execution review required')
    need(review['implementation_sha256']==initial and review['freeze_sha256']==core.sha(HOME/'FREEZE.json'),'Audit source/freeze differs')
    need(review['choice_review_sha256']==choice_digest,'Wrong accepted choice review')
    for name in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json','scalar_baselines.json'):pin(HOME/name)
    pin(__file__,initial);pin(choice_review,choice_digest)
    stages={};need(set(review['outcome_artifact_sha256'])=={'scalar_outcomes','local_outcomes','external_outcomes'},'All three outcome bindings required')
    for stage,digest in review['outcome_artifact_sha256'].items():
        folder=HOME/'stages'/stage;pin(folder/'artifact_manifest.json',digest);record_stage(folder);stages[stage]=folder
    # These stdlib metadata validators authenticate the full global populations.
    from pipeline import validate_global_predictions,validate_choices
    from data_io import validate_input_manifest
    pin(HOME/'stages/predictions_seal/all_predictions.json',review['all_predictions_sha256'])
    record_stage(HOME/'stages/predictions_seal')
    predictions=validate_global_predictions();accepted=validate_choices(choice_review)
    verify_watched()
    inputs={};truths={}
    for dataset in ('local','weeds'):
        pin(HOME/'stages'/('inputs_'+dataset)/'input_manifest.json');record_stage(HOME/'stages'/('inputs_'+dataset))
        inputs[dataset]=validate_input_manifest(dataset);verify_watched()
        folder=HOME/'stages'/('truth_'+dataset);record_stage(folder);tm=read(folder/'truth_manifest.json')
        need(tm['dataset']==dataset and tm['choice_review']==core.binding(choice_review) and tm['all_predictions_seal']==core.binding(HOME/'stages/predictions_seal/all_predictions.json'),'Truth not bound to accepted population')
        need(tm['source_manifest_sha256']==inputs[dataset]['source_manifest']['sha256'] and [x['id'] for x in tm['records']]==[x['id'] for x in inputs[dataset]['records']],'Truth membership differs')
        truths[dataset]=tm
    need(len(models)==37 and len(inputs['local']['records'])==338 and len(inputs['weeds']['records'])==26,'Incomplete actual population')
    for stage in ('scalar_outcomes','external_outcomes'):
        summary=read(stages[stage]/'summary.json')
        need(summary['freeze']==core.binding(HOME/'FREEZE.json') and summary['choice_review']==core.binding(choice_review),'Producer outcome exposure binding differs')
    return contract,{x['run_id']:x for x in models},predictions,accepted,inputs,truths,stages

def local_source_coverage(inputs,np):
    """Only after the full exposure gate: original float32 scalar target, not soft-grid mean."""
    binding=inputs['local']['source_manifest'];source=read(binding['path'],binding['sha256'])
    need(isinstance(source,list),'Local source manifest is not a record list')
    ids=[x['id'] for x in inputs['local']['records']]
    need([x['id'] for x in source]==ids and len(set(ids))==len(ids),'Local source scalar membership/order differs')
    result={}
    for row in source:
        value=row['coverage'];need(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1,'Invalid original scalar coverage')
        result[row['id']]=float(np.float32(value))
    return result

def independent_metrics(pred,truth):
    need(len(pred)==len(truth)>0,'Invalid scalar group')
    need(all(math.isfinite(x) and 0<=x<=1 for x in pred+truth),'Invalid scalar value')
    errors=[p-t for p,t in zip(pred,truth)];zero=[abs(e) for e,t in zip(errors,truth) if t==0];positive=[abs(e) for e,t in zip(errors,truth) if t>0]
    zm=math.fsum(zero)/len(zero) if zero else None;pm=math.fsum(positive)/len(positive) if positive else None
    return dict(n=len(truth),n_zero=len(zero),n_positive=len(positive),mae=math.fsum(map(abs,errors))/len(errors),bias=math.fsum(errors)/len(errors),
        mae_zero=zm,mae_positive=pm,balanced_mae=(zm+pm)/2 if zm is not None and pm is not None else None,
        balanced_mae_na_reason='missing_zero_stratum' if not zero else 'missing_positive_stratum' if not positive else '')

def audit_scalar(folder,models,ids,truth,exported,baselines,strict=True):
    """Recompute from every persisted per-sample row; independently bound caches/export scalars supplied by caller."""
    expected={(rid,R,i) for rid,m in models.items() for R in RS for i in ids[m['dataset']]};seen=set();group={}
    for row in csvrows(folder/'all_scalar_samples.csv'):
        key=row['run_id'],int(row['R']),row['id'];need(key in expected and key not in seen,'Scalar sample missing/duplicate/foreign');seen.add(key)
        rid,R,i=key;m=models[rid];fields(row,{k:m[k] for k in ('run_id','dataset','encoder','variant','lr','seed','best_epoch')},'scalar_identity')
        p,t=float(row['prediction']),float(row['truth']);check(p,exported[key],'scalar_export',1e-12);check(t,truth[m['dataset']][i],'scalar_truth',1e-12)
        check(row['signed_error'],p-t,'signed');check(row['absolute_error'],abs(p-t),'absolute');group.setdefault((rid,R),[]).append((p,t))
    need(seen==expected,'Incomplete scalar sample population');COUNTS['scalar_samples']=len(seen)
    summaries={};means={};pairs={};seedgroups={}
    for (rid,R),values in group.items():
        m=models[rid];metric=independent_metrics([x[0] for x in values],[x[1] for x in values]);meta={k:m[k] for k in ('run_id','dataset','encoder','variant','lr','seed','best_epoch')}
        summaries[(rid,str(R))]={**meta,'R':R,**metric}
    complete_table(folder/'all_model_R_metrics.csv',('run_id','R'),summaries,'scalar_R')
    for rid,m in models.items():
        means[(rid,)]=dict({k:m[k] for k in ('run_id','dataset','encoder','variant','lr','seed','best_epoch')},R_levels='1,2,4,6',checkpoint_reselected=False)
        for metric in METRICS:means[(rid,)]['mean_four_R_'+metric]=mean([summaries[rid,str(R)][metric] for R in RS])
    complete_table(folder/'all_model_four_R_means.csv',('run_id',),means,'scalar_four_R')
    for key,row in summaries.items():
        sk=tuple(str(row[x]) for x in ('dataset','encoder','lr','variant','R'));seedgroups.setdefault(sk,[]).append(row)
        if row['variant']!='regression':
            pk=tuple(str(row[x]) for x in ('dataset','encoder','lr','seed','R'));pairs.setdefault(pk,{})[row['variant']]=row
    paired={}
    for key,pair in pairs.items():
        need(set(pair)=={'hard_seg','soft_seg'},'Scalar supervision pair incomplete');h,s=pair['hard_seg'],pair['soft_seg']
        row={k:h[k] for k in ('dataset','encoder','lr','seed','R','n','n_zero','n_positive')};row.update(hard_run_id=h['run_id'],soft_run_id=s['run_id'])
        for metric in METRICS:row.update({'hard_'+metric:h[metric],'soft_'+metric:s[metric],'soft_minus_hard_'+metric+'_pp':None if h[metric] is None or s[metric] is None else 100*(s[metric]-h[metric])})
        paired[key]=row
    complete_table(folder/'hard_soft_pairs.csv',('dataset','encoder','lr','seed','R'),paired,'scalar_pairs')
    stats={}
    for key,values in seedgroups.items():
        first=values[0];wanted={m['seed'] for m in models.values() if all(m[k]==first[k] for k in ('dataset','encoder','lr','variant'))}
        need({v['seed'] for v in values}==wanted and len(values)==len(wanted),'Declared scalar seed population incomplete')
        row={k:first[k] for k in ('dataset','encoder','lr','variant','R')};row.update(n_models=len(values),seeds=','.join(map(str,sorted(wanted))))
        for metric in METRICS:
            vals=[v[metric] for v in values];row.update({metric+'_defined_models':sum(x is not None for x in vals),metric+'_mean':mean(vals),metric+'_sample_sd':sd(vals)})
        stats[key]=row
    complete_table(folder/'descriptive_seed_summaries.csv',('dataset','encoder','lr','variant','R'),stats,'scalar_seed_groups')
    constants={}
    for ds,values in truth.items():
        for name in ('zero','train_mean'):
            v=float(baselines['datasets'][ds][name]);constants[ds,name]=dict(dataset=ds,baseline=name,constant_prediction=v,R_applicability='identical at R1,2,4,6; counted once',**independent_metrics([v]*len(values),list(values.values())))
    complete_table(folder/'constant_baselines.csv',('dataset','baseline'),constants,'scalar_constants')
    if strict:need((len(seen),len(summaries),len(means),len(paired),len(stats),len(constants))==(38792,148,37,60,60,4),'Actual scalar cardinalities differ')
    return dict(samples=len(seen),model_R=len(summaries),model_means=len(means),hard_soft_pairs=len(paired),seed_groups=len(stats),constants=len(constants))

def ids_for(frame,g):return [f'{frame}:g{g:02d}:r{r:02d}:c{c:02d}' for r in range(g) for c in range(g)]
def aggregate(array,g,np):
    a=np.asarray(array,dtype=np.float64);need(a.shape==(256,256) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all(),'Invalid plane')
    side=256//g;return a.reshape(g,side,g,side).sum(axis=(1,3),dtype=np.float64).ravel()/65536
def budgets(g):
    result=[];seen={};N=g*g
    for basis in ('nominal_primary','matched_actual_secondary'):
        for code,num,den in BUDGETS:
            k=N*num//den if basis=='nominal_primary' else (16*num//den)*(N//16);effective=num/den if basis=='nominal_primary' else (16*num//den)/16
            result.append(dict(budget_basis=basis,parent_nominal_budget_code=code,parent_nominal_budget_fraction=num/den,effective_budget_fraction=effective,k=k,N=N,actual_fraction=k/N,unused_effective_fraction=effective-k/N,prior_alias=seen.get(k)))
            seen.setdefault(k,(basis,code))
    return result
def metadata(model,export_sha,manifest_sha):
    return dict(run_id=model['run_id'],variant=model['variant'],seed=model['seed'],original_selected_epoch=model['best_epoch'],checkpoint_sha256=model['checkpoint_sha256'],config_sha256=model['config_sha256'],export_sha256=export_sha,source_manifest_sha256=manifest_sha)

def audit_external(folder,models,frames,pred,truth,selections,meta,np,strict=True):
    """Independent NumPy cell integrals supplied by map loader; never creates a new selection."""
    fids=[x['id'] for x in frames];byframe={x['id']:x for x in frames}
    groups={'frame':{f:[f] for f in fids},'date':{d:[x['id'] for x in frames if x['acquisition_date']==d] for d in sorted({x['acquisition_date'] for x in frames})},'pooled':{'all'+str(len(fids)):fids}}
    expected_truth={}
    for f in fids:
        for g in GRIDS:
            a=truth[f,g];need(a.shape==(g*g,3),'Wrong truth units')
            for i,ident in enumerate(ids_for(f,g)):expected_truth[f,str(g),ident]=dict(A=1/g**2,B=a[i,0],C=a[i,1],W=a[i,2])
            if g>4:
                parent=truth[f,g//2].reshape(g//2,g//2,3);children=a.reshape(g//2,2,g//2,2,3).sum((1,3));need(np.max(abs(parent-children))<=1e-12,'Truth nested grids differ')
    complete_table(folder/'truth_units.csv',('frame_id','grid_g','id'),expected_truth,'external_truth_units')
    order_arrays={};ranking_seen=set();choice_count=0;unique_choices=set();punit_count=0
    for rid,model in models.items():
        selected=selections[rid];rankings={}
        for row in jsonlines(selected/'rankings.jsonl'):
            key=(rid,int(row['R']),row['frame_id'],int(row['grid_g']));need(key in pred and key not in ranking_seen,'Foreign/duplicate ranking');ranking_seen.add(key)
            _,R,f,g=key;ids=ids_for(f,g);P=pred[key];order=np.asarray(sorted(range(g*g),key=lambda i:(-P[i],ids[i])),dtype=np.int64)
            need(row['ordered_ids']==[ids[i] for i in order],'Map-derived ranking differs');fields(row,meta[rid],'ranking_identity');check(row['acquisition_date'],byframe[f]['acquisition_date'],'ranking_date')
            need(row['ranking_key']==f'{rid}|R{R}|{f}|g{g:02d}' and [x['id'] for x in row['units']]==ids,'Ranking key/unit identity differs')
            for i,u in enumerate(row['units']):need(set(u)=={'id','A','P'},'Label entered rank input');fields(u,{'A':1/g**2,'P':P[i]},'prediction_unit')
            order_arrays[key]=order;rankings[key]=row
        expected_keys={(rid,R,f,g) for R in RS for f in fids for g in GRIDS};need(set(rankings)==expected_keys,'Incomplete model rankings')
        pexpected={(str(R),f,str(g),ident):{'run_id':rid,'A':1/g**2,'P':pred[rid,R,f,g][i]} for R in RS for f in fids for g in GRIDS for i,ident in enumerate(ids_for(f,g))}
        complete_table(selected/'prediction_units.csv',('R','frame_id','grid_g','id'),pexpected,'prediction_units_'+rid);punit_count+=len(pexpected)
        seen=set()
        for row in jsonlines(selected/'choices.jsonl'):
            key=(rid,int(row['R']),row['frame_id'],int(row['grid_g']));need(key in rankings,'Choice rank absent');_,R,f,g=key;fields(row,meta[rid],'choice_identity')
            bd=next((x for x in budgets(g) if x['budget_basis']==row['budget_basis'] and x['parent_nominal_budget_code']==row['parent_nominal_budget_code']),None);need(bd is not None,'Foreign budget')
            ck=key+(bd['budget_basis'],bd['parent_nominal_budget_code']);need(ck not in seen,'Duplicate choice');seen.add(ck)
            fields(row,{k:v for k,v in bd.items() if k!='prior_alias'},'choice_budget');base=f'{rid}|R{R}|{f}|g{g:02d}'
            need(row['ranking_key']==base and row['selection_key']==base+f"|k{bd['k']:03d}" and row['choice_key']==base+'|'+bd['budget_basis']+'|'+bd['parent_nominal_budget_code'],'Choice identity differs')
            alias=None if bd['prior_alias'] is None else base+'|'+'|'.join(bd['prior_alias']);check(row['duplicate_selection_alias'],alias,'alias')
            indices=order_arrays[key][:bd['k']];need(row['selected_ids']==[ids_for(f,g)[i] for i in indices],'Sealed selected IDs differ')
            check(row['selected_area_FAE'],bd['k']/g**2,'choice_area');check(row['selected_predicted_weed_FAE'],float(pred[key][indices].sum()),'choice_P')
            need(row['selected_area_FAE']<=bd['effective_budget_fraction'],'Strict frame budget violated');unique_choices.add(row['selection_key'])
        need(seen=={key+(bd['budget_basis'],bd['parent_nominal_budget_code']) for key in expected_keys for bd in budgets(key[3])},'Incomplete choices');choice_count+=len(seen)
    need(ranking_seen==set(pred),'Prediction/ranking universe differs')
    curves={};ref_orders={}
    for (rid,R,f,g),order in order_arrays.items():curves[rid,R,g,'frame',f]=np.vstack((np.zeros((1,4)),np.cumsum(np.column_stack((truth[f,g],pred[rid,R,f,g]))[order],axis=0,dtype=np.float64)))
    for policy in ('label_oracle','random_expectation'):
        for f in fids:
            for g in GRIDS:
                cells=truth[f,g];ids=ids_for(f,g);order=sorted(range(g*g),key=lambda i:(-cells[i,2],ids[i])) if policy=='label_oracle' else list(range(g*g))
                ref_orders[policy,f,g]=order;c=np.zeros((g*g+1,4));c[:,:3]=(np.arange(g*g+1)/(g*g))[:,None]*cells.sum(0) if policy=='random_expectation' else np.vstack((np.zeros((1,3)),np.cumsum(cells[order],axis=0)))
                curves[policy,None,g,'frame',f]=c
        seen=set()
        for row in jsonlines(folder/(policy+'_reference_rankings.jsonl')):
            f,g=row['frame_id'],int(row['grid_g']);need((f,g) not in seen and f in fids and g in GRIDS,'Reference inventory');seen.add((f,g))
            need(row['ordered_ids']==[ids_for(f,g)[i] for i in ref_orders[policy,f,g]],'Label/random reference ordering changed')
            need(row['units']==[{'id':i,'A':1/g**2,'P':0.} for i in ids_for(f,g)],'Reference units altered')
            need(row['run_id']==policy and row['variant']=='reference' and row['R'] is None and row['seed'] is None,'Reference identity altered')
        need(seen==set(itertools.product(fids,GRIDS)),'Incomplete reference rankings')
    for rid,R in [(rid,R) for rid in models for R in RS]+[(x,None) for x in ('label_oracle','random_expectation')]:
        for g in GRIDS:
            for scope in ('date','pooled'):
                for sid,fs in groups[scope].items():curves[rid,R,g,scope,sid]=np.stack([curves[rid,R,g,'frame',f] for f in sorted(fs)]).sum(0,dtype=np.float64)
    def curvekey(row):return row['run_id'],int(row['R']) if row['R']!='' else None,int(row['grid_g']),row['scope'],row['scope_id']
    def metrics(key,k):
        rid,R,g,scope,sid=key;fs=groups[scope][sid];c=curves[key];B,C,W,P=c[k];tw,tc=c[-1,2],c[-1,1];A=len(fs)*k/(g*g)
        we=sum(truth[f,g][:,2].sum()>0 for f in fs);ce=sum(truth[f,g][:,1].sum()>0 for f in fs)
        return dict(k=k,N=g*g,actual_fraction=k/(g*g),selected_area_FAE=A,selected_predicted_weed_FAE=P if rid in models else None,
            selected_weed_FAE=W,total_weed_FAE=tw,unselected_weed_FAE=tw-W,selected_crop_FAE=C,total_crop_FAE=tc,selected_background_FAE=B,
            weed_capture_ratio=ratio(W,tw),weed_capture_na_reason='' if tw else 'zero_total_weed',crop_capture_ratio=ratio(C,tc),crop_capture_na_reason='' if tc else 'zero_total_crop',
            crop_fraction_of_selected_area=ratio(C,A),crop_composition_na_reason='' if A else 'zero_selected_area',frame_n=len(fs),weed_eligible_frame_n=int(we),crop_eligible_frame_n=int(ce),zero_weed_frame_n=int(len(fs)-we),zero_crop_frame_n=int(len(fs)-ce))
    def rowmeta(row,key,policy):
        rid,R,g,scope,sid=key;need(row['policy']==policy,'Policy mismatch')
        check(row['frame_id'],sid if scope=='frame' else '','scope_frame');check(row['acquisition_date'],byframe[sid]['acquisition_date'] if scope=='frame' else sid if scope=='date' else '','scope_date')
        fields(row,meta[rid] if rid in models else dict(run_id=rid,variant='reference',seed=None,original_selected_epoch=None,checkpoint_sha256='',config_sha256='',export_sha256='',source_manifest_sha256=next(iter(meta.values()))['source_manifest_sha256']),'outcome_identity')
    fixed={};aucs={}
    for policy in ('learned','label_oracle','random_expectation'):
        for scope in ('frame','date','pooled'):
            universe={k for k in curves if k[3]==scope and (k[0] in models if policy=='learned' else k[0]==policy)};seen=set()
            name=f'{policy}_{scope}_complete_prefix_curves.csv'
            for row in csvrows(folder/name):
                key=curvekey(row);k=int(row['k']);need(key in universe and 0<=k<=key[2]**2 and (key,k) not in seen,'Curve population differs');seen.add((key,k));rowmeta(row,key,policy);fields(row,metrics(key,k),'full_curve')
            need(seen=={(key,k) for key in universe for k in range(key[2]**2+1)},'Missing full curve points');COUNTS[name]=len(seen)
            seen=set();name=f'{policy}_{scope}_budget_metrics.csv'
            for row in csvrows(folder/name):
                key=curvekey(row);need(key in universe,'Foreign fixed outcome');rowmeta(row,key,policy)
                basis,code=row['budget_basis'],row['parent_nominal_budget_code'];bd=next((x for x in budgets(key[2]) if x['budget_basis']==basis and x['parent_nominal_budget_code']==code),None);need(bd is not None,'Foreign fixed budget')
                sk=key+(basis,code);need(sk not in seen,'Duplicate fixed outcome');seen.add(sk);value=metrics(key,bd['k']);fields(row,value,'fixed');fields(row,{k:v for k,v in bd.items() if k!='prior_alias'},'fixed_budget')
                rid,R,g,sc,sid=key;base=f'{rid}|R{R}|'+(sid if sc=='frame' else sc+'|'+sid)+f'|g{g:02d}';selection=base+f"|k{bd['k']:03d}";alias='' if bd['prior_alias'] is None else base+'|'+'|'.join(bd['prior_alias'])
                fields(row,dict(selection_key=selection,choice_key=base+'|'+basis+'|'+code,duplicate_selection_alias=alias,unused_effective_budget_area_FAE=bd['unused_effective_fraction']*value['frame_n']),'fixed_alias')
                fixed[sk]={**value,**bd,'selection_key':selection,'duplicate_selection_alias':alias}
            need(seen=={key+(bd['budget_basis'],bd['parent_nominal_budget_code']) for key in universe for bd in budgets(key[2])},'Missing fixed budgets');COUNTS[name]=len(seen)
            seen=set();name=f'{policy}_{scope}_auc.csv'
            for row in csvrows(folder/name):
                key=curvekey(row);need(key in universe and key not in seen,'AUC population differs');seen.add(key);rowmeta(row,key,policy)
                N=key[2]**2;w=curves[key][:,2];tw=w[-1];full=float(w[:-1].mean()/tw) if tw else None;half=float(w[:N//2].mean()/tw) if tw else None
                value=dict(prefix_points=N+1,frame_n=len(groups[scope][key[4]]),weed_eligible_frame_n=metrics(key,0)['weed_eligible_frame_n'],total_weed_FAE=tw,auc_0_1=full,auc_0_0p5_normalized=half,auc_na_reason='' if tw else 'zero_total_weed');fields(row,value,'AUC');aucs[key]=value
                if policy=='random_expectation' and tw:check(full,(N-1)/(2*N),'random_AUC');check(half,.25-1/(2*N),'random_half_AUC')
            need(seen==universe,'Missing AUC rows');COUNTS[name]=len(seen)
    # Independently pair all saved learned budget and AUC rows after complete matrices.
    byvs={(m['variant'],m['seed']):rid for rid,m in models.items()};seeds=sorted({m['seed'] for m in models.values()})
    for scope in ('frame','date','pooled'):
        expected={};expected_auc={}
        for seed,R,g,sid in itertools.product(seeds,RS,GRIDS,groups[scope]):
            h,s=byvs['hard_seg',seed],byvs['soft_seg',seed];hk=(h,R,g,scope,sid);sk=(s,R,g,scope,sid)
            row=dict(seed=seed,R=R,grid_g=g,scope=scope,scope_id=sid,hard_run_id=h,soft_run_id=s,budget_basis='complete_prefix_once',frame_n=aucs[hk]['frame_n'])
            for metric in ('auc_0_1','auc_0_0p5_normalized'):row.update({'hard_'+metric:aucs[hk][metric],'soft_'+metric:aucs[sk][metric],'soft_minus_hard_'+metric:None if aucs[hk][metric] is None or aucs[sk][metric] is None else aucs[sk][metric]-aucs[hk][metric]})
            expected_auc[tuple(map(str,(seed,R,g,scope,sid)))]=row
            for bd in budgets(g):
                basis,code=bd['budget_basis'],bd['parent_nominal_budget_code'];hh,ss=fixed[hk+(basis,code)],fixed[sk+(basis,code)]
                row=dict(seed=seed,R=R,grid_g=g,scope=scope,scope_id=sid,budget_basis=basis,parent_nominal_budget_code=code,hard_run_id=h,soft_run_id=s,
                    hard_selection_key=hh['selection_key'],soft_selection_key=ss['selection_key'],duplicate_budget_selection=bool(hh['duplicate_selection_alias']),**{k:hh[k] for k in ('effective_budget_fraction','k','N','frame_n','selected_area_FAE')})
                for metric in ('weed_capture_ratio','crop_capture_ratio','crop_fraction_of_selected_area'):row.update({'hard_'+metric:hh[metric],'soft_'+metric:ss[metric],'soft_minus_hard_'+metric+'_pp':None if hh[metric] is None or ss[metric] is None else 100*(ss[metric]-hh[metric])})
                expected[tuple(map(str,(seed,R,g,scope,sid,basis,code)))]=row
        complete_table(folder/f'hard_soft_{scope}_budget_pairs.csv',('seed','R','grid_g','scope','scope_id','budget_basis','parent_nominal_budget_code'),expected,'external_'+scope+'_pairs')
        complete_table(folder/f'hard_soft_{scope}_auc_pairs.csv',('seed','R','grid_g','scope','scope_id'),expected_auc,'external_'+scope+'_auc_pairs')
    if strict:need((len(models),len(fids),len(ranking_seen),choice_count,len(unique_choices),punit_count,len(expected_truth))==(6,26,1872,18720,12480,209664,8736),'Actual external cardinalities differ')
    return dict(models=len(models),frames=len(fids),dates=len(groups['date']),rankings=len(ranking_seen),choices=choice_count,unique_choices=len(unique_choices),prediction_units=punit_count,truth_units=len(expected_truth),new_solver_calls=0,new_selected_ids_persisted=0)

def execute(root_review,choice_review):
    contract,models,predictions,accepted,inputs,truthmanifests,stages=gate(root_review,choice_review)
    out=ROOT/'provenance'/datetime.datetime.now().strftime('confirmation_external_scalar_independent_%Y%m%d_%H%M%S_%f');out.mkdir()
    try:
        core.write(out/'begin_binding.json',{'watched_sha256':WATCH,'scope':'all gates passed before NumPy/array decoding','choice_review':core.binding(choice_review)})
        os.environ['CUDA_VISIBLE_DEVICES']=''
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
        import numpy as np
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            truths={};grid_truth={};ids={ds:[x['id'] for x in inputs[ds]['records']] for ds in ('local','weeds')}
            original_local_coverage=local_source_coverage(inputs,np)
            for ds,tm in truthmanifests.items():
                truths[ds]={}
                for item in tm['records']:
                    path=pin(item['truth_cache'],item['truth_cache_sha256'])
                    with np.load(path,allow_pickle=False) as z:
                        cov=z['coverage'];need(cov.dtype==np.float32 and cov.shape==(),'Scalar truth dtype/shape');truths[ds][item['id']]=float(cov)
                        if ds=='local':check(float(cov),original_local_coverage[item['id']],'local_original_float32_scalar_truth',0.)
                        if ds=='weeds':
                            a=z['soft3'];counts=z['native_class_counts'];v=z['valid'];need(set(z.files)=={'soft','soft3','native_class_counts','valid','coverage'},'Unexpected external truth arrays')
                            need(a.shape==(3,256,256) and a.dtype==np.float32 and np.isfinite(a).all() and ((a>=0)&(a<=1)).all(),'Invalid external truth')
                            need(v.shape==(256,256) and v.dtype==np.float32 and np.all(v==1) and np.array_equal(z['soft'],a[2]),'External validity/pooled class differs')
                            need(counts.shape==(3,) and counts.dtype==np.uint32 and int(counts.sum())==360000,'Native class counts differ')
                            need(np.max(abs(a.astype(np.float64).mean((1,2))-counts/360000))<=1e-7 and np.max(abs(a.sum(0,dtype=np.float64)-1))<=1e-6,'Native-to-soft mass differs')
                            check(float(cov),float(np.float32(int(counts[2])/360000)),'external_native_scalar_truth',1e-12)
                            for g in GRIDS:grid_truth[item['id'],g]=np.stack([aggregate(a[c],g,np) for c in range(3)],axis=1)
                    pin(path,item['truth_cache_sha256'])
            exported={};export_by={x['run_id']:x for x in predictions['models']}
            for rid,m in models.items():
                ex=export_by[rid];path=pin(ex['predicted_scalars']['path'],ex['predicted_scalars']['sha256']);seen=[]
                for row in csvrows(path):
                    fields(row,{k:m[k] for k in ('run_id','dataset','encoder','variant','seed','best_epoch')},'export_scalar_identity');key=(rid,int(row['R']),row['id']);need(key not in exported,'Duplicate predicted scalar');exported[key]=float(row['prediction_cpu']);seen.append((key[1],key[2]))
                need(seen==[(R,i) for R in RS for i in ids[m['dataset']]],'Export population/order mismatch')
            baselines=read(HOME/'scalar_baselines.json');scalar=audit_scalar(stages['scalar_outcomes'],models,ids,truths,exported,baselines)
            fields(read(stages['scalar_outcomes']/'summary.json'),{'models':37,'sample_rows':38792,'model_R_rows':148,'paired_hard_soft_rows':60,'constant_baseline_rows':4,'checkpoint_reselected':False,'truth_used_for_tuning':False},'scalar_summary')
            seg={rid:m for rid,m in models.items() if m['dataset']=='weeds' and m['variant']!='regression'};P={};selection={};metadata_by={}
            for rid,m in seg.items():
                ex=export_by[rid];maps=read(ex['map_index']['path'],ex['map_index']['sha256']);need([x['id'] for x in maps]==ids['weeds'],'External map membership/order')
                accepted_row=next(x for x in accepted['models'] if x['run_id']==rid);sealpath=pin(accepted_row['selection_seal']['path'],accepted_row['selection_seal']['sha256']);selected=sealpath.parent;record_stage(selected);seal=read(sealpath)
                need(seal['run_id']==rid and seal['freeze_sha256']==core.sha(HOME/'FREEZE.json') and seal['all_predictions_sha256']==core.sha(HOME/'stages/predictions_seal/all_predictions.json'),'Choice freeze/prediction identity')
                need(seal['case_count']==3120 and seal['rankings']==312 and seal['truth_used_for_choices'] is False,'Incomplete accepted choices')
                for name,digest in seal['files_sha256'].items():pin(core.resolve_under(selected,name),digest)
                selection[rid]=selected;metadata_by[rid]=metadata(m,ex['artifact_manifest']['sha256'],inputs['weeds']['source_manifest']['sha256'])
                for item in maps:
                    fields(item,{'run_id':rid,'dataset':'weeds'},'map_index_identity');need(item['R']==list(RS),'Map index R mismatch');path=pin(item['path'],item['sha256'])
                    with np.load(path,allow_pickle=False) as z:
                        need(set(z.files)=={'probabilities','valid','R','id'} and z['R'].dtype==np.int64 and z['R'].tolist()==list(RS) and str(z['id'].item())==item['id'],'Map fields/ID/R mismatch')
                        arr=z['probabilities'];v=z['valid'];need(arr.shape==(4,256,256) and arr.dtype==np.float32 and np.isfinite(arr).all() and ((arr>=0)&(arr<=1)).all(),'Map probability domain')
                        need(v.shape==(256,256) and v.dtype==np.float32 and np.all(v==1),'Map external support')
                    for ri,R in enumerate(RS):
                        check(exported[rid,R,item['id']],float(arr[ri].mean(dtype=np.float64)),'external_map_scalar',1e-7)
                        for g in GRIDS:
                            vals=aggregate(arr[ri],g,np)
                            # Exact finite binary32 sums resolve potential floating-order ties independently of saved rank values.
                            side=256//g;exact=np.array([math.fsum(map(float,arr[ri,r*side:(r+1)*side,c*side:(c+1)*side].ravel()))/65536 for r in range(g) for c in range(g)])
                            need(np.max(abs(exact-vals))<=2e-13,'Independent cell reductions differ');P[rid,R,item['id'],g]=exact
                    pin(path,item['sha256'])
            external=audit_external(stages['external_outcomes'],seg,inputs['weeds']['records'],P,grid_truth,selection,metadata_by,np)
            saved_summary=read(stages['external_outcomes']/'summary.json')
            need(saved_summary['models']==list(seg) and saved_summary['R']==list(RS) and saved_summary['grids']==list(GRIDS),'External summary population differs')
            fields(saved_summary,{'frames':26,'truth_units':8736,'learned_rankings':1872,'learned_budget_rows':18720,'learned_frame_curve_points':211536},'external_summary')
        for path,digest in WATCH.items():core.bound_file(path,digest)
        core.validate_source_lock()
        for folder in stages.values():core.completed_stage(folder)
        common={'passed':True,'freeze_sha256':core.sha(HOME/'FREEZE.json'),'implementation_sha256':core.sha(__file__),'audited_directory':None,'new_model_calls':0,'new_solver_calls':0,
            'numeric_checks':NUMERIC,'max_errors':dict(ERRORS),'counts':COUNTS,'choice_review':core.binding(choice_review),'begin_end_sources_unchanged':True,
            'interpretation':'Independent arithmetic on already sealed results; no new choices, fitting, significance, field independence or agronomic efficacy claims.'}
        for stage,result in [('scalar_outcomes',scalar),('external_outcomes',external)]:
            core.write(out/(stage+'_review.json'),{**common,'stage':stage,'audited_directory':str(stages[stage]),'audited_artifact_manifest_sha256':core.sha(stages[stage]/'artifact_manifest.json'),'summary':result})
        core.write(out/'end_binding.json',{'watched_sha256':WATCH,'passed':True});core.write(out/'artifact_manifest.json',core.stage_manifest(out))
        return out
    except BaseException as error:
        core.write(out/'failure.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc(),'counts':COUNTS,'max_errors':dict(ERRORS),'watched':WATCH});raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true',required=True);p.add_argument('--root-review',required=True);p.add_argument('--choice-review',required=True)
    a=p.parse_args();print(execute(a.root_review,a.choice_review))
