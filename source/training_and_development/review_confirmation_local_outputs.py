"""Independent saved-artifact outcome audit. No model, native mask decoding or solver.

CLI requires formal FREEZE, accepted all-30 choices, complete truth/outcomes and
a root grant. Preparation checks exercise pure routines with artificial tables.
"""
from pathlib import Path
import argparse,bisect,csv,json,math,sys
ROOT=Path(__file__).resolve().parent
HOME=ROOT/'confirmation_v1'
MS=(1,2,4);RS=(1,2,4,6);BS=(.1,.2,.3,.5,1.)
CASES=[(m,r,b) for m in MS for r in RS for b in BS]
def require(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def na(x):return x is None or x==''
def number(x):
    require(not isinstance(x,bool) and not na(x),'Number missing/bool')
    value=float(x);require(math.isfinite(value),'Nonfinite number');return value
def boolean(x):
    require(x in (True,False,'True','False'),'Boolean encoding')
    return x is True or x=='True'
def case_name(m,r,b):return f'M{m}_R{r}_B{int(b*100):03d}.json'
def unique(records,fields):
    result={}
    for row in records:
        key=tuple(row[k] for k in fields);require(key not in result,'Duplicate '+str(key));result[key]=row
    return result

class Checks:
    def __init__(self):self.comparisons=0;self.maximum_residual=0.
    def same(self,actual,expected,label):
        self.comparisons+=1
        if expected is None:require(na(actual),'Expected NA: '+label);return
        if isinstance(expected,bool):require(boolean(actual)==expected,'Flag: '+label);return
        if isinstance(expected,(str,list,dict)):require(actual==expected,'Identity: '+label);return
        value=number(actual);error=abs(value-float(expected));self.maximum_residual=max(self.maximum_residual,error)
        require(error<=max(1e-10,abs(float(expected))*1e-10),'Numeric: '+label)
    def fields(self,row,expected,label):
        for key,value in expected.items():self.same(row[key],value,label+'.'+key)

def curve(A,T,order):
    require(set(order)==set(A) and len(order)==len(A),'Order not permutation')
    totalA=math.fsum(A.values());totalT=math.fsum(T.values());require(totalA>0,'Empty area')
    a=[0.];t=[0.]
    for i in order:a.append(a[-1]+A[i]);t.append(t[-1]+T[i])
    a[-1]=totalA;t[-1]=totalT
    return a,t,[x/totalA for x in a],[x/totalT if totalT else None for x in t]
def auc(s,c,u):
    if c[0] is None:return None
    return math.fsum(c[k]*max(0.,min(s[k+1],u)-s[k]) for k in range(len(s)-1))/u
def prefix(A,P,b):
    order=sorted(A,key=lambda i:(-P[i]/A[i],i))
    a,_,_,_=curve(A,P,order);k=bisect.bisect_right(a,b*math.fsum(A.values()))-1;original=k
    while math.fsum(A[i] for i in order[:k])>b*math.fsum(A.values()):k-=1
    return order[:k],original-k,order
def selection_values(A,T,selected,b):
    require(isinstance(selected,list) and len(selected)==len(set(selected)) and set(selected)<=set(A),'Selected IDs')
    area=math.fsum(A[i] for i in selected);target=math.fsum(T[i] for i in selected)
    D=math.fsum(A.values());total=math.fsum(T.values());require(area<=b*D,'Strict original budget')
    return dict(actual_area_m2=area,captured_target_m2=target,missed_target_m2=total-target,capture_fraction=target/total if total else None,
      selected_non_target_area_m2=area-target,unused_area_m2=b*D-area,unused_budget_m2=b*D-area,budget_area_m2=b*D,actual_area_fraction=area/D)

def independent_bound(final,prefix_p,total,raw,retained,original,fallback):
    """Re-derive witness rule without importing production arithmetic."""
    require(all(finite(v) and v>=0 for v in (final,prefix_p,total)),'Bound objective')
    tolerance=max(1e-10,total*1e-10);witness=max(final,prefix_p);require(witness<=total+tolerance,'Witness > total')
    contradicted=[name for name,value in [('raw_solver_upper',raw),('retained_upper',retained)] if finite(value) and value<witness-tolerance]
    reasons=contradicted+([] if finite(retained) else ['missing_or_nonfinite_retained_upper']);clamp=0.
    if reasons:upper=total;source='sum_predicted_safe_upper_after_bound_anomaly'
    else:
        upper=min([total,retained]+([raw] if finite(raw) else []));clamp=max(0.,witness-upper)
        require(clamp<=tolerance,'Clamp tolerance');upper=max(upper,witness);source='min(totalP,finite_raw_upper,retained_upper)'
    require(upper>=final,'Upper < final')
    gap=(upper-final)/max(abs(final),1e-30) if upper else 0.
    proven=bool(original and not fallback and not reasons and gap<=1e-8+1e-10)
    return dict(raw_solver_upper_m2=raw if finite(raw) else None,retained_upper_m2=retained if finite(retained) else None,
      raw_upper_was_missing_or_nonfinite=not finite(raw),bound_tolerance_m2=tolerance,strict_feasible_prefix_used_as_witness=True,
      feasible_objective_witness_m2=witness,contradicted_upper_fields=contradicted,bound_anomaly_reasons=reasons,bound_anomaly=bool(reasons),
      updated_upper_m2=upper,updated_primal_m2=final,updated_absolute_gap_m2=upper-final,updated_relative_gap=gap,updated_bound_source=source,
      roundoff_clamp_m2=clamp,original_proven_flag=bool(original),fallback_used=bool(fallback),updated_proven_flag=proven,
      certification_downgraded=bool(original and not proven),requested_relative_gap=1e-8,no_certification_upgrade=True)

def verify_units(unit,area_rows,checks):
    geometry={(int(x['M']),x['id']):x for x in area_rows};require(len(geometry)==len(area_rows),'Duplicate geometry')
    output={m:({}, {}) for m in MS};lookup={}
    for row in unit:
        m=int(row['M']);ix=int(row['ix']);iy=int(row['iy']);key=m,row['id']
        require(m in MS and row['id']==f'M{m}_E{ix}_N{iy}' and key not in lookup,'Truth unit ID')
        A=number(row['area_m2']);T=number(row['target_m2']);require(A>0 and 0<=T<=A+1e-9,'Truth area')
        ref=geometry.pop(key);checks.fields(ref,dict(ix=ix,iy=iy,area_m2=A),'geometry/unit')
        output[m][0][row['id']]=A;output[m][1][row['id']]=T;lookup[key]=(ix,iy,A,T)
    require(not geometry and all(output[m][0] for m in MS),'Incomplete unit domain')
    for m in (2,4):
        nested={}
        for (base,_),(ix,iy,A,T) in lookup.items():
            if base==1:
                nested.setdefault((ix//m,iy//m),[[],[]]);nested[ix//m,iy//m][0].append(A);nested[ix//m,iy//m][1].append(T)
        actual={(v[0],v[1]):v for k,v in lookup.items() if k[0]==m};require(set(actual)==set(nested),'Nested membership')
        for key,(aa,tt) in nested.items():checks.same(actual[key][2],math.fsum(aa),'nested A');checks.same(actual[key][3],math.fsum(tt),'nested T')
    return output
def decode_backend_number(value):
    if finite(value):return float(value)
    if value is None:return None
    if isinstance(value,dict) and set(value)=={'nonfinite_float'} and value['nonfinite_float'] in ('NaN','+Infinity','-Infinity'):return None
    raise ValueError('Raw solver number encoding')

def verify_raw_choice(raw,choice,backend,ids,A,P,b,checks):
    total=math.fsum(P.values());candidate=raw.get('solver_candidate_ids')
    require(candidate is None or isinstance(candidate,list) and len(candidate)==len(set(candidate)) and set(candidate)<=set(A),'Raw candidate IDs')
    integrality=raw.get('candidate_integrality_error');valid=isinstance(candidate,list) and finite(integrality) and 0<=integrality<=1e-6
    valid=valid and finite(raw.get('candidate_area_m2')) and finite(raw.get('candidate_predicted_m2')) and raw.get('candidate_feasible_in_original_units') is True
    if candidate is not None:
        ca=math.fsum(A[i] for i in candidate);cp=math.fsum(P[i] for i in candidate)
        if finite(raw.get('candidate_area_m2')):checks.same(raw['candidate_area_m2'],ca,'raw candidate A')
        if finite(raw.get('candidate_predicted_m2')):checks.same(raw['candidate_predicted_m2'],cp,'raw candidate P')
        valid=valid and ca<=b*math.fsum(A.values())
    chosen=raw['selected_ids'] if valid else prefix(A,P,b)[0]
    if valid:require(chosen==candidate,'Valid candidate passthrough')
    checks.same(choice['selected_ids'],chosen,'Recovery IDs');checks.same(choice['fallback_used'],not valid,'Recovery fallback')
    final=math.fsum(P[i] for i in chosen);values=selection_values(A,P,chosen,b)
    checks.fields(choice,dict(selected_area_m2=values['actual_area_m2'],predicted_primal_m2=final,strict_original_unit_feasible=True,original_area_tolerance_m2=0.,budget_area_m2=b*math.fsum(A.values()),total_predicted_m2=total,total_area_m2=math.fsum(A.values())),'choice')
    upper=raw.get('raw_solver_dual_upper_m2');tol=max(1e-10,total*1e-10)
    anomaly='missing_or_nonfinite_raw_upper' if not finite(upper) else 'raw_upper_below_final_primal' if final-upper>tol else None
    retained=total if anomaly else max(final,min(total,upper));gap=(retained-final)/max(abs(final),1e-30) if retained else 0.
    checks.fields(choice['recomputed_bound'],dict(upper_m2=retained,primal_m2=final,absolute_gap_m2=retained-final,relative_gap=gap,bound_anomaly=anomaly),'retained')
    proven=bool(valid and raw['solver_status']==0 and gap<=1e-8+1e-10);checks.same(choice['optimality_proven'],proven,'Original flag')
    checks.fields(choice['raw_candidate_audit'],dict(candidate_ids=candidate,accepted_for_passthrough=bool(valid),
      recorded_feasible_flag=raw.get('candidate_feasible_in_original_units'),raw_solver_status=raw.get('solver_status'),
      raw_solver_upper_m2=upper,raw_selected_ids=raw.get('selected_ids')),'Raw candidate audit')
    require(backend['actual_solver_calls_for_case']==1 and backend['persisted_before_selector_audit'] is True,'Backend persistence')
    require(backend['solver_options_passed']['threads']==1 and backend['solver_options_passed']['mip_rel_gap']==1e-8,'P resources')
    result=backend['result'];checks.same(result['status'],raw['solver_status'],'Backend status')
    dual=decode_backend_number(result.get('mip_dual_bound'));checks.same(upper,-dual*(total if total>0 else 1.) if dual is not None else None,'Backend upper')
    x=result.get('x')
    if isinstance(x,list) and all(finite(v) for v in x):
        require(len(x)==len(ids),'Backend x dimension');err=max(abs(v-round(v)) for v in x)
        if err<=1e-6 and all(-1e-6<=v<=1+1e-6 for v in x):checks.same(candidate,[i for i,v in zip(ids,x) if round(v)],'Backend candidate')
    return final,retained,proven,not valid

def verify_selection(folder,item,checks,core):
    core.completed_stage(folder)
    for key in ('selection_seal','artifact_manifest'):core.bound_file(item[key]['path'],item[key]['sha256'])
    seal=read(folder/'selection_seal.json');require(seal['run_id']==item['run_id'] and seal['case_order']==[list(x) for x in CASES] and seal['truth_used_for_choices'] is False,'Outer selection scope')
    require(seal['case_count']==120 and seal['prefix_count']==60 and seal['P_solver_calls']==60 and seal['ranking_count']==12,'Outer selection counts')
    for pin in [*seal['files'].values(),seal['A_P_only_inputs']]:core.bound_file(pin['path'],pin['sha256'])
    inner=read(folder/'policy/selection_seal.json')
    require(inner['all_60_final_choices_saved'] is True and inner['target_join_started'] is False and inner['case_order']==[list(x) for x in CASES],'Inner label-free seal')
    require(inner['context']=={k:seal[k] for k in ('run_id','freeze_sha256','all_predictions_sha256','geometry_metadata_sha256','checkpoint_sha256')},'Inner/outer selection context')
    core.bound_file(folder/'policy/selection_inputs_without_annotations.csv',inner['allowed_inputs_sha256'])
    for group in ('solver_raw_sha256','raw_sha256','choices_sha256'):
        require(len(inner[group])==60,'Inner case count')
        for rel,digest in inner[group].items():core.bound_file(core.resolve_under(folder/'policy',rel),digest)
    allowed=rows(folder/'policy/selection_inputs_without_annotations.csv')
    require(all(set(x)=={'M','id','area','predicted_R1','predicted_R2','predicted_R4','predicted_R6'} for x in allowed),'Selection whitelist')
    return seal,allowed

def verify_models(population,unit,folder,checks,core):
    fixed=rows(folder/'models/all_fixed_budgets.csv');aucs=rows(folder/'models/all_prefix_auc.csv');curves=rows(folder/'models/all_complete_prefix_curves.csv')
    require(len(population['models'])==24 and len(fixed)==2880 and len(aucs)==288,'Full model table counts')
    F=unique(fixed,('run_id','M','R','budget','policy'));U=unique(aucs,('run_id','M','R'));C={}
    for row in curves:C.setdefault((row['run_id'],int(row['M']),int(row['R'])),[]).append(row)
    require(len(C)==288,'Curve population')
    bindings=read(folder/'models/model_bindings.json')
    require(len(bindings)==24 and [x['run_id'] for x in bindings]==[x['run_id'] for x in population['models']],'Outcome model bindings')
    for slot,(binding,item) in enumerate(zip(bindings,population['models']),1):
        require(binding['source_selection']==item['selection_seal'] and Path(binding['result_directory']).resolve()==(folder/'models'/f'{slot:02d}').resolve(),'Outcome source/destination binding')
    totals=dict(models=24,fixed_budget_rows=0,prefix_curves=0,auc_rows=0,prefix_curve_points=0,raw_P_cases=0,bound_anomalies=0,fallbacks=0,recorded_proven=0)
    for slot,item in enumerate(population['models'],1):
        src=Path(item['selection_seal']['path']).parent;seal,allowed=verify_selection(src,item,checks,core);rid=item['run_id']
        rankings=unique(read(src/'rankings.json'),('M','R'));prefixes=unique(read(src/'prefix_choices.json'),('M','R','budget'));bounds=unique(read(src/'bound_audit.json'),('M','R','budget'))
        require(set(rankings)=={(m,r) for m in MS for r in RS} and set(prefixes)==set(bounds)==set(CASES),'Saved case grids')
        require(len(allowed)==sum(len(unit[m][0]) for m in MS),'Selection unit count')
        for m in MS:
            A,T=unit[m];data=[x for x in allowed if int(x['M'])==m];ids=[x['id'] for x in data]
            require(ids==sorted(A),'ASCII input order')
            for x in data:checks.same(x['area'],A[x['id']],'Prediction/truth A')
            for r in RS:
                P={x['id']:number(x[f'predicted_R{r}']) for x in data};require(all(0<=P[i]<=A[i]+1e-9 for i in ids),'Invalid P')
                order=sorted(ids,key=lambda i:(-P[i]/A[i],i));checks.same(rankings[m,r]['ordered_ids'],order,'P/A ranking')
                a,t,s,c=curve(A,T,order);actual=C.pop((rid,m,r));require([int(x['k']) for x in actual]==list(range(len(ids)+1)),'Full ordered prefix points')
                for k,row in enumerate(actual):checks.fields(row,dict(actual_area_m2=a[k],target_m2=t[k],area_fraction=s[k],capture_fraction=c[k],selection_seal_sha256=item['selection_seal']['sha256']),'curve')
                u=U.pop((rid,str(m),str(r)));checks.fields(u,dict(auc_0_1=auc(s,c,1.),auc_0_half_normalized=auc(s,c,.5),selection_seal_sha256=item['selection_seal']['sha256']),'AUC')
                totals['prefix_curves']+=1;totals['auc_rows']+=1;totals['prefix_curve_points']+=len(actual)
                for b in BS:
                    case=m,r,b;name=case_name(*case);pre,trim,_=prefix(A,P,b);pr=prefixes[case]
                    checks.fields(pr,dict(selected_ids=pre,area_m2=math.fsum(A[i] for i in pre),predicted_m2=math.fsum(P[i] for i in pre),budget_area_m2=b*math.fsum(A.values()),tail_trim_count=trim,strict_original_budget=True),'prefix')
                    choice=read(src/'policy/choices'/name);raw=read(src/'policy/raw'/name);backend=read(src/'policy/solver_raw'/name)
                    require((choice['M'],choice['R'],choice['budget'])==case and choice['raw_sha256']==core.sha(src/'policy/raw'/name),'Choice/raw binding')
                    require(backend['case']==list(case) and backend['selection_input_sha256']==choice['selection_input_sha256'],'Backend case/input identity')
                    final,retained,oldproven,fallback=verify_raw_choice(raw,choice,backend,ids,A,P,b,checks)
                    ba=independent_bound(final,math.fsum(P[i] for i in pre),math.fsum(P.values()),raw.get('raw_solver_dual_upper_m2'),retained,oldproven,fallback)
                    checks.fields(bounds[case],dict(selected_ids=choice['selected_ids'],**ba),'uniform bound')
                    totals['raw_P_cases']+=1;totals['bound_anomalies']+=ba['bound_anomaly'];totals['fallbacks']+=fallback;totals['recorded_proven']+=ba['updated_proven_flag']
                    for policy,selected in [('model_prefix',pre),('predicted_area',choice['selected_ids'])]:
                        row=F.pop((rid,str(m),str(r),str(b),policy));checks.same(json.loads(row['selected_ids_json']),selected,'Outcome IDs')
                        val=selection_values(A,T,selected,b);checks.fields(row,{k:val[k] for k in ('actual_area_m2','unused_area_m2','captured_target_m2','missed_target_m2','capture_fraction','selected_non_target_area_m2')},'outcome')
                        checks.same(row['selection_seal_sha256'],item['selection_seal']['sha256'],'Outcome seal')
                        for key in ('fallback_used','bound_anomaly','updated_upper_m2','updated_relative_gap','updated_proven_flag','certification_downgraded'):checks.same(row[key],ba[key] if policy=='predicted_area' else None,'outcome '+key)
                        totals['fixed_budget_rows']+=1
        for filename,allrows in [('fixed_budgets.csv',fixed),('prefix_auc.csv',aucs),('complete_prefix_curves.csv',curves)]:
            require(rows(folder/'models'/f'{slot:02d}'/filename)==[x for x in allrows if x['run_id']==rid],'Combined/per-model mismatch')
    require(not F and not U and not C,'Extra model rows')
    return totals

def verify_label_backend(backend,raw,ids,A,T,b,checks):
    """Reconstruct the saved label candidate, including every rejected-x branch."""
    total=math.fsum(T.values());D=math.fsum(A.values());require(total>0,'Positive-T backend only')
    options=dict(time_limit=30.,mip_rel_gap=1e-8,presolve=True,threads=1,random_seed=20260916,
      mip_feasibility_tolerance=1e-9,primal_feasibility_tolerance=1e-9)
    require(backend['actual_calls']==1 and backend['persisted_before_selector_audit'] is True,'Label backend call persistence')
    require(backend['options']==options and raw['options']==options,'Label complete options')
    result=backend['result'];positive=[i for i in ids if T[i]>0];x=result.get('x')
    require(x is None or isinstance(x,list) and len(x)==len(positive),'Label raw x shape')
    values=None if x is None else [decode_backend_number(v) for v in x]
    error=None;violation=None;feasible=False;selected=[]
    if values is not None and all(v is not None for v in values):
        # The wrapper rejects a finite vector outside this range before adapting it.
        require(all(-1e-6<=v<=1+1e-6 for v in values),'Completed label x outside binary bounds')
        error=max(abs(v-round(v)) for v in values)
        if error<=1e-6:
            candidate=[i for i,v in zip(positive,values) if round(v)]
            violation=max(0.,math.fsum(A[i] for i in candidate)-b*D)
            feasible=violation==0.
            if feasible:selected=candidate
    p=math.fsum(T[i] for i in selected);capture=p/total;selectedA=math.fsum(A[i] for i in selected)
    dual=decode_backend_number(result.get('mip_dual_bound'));upper_raw=-dual if dual is not None else None
    upper=min(1.,upper_raw) if upper_raw is not None else 1.
    if upper<capture:upper=capture if capture-upper<=1e-10 else 1.
    gap=(upper-capture)/max(abs(capture),1e-30)
    proven=bool(result['status']==0 and feasible and gap<=1e-8+1e-10)
    expected=dict(status='optimal' if proven else 'solver_result_not_proven_optimal',selected_ids=selected,candidate_integrality_error=error,candidate_budget_violation=violation,
      candidate_feasible=feasible,selected_area_feasibility_audit=selectedA,budget_area=b*D,
      primal_target=p,primal_capture_fraction=capture,dual_capture_bound=upper,dual_target_bound=upper*total,
      relative_gap=gap,absolute_target_gap=(upper-capture)*total,optimality_proven=proven,
      solver_status=result['status'],raw_solver_minimization_primal=decode_backend_number(result.get('fun')),
      raw_solver_minimization_dual=dual,raw_solver_capture_upper=upper_raw,
      raw_solver_relative_gap=decode_backend_number(result.get('mip_gap')),
      positive_target_variables=len(positive),zero_target_variables_excluded=len(ids)-len(positive),
      minimum_selected_area_claim=False,secondary_min_area_solved=False)
    checks.fields(raw,expected,'Independent label raw reconstruction')
    return expected

def verify_references(unit,folder,checks,core):
    import numpy as np
    folder=folder/'shared_reference';seal=read(folder/'reference_seal.json')
    for rel,digest in seal['files'].items():core.bound_file(core.resolve_under(folder,rel),digest)
    summary=read(folder/'summary.json');require(summary==seal['summary'],'Reference summary seal')
    require(summary['unique_label_cases']==15 and summary['random_orders_per_M']==1000 and summary['random_seed']==20260916 and summary['shared_across_all24_models'] is True and summary['reference_rows_not_model_replicates'] is True,'Reference scope')
    LP=unique(read(folder/'label_prefix_budgets.json'),('M','budget'));LM=unique(read(folder/'label_MILP_budgets.json'),('M','budget'))
    LC=read(folder/'label_prefix_curves.json');LU=unique(read(folder/'label_prefix_auc.json'),('M',));RB=unique(read(folder/'random_budgets.json'),('M','repetition','budget'));RU=unique(read(folder/'random_auc.json'),('M','repetition'));RQ=unique(read(folder/'random_summary.json'),('M','budget'))
    require(len(LP)==len(LM)==len(RQ)==15 and len(LU)==3 and len(RB)==15000 and len(RU)==3000,'Full reference inventory')
    require({row['M'] for row in LC}==set(MS) and all(row['method']=='label_informed_prefix' for row in LC),'Foreign/missing label prefix M/method')
    points=0;label_calls=0;label_anomalies=0
    for m in MS:
        A,T=unit[m];ids=sorted(A);total=math.fsum(T.values());D=math.fsum(A.values());order=sorted(ids,key=lambda i:(-T[i]/A[i],i));a,t,s,c=curve(A,T,order)
        saved=[x for x in LC if x['M']==m];require([x['k'] for x in saved]==list(range(len(ids)+1)),'Label prefix points')
        for k,row in enumerate(saved):checks.fields(row,dict(area_fraction=s[k],captured_target_m2=t[k],capture_fraction=c[k]),'Label curve')
        checks.fields(LU.pop((m,)),dict(auc_0_1=auc(s,c,1.),auc_0_half_normalized=auc(s,c,.5)),'Label AUC')
        for b in BS:
            selected,trim,_=prefix(A,T,b);val=selection_values(A,T,selected,b);pre=LP.pop((m,b))
            checks.fields(pre,{k:v for k,v in val.items() if k!='unused_area_m2'},'Label prefix');checks.fields(pre,dict(selected_ids=selected,k=len(selected),prefix_tail_trim_count=trim),'Label prefix IDs')
            label=LM.pop((m,b));name=f'M{m}_B{int(b*100):03d}';raw=read(folder/'label'/(name+'_adapted_return.json'));audited=read(folder/'label'/(name+'_audited.json'))
            require(label==audited,'Shared/per-case label mismatch')
            require(Path(label['raw_return']['path']).resolve()==(folder/'label'/(name+'_adapted_return.json')).resolve(),'Label raw return path identity')
            core.bound_file(label['raw_return']['path'],label['raw_return']['sha256'])
            checks.same(label['selected_ids'],raw['selected_ids'],'Label IDs unchanged')
            v=selection_values(A,T,raw['selected_ids'],b);checks.fields(label,{k:v[k] for k in ('actual_area_m2','unused_budget_m2','captured_target_m2','missed_target_m2','capture_fraction')},'Label MILP')
            calls=1 if total else 0;checks.same(label['solver_calls'],calls,'Label calls');label_calls+=calls
            raw_upper=raw.get('raw_solver_capture_upper');raw_upper=raw_upper*total if raw_upper is not None else (0. if not total else None)
            ba=independent_bound(v['captured_target_m2'],val['captured_target_m2'],total,raw_upper,raw['dual_target_bound'],raw['optimality_proven'],False)
            checks.fields(label['bound_audit'],ba,'Label bound');checks.fields(label,dict(updated_upper_m2=ba['updated_upper_m2'],updated_relative_gap=ba['updated_relative_gap'] if total else None,updated_proven_flag=ba['updated_proven_flag'],bound_anomaly=ba['bound_anomaly'],minimum_selected_area_claim=False,selected_ids_unchanged=True),'Label flags');label_anomalies+=ba['bound_anomaly']
            if total:
                backend=read(folder/'label'/(name+'_solver_raw.json'))
                checks.fields(backend,dict(M=m,budget=b),'Label backend case')
                verify_label_backend(backend,raw,ids,A,T,b,checks)
            else:
                require(not (folder/'label'/(name+'_solver_raw.json')).exists(),'Zero T called solver')
                checks.fields(raw,dict(status='no_target',solver_status=None,primal_target=0.,dual_target_bound=0.,primal_capture_fraction=None,
                  dual_capture_bound=None,relative_gap=None,optimality_proven=True,selected_ids=[],selected_area_feasibility_audit=0.,minimum_selected_area_claim=False),'Zero-target label return')
        with np.load(folder/f'random_full_curves_M{m}.npz',allow_pickle=False) as archive:
            require(set(archive.files)=={'area_fraction','capture_fraction','actual_area_m2','captured_target_m2','order_indices','unit_ids'},'Random keys')
            z={key:archive[key] for key in archive.files}
            require(z['unit_ids'].tolist()==ids and z['order_indices'].dtype==np.int32 and z['order_indices'].shape==(1000,len(ids)),'Random IDs/order')
            for key in ('area_fraction','capture_fraction','actual_area_m2','captured_target_m2'):require(z[key].shape==(1000,len(ids)+1) and z[key].dtype==np.float64,'Random shape/dtype')
            rng=np.random.Generator(np.random.PCG64(20260916));bybudget={b:[] for b in BS}
            for repeat in range(1,1001):
                indices=rng.permutation(len(ids));require(np.array_equal(indices,z['order_indices'][repeat-1]),'Random RNG/order')
                rank=[ids[int(i)] for i in indices];aa,tt,ss,cc=curve(A,T,rank)
                for key,values in [('actual_area_m2',aa),('captured_target_m2',tt),('area_fraction',ss),('capture_fraction',cc)]:
                    expected=np.asarray([np.nan if v is None else v for v in values],np.float64);actual=z[key][repeat-1]
                    require(np.array_equal(np.isnan(expected),np.isnan(actual)),'Random NA mask');mask=np.isfinite(expected);require(np.isfinite(actual[mask]).all(),'Nonfinite random')
                    if mask.any():checks.same(float(np.max(np.abs(actual[mask]-expected[mask]))),0.,'Random full-step residual')
                points+=len(ids)+1;checks.fields(RU.pop((m,repeat)),dict(auc_0_1=auc(ss,cc,1.),auc_0_half_normalized=auc(ss,cc,.5)),'Random AUC')
                for b in BS:
                    k=bisect.bisect_right(aa,b*D)-1;original=k
                    while math.fsum(A[i] for i in rank[:k])>b*D:k-=1
                    v=selection_values(A,T,rank[:k],b);saved=RB.pop((m,repeat,b));checks.fields(saved,{key:value for key,value in v.items() if key!='unused_area_m2'},'Random budget');checks.fields(saved,dict(k=k,prefix_tail_trim_count=original-k),'Random k');bybudget[b].append(v)
            for b,vals in bybudget.items():
                row=RQ.pop((m,b));checks.same(row['orders'],1000,'Random summary count')
                for field in ('capture_fraction','actual_area_m2','captured_target_m2','missed_target_m2','unused_budget_m2'):
                    values=[x[field] for x in vals]
                    if values[0] is None:mean=lo=hi=None
                    else:
                        ordered=sorted(values);mean=math.fsum(values)/len(values)
                        def quant(p):
                            h=(len(ordered)-1)*p;j=math.floor(h);return ordered[j]+(h-j)*(ordered[min(j+1,len(ordered)-1)]-ordered[j])
                        lo=quant(.025);hi=quant(.975)
                    checks.fields(row,{field+'_mean':mean,field+'_q025':lo,field+'_q975':hi},'Random summary')
    require(not LP and not LM and not LU and not RB and not RU and not RQ,'Extra reference records')
    checks.fields(summary,dict(actual_label_solver_calls=label_calls,label_bound_anomalies=label_anomalies),'Reference totals')
    return dict(label_cases=15,label_prefix_curves=3,label_prefix_auc_rows=3,actual_saved_label_solver_calls=label_calls,random_full_curves=3000,random_curve_points=points,random_budget_rows=15000,random_auc_rows=3000,random_summary_rows=15)

def verify_native_proof(folder,home,contract,unit,checks,core):
    """Verify proof identities and conservation, without decoding native masks."""
    geometry=read(home/'stages/local_geometry/metadata.json');truth=read(home/'stages/truth_local/truth_manifest.json')
    native=read(folder/'native_truth_audit.json');begin=read(folder/'begin_binding.json');end=read(folder/'end_binding.json')
    require(begin['watched_sha256']==end['watched_sha256'] and end['all_rechecked'] is True,'Native begin/end identity')
    for path,digest in begin['watched_sha256'].items():core.bound_file(path,digest)
    core.bound_file(folder/'unit_truth.csv',end['artifact_truth_sha256'])
    source=contract['datasets']['local'];core.bound_file(source['source_summary_path'],source['source_summary_sha256'])
    summary=read(source['source_summary_path']);sources={x['source_id']:x for x in summary['sources']}
    require(len(native['windows'])==len(geometry['windows'])==len(truth['records'])==338,'Native proof population')
    require(native['validation_T_reused'] is False and native['target_source']=='native binary mask>0 intersect native alpha; native CSR integration','Native semantics')
    for record,window,proof in zip(truth['records'],geometry['windows'],native['windows']):
        sid=window['source_id'];require(record['id']==window['id']==proof['id'] and proof['source_id']==sid,'Native proof identity')
        ref=record['native_reference'];require(ref['source_id']==sid and ref['native_window_xywh']==[window['x'],window['y'],512,512] and Path(ref['mask_source']).resolve()==Path(sources[sid]['mask']).resolve(),'Native source window')
        require(proof['mask_sha256']==sources[sid]['mask_sha256']==begin['watched_sha256'][sources[sid]['mask']] and proof['geometry_sha256']==window['sha256'],'Native mask/CSR binding')
        core.bound_file(core.resolve_under(home/'stages/local_geometry',window['file']),window['sha256'])
        checks.same(proof['area_m2'],window['raw_area_m2'],'Native A');checks.same(proof['native_valid_pixels'],window['native_valid_pixels'],'Native valid count')
        T=number(proof['target_m2']);expected=number(proof['independent_native_target_m2']);require(0<=T<=number(proof['area_m2'])+1e-9,'Native T')
        checks.same(proof['target_relative_error'],abs(T-expected)/max(abs(expected),1e-30),'Native residual')
        require(proof['target_relative_error']<1e-9 and 0<=proof['native_target_pixels']<=proof['native_valid_pixels'],'Native proof tolerance/count')
    for key in ('area_m2','target_m2','native_valid_pixels','native_target_pixels'):checks.same(native[key],math.fsum(x[key] for x in native['windows']),'Native domain '+key)
    checks.same(native['area_m2'],math.fsum(unit[1][0].values()),'Native/unit A');checks.same(native['target_m2'],math.fsum(unit[1][1].values()),'Native/unit T')
    checks.same(native['area_m2'],geometry['area_m2'],'Native/geometry A')
    return {'windows':338,'native_CSR_to_T_source_proof_verified':True,'native_masks_decoded':0,'native_T_reintegrated':False,
      'scope':'Saved native-CSR provenance and per-window conservation proof, unit A/T nesting and source hashes; no new mask integration or independent field measurement.'}

def gate(root_review,choice_review):
    sys.path.insert(0,str(HOME));import core
    watched={str(Path(__file__).resolve()):core.sha(__file__),str(Path(root_review).resolve()):core.sha(root_review),str(Path(choice_review).resolve()):core.sha(choice_review)}
    for name in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json'):watched[str(HOME/name)]=core.sha(HOME/name)
    contract,models,freeze=core.validate_freeze(check_models=False)
    grant=read(root_review);require(grant.get('passed') is True and grant.get('scope')=='confirmation_local_outputs_independent','Root output-audit grant required')
    require(grant['implementation_sha256']==core.sha(__file__) and grant['freeze_sha256']==core.sha(HOME/'FREEZE.json') and grant['choice_review_sha256']==core.sha(choice_review),'Root source/freeze/selection binding')
    require(grant['model_forward_authorized'] is False and grant['solver_authorized'] is False and grant['native_mask_decode_authorized'] is False,'Audit permissions')
    from pipeline import validate_global_predictions,validate_choices
    validate_global_predictions();accepted=validate_choices(choice_review)
    for stage in ('local_outcomes','truth_local','local_geometry','local_selections_seal','predictions_seal'):
        folder=HOME/'stages'/stage;core.completed_stage(folder);watched[str(folder/'artifact_manifest.json')]=core.sha(folder/'artifact_manifest.json')
    folder=HOME/'stages/local_outcomes';require(grant['target_artifact_manifest_sha256']==core.sha(folder/'artifact_manifest.json'),'Outcome binding')
    gp=grant['geometry_review'];core.bound_file(gp['path'],gp['sha256']);watched[str(Path(gp['path']).resolve())]=gp['sha256'];review=read(gp['path'])
    require(review.get('passed') is True and review['scope']=='confirmation_geometry_independent' and review['freeze_sha256']==core.sha(HOME/'FREEZE.json') and review['metadata_sha256']==core.sha(HOME/'stages/local_geometry/metadata.json'),'Geometry review')
    population=read(HOME/'stages/local_selections_seal/all_local_selections.json');roster=[x for x in models if x['dataset']=='local' and x['variant']!='regression']
    require(len(roster)==24 and [x['run_id'] for x in population['models']]==[x['run_id'] for x in roster]==grant['local_run_ids'],'Frozen local24')
    require(population['P_solver_cases']==1440 and population['policy_choices']==2880 and population['truth_used_for_choices'] is False,'Population scope')
    accepted_by={x['run_id']:x for x in accepted['models']}
    for item,model in zip(population['models'],roster):
        require(item['selection_seal']==accepted_by[model['run_id']]['selection_seal'],'All30/local acceptance')
        seal=read(item['selection_seal']['path'])
        require(seal['checkpoint_sha256']==model['checkpoint_sha256'] and seal['freeze_sha256']==core.sha(HOME/'FREEZE.json') and seal['all_predictions_sha256']==core.sha(HOME/'stages/predictions_seal/all_predictions.json') and seal['geometry_metadata_sha256']==core.sha(HOME/'stages/local_geometry/metadata.json'),'Original model/seal identity')
        for key in ('selection_seal','artifact_manifest'):watched[str(Path(item[key]['path']).resolve())]=item[key]['sha256']
    truth=read(HOME/'stages/truth_local/truth_manifest.json')
    require(truth['choice_review']==core.binding(choice_review) and truth['all_predictions_seal']==core.binding(HOME/'stages/predictions_seal/all_predictions.json') and truth['input_manifest']==core.binding(HOME/'stages/inputs_local/input_manifest.json'),'Truth follows accepted all37/all30 on same input')
    for path,digest in read(folder/'begin_binding.json')['watched_sha256'].items():
        key=str(Path(path).resolve());require(key not in watched or watched[key]==digest,'Producer/auditor source conflict');watched[key]=digest
    for p,h in watched.items():core.bound_file(p,h)
    return core,contract,population,watched

def execute(root_review,choice_review):
    core,contract,population,watched=gate(root_review,choice_review)
    with core.attempt('audit_local_outcomes') as out:
        core.configure_cpu(out);core.write(out/'begin_binding.json',watched);checks=Checks();folder=HOME/'stages/local_outcomes'
        unit=verify_units(rows(folder/'unit_truth.csv'),rows(HOME/'stages/local_geometry/area_units.csv'),checks)
        proof=verify_native_proof(folder,HOME,contract,unit,checks,core)
        modelcounts=verify_models(population,unit,folder,checks,core);references=verify_references(unit,folder,checks,core)
        summary=read(folder/'summary.json')
        checks.fields(summary['models'],dict(models=24,fixed_rows=2880,auc_rows=288,full_curve_rows=modelcounts['prefix_curve_points']),'Summary')
        require(summary['model_choice_IDs_modified'] is False and summary['validation_truth_reused'] is False and summary['all_predictions_and_all30_choices_preceded_truth'] is True and summary['field_independence_claim'] is False and summary['herbicide_or_yield_claim'] is False,'Outcome claims')
        from pipeline import validate_global_predictions,validate_choices
        validate_global_predictions();validate_choices(choice_review)
        for stage in ('local_outcomes','truth_local','local_geometry','local_selections_seal'):core.completed_stage(HOME/'stages'/stage)
        for p,h in watched.items():core.bound_file(p,h)
        core.validate_source_lock()
        review={'passed':True,'stage':'local_outcomes','scope':'confirmation_local_outcomes_independent','audited_artifact_manifest_sha256':core.sha(folder/'artifact_manifest.json'),'freeze_sha256':core.sha(HOME/'FREEZE.json'),'implementation_sha256':core.sha(__file__),'root_review':core.binding(root_review),'all30_choice_acceptance':core.binding(choice_review),'counts':dict(**modelcounts,**references),'native_truth_proof':proof,'numeric_comparisons':checks.comparisons,'maximum_numeric_residual':checks.maximum_residual,'model_forward_calls':0,'solver_calls':0,'selected_IDs_changed':0,'cross_field_inference':False}
        core.write(out/'review.json',review);core.write(out/'end_binding.json',{'watched_sha256':watched,'all_rechecked':True})
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root-review',required=True);parser.add_argument('--choice-review',required=True);args=parser.parse_args();print(execute(args.root_review,args.choice_review))
