"""Independent label-free audit of all fixed selections; never opens truth or fits/solves."""
from pathlib import Path
import argparse,bisect,csv,hashlib,json,math
import core
HERE=Path(__file__).resolve().parent
RS=(1,2,4,6);MS=(1,2,4);BUDGETS=(('b10',1,10),('b20',1,5),('b30',3,10),('b50',1,2),('b100',1,1))
CASES=[(m,r,n/d) for m in MS for r in RS for _,n,d in BUDGETS]

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def lines(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(x) for x in f]
def close(a,b):
    core.require(math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-10),'Independent arithmetic differs: '+repr((a,b)))
def subset(ids,units):
    core.require(len(ids)==len(set(ids)) and set(ids)<=set(units),'Duplicate/unknown selected IDs')
    return math.fsum(units[i][0] for i in ids),math.fsum(units[i][1] for i in ids)
def rank_prefix(units,budget):
    order=sorted(units,key=lambda i:(-units[i][1]/units[i][0],i))
    # Match the declared sequential double accumulation, then strictly trim via fsum.
    accum=[0.]
    for i in order:accum.append(accum[-1]+units[i][0])
    accum[-1]=math.fsum(x[0] for x in units.values());limit=budget*accum[-1]
    k=bisect.bisect_right(accum,limit)-1;chosen=order[:k]
    while math.fsum(units[i][0] for i in chosen)>limit:chosen.pop()
    return order,chosen,k-len(chosen)

def independent_bound(primal,prefix,total,raw,retained,original,fallback,gap=1e-8):
    finite=lambda x:type(x) in (float,int) and math.isfinite(x)
    tol=max(1e-10,1e-10*total);witness=max(primal,prefix)
    core.require(0<=primal<=total+tol and 0<=prefix<=total+tol,'Invalid feasible prediction')
    contradicted=[k for k,v in [('raw_solver_upper',raw),('retained_upper',retained)] if finite(v) and v<witness-tol]
    reasons=contradicted+([] if finite(retained) else ['missing_or_nonfinite_retained_upper'])
    clamp=0.
    if reasons:upper=total;source='sum_predicted_safe_upper_after_bound_anomaly'
    else:
        initial=min([total,retained]+([raw] if finite(raw) else []));clamp=max(0.,witness-initial)
        core.require(clamp<=tol,'Bound clamp exceeds declared tolerance');upper=max(witness,initial);source='min(totalP,finite_raw_upper,retained_upper)'
    absolute=upper-primal;relative=absolute/max(abs(primal),1e-30) if upper else 0.
    qualified=bool(original and not fallback and not reasons and relative<=gap+1e-10)
    return dict(raw_solver_upper_m2=raw if finite(raw) else None,retained_upper_m2=retained if finite(retained) else None,
      raw_upper_was_missing_or_nonfinite=not finite(raw),bound_tolerance_m2=tol,strict_feasible_prefix_used_as_witness=True,
      feasible_objective_witness_m2=witness,contradicted_upper_fields=contradicted,bound_anomaly_reasons=reasons,bound_anomaly=bool(reasons),
      updated_upper_m2=upper,updated_primal_m2=primal,updated_absolute_gap_m2=absolute,updated_relative_gap=relative,
      updated_bound_source=source,roundoff_clamp_m2=clamp,original_proven_flag=bool(original),fallback_used=bool(fallback),
      updated_proven_flag=qualified,certification_downgraded=bool(original and not qualified),requested_relative_gap=gap,no_certification_upgrade=True)

def compare_record(saved,expected):
    for k,v in expected.items():
        if type(v) in (int,float):close(saved[k],v)
        elif isinstance(v,dict):compare_record(saved[k],v)
        else:core.require(saved[k]==v,'Changed record field: '+k)

def finite(v):return type(v) in (int,float) and math.isfinite(v)
def decode(v):
    if isinstance(v,dict) and set(v)=={'nonfinite_float'}:
        core.require(v['nonfinite_float'] in ('NaN','+Infinity','-Infinity'),'Invalid IEEE marker')
        return {'NaN':float('nan'),'+Infinity':float('inf'),'-Infinity':-float('inf')}[v['nonfinite_float']]
    if isinstance(v,list):return [decode(x) for x in v]
    return v

def solver_to_raw(units,budget,record):
    """Reconstruct the frozen selector's numerical fields from the persisted backend return."""
    ids=sorted(units);A=math.fsum(units[i][0] for i in ids);P=math.fsum(units[i][1] for i in ids);limit=budget*A
    x=decode(record.get('x'));candidate=[];chosen=[];ca=cp=error=violation=None;binary=False
    if x is not None:
        core.require(isinstance(x,list) and len(x)==len(ids),'Solver x dimension differs')
        if all(finite(v) for v in x):
            error=max(abs(v-round(v)) for v in x);binary=error<=1e-6 and all(-1e-6<=v<=1+1e-6 for v in x)
            if binary:
                candidate=[i for i,v in zip(ids,x) if round(v)!=0];ca,cp=subset(candidate,units);violation=max(0.,ca-limit)
                if violation==0:chosen=list(candidate)
    feasible=bool(binary and violation==0);a,p=subset(chosen,units)
    value=lambda key:decode(record.get(key)) if finite(decode(record.get(key))) else None
    dual=value('mip_dual_bound');raw_upper=-dual*(P if P>0 else 1.) if dual is not None else None
    upper=min(P,raw_upper) if raw_upper is not None else P;inversion=max(0.,p-upper)
    core.require(inversion<=max(1e-10,1e-10*P),'Raw solver bound violates frozen selector assertion');upper=max(p,upper)
    gap=(upper-p)/max(abs(p),1e-30) if upper else 0.;proven=record['status']==0 and feasible and gap<=1e-8+1e-10
    return dict(selected_ids=chosen,solver_candidate_ids=candidate,selected_area_m2=a,predicted_primal_m2=p,predicted_dual_upper_m2=upper,predicted_absolute_gap_m2=upper-p,
      predicted_relative_gap=gap,raw_solver_dual_upper_m2=raw_upper,bound_roundoff_clamp_m2=inversion,raw_minimization_fun=value('fun'),raw_minimization_dual=dual,
      raw_solver_gap=value('mip_gap'),solver_status=int(record['status']),solver_message=str(record['message']),status='optimal_within_recorded_gap' if proven else 'not_proven_optimal',
      optimality_proven_within_tolerance=bool(proven),candidate_feasible_in_original_units=feasible,candidate_area_m2=ca,candidate_predicted_m2=cp,candidate_budget_violation_m2=violation,
      candidate_integrality_error=error,rejected_candidate_fallback=None if feasible else 'empty known-feasible set; not asserted to be solver optimum',budget_fraction=budget,
      budget_area_m2=limit,total_area_m2=A,total_predicted_m2=P,n_variables=len(ids),mip_node_count=value('mip_node_count'),original_unit_feasibility_tolerance_m2=0.,
      binary_audit_tolerance=1e-6,normalization='cost/total_area; predicted/total_predicted unless zero',no_secondary_objective=True)

def recover_raw(units,budget,raw):
    """Independent reconstruction of strict feasibility/fallback and its first bound record."""
    A=math.fsum(v[0] for v in units.values());P=math.fsum(v[1] for v in units.values());limit=budget*A
    candidate=raw.get('solver_candidate_ids');ids_valid=isinstance(candidate,list);ca=cp=violation=None;reasons=[]
    if candidate is not None:core.require(ids_valid and all(isinstance(i,str) for i in candidate),'Malformed candidate')
    if not ids_valid:reasons.append('missing_candidate_ids')
    error=raw.get('candidate_integrality_error');binary=finite(error) and 0<=error<=1e-6
    if not binary:reasons.append('missing_nonfinite_or_nonbinary_candidate_audit')
    summaries=finite(raw.get('candidate_area_m2')) and finite(raw.get('candidate_predicted_m2'))
    if not summaries:reasons.append('missing_or_nonfinite_candidate_summary')
    if ids_valid:
        ca,cp=subset(candidate,units);violation=max(0.,ca-limit)
        if summaries:close(ca,raw['candidate_area_m2']);close(cp,raw['candidate_predicted_m2'])
        if violation>0:reasons.append('strict_original_budget_exceeded')
    if raw.get('candidate_feasible_in_original_units') is not True:reasons.append('raw_selector_rejected_candidate')
    valid=not reasons;before=None;trim=0
    if valid:
        core.require(raw.get('selected_ids')==candidate,'Feasible raw candidate changed');final=list(candidate)
    else:
        order,final,trim=rank_prefix(units,budget);before=order[:len(final)+trim]
    a,p=subset(final,units);core.require(a<=limit,'Recovery violated strict area')
    upper_raw=raw.get('raw_solver_dual_upper_m2');tol=max(1e-10,1e-10*P);anomaly=None;clamp=0.
    if not finite(upper_raw):anomaly='missing_or_nonfinite_raw_upper';upper=P
    elif p-upper_raw>tol:anomaly='raw_upper_below_final_primal';upper=P
    else:upper=min(P,upper_raw);clamp=max(0.,p-upper);upper=max(p,upper)
    gap=(upper-p)/max(abs(p),1e-30) if upper else 0.
    bound=dict(upper_m2=upper,primal_m2=p,absolute_gap_m2=upper-p,relative_gap=gap,bound_anomaly=anomaly,
      bound_source='sum_predicted_safe_upper' if anomaly else 'min(raw_solver_upper,sum_predicted)',raw_upper_m2=upper_raw if finite(upper_raw) else None,roundoff_clamp_m2=clamp,bound_inversion_tolerance_m2=tol)
    proven=False
    if valid:
        core.require(anomaly is None,'Accepted original candidate has anomalous bound')
        for k,v in [('selected_area_m2',a),('predicted_primal_m2',p),('predicted_dual_upper_m2',upper),('predicted_relative_gap',gap)]:close(raw[k],v)
        proven=bool(raw['solver_status']==0 and gap<=1e-8+1e-10);core.require(raw['optimality_proven_within_tolerance']==proven,'Raw certification flag differs')
    return dict(policy_label='predicted_area_milp_with_protocol_prefix_fallback_v1',selected_ids=final,selected_area_m2=a,predicted_primal_m2=p,budget_fraction=budget,budget_area_m2=limit,total_area_m2=A,total_predicted_m2=P,
      strict_original_unit_feasible=True,original_area_tolerance_m2=0.,fallback_used=not valid,fallback_reasons=reasons,prefix_before_tail_trim_ids=before,prefix_tail_trim_count=trim,
      status='feasible_fallback_not_proven' if not valid else raw['status'],optimality_proven=proven,recomputed_bound=bound,
      raw_candidate_audit=dict(candidate_ids=candidate,ids_valid=ids_valid,recorded_integrality_error=error,recorded_binary_tolerance=1e-6,binary_audit_passed=binary,summaries_finite=summaries,
        area_m2_recomputed=ca,predicted_m2_recomputed=cp,budget_violation_m2_recomputed=violation,recorded_feasible_flag=raw.get('candidate_feasible_in_original_units'),accepted_for_passthrough=valid,
        raw_selected_ids=raw.get('selected_ids'),raw_solver_status=raw.get('solver_status'),raw_optimality_proven=raw.get('optimality_proven_within_tolerance'),raw_solver_gap=raw.get('raw_solver_gap'),
        raw_reported_relative_gap=raw.get('predicted_relative_gap'),raw_solver_upper_m2=upper_raw))

class Watch:
    def __init__(self):self.files={}
    def add(self,path,digest=None):
        p=str(Path(path).resolve());actual=core.sha(p)
        if digest is not None:core.require(actual==digest,'Input binding mismatch: '+p)
        core.require(p not in self.files or self.files[p]==actual,'Input changed: '+p);self.files[p]=actual;return Path(p)
    def read(self,path,digest=None):return core.read(self.add(path,digest))
    def binding(self,item):return self.add(item['path'],item['sha256'])
    def stage(self,folder):
        self.add(Path(folder)/'artifact_manifest.json');manifest=core.completed_stage(folder)
        for name,digest in manifest['files'].items():self.add(core.resolve_under(folder,name),digest)
    def finish(self):
        for p,h in self.files.items():core.bound_file(p,h)

def local_integrals(maps,geometry,folder,watch):
    import numpy as np
    core.require([x['id'] for x in maps]==[x['id'] for x in geometry['windows']],'Local window join differs')
    accumulated={}
    for record,window in zip(maps,geometry['windows']):
        watch.add(record['path'],record['sha256']);path=core.resolve_under(folder,window['file']);watch.add(path,window['sha256'])
        with np.load(record['path'],allow_pickle=False) as z:
            core.require(set(z.files)=={'probabilities','valid','R','id'} and str(z['id'].item())==record['id'] and z['R'].tolist()==list(RS),'Map whitelist/identity differs')
            p=z['probabilities'].astype(np.float64).reshape(4,-1)
        with np.load(path,allow_pickle=False) as z:
            cells=z['cells'];a=z['area'];ix=z['indices'];weights=z['weights'];ptr=z['indptr']
        owner=np.repeat(np.arange(len(a)),np.diff(ptr))
        values=np.column_stack([np.bincount(owner,weights=weights*p[j,ix],minlength=len(a)) for j in range(4)])
        for j,(x,y) in enumerate(cells):
            for m in MS:
                k=(m,f'M{m}_E{int(x)//m}_N{int(y)//m}')
                accumulated.setdefault(k,[[],[],[],[],[]])
                for c,v in enumerate([float(a[j]),*values[j].tolist()]):accumulated[k][c].append(v)
    return {k:[math.fsum(v) for v in components] for k,components in accumulated.items()}

def audit_local(entry,slot,prediction,geometry,watch):
    folder=HERE/'stages'/f'local_select_{slot:02d}';watch.stage(folder)
    seal=watch.read(folder/'selection_seal.json');geo=HERE/'stages/local_geometry'
    compare_record(seal,dict(version='confirmation_local_selection_v1',run_id=entry['run_id'],checkpoint_sha256=entry['checkpoint_sha256'],
      freeze_sha256=core.sha(HERE/'FREEZE.json'),all_predictions_sha256=core.sha(HERE/'stages/predictions_seal/all_predictions.json'),
      geometry_metadata_sha256=core.sha(geo/'metadata.json'),case_order=[list(x) for x in CASES],case_count=120,P_solver_calls=60,prefix_count=60,ranking_count=12,
      truth_used_for_choices=False,selected_ids_unchanged_by_bound_audit=True,implementation_sha256=core.sha(HERE/'local_management.py')))
    expected_names={'prefix_choices.json','rankings.json','bound_audit.json','extraction_ledger.json','policy/selection_seal.json'}
    core.require(set(seal['files'])==expected_names,'Local file inventory changed')
    for name,item in seal['files'].items():
        core.require(Path(item['path']).resolve()==(folder/name).resolve(),'Local seal path differs');watch.binding(item)
    allowed_path=folder/'policy/selection_inputs_without_annotations.csv'
    core.require(Path(seal['A_P_only_inputs']['path']).resolve()==allowed_path.resolve(),'Local A/P path differs');watch.binding(seal['A_P_only_inputs'])
    allowed=rows(allowed_path);fields={'M','id','area',*[f'predicted_R{r}' for r in RS]}
    core.require(allowed and all(set(x)==fields for x in allowed),'Non-whitelisted selection inputs')
    ap={(int(x['M']),x['id']):[float(x['area']),*[float(x[f'predicted_R{r}']) for r in RS]] for x in allowed}
    core.require(len(ap)==len(allowed) and all(i.isascii() for m,i in ap),'Duplicate/non-ASCII units')
    maps=watch.read(prediction['map_index']['path'],prediction['map_index']['sha256'])
    integrated=local_integrals(maps,geometry,geo,watch);core.require(set(ap)==set(integrated),'Local integrated unit identity differs')
    for key,vals in ap.items():
        core.require(vals[0]>0 and all(math.isfinite(x) and 0<=x<=vals[0]+1e-9 for x in vals[1:]),'Invalid unit A/P')
        for a,b in zip(vals,integrated[key]):close(a,b)
    rankings=core.read(folder/'rankings.json');prefixes=core.read(folder/'prefix_choices.json');bounds=core.read(folder/'bound_audit.json')
    core.require([(x['M'],x['R']) for x in rankings]==[(m,r) for m in MS for r in RS],'Local ranking inventory')
    for items in (prefixes,bounds):core.require([(x['M'],x['R'],x['budget']) for x in items]==CASES,'Local case inventory')
    ranks={(x['M'],x['R']):x for x in rankings};inner=core.read(folder/'policy/selection_seal.json')
    compare_record(inner,dict(all_60_final_choices_saved=True,target_join_started=False,case_order=[list(x) for x in CASES],solver_calls=60,allowed_inputs_sha256=core.sha(allowed_path)))
    core.require(inner['context']=={k:seal[k] for k in ('run_id','freeze_sha256','all_predictions_sha256','geometry_metadata_sha256','checkpoint_sha256')},'Inner context differs')
    for category in ('raw','choices','solver_raw'):
        expected={f'{category}/M{m}_R{r}_B{round(b*100):03d}.json' for m,r,b in CASES}
        core.require(set(inner[category+'_sha256'])==expected,'Raw choice case inventory differs')
        for name,digest in inner[category+'_sha256'].items():watch.add(folder/'policy'/name,digest)
    qualified=anomalies=fallbacks=0
    for case,prefix,bound in zip(CASES,prefixes,bounds):
        m,r,b=case;units={i:(v[0],v[1+RS.index(r)]) for (mm,i),v in ap.items() if mm==m}
        A=math.fsum(x[0] for x in units.values());close(A,geometry['area_m2'])
        order,chosen,trim=rank_prefix(units,b)
        core.require(ranks[m,r]['ordered_ids']==order and ranks[m,r]['ASCII_ties'] is True,'Rank/tie ordering differs')
        a,p=subset(chosen,units);compare_record(prefix,dict(selected_ids=chosen,area_m2=a,predicted_m2=p,budget_area_m2=b*A,tail_trim_count=trim,strict_original_budget=True))
        name=f'M{m}_R{r}_B{round(b*100):03d}.json';choice_path=folder/'policy/choices'/name
        core.require(bound['choice']==core.binding(choice_path),'Bound choice binding differs');choice=core.read(choice_path);raw=core.read(folder/'policy/raw'/name);solver=core.read(folder/'policy/solver_raw'/name)
        core.require(choice['selected_ids']==bound['selected_ids'],'Final IDs changed');a,primal=subset(choice['selected_ids'],units);core.require(a<=b*A,'Original strict budget exceeded')
        close(a,choice['selected_area_m2']);close(primal,choice['predicted_primal_m2']);close(raw['raw_solver_dual_upper_m2'],choice['raw_candidate_audit']['raw_solver_upper_m2']) if type(raw.get('raw_solver_dual_upper_m2')) in (int,float) else core.require(raw.get('raw_solver_dual_upper_m2')==choice['raw_candidate_audit']['raw_solver_upper_m2'],'Missing raw upper differs')
        compare_record(solver,dict(case=list(case),actual_solver_calls_for_case=1,persisted_before_selector_audit=True,backend='scipy.optimize.milp'))
        compare_record(solver['solver_options_passed'],dict(mip_rel_gap=1e-8,time_limit=30.,presolve=True,threads=1,parallel=False,random_seed=20260916,mip_feasibility_tolerance=1e-9,primal_feasibility_tolerance=1e-9))
        payload={'ids':sorted(units),'area':[units[i][0] for i in sorted(units)],'predicted':[units[i][1] for i in sorted(units)],'budget':b}
        expected_sha=hashlib.sha256(json.dumps(payload,ensure_ascii=True,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        core.require(solver['selection_input_sha256']==expected_sha,'Recorded solver inputs differ')
        compare_record(raw,solver_to_raw(units,b,solver['result']))
        compare_record(raw['options'],solver['solver_options_passed'])
        compare_record(choice,recover_raw(units,b,raw))
        compare_record(choice,dict(M=m,R=r,budget=b,raw_origin='new_predeclared_case_solve',raw_file='raw/'+name,raw_sha256=core.sha(folder/'policy/raw'/name),
          recovery_source_sha256='7fdc3fac309b8b6e98d7c53934ad676fc305670643fd933f5d762443ca550225',requested_relative_gap=1e-8,selection_input_sha256=expected_sha))
        result=independent_bound(primal,p,math.fsum(x[1] for x in units.values()),raw.get('raw_solver_dual_upper_m2'),choice['recomputed_bound']['upper_m2'],choice['optimality_proven'],choice['fallback_used'])
        compare_record(bound,result);qualified+=result['updated_proven_flag'];anomalies+=result['bound_anomaly'];fallbacks+=result['fallback_used']
    return dict(run_id=entry['run_id'],selection_seal_sha256=core.sha(folder/'selection_seal.json'),selection_seal=core.binding(folder/'selection_seal.json'),case_count=120,integrated_windows=len(maps),unit_rows=len(ap),independently_checked_P_bounds=60,qualified_flags=qualified,bound_anomalies=anomalies,fallbacks=fallbacks)

def grid_values(plane,frame,g):
    import numpy as np
    core.require(plane.shape==(256,256) and np.isfinite(plane).all() and ((plane>=0)&(plane<=1)).all(),'Invalid external prediction')
    size=256//g;values=plane.astype(np.float64).reshape(g,size,g,size).sum(axis=(1,3))/65536;result={}
    for r in range(g):
        for c in range(g):
            value=math.fsum(map(float,plane[r*size:(r+1)*size,c*size:(c+1)*size].ravel()))/65536
            close(value,values[r,c]);result[f'{frame}:g{g:02d}:r{r:02d}:c{c:02d}']=value
    return result

def audit_external(entry,prediction,inp,watch):
    import numpy as np
    folder=HERE/'stages'/('select_'+entry['run_id']);watch.stage(folder);seal=core.read(folder/'selection_seal.json')
    compare_record(seal,dict(scope='confirmation_external_label_free_selection',run_id=entry['run_id'],dataset='weeds',freeze_sha256=core.sha(HERE/'FREEZE.json'),all_predictions_sha256=core.sha(HERE/'stages/predictions_seal/all_predictions.json'),R=list(RS),grids=[4,8,16],case_count=3120,rankings=312,unique_selections=2080,prediction_unit_rows=34944,truth_used_for_choices=False))
    core.require(set(seal['files_sha256'])=={'input_binding.json','prediction_units.csv','rankings.jsonl','choices.jsonl'},'External sealed files differ')
    for name,digest in seal['files_sha256'].items():watch.add(folder/name,digest)
    maps=watch.read(prediction['map_index']['path'],prediction['map_index']['sha256'])
    core.require([x['id'] for x in maps]==[x['id'] for x in inp['records']],'External map/frame identity differs')
    rankings=lines(folder/'rankings.jsonl');choices=lines(folder/'choices.jsonl');unit_rows=rows(folder/'prediction_units.csv')
    ranks={(x['R'],x['frame_id'],x['grid_g']):x for x in rankings};cases={(x['R'],x['frame_id'],x['grid_g'],x['budget_basis'],x['parent_nominal_budget_code']):x for x in choices}
    unit_lookup={(int(x['R']),x['frame_id'],int(x['grid_g']),x['id']):x for x in unit_rows}
    core.require(len(ranks)==len(rankings)==312 and len(cases)==len(choices)==3120 and len(unit_lookup)==len(unit_rows)==34944,'External duplicate/missing records')
    metadata=dict(run_id=entry['run_id'],variant=entry['variant'],seed=entry['seed'],original_selected_epoch=entry['best_epoch'],checkpoint_sha256=entry['checkpoint_sha256'],config_sha256=entry['config_sha256'],export_sha256=prediction['artifact_manifest']['sha256'],source_manifest_sha256=inp['source_manifest']['sha256'])
    for frame,record in zip(inp['records'],maps):
        watch.add(record['path'],record['sha256'])
        with np.load(record['path'],allow_pickle=False) as z:
            core.require(set(z.files)=={'probabilities','valid','R','id'} and np.all(z['valid']==1) and z['R'].tolist()==list(RS) and str(z['id'].item())==frame['id'],'External map whitelist/identity')
            p=z['probabilities'];core.require(p.shape==(4,256,256) and p.dtype==np.float32,'External map schema')
        for ri,r in enumerate(RS):
            for g in (4,8,16):
                values=grid_values(p[ri],frame['id'],g);order=sorted(values,key=lambda i:(-values[i],i));n=g*g;key=(r,frame['id'],g);rank=ranks.pop(key)
                ranking_key=f"{entry['run_id']}|R{r}|{frame['id']}|g{g:02d}";compare_record(rank,{**metadata,'acquisition_date':frame['acquisition_date'],'ranking_key':ranking_key,'ordered_ids':order})
                units=rank['units'];core.require(len(units)==n and [u['id'] for u in units]==sorted(values),'External unit order differs')
                for u in units:
                    core.require(set(u)=={'id','A','P'},'Non-whitelisted external selection unit');close(u['A'],1/n);close(u['P'],values[u['id']])
                    ur=unit_lookup.pop((*key,u['id']));core.require(ur['run_id']==entry['run_id'],'Wrong external unit model');close(ur['A'],1/n);close(ur['P'],values[u['id']])
                seen={}
                for basis in ('nominal_primary','matched_actual_secondary'):
                    for code,num,den in BUDGETS:
                        k=n*num//den if basis=='nominal_primary' else n*(16*num//den)//16
                        effective=num/den if basis=='nominal_primary' else (16*num//den)/16
                        row=cases.pop((*key,basis,code));selected=order[:k];choice_key=ranking_key+'|'+basis+'|'+code
                        compare_record(row,{**metadata,'acquisition_date':frame['acquisition_date'],'ranking_key':ranking_key,'choice_key':choice_key,'selection_key':ranking_key+f'|k{k:03d}',
                          'duplicate_selection_alias':seen.get(k),'parent_nominal_budget_fraction':num/den,'effective_budget_fraction':effective,'k':k,'N':n,'actual_fraction':k/n,
                          'unused_effective_fraction':effective-k/n,'selected_ids':selected,'selected_area_FAE':k/n,'selected_predicted_weed_FAE':math.fsum(values[i] for i in sorted(selected))})
                        seen.setdefault(k,choice_key)
    core.require(not ranks and not cases and not unit_lookup,'Extraneous external case')
    core.require(len({x['selection_key'] for x in choices})==2080,'External unique selection count differs')
    return dict(run_id=entry['run_id'],selection_seal_sha256=core.sha(folder/'selection_seal.json'),selection_seal=core.binding(folder/'selection_seal.json'),case_count=3120,prediction_unit_rows=34944,rankings=312,unique_selections=2080)

def validate(review_path,watch):
    watch.add(__file__);review=watch.read(review_path);watch.add(HERE/'FREEZE.json');watch.add(HERE/'SOURCE_LOCK.json')
    contract,models,_=core.validate_freeze()
    compare_record(review,dict(passed=True,scope='confirmation_all_selections_audit_execution',implementation_sha256=core.sha(__file__),freeze_sha256=core.sha(HERE/'FREEZE.json'),truth_access_authorized=False,new_model_or_solver_authorized=False))
    core.require(review['run_ids']==[x['run_id'] for x in models if x['variant']!='regression'],'Review model population')
    from pipeline import validate_global_predictions
    predictions=validate_global_predictions();watch.stage(HERE/'stages/predictions_seal')
    for key,scope,digest_name,target in [('predictions_review','confirmation_predictions_independent','all_predictions_sha256','stages/predictions_seal/all_predictions.json'),('geometry_review','confirmation_geometry_independent','metadata_sha256','stages/local_geometry/metadata.json')]:
        item=review[key];accepted=watch.read(item['path'],item['sha256'])
        core.require(accepted.get('passed') is True and accepted['scope']==scope and accepted[digest_name]==core.sha(HERE/target),'Missing independent input acceptance')
    for stage in ('inputs_local','inputs_weeds','local_geometry','local_selections_seal'):watch.stage(HERE/'stages'/stage)
    for p in predictions['models']:watch.stage(p['stage'])
    watch.finish();return models,predictions

def execute(review_path):
    watch=Watch();models,predictions=validate(review_path,watch)
    geometry=watch.read(HERE/'stages/local_geometry/metadata.json');inp=watch.read(HERE/'stages/inputs_weeds/input_manifest.json');population=watch.read(HERE/'stages/local_selections_seal/all_local_selections.json')
    local=[x for x in models if x['dataset']=='local' and x['variant']!='regression']
    core.require([x['run_id'] for x in population['models']]==[x['run_id'] for x in local] and population['truth_used_for_choices'] is False,'Local population identity')
    core.require(population['freeze_sha256']==core.sha(HERE/'FREEZE.json') and population['all_predictions_sha256']==core.sha(HERE/'stages/predictions_seal/all_predictions.json'),'Local population seal mismatch')
    core.require(not (HERE/'stages/truth_local').exists() and not (HERE/'stages/truth_weeds').exists(),'Choices must be independently accepted before truth stages')
    with core.attempt('audit_all_selections') as out:
        core.configure_cpu(out);results=[];by_prediction={x['run_id']:x for x in predictions['models']}
        for entry in models:
            if entry['variant']=='regression':continue
            if entry['dataset']=='local':
                slot=[x['run_id'] for x in local].index(entry['run_id'])+1;result=audit_local(entry,slot,by_prediction[entry['run_id']],geometry,watch)
                item=population['models'][slot-1];core.require(item['selection_seal']==result['selection_seal'] and item['case_count']==120,'Local population choice changed');watch.binding(item['artifact_manifest'])
            else:result=audit_external(entry,by_prediction[entry['run_id']],inp,watch)
            results.append(result);core.write(out/'progress.json',{'completed_models':len(results),'run_id':entry['run_id'],'truth_opened':False})
        core.require(len(results)==30,'Incomplete selection review');watch.finish();core.validate_source_lock()
        core.write(out/'review.json',dict(passed=True,scope='confirmation_all_selections_independent',implementation_sha256=core.sha(__file__),freeze_sha256=core.sha(HERE/'FREEZE.json'),
          all_predictions_sha256=core.sha(HERE/'stages/predictions_seal/all_predictions.json'),models=results,truth_used_for_choices=False,complete_case_inventory_verified=True,
          new_model_calls=0,new_solver_calls=0,annotation_arrays_opened=0,source_bindings=watch.files,
          limitations=['Numerical consistency of saved bounds is not an independent proof of optimality.','Local P is independently integrated from previously audited geometry weights; no second ground survey or raw polygon reconstruction.']))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true',required=True);p.add_argument('--root-review',required=True);a=p.parse_args();print(execute(a.root_review))
if __name__=='__main__':main()
