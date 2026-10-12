"""Independent stdlib arithmetic audit of completed common management v2 results.

No production evaluator imports, model inference, raw/cache pixels, GPU, or MILP.
Original results and historical independent checks remain read-only.
"""
from pathlib import Path
import collections
import csv
import datetime
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parent
PA=ROOT/'precision_agriculture'
BUDGETS=(.1,.2,.3,.5,1.)
COMBINATIONS={(m,r,b) for m in (1,2,4) for r in (1,2,4) for b in BUDGETS}
PIPELINE_SHA='99315d78bd31e721f024601227e032291962075336ed65a7e9bd157caf643d25'
PROTOCOL_SHA='d3a82af875ac95e18ebdd30d124f668b7ef8916244b57d07686af6ec6be56bc7'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def close(a,b,tol=1e-10):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a,b,rel_tol=tol,abs_tol=tol),(a,b)


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def write_csv(path,values):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(values[0]))
        writer.writeheader();writer.writerows(values)


def get_units(path):
    data=[]
    for row in read_csv(path):
        u={k:int(row[k]) for k in ('M','ix','iy')}
        u['id']=row['id']
        assert u['id'].isascii()
        for k in ('area','target','q_true','predicted_R1','predicted_R2','predicted_R4','q_R1','q_R2','q_R4'):
            u[k]=float(row[k]);assert math.isfinite(u[k])
        assert u['area']>0 and 0<=u['target']<=u['area']+1e-12
        close(u['q_true'],u['target']/u['area'],1e-12)
        for r in (1,2,4):
            assert 0<=u[f'predicted_R{r}']<=u['area']+1e-12
            close(u[f'q_R{r}'],u[f'predicted_R{r}']/u['area'],1e-12)
        data.append(u)
    return {m:[u for u in data if u['M']==m] for m in (1,2,4)}


def aggregate(ids,by_id,R):
    assert len(ids)==len(set(ids)) and set(ids)<=by_id.keys()
    a=math.fsum(by_id[i]['area'] for i in ids)
    t=math.fsum(by_id[i]['target'] for i in ids)
    p=math.fsum(by_id[i][f'predicted_R{R}'] for i in ids)
    zero=math.fsum(by_id[i]['area'] for i in ids if by_id[i]['target']==0)
    return a,t,p,zero


def payload_sha(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=True,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def number(x):return isinstance(x,(int,float)) and math.isfinite(x)


def same_optional(a,b):
    if a is None or b is None:assert a is b,(a,b)
    else:close(float(a),float(b),1e-12)


def audit_seal(directory):
    protocol=read(PA/'RM_common_pipeline_v2_protocol.json')
    assert sha(PA/'RM_common_pipeline_v2_protocol.json')==PROTOCOL_SHA
    assert sha(ROOT/'evaluate_common_management_v2.py')==PIPELINE_SHA
    summary=read(directory/'summary.json');inputs=read(directory/'inputs.json')
    assert summary['model_run'] in protocol['remaining_run_ids']
    assert summary['pipeline_protocol_sha256']==inputs['pipeline_protocol_sha256']==PROTOCOL_SHA
    assert summary['model_run'] not in protocol['known_completed_rejected']
    dest=directory/'predicted_area_policy';seal=read(dest/'selection_seal.json')
    assert seal['all_45_final_choices_saved'] and seal['target_join_started'] is False and seal['solver_calls']==45
    assert seal['implementation_sha256']==PIPELINE_SHA
    assert seal['case_order']==[list(x) for x in sorted(COMBINATIONS)]
    for category,key in [('raw','raw_sha256'),('choices','choices_sha256'),('solver_raw','solver_raw_sha256')]:
        expected={f'{category}/M{m}_R{r}_B{round(b*100):03d}.json' for m,r,b in COMBINATIONS}
        assert set(seal[key])==expected
        assert len(list((dest/category).glob('*.json')))==45
        for rel,digest in seal[key].items():assert sha(dest/rel)==digest
    assert sha(dest/'selection_inputs_without_annotations.csv')==seal['allowed_inputs_sha256']
    assert seal['context']['inputs_sha256']==sha(directory/'inputs.json')
    assert seal['context']['unit_predictions_sha256']==sha(directory/'unit_predictions.csv')
    assert seal['context']['run_id']==summary['model_run']
    ledger=read(directory/'source_ledger.json')
    assert sha(directory/'source_ledger.json')==inputs['source_ledger_sha256']
    for entry in ledger['records']:assert sha(entry['original'])==sha(directory/entry['copy'])==entry['sha256']
    allowed=read_csv(dest/'selection_inputs_without_annotations.csv');actual=read_csv(directory/'unit_predictions.csv')
    fields=('M','id','area','predicted_R1','predicted_R2','predicted_R4')
    assert all(set(x)==set(fields) for x in allowed)
    assert allowed==[{k:x[k] for k in fields} for x in actual]


def audit_policy_case(directory,case,by_id,row,prefix):
    m,r,b=case;dest=directory/'predicted_area_policy';name=f'M{m}_R{r}_B{round(b*100):03d}.json'
    choice=read(dest/'choices'/name);raw=read(dest/'raw'/name);solver=read(dest/'solver_raw'/name)
    evaluation=read(dest/(Path(name).stem+'_evaluation.json'))
    ids=sorted(by_id);aa=[by_id[i]['area'] for i in ids];pp=[by_id[i][f'predicted_R{r}'] for i in ids]
    A=math.fsum(aa);P=math.fsum(pp);T=math.fsum(u['target'] for u in by_id.values());B=b*A
    input_sha=payload_sha({'ids':ids,'area':aa,'predicted':pp,'budget':b})
    assert solver['selection_input_sha256']==choice['selection_input_sha256']==input_sha
    assert solver['case']==list(case) and solver['actual_solver_calls_for_case']==1 and solver['persisted_before_selector_audit']
    assert solver['solver_options_passed']==raw['options']
    assert raw['options']==read(PA/'RM_common_pipeline_v2_protocol.json')['solver_options']
    assert raw['n_variables']==len(ids) and raw['original_unit_feasibility_tolerance_m2']==0.
    for k,v in [('total_area_m2',A),('total_predicted_m2',P),('budget_area_m2',B),('budget_fraction',b)]:close(raw[k],v,1e-12)
    backend=solver['result'];x=backend.get('x');binary=False;integrality=None;candidates=[];candidate_a=candidate_p=violation=None
    if x is not None and all(number(v) for v in x):
        assert len(x)==len(ids)
        integrality=max(abs(v-round(v)) for v in x)
        binary=integrality<=1e-6 and all(-1e-6<=v<=1+1e-6 for v in x)
        if binary:
            candidates=[i for i,v in zip(ids,x) if bool(round(v))]
            candidate_a,_,candidate_p,_=aggregate(candidates,by_id,r);violation=max(0.,candidate_a-B)
    assert candidates==raw['solver_candidate_ids']
    for expected,key in [(integrality,'candidate_integrality_error'),(candidate_a,'candidate_area_m2'),(candidate_p,'candidate_predicted_m2'),(violation,'candidate_budget_violation_m2')]:same_optional(expected,raw[key])
    feasible=bool(binary and violation==0.)
    assert raw['candidate_feasible_in_original_units']==feasible
    assert raw['selected_ids']==(candidates if feasible else [])
    raw_a,_,raw_p,_=aggregate(raw['selected_ids'],by_id,r)
    same_optional(raw_a,raw['selected_area_m2']);same_optional(raw_p,raw['predicted_primal_m2'])
    assert int(backend['status'])==raw['solver_status'] and str(backend['message'])==raw['solver_message']
    for backend_key,raw_key in [('fun','raw_minimization_fun'),('mip_dual_bound','raw_minimization_dual'),('mip_gap','raw_solver_gap'),('mip_node_count','mip_node_count')]:
        val=backend.get(backend_key);same_optional(float(val) if number(val) else None,raw[raw_key])
    dual=backend.get('mip_dual_bound');raw_upper=-dual*(P if P>0 else 1.) if number(dual) else None
    same_optional(raw_upper,raw['raw_solver_dual_upper_m2'])
    prior_upper=min(P,raw_upper) if raw_upper is not None else P
    prior_clamp=max(0.,raw_p-prior_upper);assert prior_clamp<=max(1e-10,1e-10*P)
    prior_upper=max(raw_p,prior_upper);prior_gap=(prior_upper-raw_p)/max(abs(raw_p),1e-30) if prior_upper else 0.
    close(prior_upper,raw['predicted_dual_upper_m2'],1e-12);close(prior_gap,raw['predicted_relative_gap'],1e-12)
    raw_proven=raw['solver_status']==0 and feasible and prior_gap<=1e-8+1e-10
    assert raw['optimality_proven_within_tolerance']==raw_proven
    assert (choice['M'],choice['R'],choice['budget'])==case and choice['raw_sha256']==sha(dest/'raw'/name)
    assert choice['raw_origin']=='new_predeclared_case_solve' and choice['fallback_used']==(not feasible)
    for key,value in [('total_area_m2',A),('total_predicted_m2',P),('budget_area_m2',B),('budget_fraction',b)]:close(choice[key],value,1e-12)
    caudit=choice['raw_candidate_audit']
    assert caudit['candidate_ids']==candidates and caudit['ids_valid']
    same_optional(caudit['recorded_integrality_error'],integrality)
    assert caudit['recorded_binary_tolerance']==1e-6
    assert caudit['binary_audit_passed']==(integrality is not None and 0<=integrality<=1e-6)
    assert caudit['summaries_finite']==(candidate_a is not None and candidate_p is not None)
    ca,_,cp,_=aggregate(candidates,by_id,r)
    close(caudit['area_m2_recomputed'],ca,1e-12);close(caudit['predicted_m2_recomputed'],cp,1e-12)
    close(caudit['budget_violation_m2_recomputed'],max(0.,ca-B),1e-12)
    assert caudit['recorded_feasible_flag']==caudit['accepted_for_passthrough']==feasible
    assert caudit['raw_selected_ids']==raw['selected_ids'] and caudit['raw_solver_status']==raw['solver_status']
    assert caudit['raw_optimality_proven']==raw_proven
    for ck,rk in [('raw_solver_gap','raw_solver_gap'),('raw_reported_relative_gap','predicted_relative_gap'),('raw_solver_upper_m2','raw_solver_dual_upper_m2')]:same_optional(caudit[ck],raw[rk])
    final_ids=choice['selected_ids']
    if feasible:
        assert final_ids==raw['selected_ids'] and not choice['fallback_reasons'] and choice['prefix_tail_trim_count']==0
    else:
        ordered=sorted(ids,key=lambda i:(-by_id[i][f'predicted_R{r}']/by_id[i]['area'],i))
        cum=0.;before=[]
        for k,i in enumerate(ordered):
            cum+=by_id[i]['area']
            if k==len(ordered)-1:cum=A
            if cum>B:break
            before.append(i)
        expected=before.copy()
        while math.fsum(by_id[i]['area'] for i in expected)>B:expected.pop()
        assert final_ids==expected and choice['prefix_before_tail_trim_ids']==before
        assert choice['prefix_tail_trim_count']==len(before)-len(expected)
        assert choice['status']=='feasible_fallback_not_proven' and not choice['optimality_proven']
    a,t,p,zero=aggregate(final_ids,by_id,r);assert a<=B and choice['original_area_tolerance_m2']==0.
    assert choice['strict_original_unit_feasible']
    close(a,choice['selected_area_m2'],1e-12);close(p,choice['predicted_primal_m2'],1e-12)
    anomaly=None;clamp=0.;tol=max(1e-10,1e-10*P)
    if raw_upper is None:upper=P;anomaly='missing_or_nonfinite_raw_upper'
    elif p-raw_upper>tol:upper=P;anomaly='raw_upper_below_final_primal'
    else:
        upper=min(P,raw_upper)
        if upper<p:clamp=p-upper;upper=p
    gap=(upper-p)/max(abs(p),1e-30) if upper else 0.
    bound=choice['recomputed_bound'];assert bound['bound_anomaly']==anomaly
    for key,value in [('upper_m2',upper),('primal_m2',p),('absolute_gap_m2',upper-p),('relative_gap',gap),('roundoff_clamp_m2',clamp),('bound_inversion_tolerance_m2',tol)]:close(bound[key],value,1e-12)
    proven=bool(feasible and raw_proven);assert choice['optimality_proven']==proven
    assert evaluation['prefix_selected_ids']==prefix['ids']
    for label,vals in [('prefix',prefix),('pmilp',{'ids':final_ids,'area':a,'target':t,'predicted':p,'zero_units_area':zero,'capture':t/T})]:
        fields={'selected_count':len(vals['ids']),'actual_area_m2':vals['area'],'actual_area_fraction':vals['area']/A,'captured_target_m2':vals['target'],'capture_fraction':vals['capture'],'missed_target_m2':T-vals['target'],'zero_annotation_area_m2':vals['area']-vals['target'],'selected_zero_target_unit_area_m2':vals['zero_units_area'],'predicted_objective_m2':vals['predicted']}
        for key,value in fields.items():close(float(row[label+'_'+key]),value);close(float(evaluation['metrics'][label+'_'+key]),value)
    close(float(row['delta_capture_pp']),100*(t-prefix['target'])/T)
    close(float(row['delta_predicted_objective_m2']),p-prefix['predicted']);close(float(row['unused_budget_m2']),B-a)
    close(float(row['predicted_relative_gap']),gap,1e-12);close(float(row['predicted_upper_m2']),upper,1e-12)
    assert (row['optimality_proven']=='True')==proven==evaluation['metrics']['optimality_proven']
    assert (row['fallback_used']=='True')==choice['fallback_used']==evaluation['metrics']['fallback_used']
    assert row['final_status']==choice['status'] and row['fallback_reasons']==';'.join(choice['fallback_reasons'])
    assert int(row['prefix_tail_trim_count'])==choice['prefix_tail_trim_count']
    assert (row['raw_candidate_feasible']=='True')==feasible
    close(float(row['raw_candidate_budget_violation_m2']),caudit['budget_violation_m2_recomputed'],1e-12)
    assert int(row['raw_solver_status'])==raw['solver_status']
    if raw['raw_solver_gap'] is None:assert row['raw_solver_gap']==''
    else:close(float(row['raw_solver_gap']),raw['raw_solver_gap'],1e-12)
    assert row['choice_sha256']==sha(dest/'choices'/name)
    values={'ids':final_ids,'area':a,'target':t,'predicted':p,'capture':t/T,'proven':proven,'relative_gap':gap,'upper_P':upper,'solver_status':raw['solver_status'],'fallback':not feasible}
    record={'method':'predicted_area_milp_with_protocol_prefix_fallback_v1','M':m,'R':r,'budget':b,'selected_ids':final_ids,'A':a,'T':t,'P':p,'recorded_upper_P':upper,'recorded_gap':gap,'optimality_within_recorded_tolerance':proven,'fallback':not feasible,'three_layer_raw_checked':True,'all_checks_passed':True}
    return values,record


def audit_one(label,directory,geometry,baseline):
    state=read(directory/'status.json');summary=read(directory/'summary.json');inputs=read(directory/'inputs.json')
    assert state['state']==summary['state']=='completed'
    assert not state['test_accessed'] and not state['R6_accessed'] and not summary['model_tuned']
    assert summary['model_run']==label
    maps=read(Path(inputs['maps_directory'])/'metadata.json')
    assert maps['run_id']==label and maps['checkpoint_epoch']==summary['checkpoint_epoch']
    assert maps['checkpoint_sha256']==summary['checkpoint_sha256']
    audit_seal(directory)
    for key in ('native_valid_pixels','native_target_pixels','origin_utm_m','crs'):
        assert summary['domain'][key]==geometry[key]
    for key in ('area_m2','target_m2'):
        close(summary['domain'][key],geometry[key])
    assert geometry['native_valid_pixels']==91061049 and geometry['native_target_pixels']==885226
    assert inputs['no_inference'] and inputs['no_test_or_R6']
    assert inputs['ownership']['source_clipped_native_rectangles_pairwise_disjoint']
    for name,digest in state['files_sha256'].items():
        assert sha(directory/name)==digest, name
    for name,digest in inputs['file_sha256'].items():
        assert sha(name)==digest,name
    assert summary['source_sha256']==inputs['file_sha256']
    integration=read(directory/'integration_checks.json')
    assert integration['max_window_probability_relative_error']<=1e-9
    units=get_units(directory/'unit_predictions.csv')
    assert [len(units[m]) for m in (1,2,4)]==[1614,443,126]
    curve_rows=read_csv(directory/'decision_curves.csv')
    fixed_rows=read_csv(directory/'fixed_budgets.csv')
    fixed={(int(r['M']),int(r['R']),float(r['budget'])):r for r in fixed_rows if r['method']=='model_prefix'}
    assert len(fixed)==45 and set(fixed)==COMBINATIONS
    policies=read_csv(directory/'predicted_area_policy'/'all_45_deltas.csv')
    policies={(int(r['M']),int(r['R']),float(r['budget'])):r for r in policies}
    assert len(policies)==45 and set(policies)==COMBINATIONS
    set_audits=[];case_values={};label_certificates=0;curve_points=0
    total_area=math.fsum(u['area'] for u in units[1]);total_target=math.fsum(u['target'] for u in units[1])
    close(total_area,geometry['area_m2']);close(total_target,geometry['target_m2'])
    for m,group in units.items():
        by_id={u['id']:u for u in group};assert len(by_id)==len(group)
        assert len({(u['ix'],u['iy']) for u in group})==len(group)
        reference={u['id']:u for u in baseline[m]}
        assert set(by_id)==set(reference)
        for ident,u in by_id.items():
            assert (u['ix'],u['iy'],u['area'],u['target'])==(reference[ident]['ix'],reference[ident]['iy'],reference[ident]['area'],reference[ident]['target'])
        A=math.fsum(u['area'] for u in group);T=math.fsum(u['target'] for u in group)
        close(A,total_area);close(T,total_target)
        if m>1:
            children=collections.defaultdict(list)
            for u in units[1]:children[(u['ix']//m,u['iy']//m)].append(u)
            assert set(children)=={(u['ix'],u['iy']) for u in group}
            for u in group:
                for col in ('area','target','predicted_R1','predicted_R2','predicted_R4'):
                    close(math.fsum(c[col] for c in children[(u['ix'],u['iy'])]),u[col])
        for R in (1,2,4,'label'):
            method='label_prefix' if R=='label' else 'model_prefix'
            key='q_true' if R=='label' else f'q_R{R}'
            ordered=sorted(group,key=lambda u:(-u[key],u['id']))
            area=[u['area'] for u in ordered];target=[u['target'] for u in ordered]
            # Accurately recompute all complete prefixes, not only selected budgets.
            ca=[math.fsum(area[:k]) for k in range(len(ordered)+1)]
            ct=[math.fsum(target[:k]) for k in range(len(ordered)+1)]
            points=[r for r in curve_rows if int(r['M'])==m and r['method']==method and (R=='label' or int(r['R'])==R)]
            points.sort(key=lambda r:int(r['k']))
            assert len(points)==len(ordered)+1
            curve_points+=len(points)
            for k,row in enumerate(points):
                assert int(row['k'])==k
                close(float(row['actual_area']),ca[k]);close(float(row['captured_target']),ct[k])
                close(float(row['actual_area_fraction']),ca[k]/A);close(float(row['capture_fraction']),ct[k]/T)
                close(float(row['missed_target']),T-ct[k]);close(float(row['zero_annotation_area']),ca[k]-ct[k])
                if k:assert row['last_added_id']==ordered[k-1]['id']
            prefix_summary=next(x for x in summary['results']['by_M'][str(m)]['prefix_results'] if x['method']==method and (R=='label' or x['R']==R))
            for endpoint,name in ((1.,'auc_0_1'),(.5,'auc_0_0p5_normalized')):
                integral=math.fsum(max(0.,min(endpoint,ca[k+1]/A)-min(endpoint,ca[k]/A))*ct[k]/T for k in range(len(ordered)))/endpoint
                close(integral,prefix_summary[name])
            for b in BUDGETS:
                k=len(ordered) if b==1 else max(i for i,a in enumerate(ca) if a<=b*A)
                selected=[u['id'] for u in ordered[:k]]
                if R=='label':
                    row=next(r for r in fixed_rows if int(r['M'])==m and r['method']==method and float(r['budget'])==b)
                else:row=fixed[(m,R,b)]
                assert int(row['k'])==k
                close(float(row['actual_area']),ca[k]);close(float(row['captured_target']),ct[k])
                close(float(row['capture_fraction']),ct[k]/T);close(float(row['unused_budget_area']),b*A-ca[k])
                assert ca[k]<=b*A
                if R!='label':
                    a,t,p,zero=aggregate(selected,by_id,R)
                    case_values[(m,R,b)]={'prefix':{'ids':selected,'area':a,'target':t,'predicted':p,'zero_units_area':zero,'capture':t/T}}
                    set_audits.append({'model':label,'method':'prefix','M':m,'R':R,'budget':b,
                                       'selected_ids':selected,'A':a,'T':t,'P':p,'all_checks_passed':True})
        for optimum in summary['results']['by_M'][str(m)]['label_milp']:
            ids=optimum['selected_ids'];assert len(ids)==len(set(ids)) and set(ids)<=by_id.keys()
            a=math.fsum(by_id[i]['area'] for i in ids);t=math.fsum(by_id[i]['target'] for i in ids)
            assert a<=optimum['budget']*A
            close(a,optimum['selected_area_feasibility_audit']);close(t,optimum['primal_target'])
            assert optimum['dual_target_bound']>=t-1e-9
            if optimum['optimality_proven']:
                assert optimum['solver_status']==0
                close(t,optimum['dual_target_bound'],1e-8)
            label_certificates+=1
        for R in (1,2,4):
            P=math.fsum(u[f'predicted_R{R}'] for u in group)
            close(P,summary['results']['by_M'][str(m)]['total_predicted'][str(R)])
            for b in BUDGETS:
                values,record=audit_policy_case(directory,(m,R,b),by_id,policies[(m,R,b)],case_values[(m,R,b)]['prefix'])
                case_values[(m,R,b)]['pmilp']=values
                set_audits.append({'model':label,**record})
    deltas=[100*(v['pmilp']['capture']-v['prefix']['capture']) for v in case_values.values()]
    policy_counts={'combinations':45,'optimal_within_tolerance':sum(v['pmilp']['proven'] for v in case_values.values()),
                   'fallback_count':sum(v['pmilp']['fallback'] for v in case_values.values()),
                   'improved':sum(d>1e-8 for d in deltas),'worsened':sum(d< -1e-8 for d in deltas),
                   'unchanged':sum(abs(d)<=1e-8 for d in deltas)}
    for k,v in policy_counts.items():assert v==state['policy_summary'][k]==summary['predicted_area_policy'][k]
    return {'label':label,'directory':str(directory),'model_run':summary['model_run'],
            'checkpoint_epoch':summary['checkpoint_epoch'],'status_sha256':sha(directory/'status.json'),
            'area_m2':total_area,'target_m2':total_target,'units_by_M':{m:len(u) for m,u in units.items()},
            'prefix_curve_points_checked':curve_points,'model_prefix_budgets_checked':45,'label_prefix_budgets_checked':15,
            'P_MILP_selected_sets_checked':45,'label_reference_selected_sets_checked':label_certificates,
            'nested_area_target_prediction_checks':[2,4],'policy_counts':policy_counts,
            'all_recorded_hashes_verified':True,'arithmetic_passed':True},case_values,set_audits


def main():
    import argparse,shutil
    parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path,required=True);args=parser.parse_args()
    directory=args.result.resolve();assert directory.parent==PA.resolve() and directory.name.startswith('cmv2_')
    geometry=read(PA/'geometry_20260916_v1/metadata.json')
    baseline=get_units(PA/'RM_evaluation_20260916_020003_962711/unit_predictions.csv')
    run_id=read(directory/'summary.json')['model_run']
    checked,values,sets=audit_one(run_id,directory,geometry,baseline)
    assert len(values)==45 and len(sets)==90
    rows=[]
    for (m,r,b),v in sorted(values.items()):
        p=v['prefix'];q=v['pmilp']
        rows.append({'run_id':run_id,'M':m,'R':r,'budget':b,
            'prefix_area':p['area'],'prefix_target':p['target'],'prefix_predicted':p['predicted'],'prefix_capture':p['capture'],
            'policy_area':q['area'],'policy_target':q['target'],'policy_predicted':q['predicted'],'policy_capture':q['capture'],
            'delta_capture_pp':100*(q['capture']-p['capture']),'fallback':q['fallback'],'upper':q['upper_P'],
            'relative_gap':q['relative_gap'],'proven_within_recorded_tolerance':q['proven']})
    report={'passed':True,'scope':'Independent standard-library recomputation; no production evaluator imports, models, pixels or solvers',
        'created_at':datetime.datetime.now().astimezone().isoformat(),'run':checked,'all_45_solver_raw_selector_raw_choice_links_checked':True,
        'selection_seal_sha256':sha(directory/'predicted_area_policy/selection_seal.json'),
        'fixed_D_A_T_and_nested_M_verified':True,'full_prefix_curves_and_step_AUC_verified':True,
        'model_prefix_sets':45,'final_policy_sets':45,'solver_returns':45,'selector_returns':45,'final_choices':45,
        'original_budget_tolerance_m2':0.0,'all_45_cases_included':True,'source_sha256':sha(__file__),
        'no_MILP_solver_invoked':True,'no_model_inference':True,'gpu_used':False,'test_accessed':False,'R6_accessed':False,
        'limits':['Stored solver primal/dual values and their transformations were independently recomputed; no separate optimization proof is claimed.',
            'Recorded solver execution count is supported by sealed returns and reviewed control flow, not a separate operating-system trace.',
            'All outcomes are descriptive one-scene seed17 validation results; neither statistical significance nor field efficacy is implied.']}
    out=PA/('cmv2_independent_audit_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    dump(out/'audit.json',report);dump(out/'selection_sets.json',sets);write_csv(out/'cases_45.csv',rows)
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    counts=checked['policy_counts']
    (out/'review.md').write_text('# Independent common management v2 result audit\n\n'+
        '**Passed all arithmetic, identity, prefix, nested geometry and recorded-bound checks.**\n\n'+
        f"Run: {run_id}. Original selected epoch: {checked['checkpoint_epoch']}.\n\n"+
        f"All 45 original prefix budgets and 45 final selections were independently recomputed. All 45 solver returns, selector audits and final choices were linked; all source/product/seal hashes passed. Final areas obey the original budget with zero tolerance. Fallbacks: {counts['fallback_count']}; recorded optimality within tolerance: {counts['optimal_within_tolerance']}/45.\n\n"+
        f"Compared with the original prefix, annotated capture improved/decreased/was unchanged in {counts['improved']}/{counts['worsened']}/{counts['unchanged']} cases. The complete 45-row table and all 90 selected ID sets are retained.\n\n"+
        'The shared native domain, exact A/T, 1-to-2/4 m nesting, full prefix staircases and step AUCs were checked against the original baseline. No solver, model, GPU or test/R6 access occurred. Stored bounds are audited for numerical consistency, not independently re-solved proofs.\n','utf8')
    hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()};dump(out/'artifact_hashes.json',hashes)
    print(json.dumps({'output':str(out),'run_id':run_id,'passed':True,'counts':counts,'audit_sha256':hashes['audit.json']},ensure_ascii=True))


if __name__=='__main__':main()

