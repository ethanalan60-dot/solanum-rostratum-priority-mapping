"""Independent stdlib arithmetic audit of two fixed common R18 R x M results.

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
DIRECTORIES={
    'hard':PA/'common_RM_dev_common_v2_lr1e4_resnet18_hard_seg_L256_s17_20260916_031336_012442',
    'soft':PA/'common_RM_dev_common_v2_lr1e4_resnet18_soft_seg_L256_s17_20260916_031402_620370',
}
OUTPUT=PA/'common_rm_review_v1'
BUDGETS=(.1,.2,.3,.5,1.)
COMBINATIONS={(m,r,b) for m in (1,2,4) for r in (1,2,4) for b in BUDGETS}


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


def audit_one(label,directory,geometry,baseline):
    state=read(directory/'status.json');summary=read(directory/'summary.json');inputs=read(directory/'inputs.json')
    assert state['state']==summary['state']=='completed'
    assert not state['test_accessed'] and not state['R6_accessed'] and not summary['model_tuned']
    assert summary['checkpoint_epoch']==(22 if label=='hard' else 23)
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
            assert optimum['solver_status']==0 and optimum['optimality_proven']
            close(t,optimum['dual_target_bound'],1e-8);label_certificates+=1
        for R in (1,2,4):
            P=math.fsum(u[f'predicted_R{R}'] for u in group)
            close(P,summary['results']['by_M'][str(m)]['total_predicted'][str(R)])
            for b in BUDGETS:
                row=policies[(m,R,b)]
                solution_path=directory/'predicted_area_policy'/row['solution_file']
                solution=read(solution_path);evaluation=read(solution_path.with_name(solution_path.stem+'_evaluation.json'))
                assert (solution['M'],solution['R'],solution['budget'])==(m,R,b)
                assert solution['selected_ids']==sorted(solution['selected_ids'])
                ids=solution['selected_ids'];a,t,p,zero=aggregate(ids,by_id,R)
                prefix=case_values[(m,R,b)]['prefix']
                assert evaluation['prefix_selected_ids']==prefix['ids']
                assert a<=b*A and solution['candidate_feasible_in_original_units']
                assert solution['solver_candidate_ids']==ids and solution['candidate_integrality_error']<=1e-6
                assert solution['candidate_budget_violation_m2']==0 and solution['rejected_candidate_fallback'] is None
                close(a,solution['selected_area_m2']);close(a,solution['candidate_area_m2'])
                close(p,solution['predicted_primal_m2']);close(p,solution['candidate_predicted_m2'])
                close(P,solution['total_predicted_m2']);close(A,solution['total_area_m2']);close(b*A,solution['budget_area_m2'])
                raw=solution['raw_minimization_dual']
                upper=min(P,-raw*(P if P>0 else 1.)) if raw is not None else P
                clamp=max(0.,p-upper);assert clamp<=max(1e-10,1e-10*P)
                upper=max(p,upper);gap=(upper-p)/max(abs(p),1e-30) if upper else 0.
                close(upper,solution['predicted_dual_upper_m2']);close(gap,solution['predicted_relative_gap'])
                close(upper-p,solution['predicted_absolute_gap_m2']);close(clamp,solution['bound_roundoff_clamp_m2'])
                proven=bool(solution['solver_status']==0 and gap<=solution['options']['mip_rel_gap']+1e-10)
                assert solution['options']['mip_rel_gap']==1e-8 and proven==solution['optimality_proven_within_tolerance']
                assert solution['status']==('optimal_within_recorded_gap' if proven else 'not_proven_optimal')
                assert solution['no_secondary_objective']
                for prefix_name,values in (('prefix',prefix),('pmilp',{'ids':ids,'area':a,'target':t,'predicted':p,'zero_units_area':zero,'capture':t/T})):
                    expected={'selected_count':len(values['ids']),'actual_area_m2':values['area'],
                              'actual_area_fraction':values['area']/A,'captured_target_m2':values['target'],
                              'capture_fraction':values['capture'],'missed_target_m2':T-values['target'],
                              'zero_annotation_area_m2':values['area']-values['target'],
                              'selected_zero_target_unit_area_m2':values['zero_units_area'],
                              'predicted_objective_m2':values['predicted']}
                    for name,value in expected.items():
                        close(float(row[prefix_name+'_'+name]),value)
                        close(float(evaluation['metrics'][prefix_name+'_'+name]),value)
                close(float(row['delta_capture_pp']),100*(t-prefix['target'])/T)
                close(float(row['delta_predicted_objective_m2']),p-prefix['predicted'])
                close(float(row['unused_budget_m2']),b*A-a)
                close(float(row['predicted_relative_gap']),gap)
                assert (row['optimality_proven']=='True')==proven==evaluation['metrics']['optimality_proven']
                if proven:assert p>=prefix['predicted']-max(1e-10,1e-8*P)
                added=set(ids)-set(prefix['ids']);removed=set(prefix['ids'])-set(ids)
                close(t-prefix['target'],math.fsum(by_id[i]['target'] for i in added)-math.fsum(by_id[i]['target'] for i in removed))
                case_values[(m,R,b)]['pmilp']={'ids':ids,'area':a,'target':t,'predicted':p,'capture':t/T,
                    'proven':proven,'relative_gap':gap,'upper_P':upper,'solver_status':solution['solver_status'],
                    'added_target':math.fsum(by_id[i]['target'] for i in added),'removed_target':math.fsum(by_id[i]['target'] for i in removed)}
                set_audits.append({'model':label,'method':'predicted_area_milp','M':m,'R':R,'budget':b,
                    'selected_ids':ids,'A':a,'T':t,'P':p,'recorded_upper_P':upper,'recorded_gap':gap,
                    'optimality_within_recorded_tolerance':proven,'all_checks_passed':True})
    deltas=[100*(v['pmilp']['capture']-v['prefix']['capture']) for v in case_values.values()]
    policy_counts={'combinations':45,'optimal_within_tolerance':sum(v['pmilp']['proven'] for v in case_values.values()),
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


def run():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    geometry=read(PA/'geometry_20260916_v1'/'metadata.json')
    baseline=get_units(PA/'RM_evaluation_20260916_020003_962711'/'unit_predictions.csv')
    reports=[];all_values={};all_sets=[]
    for label,directory in DIRECTORIES.items():
        report,values,sets=audit_one(label,directory,geometry,baseline)
        reports.append(report);all_values[label]=values;all_sets.extend(sets)
    pairs=[]
    for key in sorted(COMBINATIONS):
        m,r,b=key;hard=all_values['hard'][key];soft=all_values['soft'][key]
        row={'M':m,'R':r,'budget':b,'total_valid_area_m2':geometry['area_m2'],'total_target_m2':geometry['target_m2']}
        for policy in ('prefix','pmilp'):
            for label,values in (('hard',hard[policy]),('soft',soft[policy])):
                for name in ('area','target','predicted','capture'):
                    row[f'{label}_{policy}_{name}']=values[name]
                row[f'{label}_{policy}_selected_count']=len(values['ids'])
                if policy=='pmilp':
                    row[f'{label}_pmilp_optimal_within_tolerance']=values['proven']
                    row[f'{label}_pmilp_relative_gap']=values['relative_gap']
                    row[f'{label}_pmilp_solver_status']=values['solver_status']
            row[f'soft_minus_hard_{policy}_capture_pp']=100*(soft[policy]['capture']-hard[policy]['capture'])
            row[f'soft_minus_hard_{policy}_area_m2']=soft[policy]['area']-hard[policy]['area']
            row[f'soft_minus_hard_{policy}_target_m2']=soft[policy]['target']-hard[policy]['target']
            # Target/missed/non-target differences remain in raw annotated units.
            row[f'soft_minus_hard_{policy}_missed_target_m2']=hard[policy]['target']-soft[policy]['target']
            row[f'soft_minus_hard_{policy}_zero_annotation_area_m2']=(soft[policy]['area']-soft[policy]['target'])-(hard[policy]['area']-hard[policy]['target'])
        for label,values in (('hard',hard),('soft',soft)):
            row[f'{label}_pmilp_minus_prefix_capture_pp']=100*(values['pmilp']['capture']-values['prefix']['capture'])
        pairs.append(row)
    assert len(pairs)==45 and len(all_sets)==180
    counts={}
    for policy in ('prefix','pmilp'):
        values=[r[f'soft_minus_hard_{policy}_capture_pp'] for r in pairs]
        counts[policy]={'soft_higher':sum(d>1e-8 for d in values),'soft_lower':sum(d< -1e-8 for d in values),
                        'tied':sum(abs(d)<=1e-8 for d in values),'minimum_delta_pp':min(values),'maximum_delta_pp':max(values)}
    report={'passed':True,'reviewer':'tile_audit','created_at':datetime.datetime.now().astimezone().isoformat(),
            'scope':'Independent Python standard-library arithmetic and recorded-certificate audit; no production evaluator imports',
            'runs':reports,'paired_cases':45,'selection_sets_recomputed':180,'paired_counts':counts,
            'fixed_D_equal_to_original':True,'fixed_A_T_equal_across_models_R_M':True,
            'M4_budget10pct_all_R':[r for r in pairs if r['M']==4 and r['budget']==.1],
            'no_MILP_solver_invoked':True,'no_model_inference':True,'gpu_used':False,'test_accessed':False,'R6_accessed':False,
            'limits':'Supplied solver primal/dual arithmetic is verified, not an independent solver proof. Not all P-MILP cases meet the stated gap. Paired differences are descriptive within one validation domain, not independent field replications.',
            'source_sha256':sha(__file__)}
    OUTPUT.mkdir()
    write_csv(OUTPUT/'paired_45.csv',pairs)
    dump(OUTPUT/'selection_sets.json',all_sets)
    dump(OUTPUT/'audit.json',report)
    lines=['# Independent audit of common ResNet18 R×M results','',
           '**All arithmetic, identity, prefix, and recorded-bound checks pass.** Both original result directories remain unchanged.','',
           'The audit uses only the Python standard library. It does not import the production evaluators, invoke any MILP solver, run a model, read image/cache pixels, or access test/R6.','',
           '## Complete checks','',
           '- Both domains reproduce the original unit IDs, integer coordinates, valid areas and annotated target areas exactly. Total valid area is 1457.4861587858427 m²; annotated target area is 14.168571381451857 m². M1/M2/M4 contain 1614/443/126 units. Native D contains 91,061,049 valid and 885,226 annotated target pixels.',
           '- All original complete-prefix curves and their [0,1] and normalized [0,0.5] step AUCs reproduce; all 90 model-prefix budgets and 30 label-prefix budgets pass.',
           '- All 90 P-MILP selected ID sets and their A/T/P totals are recomputed. Original prefix ID order reproduces exactly. Every set respects the original area budget. Label-reference 30 selected sets and bounds are checked as well.',
           '- Integer M1→M2/M4 sums conserve each area, target and predicted-area column. All recorded input/product hashes are checked.','',
           '## Recorded optimality and policy outcomes','',
           '| Model | P-MILP within recorded tolerance | Higher target capture vs prefix | Lower | Tied |',
           '|---|---:|---:|---:|---:|']
    for item in reports:
        c=item['policy_counts'];lines.append(f"| {item['label']} | {c['optimal_within_tolerance']}/45 | {c['improved']} | {c['worsened']} | {c['unchanged']} |")
    lines+=['','Ten P-MILP solutions per model do not meet the recorded optimality criterion; retain their solver status and gap, and call them feasible solutions. The audit recomputes bounds/gaps from saved solver evidence; it is not a second solver certificate. No secondary minimum-area objective was solved. Optimizing predicted area need not improve annotated target capture.','',
            '## All45 paired soft−hard differences','',
            'The paired_45.csv file includes every fixed M×R×budget combination for both prefix and P-MILP policies, with areas, target capture, predicted objectives, raw target differences, solver gaps and flags. No budget, R, M, or unfavorable point is omitted.','',
            '| Policy | Soft higher | Soft lower | Tied | Minimum delta (pp) | Maximum delta (pp) |',
            '|---|---:|---:|---:|---:|---:|']
    for p,c in counts.items():lines.append(f"| {p} | {c['soft_higher']} | {c['soft_lower']} | {c['tied']} | {c['minimum_delta_pp']:.9f} | {c['maximum_delta_pp']:.9f} |")
    lines+=['','### M4, 10% budget: every R','', '| R | Prefix soft−hard (pp) | P-MILP soft−hard (pp) |','|---|---:|---:|']
    for row in report['M4_budget10pct_all_R']:
        lines.append(f"| {row['R']} | {row['soft_minus_hard_prefix_capture_pp']:.9f} | {row['soft_minus_hard_pmilp_capture_pp']:.9f} |")
    lines+=['','These are same-domain descriptive development comparisons. Unit and budget counts are not independent field or flight counts.','',
            'Artifacts: audit.json contains machine-readable checks; selection_sets.json contains all180 audited ID sets and recomputed A/T/P; paired_45.csv contains the complete paired table.']
    (OUTPUT/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'output':str(OUTPUT),'paired_counts':counts,'policy_counts':[r['policy_counts'] for r in reports]},ensure_ascii=False))


if __name__=='__main__':run()
