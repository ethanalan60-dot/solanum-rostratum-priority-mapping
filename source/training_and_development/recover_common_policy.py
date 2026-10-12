"""Bounded numerical recovery:42 original results,1 saved replay,2 new cases.

No frozen source edits, model inference, geometry computation, test or R6.
Selection accepts only IDs, valid areas, predicted areas and a budget fraction.
"""
from pathlib import Path
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,datetime,json,math,shutil,sys,time,traceback
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
import diagnose_predicted_knapsack as policy
import evaluate_common_management as wrapper

ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture'
PROTOCOL=PA/'RM_numerical_recovery_protocol_v1.json'
PROTOCOL_SHA='97b12d1178dcba1357860c4b2517a041ca3aaf456eb8a575c2c931db5579cfdd'
PARENT=PA/'common_RM_dev_common_v2_lr1e4_convnextv2_tiny_hard_seg_L256_s17_20260916_033131_837962'
PARTIAL=PA/'convnext_hard_lr1e4_appendix_20260916_032658_994435'/'partial_rm_independent_audit.json'
REPLAY_DIR=PA/'convnext_failed_policy_diagnostic_20260916_033601_083035'
REPLAY=REPLAY_DIR/'raw_selector_return.json'
REPLAY_SHA='dcbf0df0183e9e9576984000603a0d7bca094639562d5934082b0781b2565e05'
MS=(1,2,4);RS=(1,2,4);BS=(.1,.2,.3,.5,1.)
CASES=[(m,r,b) for m in MS for r in RS for b in BS]
NEW_CASES=((4,4,.5),(4,4,1.))
FIELDS=('M','id','area','predicted_R1','predicted_R2','predicted_R4')
sha=policy.sha;write_json=policy.write_json;write_csv=policy.write_csv;stamp=policy.stamp
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def case_name(case):return f'M{case[0]}_R{case[1]}_B{int(case[2]*100):03d}.json'
def number(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-10),(a,b)

def validate_arrays(ids,area,predicted,budget):
    ids=list(ids);area=np.asarray(area,dtype=np.float64);predicted=np.asarray(predicted,dtype=np.float64)
    assert len(ids)>0 and len(ids)==len(area)==len(predicted) and area.ndim==predicted.ndim==1
    assert len(set(ids))==len(ids) and all(isinstance(i,str) and i.isascii() for i in ids)
    assert np.isfinite(area).all() and np.all(area>0)
    assert np.isfinite(predicted).all() and np.all(predicted>=0) and np.all(predicted<=area+1e-9)
    assert number(budget) and 0<=budget<=1
    return ids,area,predicted,{i:j for j,i in enumerate(ids)}

def bounds_for_final(raw_upper,primal,total_predicted):
    """A safe total-P bound replaces missing/nonfinite/contradictory raw bounds."""
    tolerance=max(1e-10,1e-10*total_predicted)
    anomaly=None;clamp=0.
    if not number(raw_upper):
        anomaly='missing_or_nonfinite_raw_upper';upper=total_predicted
    elif primal-float(raw_upper)>tolerance:
        anomaly='raw_upper_below_final_primal';upper=total_predicted
    else:
        upper=min(total_predicted,float(raw_upper))
        if upper<primal:
            clamp=primal-upper;upper=primal
    assert upper>=primal and math.isfinite(upper)
    absolute=upper-primal
    gap=absolute/max(abs(primal),1e-30) if upper else 0.
    return {'upper_m2':upper,'primal_m2':primal,'absolute_gap_m2':absolute,'relative_gap':gap,
            'bound_anomaly':anomaly,'bound_source':'sum_predicted_safe_upper' if anomaly else 'min(raw_solver_upper,sum_predicted)',
            'raw_upper_m2':raw_upper if number(raw_upper) else None,'roundoff_clamp_m2':clamp,
            'bound_inversion_tolerance_m2':tolerance}

def recover_choice(ids,area,predicted,budget,raw):
    """Pure numerical recovery. It receives no annotation/reference argument."""
    ids,area,predicted,index=validate_arrays(ids,area,predicted,budget)
    total_area=math.fsum(area);total_predicted=math.fsum(predicted);limit=budget*total_area
    candidate=raw.get('solver_candidate_ids')
    reasons=[];candidate_area=candidate_predicted=violation=None
    if candidate is not None:
        # Malformed/unknown IDs indicate a provenance/programming problem, not
        # an eligible numerical solver failure. Never absorb it as fallback.
        assert isinstance(candidate,list) and all(isinstance(i,str) for i in candidate)
        assert len(set(candidate))==len(candidate) and set(candidate).issubset(index)
    ids_valid=isinstance(candidate,list)
    if not ids_valid:reasons.append('missing_candidate_ids')
    integrality=raw.get('candidate_integrality_error')
    binary_audit=number(integrality) and 0<=integrality<=1e-6
    if not binary_audit:reasons.append('missing_nonfinite_or_nonbinary_candidate_audit')
    summaries_finite=number(raw.get('candidate_area_m2')) and number(raw.get('candidate_predicted_m2'))
    if not summaries_finite:reasons.append('missing_or_nonfinite_candidate_summary')
    if ids_valid:
        candidate_area=math.fsum(area[index[i]] for i in candidate)
        candidate_predicted=math.fsum(predicted[index[i]] for i in candidate)
        violation=max(0.,candidate_area-limit)
        if summaries_finite:
            close(candidate_area,raw['candidate_area_m2']);close(candidate_predicted,raw['candidate_predicted_m2'])
        if violation>0:reasons.append('strict_original_budget_exceeded')
    if raw.get('candidate_feasible_in_original_units') is not True:
        reasons.append('raw_selector_rejected_candidate')
    valid=not reasons
    prefix_before=None;tail_trim=0
    if valid:
        # Do not replace an already-feasible candidate with a competing prefix.
        assert raw.get('selected_ids')==candidate
        final=list(raw['selected_ids'])
    else:
        prefix_before=policy.prefix_ids(ids,area,predicted,budget)
        assert len(set(prefix_before))==len(prefix_before) and set(prefix_before).issubset(index)
        final=list(prefix_before)
        while math.fsum(area[index[i]] for i in final)>limit:
            assert final
            final.pop();tail_trim+=1
        assert final==prefix_before[:len(prefix_before)-tail_trim]
    a=math.fsum(area[index[i]] for i in final);p=math.fsum(predicted[index[i]] for i in final)
    assert a<=limit and len(set(final))==len(final) and set(final).issubset(index)
    bound=bounds_for_final(raw.get('raw_solver_dual_upper_m2'),p,total_predicted)
    if valid:
        close(a,raw['selected_area_m2']);close(p,raw['predicted_primal_m2'])
        close(bound['upper_m2'],raw['predicted_dual_upper_m2']);close(bound['relative_gap'],raw['predicted_relative_gap'])
        assert bound['bound_anomaly'] is None,'Original feasible result has inconsistent bound audit'
        proven=bool(raw['optimality_proven_within_tolerance'])
        assert proven==bool(raw['solver_status']==0 and bound['relative_gap']<=policy.OPTIONS['mip_rel_gap']+1e-10)
    else:proven=False
    return {'policy_label':'predicted_area_milp_with_protocol_prefix_fallback_v1',
        'selected_ids':final,'selected_area_m2':a,'predicted_primal_m2':p,
        'budget_fraction':budget,'budget_area_m2':limit,'total_area_m2':total_area,'total_predicted_m2':total_predicted,
        'strict_original_unit_feasible':a<=limit,'original_area_tolerance_m2':0.,
        'fallback_used':not valid,'fallback_reasons':reasons,'prefix_before_tail_trim_ids':prefix_before,'prefix_tail_trim_count':tail_trim,
        'status':'feasible_fallback_not_proven' if not valid else raw['status'],
        'optimality_proven':proven,'recomputed_bound':bound,
        'raw_candidate_audit':{'candidate_ids':candidate,'ids_valid':ids_valid,'recorded_integrality_error':integrality,
            'recorded_binary_tolerance':1e-6,'binary_audit_passed':binary_audit,'summaries_finite':summaries_finite,
            'area_m2_recomputed':candidate_area,'predicted_m2_recomputed':candidate_predicted,'budget_violation_m2_recomputed':violation,
            'recorded_feasible_flag':raw.get('candidate_feasible_in_original_units'),'accepted_for_passthrough':valid,
            'raw_selected_ids':raw.get('selected_ids'),'raw_solver_status':raw.get('solver_status'),
            'raw_optimality_proven':raw.get('optimality_proven_within_tolerance'),'raw_solver_gap':raw.get('raw_solver_gap'),
            'raw_reported_relative_gap':raw.get('predicted_relative_gap'),'raw_solver_upper_m2':raw.get('raw_solver_dual_upper_m2')},
        'selection_input_sha256':policy.payload_sha({'ids':ids,'area':area.tolist(),'predicted':predicted.tolist(),'budget':budget})}

def toy_raw(ids,area,predicted,budget,chosen,*,candidate=None,feasible=True,upper=None):
    index={i:j for j,i in enumerate(ids)};candidate=chosen if candidate is None else candidate
    a=math.fsum(area[index[i]] for i in candidate);p=math.fsum(predicted[index[i]] for i in candidate)
    chosen_a=math.fsum(area[index[i]] for i in chosen);chosen_p=math.fsum(predicted[index[i]] for i in chosen)
    if upper is None:upper=chosen_p
    gap=(upper-chosen_p)/max(abs(chosen_p),1e-30) if upper else 0.
    proven=feasible and gap<=policy.OPTIONS['mip_rel_gap']+1e-10
    return {'solver_candidate_ids':list(candidate),'selected_ids':list(chosen),
        'candidate_area_m2':a,'candidate_predicted_m2':p,'candidate_integrality_error':0.,
        'candidate_feasible_in_original_units':feasible,'selected_area_m2':chosen_a,'predicted_primal_m2':chosen_p,
        'predicted_dual_upper_m2':upper,'predicted_relative_gap':gap,'raw_solver_dual_upper_m2':upper,
        'solver_status':0,'optimality_proven_within_tolerance':proven,
        'status':'optimal_within_recorded_gap' if proven else 'not_proven_optimal','raw_solver_gap':gap}

def self_checks():
    ids=['a','b','c'];a=np.array([1.,1.,1.]);p=np.array([.8,.6,.2]);b=.5
    valid=toy_raw(ids,a,p,b,['a'],upper=.8)
    one=recover_choice(ids,a,p,b,valid);assert one['selected_ids']==['a'] and not one['fallback_used']
    inferior_valid=toy_raw(ids,a,p,b,['b'],upper=.8)
    unchanged=recover_choice(ids,a,p,b,inferior_valid)
    assert unchanged['selected_ids']==['b'] and not unchanged['fallback_used']
    nonfinite=dict(valid,candidate_integrality_error=float('nan'),candidate_feasible_in_original_units=False)
    assert recover_choice(ids,a,p,b,nonfinite)['fallback_used']
    rejected=toy_raw(ids,a,p,b,[],candidate=['a','b'],feasible=False,upper=1.4)
    two=recover_choice(ids,a,p,b,rejected)
    assert two['selected_ids']==['a'] and two['fallback_used'] and not two['optimality_proven']
    missing=dict(rejected,solver_candidate_ids=[],candidate_area_m2=None,candidate_predicted_m2=None,candidate_integrality_error=None)
    three=recover_choice(ids,a,p,b,missing);assert three['fallback_used'] and three['selected_ids']==['a']
    nonbinary=dict(rejected,candidate_integrality_error=.25)
    assert recover_choice(ids,a,p,b,nonbinary)['fallback_used']
    legal_empty=toy_raw(ids,a,p,0.,[],upper=0.)
    assert not recover_choice(ids,a,p,0.,legal_empty)['fallback_used']
    rid=[f'a{i}' for i in range(10)];ra=np.full(10,.1);rp=np.full(10,.05)
    rounding=recover_choice(rid,ra,rp,.6,missing)
    assert len(rounding['prefix_before_tail_trim_ids'])==6 and rounding['prefix_tail_trim_count']==1 and len(rounding['selected_ids'])==5
    invalid_bound=recover_choice(ids,a,p,b,dict(rejected,raw_solver_dual_upper_m2=.1))
    assert invalid_bound['recomputed_bound']['upper_m2']==math.fsum(p) and invalid_bound['recomputed_bound']['bound_anomaly']
    finite_missing=recover_choice(ids,a,p,b,dict(rejected,raw_solver_dual_upper_m2=None))
    assert finite_missing['recomputed_bound']['upper_m2']==math.fsum(p)
    small_inversion=recover_choice(ids,a,p,b,dict(rejected,raw_solver_dual_upper_m2=.8-5e-11))
    assert small_inversion['recomputed_bound']['roundoff_clamp_m2']>0 and not small_inversion['optimality_proven']
    rows=[{'M':'1','id':i,'area':str(x),'predicted_R1':str(y),'predicted_R2':str(y),'predicted_R4':str(y),'target':t}
          for i,x,y,t in zip(ids,a,p,['0','0','1'])]
    altered=[dict(r,target='UNREADABLE_LABEL',q_true='FORBIDDEN') for r in rows]
    projected=policy.allowed_selection_rows(rows);changed=policy.allowed_selection_rows(altered)
    assert policy.payload_sha(projected)==policy.payload_sha(changed)
    recovered=[]
    for payload in (projected,changed):
        recovered.append(recover_choice([r['id'] for r in payload],[float(r['area']) for r in payload],
            [float(r['predicted_R1']) for r in payload],b,rejected))
    assert recovered[0]==recovered[1]
    try:recover_choice(ids,a,p,b,dict(rejected,solver_candidate_ids=['unknown']))
    except AssertionError:pass
    else:raise AssertionError('Unknown candidate ID must hard fail')
    return {'passed':True,'solver_calls':0,'checks':['valid_candidate_passthrough','valid_candidate_retained_even_when_prefix_has_higher_P','nonfinite_candidate_audit','overshoot_uses_prefix_not_empty_raw_selected',
        'missing_candidate','nonbinary_candidate','legal_empty_candidate','actual_float64_prefix_rounding_tail_trim',
        'contradictory_bound_uses_sumP','missing_bound_uses_sumP','small_bound_inversion_clamp_not_proven','label_projection_and_recovery_invariance','unknown_id_hard_failure'],
        'rounding_case':{'prefix_count':6,'fsum_prefix_area':math.fsum([.1]*6),'budget':.6,'tail_trim':1},
        'fallback_not_proven_asserted':True}

def verify_inputs():
    assert sha(PROTOCOL)==PROTOCOL_SHA
    protocol=read(PROTOCOL)
    assert sha(ROOT/'diagnose_predicted_knapsack.py')==protocol['frozen_selector_sha256']
    assert sha(ROOT/'evaluate_common_management.py')==protocol['frozen_common_wrapper_sha256']
    assert sha(PA/protocol['parent_protocol'])==protocol['parent_protocol_sha256']
    partial=read(PARTIAL);assert partial['audit_passed_for_preserved_partial_outputs'] and not partial['completed_full_RM']
    assert Path(partial['rm_directory']).resolve()==PARENT.resolve()
    assert read(PARENT/'status.json')['state']=='failed'
    hashes={str(PROTOCOL):sha(PROTOCOL),str(PARTIAL):sha(PARTIAL),str(REPLAY):sha(REPLAY),str(Path(__file__)):sha(__file__)}
    for rel,h in partial['all_preserved_product_sha256'].items():
        path=(PARENT/rel).resolve();assert path.is_relative_to(PARENT.resolve()) and sha(path)==h
        hashes[str(path)]=h
    assert sha(REPLAY)==REPLAY_SHA
    diagnostic=read(REPLAY_DIR/'diagnostic.json');assert diagnostic['raw_result_sha256']==REPLAY_SHA
    assert diagnostic['labels_read_or_evaluated'] is False and diagnostic['candidate_feasible'] is False
    hashes[str(REPLAY_DIR/'diagnostic.json')]=sha(REPLAY_DIR/'diagnostic.json')
    parent_inputs=read(PARENT/'inputs.json')
    for path,h in parent_inputs['file_sha256'].items():assert sha(path)==h;hashes[path]=h
    checked=wrapper.validate(Path(parent_inputs['maps_directory']).resolve())
    extension,original,maps,geometry,manifest,ownership,wrapper_hashes=checked
    hashes.update(wrapper_hashes)
    assert len(extension['declared_runs'])==8 and maps['run_id']==partial['run_id']
    assert parent_inputs['checkpoint_sha256']==maps['checkpoint_sha256']
    assert maps['encoder']=='convnextv2_tiny' and maps['variant']=='hard_seg'
    expected={case_name(c) for c in CASES[:42]}
    found={p.name for p in (PARENT/'predicted_area_policy').glob('M*_R*_B*.json')}
    assert found==expected and partial['preserved_pmilp_solutions']==42
    for case in CASES[42:]:assert not (PARENT/'predicted_area_policy'/case_name(case)).exists()
    for previous in PA.glob('common_policy_recovery_*'):
        if (previous/'manifest.json').exists() and read(previous/'manifest.json').get('parent_result')==str(PARENT):
            raise RuntimeError('Prior recovery exists; do not repeat remaining-case solves: '+str(previous))
    return protocol,parent_inputs,partial,maps,geometry,hashes

def allowed_input_rows():
    # The file contains annotations, but only these six fields are retained.
    # No target conversion, selection, join or label-reference read occurs here.
    with (PARENT/'unit_predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
        return [{k:row[k] for k in FIELDS} for row in csv.DictReader(stream)]

def main():
    start=time.time();protocol,parent_inputs,partial,maps,geometry,hashes=verify_inputs()
    out=PA/('common_policy_recovery_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    for name in ('source','raw','choices'):(out/name).mkdir()
    state={'state':'running','stage':'preflight','pid':os.getpid(),'parent_result':str(PARENT),'test_accessed':False,'R6_accessed':False,
        'solver_calls':0,'completed_choices':0,'labels_joined':False,'started_at':stamp()}
    write_json(out/'status.json',state);print(json.dumps({'output':str(out),'state':'running'}),flush=True)
    try:
        # Archive implementation and frozen policy before any choice is constructed.
        for source in (Path(__file__),PROTOCOL,ROOT/'diagnose_predicted_knapsack.py',ROOT/'evaluate_common_management.py',
                       ROOT/'evaluate_management.py',PA/protocol['parent_protocol'],PARTIAL,REPLAY_DIR/'diagnostic.json'):
            shutil.copy2(source,out/'source'/source.name);assert sha(source)==sha(out/'source'/source.name)
        manifest={'created_at':stamp(),'policy_label':'predicted_area_milp_with_protocol_prefix_fallback_v1',
            'protocol_sha256':PROTOCOL_SHA,'parent_result':str(PARENT),'run_id':maps['run_id'],
            'maps_directory':parent_inputs['maps_directory'],'checkpoint_sha256':maps['checkpoint_sha256'],
            'checkpoint_epoch':maps['checkpoint_epoch'],'domain':{k:geometry[k] for k in ('area_m2','target_m2','native_valid_pixels','native_target_pixels')},
            'input_hashes':hashes,'required_case_order':[list(c) for c in CASES],
            'planned_raw_sources':{'original_saved':42,'saved_one_shot_replay':1,'new_remaining_solves':2},
            'new_solve_cases':[list(c) for c in NEW_CASES],'frozen_solver_options':policy.OPTIONS,
            'versions':{'python':sys.version,'numpy':np.__version__,'scipy':policy.scipy.__version__},
            'all_eight_configs_same_policy_declared':True,'no_new_training_inference_geometry':True,
            'selection_interface':'ids,valid area,predicted area,budget,raw numerical audit; no targets',
            'metadata_validation_note':'Frozen wrapper validates manifests/geometry including annotation metadata; these are not selection inputs.',
            'test_accessed':False,'R6_accessed':False}
        write_json(out/'manifest.json',manifest)
        write_json(out/'self_checks.json',self_checks())
        allowed=allowed_input_rows();assert all(set(r)==set(FIELDS) for r in allowed)
        write_csv(out/'selection_inputs_without_annotations.csv',allowed)
        data={}
        for m in MS:
            records=sorted([r for r in allowed if int(r['M'])==m],key=lambda r:r['id'])
            assert len(records)==partial['units_per_M'][str(m)]
            ids=[r['id'] for r in records];area=np.array([float(r['area']) for r in records])
            predicted={r:np.array([float(row[f'predicted_R{r}']) for row in records]) for r in RS}
            for r in RS:
                validate_arrays(ids,area,predicted[r],.1)
                close(math.fsum(predicted[r]),partial['totals_per_M'][str(m)][f'predicted_R{r}'])
            close(math.fsum(area),geometry['area_m2']);data[m]=(ids,area,predicted)
        choices=[];new_calls=[]
        with threadpool_limits(limits=2):
            manifest['threadpool_info']=threadpool_info();write_json(out/'manifest.json',manifest)
            for index,case in enumerate(CASES):
                m,r,b=case;ids,area,predicted=data[m];pred=predicted[r];name=case_name(case)
                if index<42:
                    origin='original_saved';source=PARENT/'predicted_area_policy'/name;raw=read(source)
                    assert (raw['M'],raw['R'],raw['budget'])==case
                elif index==42:
                    assert case==(4,4,.3);origin='saved_one_shot_replay';source=REPLAY;raw=read(source)
                else:
                    assert case in NEW_CASES and case not in new_calls
                    origin='new_remaining_case_solve';source=None
                    raw=policy.select_predicted_area(ids,area,pred,b);new_calls.append(case)
                    raw.update(M=m,R=r,budget=b)
                assert raw['options']==policy.OPTIONS and raw['original_unit_feasibility_tolerance_m2']==0.
                close(raw['budget_fraction'],b);close(raw['total_area_m2'],math.fsum(area))
                close(raw['total_predicted_m2'],math.fsum(pred));close(raw['budget_area_m2'],b*math.fsum(area))
                assert raw['n_variables']==len(ids)
                raw_path=out/'raw'/name
                if source is not None:
                    shutil.copy2(source,raw_path);assert sha(source)==sha(raw_path)
                else:write_json(raw_path,raw)
                choice=recover_choice(ids,area,pred,b,raw)
                choice.update(M=m,R=r,budget=b,raw_origin=origin,raw_source=str(source) if source else None,
                    raw_archived_file='raw/'+name,raw_sha256=sha(raw_path),protocol_sha256=PROTOCOL_SHA,
                    recovery_source_sha256=sha(__file__),unit_source_sha256=sha(PARENT/'unit_predictions.csv'),created_at=stamp())
                if index<42:assert not choice['fallback_used'] and choice['selected_ids']==raw['selected_ids']
                if index==42:
                    assert choice['fallback_used'] and choice['raw_candidate_audit']['budget_violation_m2_recomputed']>0
                    close(choice['raw_candidate_audit']['budget_violation_m2_recomputed'],diagnostic_violation())
                write_json(out/'choices'/name,choice);choices.append(choice)
                state.update(stage='label_free_choices',completed_choices=len(choices),solver_calls=len(new_calls),updated_at=stamp())
                write_json(out/'status.json',state)
            assert new_calls==list(NEW_CASES) and len(choices)==45
        # Freeze all final IDs before target values or label-reference rows are joined.
        frozen={str(p.relative_to(out)):sha(p) for p in (out/'choices').glob('*.json')}
        raw_hashes={str(p.relative_to(out)):sha(p) for p in (out/'raw').glob('*.json')}
        assert len(frozen)==len(raw_hashes)==45
        seal={'sealed_at':stamp(),'all_45_final_choices_saved':True,'target_join_started':False,
              'choices_sha256':frozen,'raw_sha256':raw_hashes,'allowed_inputs_sha256':sha(out/'selection_inputs_without_annotations.csv'),
              'case_order':[list(c) for c in CASES],'new_solver_calls':[list(c) for c in new_calls],
              'protocol_sha256':PROTOCOL_SHA,'implementation_sha256':sha(__file__)}
        write_json(out/'selection_seal.json',seal)
        state.update(stage='annotation_evaluation',labels_joined=True,labels_join_started_at=stamp(),selection_seal_sha256=sha(out/'selection_seal.json'))
        write_json(out/'status.json',state)
        annotated=policy.read_csv(PARENT/'unit_predictions.csv')
        assert policy.payload_sha(policy.allowed_selection_rows(annotated))==policy.payload_sha(allowed)
        truth={(int(row['M']),row['id']):float(row['target']) for row in annotated}
        assert len(truth)==len(annotated)==len(allowed)
        original_refs=policy.read_csv(PARENT/'fixed_budgets.csv')
        prefixes={(int(x['M']),int(x['R']),float(x['budget'])):x for x in original_refs if x['method']=='model_prefix'}
        assert set(prefixes)==set(CASES)
        results=[];prefix_rows=[]
        for choice in choices:
            m,r,b=choice['M'],choice['R'],choice['budget'];ids,area,predicted=data[m];pred=predicted[r]
            target=np.array([truth[(m,i)] for i in ids]);assert np.isfinite(target).all() and np.all((target>=0)&(target<=area+1e-9))
            close(math.fsum(target),geometry['target_m2'])
            final=policy.evaluate_selected(ids,choice['selected_ids'],area,pred,target)
            before=policy.prefix_ids(ids,area,pred,b)
            reference=policy.evaluate_selected(ids,before,area,pred,target)
            old=prefixes[(m,r,b)]
            for field,column in [('selected_count','k'),('actual_area_m2','actual_area'),('captured_target_m2','captured_target'),
                ('capture_fraction','capture_fraction'),('zero_annotation_area_m2','zero_annotation_area')]:close(reference[field],old[column])
            assert final['actual_area_m2']<=choice['budget_area_m2']
            close(final['predicted_objective_m2'],choice['predicted_primal_m2'])
            prefix_rows.append({'M':m,'R':r,'budget':b,'budget_area_m2':choice['budget_area_m2'],**reference,
                'selected_ids_json':json.dumps(before),'strict_original_unit_feasible':reference['actual_area_m2']<=choice['budget_area_m2']})
            result={'M':m,'R':r,'budget':b,'policy_label':choice['policy_label'],'raw_origin':choice['raw_origin'],
                'budget_area_m2':choice['budget_area_m2'],'fallback_used':choice['fallback_used'],'fallback_reasons':';'.join(choice['fallback_reasons']),
                'prefix_tail_trim_count':choice['prefix_tail_trim_count'],'final_status':choice['status'],'optimality_proven':choice['optimality_proven'],
                'final_predicted_upper_m2':choice['recomputed_bound']['upper_m2'],'final_predicted_relative_gap':choice['recomputed_bound']['relative_gap'],
                'bound_anomaly':choice['recomputed_bound']['bound_anomaly'],'bound_roundoff_clamp_m2':choice['recomputed_bound']['roundoff_clamp_m2'],
                'raw_candidate_feasible':choice['raw_candidate_audit']['recorded_feasible_flag'],
                'raw_candidate_budget_violation_m2':choice['raw_candidate_audit']['budget_violation_m2_recomputed'],
                'raw_solver_status':choice['raw_candidate_audit']['raw_solver_status'],'raw_solver_gap':choice['raw_candidate_audit']['raw_solver_gap'],
                **{'recovered_'+k:v for k,v in final.items()},**{'prefix_'+k:v for k,v in reference.items()},
                'delta_capture_pp':100*(final['capture_fraction']-reference['capture_fraction']),
                'delta_predicted_objective_m2':final['predicted_objective_m2']-reference['predicted_objective_m2'],
                'unused_budget_m2':choice['budget_area_m2']-final['actual_area_m2'],
                'choice_file':'choices/'+case_name((m,r,b)),'choice_sha256':frozen['choices/'+case_name((m,r,b))]}
            results.append(result)
        assert len(results)==len(prefix_rows)==45
        write_csv(out/'all_45_policy_outcomes.csv',results);write_csv(out/'all_45_prefix_reference.csv',prefix_rows)
        shutil.copy2(PARENT/'fixed_budgets.csv',out/'source'/'full_original_fixed_budgets.csv')
        shutil.copy2(PARENT/'decision_curves.csv',out/'source'/'full_original_decision_curves.csv')
        for rel,h in {**frozen,**raw_hashes}.items():assert sha(out/rel)==h
        for path,h in hashes.items():assert sha(path)==h,'Changed input: '+path
        summary={'state':'completed','stage':'development numerical recovery','run_id':maps['run_id'],'policy_label':choices[0]['policy_label'],
            'cases':45,'original_results_reused':42,'saved_replay_reused':1,'new_solver_calls':len(new_calls),
            'all_final_choices_strictly_feasible':True,'all_choices_sealed_before_target_join':True,'selection_seal_sha256':sha(out/'selection_seal.json'),
            'fallback_count':sum(x['fallback_used'] for x in choices),'fallback_cases':[{k:x[k] for k in ('M','R','budget','selected_area_m2','predicted_primal_m2','prefix_tail_trim_count','recomputed_bound','raw_candidate_audit')} for x in choices if x['fallback_used']],
            'optimality_proven_within_tolerance':sum(x['optimality_proven'] for x in choices),
            'capture_increased_vs_prefix':sum(x['delta_capture_pp']>1e-8 for x in results),
            'capture_decreased_vs_prefix':sum(x['delta_capture_pp']< -1e-8 for x in results),
            'capture_unchanged_vs_prefix':sum(abs(x['delta_capture_pp'])<=1e-8 for x in results),
            'domain':manifest['domain'],'protocol_sha256':PROTOCOL_SHA,'parent_failed_preserved':True,
            'all_eight_configs_same_policy_declared':True,'self_checks_passed':True,'test_accessed':False,'R6_accessed':False,
            'no_training_inference_or_geometry':True,'elapsed_seconds':time.time()-start}
        write_json(out/'summary.json',summary)
        report=['# Numerical policy recovery — development','',
            'A separate recovery policy was applied under the frozen numerical-recovery protocol. The failed parent and all raw candidates remain unchanged.',
            '',f"Reused42 original valid results and one saved replay; called the unchanged selector only for M4/R4/budget0.5 and1.0. All45 final choices were sealed before joining target annotations.",
            '',f"Final strictly feasible choices:45/45. Prefix fallbacks:{summary['fallback_count']}. Recorded optimality within tolerance:{summary['optimality_proven_within_tolerance']}/45. Every fallback remains feasible_fallback_not_proven.",
            '',f"Against the complete original prefix reference, annotated capture increased/decreased/was unchanged in {summary['capture_increased_vs_prefix']}/{summary['capture_decreased_vs_prefix']}/{summary['capture_unchanged_vs_prefix']} cases.",
            '', '## Fallback records','']
        for x in summary['fallback_cases']:
            report.append(f"- M{x['M']}/R{x['R']}/budget{x['budget']}: raw candidate exceeded budget by {x['raw_candidate_audit']['budget_violation_m2_recomputed']:.15g} m²; final area {x['selected_area_m2']:.15g} m², predicted objective {x['predicted_primal_m2']:.15g} m², recomputed relative gap {x['recomputed_bound']['relative_gap']:.15g}; tail trims {x['prefix_tail_trim_count']}.")
        report+=['','The actual rejected candidate was obtained from solver_candidate_ids, not the raw selected_ids empty fallback. Its budget excess is not described as last-bit rounding. The cause inside the solver has not been established.',
            '', 'Complete45-case policy and prefix tables, raw/choice JSONs, frozen code/protocol, input hashes, synthetic checks and the selection seal are included. This is numerical recovery in one-scene development evaluation, not a new optimization algorithm or evidence of field efficacy. No test/R6, training, inference or new geometry was used.']
        (out/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
        state.update(state='completed',stage='complete',completed_at=stamp(),summary=summary,
            files_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'status.json'})
        write_json(out/'status.json',state)
        print(json.dumps({'output':str(out),'summary':summary},ensure_ascii=True),flush=True)
    except BaseException as error:
        state.update(state='failed',error=repr(error),traceback=traceback.format_exc(),failed_at=stamp())
        write_json(out/'status.json',state);raise

def diagnostic_violation():return float(read(REPLAY_DIR/'diagnostic.json')['candidate_minus_budget_m2'])
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-check-only',action='store_true')
    args=parser.parse_args()
    if args.self_check_only:print(json.dumps(self_checks(),indent=2))
    else:main()
