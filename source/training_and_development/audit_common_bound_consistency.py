"""Uniform output-only bound audit of eight fixed policies,360 existing choices.

No optimization, inference, new geometry, budget change or new final selection.
All old records remain unchanged. A feasible prefix is a bound-consistency witness.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,datetime,json,math,shutil,sys,time,traceback
import numpy as np
from threadpoolctl import threadpool_limits
import evaluate_common_management_v2 as v

ROOT=v.ROOT;PA=v.PA;PROTOCOL=PA/'RM_bound_consistency_protocol_v1.json'
V2_SHA='99315d78bd31e721f024601227e032291962075336ed65a7e9bd157caf643d25'
V2_PROTOCOL_SHA='d3a82af875ac95e18ebdd30d124f668b7ef8916244b57d07686af6ec6be56bc7'
sha=v.ev.sha;read=v.ev.read;write=v.ev.write_json

def forbid_solver(*args,**kwargs):
    raise RuntimeError('Solver calls forbidden in uniform bound audit')
# Defensive guards, not source edits. Existing prefix reconstruction is permitted.
v.policy.milp=forbid_solver
v.policy.select_predicted_area=forbid_solver
v.ev.milp=forbid_solver


def finite(x):return isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x)


def audit_bounds(final_p,prefix_p,prefix_feasible,total_p,raw_upper,retained_upper,
                 original_proven,fallback,requested_gap):
    """Label-free numerical audit; never modifies or returns selection IDs."""
    assert all(finite(x) and x>=0 for x in (final_p,prefix_p,total_p))
    tol=max(1e-10,1e-10*total_p)
    assert final_p<=total_p+tol
    witness=max(final_p,prefix_p) if prefix_feasible else final_p
    assert witness<=total_p+tol
    observed={'raw_solver_upper':raw_upper,'retained_upper':retained_upper}
    contradicted=[name for name,u in observed.items() if finite(u) and u<witness-tol]
    reasons=list(contradicted)
    if not finite(retained_upper):reasons.append('missing_or_nonfinite_retained_upper')
    clamp=0.
    if reasons:
        upper=total_p;source='sum_predicted_safe_upper_after_bound_anomaly'
    else:
        candidates=[total_p,float(retained_upper)]
        if finite(raw_upper):candidates.append(float(raw_upper))
        upper=min(candidates);source='min(totalP,finite_raw_upper,retained_upper)'
        if upper<witness:
            clamp=witness-upper
            assert clamp<=tol
            upper=witness
    assert upper>=final_p
    absolute=upper-final_p
    gap=absolute/max(abs(final_p),1e-30) if upper else 0.
    proven=bool(original_proven and not fallback and not reasons and gap<=requested_gap+1e-10)
    return {'raw_solver_upper_m2':raw_upper if finite(raw_upper) else None,
            'retained_upper_m2':retained_upper if finite(retained_upper) else None,
            'raw_upper_was_missing_or_nonfinite':not finite(raw_upper),
            'bound_tolerance_m2':tol,'strict_feasible_prefix_used_as_witness':bool(prefix_feasible),
            'feasible_objective_witness_m2':witness,'contradicted_upper_fields':contradicted,
            'bound_anomaly_reasons':reasons,'bound_anomaly':bool(reasons),
            'updated_upper_m2':upper,'updated_primal_m2':final_p,'updated_absolute_gap_m2':absolute,
            'updated_relative_gap':gap,'updated_bound_source':source,'roundoff_clamp_m2':clamp,
            'original_proven_flag':bool(original_proven),'fallback_used':bool(fallback),
            'updated_proven_flag':proven,'certification_downgraded':bool(original_proven and not proven),
            'requested_relative_gap':requested_gap,'no_certification_upgrade':True}


def self_checks():
    def f(**extra):
        args=dict(final_p=.6,prefix_p=.5,prefix_feasible=True,total_p=1.,raw_upper=.7,
                  retained_upper=.7,original_proven=False,fallback=False,requested_gap=1e-8)
        args.update(extra);return audit_bounds(**args)
    x=f(prefix_p=.8,raw_upper=.6,retained_upper=.6,original_proven=True)
    assert x['bound_anomaly'] and x['updated_upper_m2']==1 and not x['updated_proven_flag']
    assert math.isclose(x['updated_relative_gap'],(.4/.6),abs_tol=1e-15)
    y=f(final_p=.6,prefix_p=.6+5e-11,raw_upper=.6,retained_upper=.6,original_proven=True)
    assert not y['bound_anomaly'] and y['roundoff_clamp_m2']>0 and y['updated_upper_m2']==.6+5e-11
    assert y['updated_proven_flag']
    z=f(prefix_p=.9,prefix_feasible=False,raw_upper=.6,retained_upper=.6,original_proven=True)
    assert not z['bound_anomaly'] and z['updated_proven_flag']
    assert f(retained_upper=None)['updated_upper_m2']==1
    assert f(raw_upper=.4,retained_upper=.9)['bound_anomaly']
    assert not f(final_p=1.,prefix_p=1.,raw_upper=1.,retained_upper=1.,fallback=True,original_proven=True)['updated_proven_flag']
    assert not f(raw_upper=.6,retained_upper=.6,original_proven=False)['updated_proven_flag']
    assert f(raw_upper=None,retained_upper=1.)['raw_upper_was_missing_or_nonfinite']
    # At the strict threshold equality is a disclosed clamp, not a larger anomaly.
    w=f(final_p=0.,prefix_p=5e-11,raw_upper=0.,retained_upper=0.)
    assert w['roundoff_clamp_m2']==5e-11
    return {'passed':True,'checks':['feasible_prefix_contradiction_uses_safe_sumP','updated_gap_recomputed',
            'within_threshold_inversion_disclosed_clamp','infeasible_prefix_excluded_as_witness',
            'missing_retained_bound_safe_upper','raw_contradiction_detected_despite_higher_retained_bound',
            'fallback_never_proven','no_upgrade_of_original_false_flag','missing_raw_audit_retained','near_zero_clamp'],
            'solver_calls':0,'model_calls':0}


def load_inputs(expected_sha):
    assert sha(PROTOCOL)==expected_sha
    protocol=read(PROTOCOL)
    assert protocol['schema']=='common_bound_consistency_v1'
    assert protocol['implementation_sha256']==sha(__file__)
    assert protocol['v2_pipeline_source_sha256']==V2_SHA==sha(v.__file__)
    assert protocol['v2_pipeline_protocol_sha256']==V2_PROTOCOL_SHA==sha(v.PIPELINE)
    assert protocol['recovery_protocol_sha256']==v.RECOVERY_SHA==sha(v.recovery.PROTOCOL)
    assert protocol['extension_protocol_sha256']==v.EXTENSION_SHA==sha(v.original.PROTOCOL)
    assert protocol['bound_tolerance']=='max(1e-10,1e-10*sumP)' and protocol['strict_area_tolerance_m2']==0.
    assert protocol['final_selected_ids_changed'] is False and protocol['solver_calls']==0
    declared=[x['run_id'] for x in read(v.original.PROTOCOL)['declared_runs']]
    assert [x['run_id'] for x in protocol['models']]==declared and len(declared)==8
    assert [x['index'] for x in protocol['models']]==list(range(1,9))
    watched=dict(protocol['pinned_files_sha256'])
    watched.update({str(PROTOCOL):expected_sha,str(Path(__file__)):sha(__file__)})
    v.verify_hashes(watched)
    inputs=[]
    for model in protocol['models']:
        output=Path(model['output']);units=Path(model['unit_tables_directory'])
        assert output.is_relative_to(PA) and units.is_relative_to(PA)
        metadata=read(Path(model['maps_directory'])/'metadata.json')
        checked=v.original.validate(Path(model['maps_directory']).resolve())
        assert checked[2]['run_id']==model['run_id'] and metadata['checkpoint_epoch']==model['checkpoint_epoch']
        assert metadata['checkpoint_sha256']==model['checkpoint_sha256']
        config=read(ROOT/'runs'/model['run_id']/'config.json')
        for key in ('encoder','variant','lr'):assert model[key]==config[key]
        watched.update(checked[-1])
        status=read(output/'status.json')
        assert status['state']==model['original_output_state']
        watched[str(output/'status.json')]=sha(output/'status.json')
        for p in (units/'unit_predictions.csv',units/'fixed_budgets.csv'):
            assert sha(p)==model['table_sha256'][p.name];watched[str(p)]=sha(p)
        if model['format']=='v2_sealed':
            dest=output/'predicted_area_policy';seal=v.verify_selection_seal(dest)
            assert sha(dest/'selection_seal.json')==model['selection_seal_sha256']
            watched[str(dest/'selection_seal.json')]=sha(dest/'selection_seal.json')
            for group in ('solver_raw_sha256','raw_sha256','choices_sha256'):
                for rel,digest in seal[group].items():watched[str(v.relative_path(dest,rel))]=digest
            assert status['unit_predictions_sha256']==model['table_sha256']['unit_predictions.csv']
        elif model['format']=='recovery_sealed':
            dest=output;seal=read(dest/'selection_seal.json')
            assert seal['all_45_final_choices_saved'] and seal['target_join_started'] is False
            assert sha(dest/'selection_seal.json')==model['selection_seal_sha256']
            watched[str(dest/'selection_seal.json')]=sha(dest/'selection_seal.json')
            for group in ('raw_sha256','choices_sha256'):
                assert len(seal[group])==45
                for rel,digest in seal[group].items():watched[str(v.relative_path(dest,rel))]=digest
        else:
            assert model['format']=='original_audited_R18' and model['run_id'] in v.KNOWN_COMPLETED[:2]
            assert status['state']=='completed'
            dest=output/'predicted_area_policy'
        for case in v.CASES:
            name=v.recovery.case_name(case)
            raw_path=dest/name if model['format']=='original_audited_R18' else dest/'raw'/name
            if model['format']=='original_audited_R18':
                prior_files={key.replace('\\','/'):digest for key,digest in status['files_sha256'].items()}
                assert sha(raw_path)==prior_files['predicted_area_policy/'+name]
            watched[str(raw_path)]=sha(raw_path)
            if model['format']!='original_audited_R18':watched[str(dest/'choices'/name)]=sha(dest/'choices'/name)
        allowed=v.project_selection_inputs(units/'unit_predictions.csv')
        data=v.prepare_data(allowed,checked[3]['area_m2'])
        inputs.append((model,output,units,dest,data,checked[3]))
    v.verify_hashes(watched)
    return protocol,inputs,watched


def bind_existing_choices(inputs,out,protocol_sha):
    dest=out/'records';dest.mkdir();bindings=[]
    for model,output,units,old,data,domain in inputs:
        index=model['index'];folder=dest/f'{index:02d}';folder.mkdir()
        for case in v.CASES:
            m,r,b=case;name=v.recovery.case_name(case);ids,A,Ps=data[m];P=Ps[r];lookup={i:j for j,i in enumerate(ids)}
            raw_path=old/name if model['format']=='original_audited_R18' else old/'raw'/name
            raw=read(raw_path)
            if model['format']=='original_audited_R18':
                final_ids=raw['selected_ids'];final_a=raw['selected_area_m2'];final_p=raw['predicted_primal_m2']
                retained=raw['predicted_dual_upper_m2'];old_proven=raw['optimality_proven_within_tolerance'];fallback=False
                origin='original_saved_and_independently_equivalent';choice_path=raw_path
            else:
                choice_path=old/'choices'/name;choice=read(choice_path)
                assert (choice['M'],choice['R'],choice['budget'])==case
                assert choice['raw_sha256']==sha(raw_path)
                final_ids=choice['selected_ids'];final_a=choice['selected_area_m2'];final_p=choice['predicted_primal_m2']
                retained=choice['recomputed_bound']['upper_m2'];old_proven=choice['optimality_proven'];fallback=choice['fallback_used']
                origin=choice['raw_origin']
            assert len(set(final_ids))==len(final_ids) and set(final_ids).issubset(lookup)
            a=math.fsum(A[lookup[i]] for i in final_ids);p=math.fsum(P[lookup[i]] for i in final_ids)
            v.recovery.close(a,final_a);v.recovery.close(p,final_p)
            limit=b*math.fsum(A);assert a<=limit
            prefix=v.policy.prefix_ids(ids,A,P,b)
            prefix_a=math.fsum(A[lookup[i]] for i in prefix);prefix_p=math.fsum(P[lookup[i]] for i in prefix)
            audit=audit_bounds(p,prefix_p,prefix_a<=limit,math.fsum(P),raw.get('raw_solver_dual_upper_m2'),
                               retained,old_proven,fallback,v.policy.OPTIONS['mip_rel_gap'])
            binding={'model_index':index,'run_id':model['run_id'],'checkpoint_epoch':model['checkpoint_epoch'],
                     'encoder':model['encoder'],'variant':model['variant'],'lr':model['lr'],
                     'checkpoint_sha256':model['checkpoint_sha256'],'maps_directory':model['maps_directory'],
                     'original_output':str(output),'original_format':model['format'],
                     'original_selection_seal_sha256':model.get('selection_seal_sha256'),
                     'M':m,'R':r,'budget':b,'selected_ids':final_ids,'final_area_m2':a,'final_predicted_m2':p,
                     'budget_area_m2':limit,'prefix_ids':prefix,'prefix_area_m2':prefix_a,'prefix_predicted_m2':prefix_p,
                     'prefix_strictly_feasible':prefix_a<=limit,'final_ids_unchanged':True,
                     'choice_source':str(choice_path),'choice_source_sha256':sha(choice_path),
                     'raw_source':str(raw_path),'raw_source_sha256':sha(raw_path),'raw_origin':origin,
                     'selection_input_sha256':v.policy.payload_sha({'ids':ids,'area':A.tolist(),'predicted':P.tolist(),'budget':b}),
                     'raw_solver_status':raw.get('solver_status'),'raw_solver_gap':raw.get('raw_solver_gap'),
                     'raw_candidate_feasible_flag':raw.get('candidate_feasible_in_original_units'),
                     'raw_optimality_proven_flag':raw.get('optimality_proven_within_tolerance'),
                     'raw_candidate_integrality_error':raw.get('candidate_integrality_error'),
                     'raw_selected_ids':raw.get('selected_ids'),'raw_solver_candidate_ids':raw.get('solver_candidate_ids'),
                     'original_retained_gap':raw['predicted_relative_gap'] if model['format']=='original_audited_R18' else choice['recomputed_bound']['relative_gap'],
                     'bound_audit':audit}
            write(folder/name,binding);bindings.append(binding)
    assert len(bindings)==360
    seal={'created_at':v.ev.stamp(),'all360_existing_final_ids_bound':True,'new_final_selections':0,
          'annotations_joined':False,'protocol_sha256':protocol_sha,'implementation_sha256':sha(__file__),
          'records_sha256':v.file_hashes(out,sorted(dest.rglob('*.json')))}
    assert len(seal['records_sha256'])==360
    write(out/'binding_seal.json',seal)
    return bindings,seal


def evaluate_bound_records(inputs,out,bindings,seal):
    assert seal==read(out/'binding_seal.json') and seal['annotations_joined'] is False
    for rel,digest in seal['records_sha256'].items():assert sha(v.relative_path(out,rel))==digest
    (out/'per_model').mkdir();all_rows=[];summaries=[]
    for model,output,units,old,data,domain in inputs:
        rows=v.policy.read_csv(units/'unit_predictions.csv')
        truth={(int(x['M']),x['id']):float(x['target']) for x in rows};assert len(truth)==len(rows)
        refs={(int(x['M']),int(x['R']),float(x['budget'])):x for x in v.policy.read_csv(units/'fixed_budgets.csv') if x['method']=='model_prefix'}
        assert set(refs)==set(v.CASES)
        current=[]
        for record in [x for x in bindings if x['run_id']==model['run_id']]:
            m,r,b=record['M'],record['R'],record['budget'];ids,A,Ps=data[m];P=Ps[r]
            target=np.array([truth[(m,i)] for i in ids]);assert np.isfinite(target).all() and np.all((target>=0)&(target<=A+1e-9))
            v.recovery.close(math.fsum(target),domain['target_m2'])
            final=v.policy.evaluate_selected(ids,record['selected_ids'],A,P,target)
            prefix=v.policy.evaluate_selected(ids,record['prefix_ids'],A,P,target)
            assert final['actual_area_m2']<=record['budget_area_m2']
            v.recovery.close(final['predicted_objective_m2'],record['final_predicted_m2'])
            for field,column in (('selected_count','k'),('actual_area_m2','actual_area'),('captured_target_m2','captured_target'),
                                 ('capture_fraction','capture_fraction'),('zero_annotation_area_m2','zero_annotation_area')):
                v.recovery.close(prefix[field],refs[(m,r,b)][column])
            a=record['bound_audit']
            row={'model_index':model['index'],'run_id':model['run_id'],'checkpoint_epoch':model['checkpoint_epoch'],
                 'encoder':model['encoder'],'variant':model['variant'],'lr':model['lr'],
                 'checkpoint_sha256':model['checkpoint_sha256'],'maps_directory':model['maps_directory'],
                 'original_output':str(output),'original_format':model['format'],
                 'original_selection_seal_sha256':model.get('selection_seal_sha256'),
                 'M':m,'R':r,'budget':b,'budget_area_m2':record['budget_area_m2'],'raw_origin':record['raw_origin'],
                 'final_ids_unchanged':True,'raw_solver_status':record['raw_solver_status'],'raw_solver_gap':record['raw_solver_gap'],
                 'original_retained_gap':record['original_retained_gap'],
                 'raw_candidate_feasible_flag':record['raw_candidate_feasible_flag'],
                 'raw_optimality_proven_flag':record['raw_optimality_proven_flag'],
                 'raw_candidate_integrality_error':record['raw_candidate_integrality_error'],
                 'policy_selected_ids_json':json.dumps(record['selected_ids']),
                 'prefix_selected_ids_json':json.dumps(record['prefix_ids']),
                 'raw_selected_ids_json':json.dumps(record['raw_selected_ids']),
                 'raw_solver_candidate_ids_json':json.dumps(record['raw_solver_candidate_ids']),
                 **{k:(';'.join(x) if isinstance(x,list) else x) for k,x in a.items()},
                 **{'policy_'+k:x for k,x in final.items()},**{'prefix_'+k:x for k,x in prefix.items()},
                 'prefix_strictly_feasible':record['prefix_strictly_feasible'],
                 'delta_capture_pp':100*(final['capture_fraction']-prefix['capture_fraction']),
                 'delta_predicted_m2':final['predicted_objective_m2']-prefix['predicted_objective_m2'],
                 'unused_budget_m2':record['budget_area_m2']-final['actual_area_m2'],
                 'record_file':f"records/{model['index']:02d}/"+v.recovery.case_name((m,r,b)),
                 'choice_source_sha256':record['choice_source_sha256'],'raw_source_sha256':record['raw_source_sha256']}
            current.append(row)
        assert len(current)==45;all_rows+=current
        v.ev.write_csv(out/'per_model'/f"{model['index']:02d}_all45.csv",current)
        summaries.append({'model_index':model['index'],'run_id':model['run_id'],'checkpoint_epoch':model['checkpoint_epoch'],
                          'cases':45,'old_output_state':model['original_output_state'],
                          'fallback_count':sum(x['fallback_used'] for x in current),
                          'original_proven_count':sum(x['original_proven_flag'] for x in current),
                          'updated_proven_count':sum(x['updated_proven_flag'] for x in current),
                          'certification_downgrades':sum(x['certification_downgraded'] for x in current),
                          'bound_anomalies':sum(x['bound_anomaly'] for x in current),
                          'roundoff_clamps':sum(x['roundoff_clamp_m2']>0 for x in current),
                          'capture_increased':sum(x['delta_capture_pp']>1e-8 for x in current),
                          'capture_decreased':sum(x['delta_capture_pp']< -1e-8 for x in current),
                          'capture_unchanged':sum(abs(x['delta_capture_pp'])<=1e-8 for x in current)})
    assert len(all_rows)==360
    v.ev.write_csv(out/'all360_policy_outcomes.csv',all_rows)
    return summaries


def main(protocol_sha,validate_only=False):
    started=time.time();protocol,inputs,watched=load_inputs(protocol_sha)
    if validate_only:
        print(json.dumps({'state':'validated_only','models':8,'cases':360,'source_sha256':sha(__file__),'protocol_sha256':protocol_sha}));return
    for prior in PA.glob('bca_*'):
        if (prior/'manifest.json').exists() and read(prior/'manifest.json').get('protocol_sha256')==protocol_sha:
            raise RuntimeError('Existing bound-audit attempt preserved; no automatic repeat: '+str(prior))
    out=PA/('bca_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False);(out/'source').mkdir()
    state={'state':'running','stage':'archive','pid':os.getpid(),'started_at':v.ev.stamp(),'test_accessed':False,'R6_accessed':False}
    write(out/'status.json',state)
    try:
        source_paths=[Path(__file__),PROTOCOL,Path(v.__file__),v.PIPELINE,v.original.PROTOCOL,v.recovery.PROTOCOL]+[ROOT/k for k in v.FROZEN]
        for i,p in enumerate(source_paths):
            target=out/'source'/(f'{i:02d}_'+p.name);shutil.copy2(p,target);assert sha(target)==sha(p)
        write(out/'manifest.json',{'protocol_sha256':protocol_sha,'implementation_sha256':sha(__file__),
             'models':protocol['models'],'source_sha256':watched,'solver_calls':0,'new_final_selections':0,'created_at':v.ev.stamp()})
        write(out/'self_checks.json',self_checks())
        with threadpool_limits(limits=2):
            bindings,seal=bind_existing_choices(inputs,out,protocol_sha)
            v.verify_hashes(watched)
            summaries=evaluate_bound_records(inputs,out,bindings,seal)
        v.verify_hashes(watched)
        for rel,digest in seal['records_sha256'].items():assert sha(v.relative_path(out,rel))==digest
        summary={'state':'completed','stage':'development uniform bound-consistency audit','models':8,'cases':360,
                 'all_existing_final_ids_unchanged':True,'all_final_selections_strictly_feasible':True,
                 'solver_calls':0,'new_inference_calls':0,'new_final_selections':0,'test_accessed':False,'R6_accessed':False,
                 'protocol_sha256':protocol_sha,'binding_seal_sha256':sha(out/'binding_seal.json'),
                 'by_model':summaries,'elapsed_seconds':time.time()-started}
        write(out/'summary.json',summary)
        lines=['# Uniform bound-consistency audit — development','',
               'All360 existing policy selections are unchanged. No optimizer/model was called. All eight configurations were audited by one rule before annotation values were joined for this output. Original outputs, raw records, failed statuses and seals remain untouched.','',
               'A strictly feasible original prefix supplies a known feasible objective. If either recorded raw/retained upper is below the largest known feasible objective by more than max(1e-10,1e-10sumP), the bound is flagged, the safe sumP upper replaces it and certification is false. Smaller inversions are explicitly clamped and recorded. Fallbacks stay not proven; previously false flags are never upgraded.','',
               '|Model|Original flag count|Updated flag count|Anomalies|Fallbacks|Capture up/down/tie|','|---|---:|---:|---:|---:|---|']
        for x in summaries:lines.append(f"|{x['run_id']}|{x['original_proven_count']}|{x['updated_proven_count']}|{x['bound_anomalies']}|{x['fallback_count']}|{x['capture_increased']}/{x['capture_decreased']}/{x['capture_unchanged']}|")
        lines+=['','The tighter bound threshold is reused from the audited recovery definition. This is a disclosed development numerical revision, not prospective confirmation. The updated flags are bound-consistent recorded solver claims, not an independent re-solve or proof of real-world management benefit. Same budget does not mean equal actual selected area; all outcomes and all negative differences remain in the360-row table.']
        (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        state.update(state='completed',stage='complete',completed_at=v.ev.stamp(),summary=summary,
                     files_sha256=v.file_hashes(out,[p for p in out.rglob('*') if p.is_file() and p!=out/'status.json']))
        write(out/'status.json',state);print(json.dumps({'output':str(out),'summary':summary}),flush=True)
    except BaseException as error:
        state.update(state='failed',error=repr(error),traceback=traceback.format_exc());write(out/'status.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol-sha256');parser.add_argument('--validate-only',action='store_true');parser.add_argument('--self-check-only',action='store_true')
    args=parser.parse_args()
    if args.self_check_only:print(json.dumps(self_checks(),indent=2))
    else:
        assert args.protocol_sha256
        main(args.protocol_sha256,args.validate_only)
