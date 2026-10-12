"""Output-only resume from45 sealed choices; no solver calls or final-ID selection.

The original recovery reached its45-choice seal, then failed on a Windows
slash/backslash key lookup while building the outcome table. Preserve it intact.
"""
from pathlib import Path
import csv,json,math,os,shutil,sys,time,traceback,datetime
import recover_common_policy as recovery
import numpy as np

ROOT=recovery.ROOT;PA=recovery.PA;PARENT=recovery.PARENT
FAILED=PA/'common_policy_recovery_20260916_034644_420249'
sha=recovery.sha;read=recovery.read;write_json=recovery.write_json;write_csv=recovery.write_csv;stamp=recovery.stamp
policy=recovery.policy
def forbid_solver(*args,**kwargs):raise RuntimeError('Solver calls forbidden in output-only resume')
policy.select_predicted_area=forbid_solver

def relative(root,name):
    # Interpret either separator consistently; do not alter original seal bytes.
    parts=str(name).replace('\\','/').split('/')
    assert all(p not in ('','.','..') for p in parts)
    result=root.joinpath(*parts).resolve();assert result.is_relative_to(root.resolve())
    return result

def main():
    started=time.time()
    failed_state=read(FAILED/'status.json');old=read(FAILED/'manifest.json');seal=read(FAILED/'selection_seal.json')
    assert failed_state['state']=='failed' and failed_state['completed_choices']==45 and failed_state['solver_calls']==2
    assert failed_state['error'].startswith('KeyError(') and 'choices/M1_R1_B010.json' in failed_state['error']
    assert old['parent_result']==str(PARENT) and old['protocol_sha256']==recovery.PROTOCOL_SHA==sha(recovery.PROTOCOL)
    assert seal['all_45_final_choices_saved'] and seal['target_join_started'] is False
    assert seal['new_solver_calls']==[[4,4,.5],[4,4,1.]] and len(seal['choices_sha256'])==len(seal['raw_sha256'])==45
    assert failed_state['selection_seal_sha256']==sha(FAILED/'selection_seal.json')
    assert sha(ROOT/'recover_common_policy.py')==seal['implementation_sha256']==sha(FAILED/'source'/'recover_common_policy.py')
    for path,digest in old['input_hashes'].items():assert sha(path)==digest,'Changed original source: '+path
    source_hashes={str(p):sha(p) for p in (FAILED/'status.json',FAILED/'manifest.json',FAILED/'selection_seal.json',Path(__file__))}
    for recorded in (seal['choices_sha256'],seal['raw_sha256']):
        for rel,digest in recorded.items():
            path=relative(FAILED,rel);assert sha(path)==digest;source_hashes[str(path)]=digest
    allowed=policy.read_csv(FAILED/'selection_inputs_without_annotations.csv')
    assert sha(FAILED/'selection_inputs_without_annotations.csv')==seal['allowed_inputs_sha256']
    assert all(set(row)==set(recovery.FIELDS) for row in allowed)
    data={};choices=[]
    for m in recovery.MS:
        records=sorted([row for row in allowed if int(row['M'])==m],key=lambda row:row['id'])
        ids=[row['id'] for row in records];area=np.array([float(row['area']) for row in records])
        predicted={r:np.array([float(row[f'predicted_R{r}']) for row in records]) for r in recovery.RS}
        data[m]=(ids,area,predicted)
    for index,case in enumerate(recovery.CASES):
        name=recovery.case_name(case);choice=read(FAILED/'choices'/name);raw=read(FAILED/'raw'/name)
        m,r,b=case;ids,area,predicted=data[m];pred=predicted[r]
        recovery.validate_arrays(ids,area,pred,b);lookup={i:j for j,i in enumerate(ids)}
        assert (choice['M'],choice['R'],choice['budget'])==case
        selected=choice['selected_ids'];assert len(set(selected))==len(selected) and set(selected).issubset(lookup)
        a=math.fsum(area[lookup[i]] for i in selected);p=math.fsum(pred[lookup[i]] for i in selected)
        assert a<=b*math.fsum(area)
        recovery.close(a,choice['selected_area_m2']);recovery.close(p,choice['predicted_primal_m2'])
        assert choice['raw_sha256']==sha(FAILED/'raw'/name)
        assert choice['recovery_source_sha256']==seal['implementation_sha256']
        assert choice['selection_input_sha256']==policy.payload_sha({'ids':ids,'area':area.tolist(),'predicted':pred.tolist(),'budget':b})
        if index<42:
            assert not choice['fallback_used'] and choice['raw_origin']=='original_saved'
            assert selected==raw['selected_ids'] and sha(FAILED/'raw'/name)==sha(PARENT/'predicted_area_policy'/name)
        elif index==42:
            assert choice['raw_origin']=='saved_one_shot_replay' and choice['fallback_used'] and not choice['optimality_proven']
            assert sha(FAILED/'raw'/name)==recovery.REPLAY_SHA
        else:assert choice['raw_origin']=='new_remaining_case_solve'
        choices.append(choice)
    # All final IDs are read from sealed records; no recovery/prefix call above.
    out=PA/('common_policy_recovery_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    for name in ('source','raw','choices'):(out/name).mkdir()
    state={'state':'running','stage':'output_only_resume','pid':os.getpid(),'parent_result':str(PARENT),
        'failed_recovery':str(FAILED),'new_solver_calls':0,'reselected_final_ids':False,'test_accessed':False,'R6_accessed':False}
    write_json(out/'status.json',state)
    try:
        for source in (FAILED/'source').iterdir():
            if source.is_file():shutil.copy2(source,out/'source'/source.name)
        shutil.copy2(Path(__file__),out/'source'/Path(__file__).name)
        for name in ('status.json','manifest.json'):
            shutil.copy2(FAILED/name,out/'source'/('initial_failed_'+name))
        for name in ('self_checks.json','selection_inputs_without_annotations.csv','selection_seal.json'):
            shutil.copy2(FAILED/name,out/name)
        for category,recorded in [('raw',seal['raw_sha256']),('choices',seal['choices_sha256'])]:
            for rel,h in recorded.items():
                source=relative(FAILED,rel);destination=out/category/source.name
                shutil.copy2(source,destination);assert sha(destination)==h
        manifest=dict(old)
        manifest.update(output_only_resume_at=stamp(),failed_recovery=str(FAILED),resume_source_sha256=sha(__file__),
            resume_source_hashes=source_hashes,new_solver_calls_in_resume=0,reselected_final_ids=False,
            output_key_fix='Normalize seal path separators only; preserve all raw/choice/protocol bytes.',
            original_selection_seal_sha256=sha(FAILED/'selection_seal.json'))
        write_json(out/'manifest.json',manifest)
        receipt={'at':stamp(),'all_45_final_ids_reused_from_seal':True,'all_raw_and_choice_files_byte_identical':True,
            'new_solver_calls':0,'new_inference_calls':0,'original_solver_calls':2,'sealed_before_original_target_join':True,
            'original_selection_seal_sha256':sha(FAILED/'selection_seal.json'),'resume_before_target_join':True,
            'path_fix_only':True,'final_choices_count':45}
        write_json(out/'resume_receipt.json',receipt)
        # Now join target values solely to evaluate the already-fixed choices.
        annotated=policy.read_csv(PARENT/'unit_predictions.csv')
        assert policy.payload_sha(policy.allowed_selection_rows(annotated))==policy.payload_sha(allowed)
        truth={(int(row['M']),row['id']):float(row['target']) for row in annotated}
        assert len(truth)==len(annotated)==len(allowed)
        refs=policy.read_csv(PARENT/'fixed_budgets.csv')
        prefix_refs={(int(x['M']),int(x['R']),float(x['budget'])):x for x in refs if x['method']=='model_prefix'}
        assert set(prefix_refs)==set(recovery.CASES)
        outcomes=[];prefix_rows=[]
        for choice in choices:
            m,r,b=choice['M'],choice['R'],choice['budget'];ids,area,predicted=data[m];pred=predicted[r]
            target=np.array([truth[(m,i)] for i in ids]);assert np.isfinite(target).all() and np.all((target>=0)&(target<=area+1e-9))
            recovery.close(math.fsum(target),old['domain']['target_m2'])
            final=policy.evaluate_selected(ids,choice['selected_ids'],area,pred,target)
            # Only reconstruct the existing comparison prefix, never choose final IDs.
            baseline_ids=policy.prefix_ids(ids,area,pred,b)
            baseline=policy.evaluate_selected(ids,baseline_ids,area,pred,target)
            prior=prefix_refs[(m,r,b)]
            for field,column in [('selected_count','k'),('actual_area_m2','actual_area'),('captured_target_m2','captured_target'),('capture_fraction','capture_fraction'),('zero_annotation_area_m2','zero_annotation_area')]:
                recovery.close(baseline[field],prior[column])
            assert final['actual_area_m2']<=choice['budget_area_m2']
            recovery.close(final['predicted_objective_m2'],choice['predicted_primal_m2'])
            name=recovery.case_name((m,r,b))
            prefix_rows.append({'M':m,'R':r,'budget':b,'budget_area_m2':choice['budget_area_m2'],**baseline,
                'selected_ids_json':json.dumps(baseline_ids),'strict_original_unit_feasible':baseline['actual_area_m2']<=choice['budget_area_m2']})
            outcomes.append({'M':m,'R':r,'budget':b,'policy_label':choice['policy_label'],'raw_origin':choice['raw_origin'],
                'budget_area_m2':choice['budget_area_m2'],'fallback_used':choice['fallback_used'],'fallback_reasons':';'.join(choice['fallback_reasons']),
                'prefix_tail_trim_count':choice['prefix_tail_trim_count'],'final_status':choice['status'],'optimality_proven':choice['optimality_proven'],
                'final_predicted_upper_m2':choice['recomputed_bound']['upper_m2'],'final_predicted_relative_gap':choice['recomputed_bound']['relative_gap'],
                'bound_anomaly':choice['recomputed_bound']['bound_anomaly'],'bound_roundoff_clamp_m2':choice['recomputed_bound']['roundoff_clamp_m2'],
                'raw_candidate_feasible':choice['raw_candidate_audit']['recorded_feasible_flag'],
                'raw_candidate_budget_violation_m2':choice['raw_candidate_audit']['budget_violation_m2_recomputed'],
                'raw_solver_status':choice['raw_candidate_audit']['raw_solver_status'],'raw_solver_gap':choice['raw_candidate_audit']['raw_solver_gap'],
                **{'recovered_'+k:v for k,v in final.items()},**{'prefix_'+k:v for k,v in baseline.items()},
                'delta_capture_pp':100*(final['capture_fraction']-baseline['capture_fraction']),
                'delta_predicted_objective_m2':final['predicted_objective_m2']-baseline['predicted_objective_m2'],
                'unused_budget_m2':choice['budget_area_m2']-final['actual_area_m2'],'choice_file':'choices/'+name,'choice_sha256':sha(out/'choices'/name)})
        assert len(outcomes)==len(prefix_rows)==45
        write_csv(out/'all_45_policy_outcomes.csv',outcomes);write_csv(out/'all_45_prefix_reference.csv',prefix_rows)
        for name in ('fixed_budgets.csv','decision_curves.csv'):
            shutil.copy2(PARENT/name,out/'source'/('full_original_'+name))
        for path,h in {**old['input_hashes'],**source_hashes}.items():assert sha(path)==h
        for mapping in (seal['choices_sha256'],seal['raw_sha256']):
            for rel,h in mapping.items():assert sha(relative(out,rel))==h
        summary={'state':'completed','stage':'development numerical recovery','run_id':old['run_id'],'policy_label':choices[0]['policy_label'],
            'cases':45,'original_results_reused':42,'saved_single_diagnostic_replay_reused':1,
            'new_remaining_solver_calls_in_original_recovery':2,'new_solver_calls_in_resume':0,
            'all_final_choices_strictly_feasible':True,'all_final_ids_reused_from_original_seal':True,
            'selection_seal_sha256':sha(out/'selection_seal.json'),'original_failed_recovery':str(FAILED),
            'fallback_count':sum(x['fallback_used'] for x in choices),
            'fallback_cases':[{k:x[k] for k in ('M','R','budget','selected_area_m2','predicted_primal_m2','prefix_tail_trim_count','recomputed_bound','raw_candidate_audit')} for x in choices if x['fallback_used']],
            'optimality_proven_within_tolerance':sum(x['optimality_proven'] for x in choices),
            'capture_increased_vs_prefix':sum(x['delta_capture_pp']>1e-8 for x in outcomes),
            'capture_decreased_vs_prefix':sum(x['delta_capture_pp']< -1e-8 for x in outcomes),
            'capture_unchanged_vs_prefix':sum(abs(x['delta_capture_pp'])<=1e-8 for x in outcomes),
            'domain':old['domain'],'protocol_sha256':recovery.PROTOCOL_SHA,'all_prior_failures_preserved':True,
            'all_eight_configs_same_policy_declared':True,'self_checks_passed':True,'test_accessed':False,'R6_accessed':False,
            'no_training_inference_or_geometry':True,'elapsed_resume_seconds':time.time()-started}
        write_json(out/'summary.json',summary)
        lines=['# Numerical policy recovery — completed development output','',
            'The original recovery created and sealed all45 final choices, then failed while writing a table because of a Windows path-separator key mismatch. This separate output-only resume preserves every original raw/choice byte and changes no selected ID, budget or policy. Both failed directories remain intact.',
            '', 'The original recovery reused42 original valid results and one saved single diagnostic replay for case43. Only cases44/45 (M4,R4,budget0.5/1.0) invoked the unchanged selector. This resume made zero solver calls.',
            '',f"All45 final choices are strictly feasible. Fallbacks:{summary['fallback_count']}; recorded optimality within tolerance:{summary['optimality_proven_within_tolerance']}/45. Every fallback is explicitly feasible_fallback_not_proven.",
            '',f"Compared with the complete original prefix reference, annotated capture increased/decreased/was unchanged in {summary['capture_increased_vs_prefix']}/{summary['capture_decreased_vs_prefix']}/{summary['capture_unchanged_vs_prefix']} cases.",
            '', '## Fallback audit','']
        for x in summary['fallback_cases']:
            lines.append(f"- M{x['M']}/R{x['R']}/budget{x['budget']}: saved diagnostic replay candidate excess {x['raw_candidate_audit']['budget_violation_m2_recomputed']:.15g} m²; final area {x['selected_area_m2']:.15g} m²; final predicted objective {x['predicted_primal_m2']:.15g} m²; recomputed gap {x['recomputed_bound']['relative_gap']:.15g}; prefix tail trims {x['prefix_tail_trim_count']}.")
        lines+=['', 'The saved case43 is a diagnostic replay, not proof of the unsaved original solver candidate identity. Candidate IDs were read from solver_candidate_ids, not the rejected raw selected_ids empty set. Its actual budget excess is not attributed to last-bit rounding; the internal cause is unconfirmed.',
            '', 'All45 policy outcomes and the full original prefix reference are included with code/protocol, source hashes, raw audits, selection seal and output-resume receipt. This is numerical robustness in one-scene development evaluation, not a new algorithm or evidence of field efficacy. No test/R6, training, inference or new geometry was used.']
        (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        state.update(state='completed',completed_at=stamp(),summary=summary,
            files_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'status.json'})
        write_json(out/'status.json',state)
        print(json.dumps({'output':str(out),'summary':summary},ensure_ascii=True),flush=True)
    except BaseException as error:
        state.update(state='failed',error=repr(error),traceback=traceback.format_exc());write_json(out/'status.json',state);raise
if __name__=='__main__':main()
