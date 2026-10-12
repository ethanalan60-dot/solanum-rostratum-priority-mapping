"""Independent CPU arithmetic/hash audit; never call a solver or model."""
from pathlib import Path
import argparse, csv, datetime, hashlib, importlib.util, json, math, sys

ROOT = Path(__file__).resolve().parent
PA = ROOT / 'precision_agriculture'
SEALED = PA / 'common_policy_recovery_20260916_034644_420249'
CASES = [(m,r,b) for m in (1,2,4) for r in (1,2,4) for b in (.1,.2,.3,.5,1.)]
PROTOCOL_SHA = '97b12d1178dcba1357860c4b2517a041ca3aaf456eb8a575c2c931db5579cfdd'
REPLAY_SHA = 'dcbf0df0183e9e9576984000603a0d7bca094639562d5934082b0781b2565e05'

def read(p): return json.loads(Path(p).read_text('utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-10),(a,b)
def name(case): return 'M%d_R%d_B%03d.json'%(case[0],case[1],round(case[2]*100))
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
def write_csv(p,rs):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def unit_groups(rs):
    result={}
    for m in (1,2,4):
        subset=[x for x in rs if int(x['M'])==m]
        assert len({x['id'] for x in subset})==len(subset)
        result[m]={x['id']:x for x in subset}
    return result
def prefix(unit,r,b):
    ids=sorted(unit,key=lambda i:(-float(unit[i]['predicted_R%d'%r])/float(unit[i]['area']),i))
    limit=b*math.fsum(float(x['area']) for x in unit.values())
    # Independent sequential binary64 cumulative sum reproduces frozen np.cumsum.
    cumulative=0.; selected=[]
    for k,i in enumerate(ids):
        cumulative+=float(unit[i]['area'])
        if k==len(ids)-1: cumulative=math.fsum(float(x['area']) for x in unit.values())
        if cumulative>limit: break
        selected.append(i)
    return selected
def sums(unit,ids,r):
    assert len(set(ids))==len(ids) and all(i in unit for i in ids)
    return (math.fsum(float(unit[i]['area']) for i in ids),
            math.fsum(float(unit[i]['predicted_R%d'%r]) for i in ids))
def expected_bound(raw,primal,total):
    u=raw.get('raw_solver_dual_upper_m2'); tol=max(1e-10,1e-10*total)
    if not isinstance(u,(int,float)) or not math.isfinite(u) or u<primal-tol: upper=total
    else: upper=max(primal,min(total,u))
    return upper,(upper-primal)/max(abs(primal),1e-30) if upper else 0.

def audit_selections():
    manifest=read(SEALED/'manifest.json');seal=read(SEALED/'selection_seal.json');state=read(SEALED/'status.json')
    assert state['completed_choices']==45 and state['solver_calls']==2
    assert seal['new_solver_calls']==[[4,4,.5],[4,4,1.]]
    assert seal['case_order']==[list(x) for x in CASES]
    assert seal['target_join_started'] is False and seal['all_45_final_choices_saved']
    assert datetime.datetime.fromisoformat(seal['sealed_at'])<datetime.datetime.fromisoformat(state['labels_join_started_at'])
    assert sha(SEALED/'selection_seal.json')==state['selection_seal_sha256']
    assert sha(SEALED/'source/recover_common_policy.py')==seal['implementation_sha256']
    assert sha(SEALED/'source/RM_numerical_recovery_protocol_v1.json')==PROTOCOL_SHA==seal['protocol_sha256']
    for field in ('choices_sha256','raw_sha256'):
        assert len(seal[field])==45
        for p,h in seal[field].items(): assert sha(SEALED/p)==h
    assert sha(SEALED/'selection_inputs_without_annotations.csv')==seal['allowed_inputs_sha256']
    checked_hashes=0;archived_source_substitutions=[]
    for p,h in manifest['input_hashes'].items():
        path=Path(p)
        if path==ROOT/'recover_common_policy.py':
            assert sha(SEALED/'source/recover_common_policy.py')==h
            archived_source_substitutions.append(str(path))
        else: assert sha(path)==h,p
        checked_hashes+=1
    parent=Path(manifest['parent_result']); allowed=rows(SEALED/'selection_inputs_without_annotations.csv')
    assert set(allowed[0])=={'M','id','area','predicted_R1','predicted_R2','predicted_R4'}
    groups=unit_groups(allowed);audits=[]
    for index,case in enumerate(CASES):
        m,r,b=case;n=name(case);c=read(SEALED/'choices'/n);raw=read(SEALED/'raw'/n);units=groups[m]
        assert (c['M'],c['R'],c['budget'])==case
        assert c['raw_sha256']==sha(SEALED/'raw'/n)
        assert c['recovery_source_sha256']==seal['implementation_sha256']
        assert c['protocol_sha256']==PROTOCOL_SHA
        a,p=sums(units,c['selected_ids'],r);totalA=math.fsum(float(x['area']) for x in units.values())
        totalP=math.fsum(float(x['predicted_R%d'%r]) for x in units.values());limit=b*totalA
        assert a<=limit and c['original_area_tolerance_m2']==0. and c['strict_original_unit_feasible']
        close(a,c['selected_area_m2']);close(p,c['predicted_primal_m2']);close(limit,c['budget_area_m2'])
        rawA,rawP=sums(units,raw['solver_candidate_ids'],r)
        close(rawA,c['raw_candidate_audit']['area_m2_recomputed']);close(rawP,c['raw_candidate_audit']['predicted_m2_recomputed'])
        close(max(0,rawA-limit),c['raw_candidate_audit']['budget_violation_m2_recomputed'])
        upper,gap=expected_bound(raw,p,totalP)
        close(upper,c['recomputed_bound']['upper_m2']);close(gap,c['recomputed_bound']['relative_gap'])
        assert c['recomputed_bound']['bound_anomaly'] is None
        if index<42:
            assert sha(parent/'predicted_area_policy'/n)==sha(SEALED/'raw'/n)
            assert c['raw_origin']=='original_saved' and not c['fallback_used']
            assert c['selected_ids']==raw['selected_ids']==raw['solver_candidate_ids']
        elif index==42:
            assert sha(SEALED/'raw'/n)==REPLAY_SHA and c['raw_origin']=='saved_one_shot_replay'
            assert rawA>limit and c['fallback_used'] and c['optimality_proven'] is False
            assert c['status']=='feasible_fallback_not_proven'
            pref=prefix(units,r,b);before=pref.copy()
            while sums(units,pref,r)[0]>limit: pref.pop()
            assert pref==c['selected_ids'] and before==c['prefix_before_tail_trim_ids']
            assert len(before)-len(pref)==c['prefix_tail_trim_count']==0
        else:
            assert c['raw_origin']=='new_remaining_case_solve'
            assert not c['fallback_used'] and c['selected_ids']==raw['selected_ids']
        if c['optimality_proven']:
            assert not c['fallback_used'] and raw['solver_status']==0 and gap<=1e-8+1e-10
        audits.append({'M':m,'R':r,'budget':b,'count':len(c['selected_ids']),'area':a,'predicted':p,
            'limit':limit,'unused':limit-a,'upper':upper,'relative_gap':gap,'fallback':c['fallback_used'],
            'optimality_proven':c['optimality_proven'],'raw_origin':c['raw_origin']})
    assert sum(x['fallback'] for x in audits)==1
    return manifest,seal,groups,audits,{'checked_manifest_input_hashes':checked_hashes,
        'execution_source_verified_from_archive':archived_source_substitutions}

def audit_r18_equivalence():
    # Invoke only the pure recovery function from the actual frozen execution source.
    spec=importlib.util.spec_from_file_location('frozen_recovery_for_audit',SEALED/'source/recover_common_policy.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def forbidden(*a,**kw): raise AssertionError('Independent audit must never call a solver')
    module.policy.select_predicted_area=forbidden
    import numpy as np
    results=[]
    for variant,suffix in [('hard_seg','031336_012442'),('soft_seg','031402_620370')]:
        directory=PA/('common_RM_dev_common_v2_lr1e4_resnet18_%s_L256_s17_20260916_%s'%(variant,suffix))
        allowed=[{k:v for k,v in x.items() if k in ('M','id','area','predicted_R1','predicted_R2','predicted_R4')} for x in rows(directory/'unit_predictions.csv')]
        groups=unit_groups(allowed)
        for m,r,b in CASES:
            unit=groups[m];ids=list(unit);a=np.array([float(unit[i]['area']) for i in ids]);p=np.array([float(unit[i]['predicted_R%d'%r]) for i in ids])
            raw=read(directory/'predicted_area_policy'/name((m,r,b)));got=module.recover_choice(ids,a,p,b,raw)
            assert not got['fallback_used'] and got['selected_ids']==raw['selected_ids']
            assert got['selected_area_m2']==raw['selected_area_m2']
            assert got['predicted_primal_m2']==raw['predicted_primal_m2']
            assert got['recomputed_bound']['upper_m2']==raw['predicted_dual_upper_m2']
            assert got['recomputed_bound']['relative_gap']==raw['predicted_relative_gap']
            assert got['optimality_proven']==raw['optimality_proven_within_tolerance']
            results.append({'model':variant,'M':m,'R':r,'budget':b,'exact_ids_equal':True,'exact_primal_equal':True,
                'exact_upper_and_gap_equal':True,'fallback':False,'raw_sha256':sha(directory/'predicted_area_policy'/name((m,r,b)))})
    assert len(results)==90
    return results

def audit_completed(directory,manifest,seal):
    directory=Path(directory).resolve();status=read(directory/'status.json');summary=read(directory/'summary.json')
    newmanifest=read(directory/'manifest.json');receipt=read(directory/'resume_receipt.json')
    assert status['state']==summary['state']=='completed' and status['new_solver_calls']==0
    assert not status['reselected_final_ids'] and receipt['all_45_final_ids_reused_from_seal']
    for p,h in status['files_sha256'].items(): assert sha(directory/p)==h,p
    for field in ('choices_sha256','raw_sha256'):
        for p,h in seal[field].items(): assert sha(directory/p)==h
    assert sha(directory/'selection_seal.json')==sha(SEALED/'selection_seal.json')
    assert sha(directory/'source/initial_failed_status.json')==sha(SEALED/'status.json')
    assert sha(directory/'source/initial_failed_manifest.json')==sha(SEALED/'manifest.json')
    assert sha(directory/'source/resume_recovered_policy_tables.py')==newmanifest['resume_source_sha256']
    assert sha(ROOT/'resume_recovered_policy_tables.py')==newmanifest['resume_source_sha256']
    for path,digest in newmanifest['resume_source_hashes'].items(): assert sha(path)==digest
    # All prior selections have been independently verified before annotation parsing.
    parent=Path(manifest['parent_result']);annotated=rows(parent/'unit_predictions.csv');groups=unit_groups(annotated)
    allowed=rows(SEALED/'selection_inputs_without_annotations.csv')
    assert [{k:x[k] for k in allowed[0]} for x in annotated]==allowed
    original={(int(x['M']),int(x['R']),float(x['budget'])):x for x in rows(parent/'fixed_budgets.csv') if x['method']=='model_prefix'}
    outcomes=rows(directory/'all_45_policy_outcomes.csv');prefixes=rows(directory/'all_45_prefix_reference.csv')
    def indexed(rs):
        result={(int(x['M']),int(x['R']),float(x['budget'])):x for x in rs}
        assert len(rs)==len(result)==45 and set(result)==set(CASES)
        return result
    outcomes=indexed(outcomes);prefixes=indexed(prefixes);checks=[]
    for case in CASES:
        m,r,b=case;units=groups[m];choice=read(directory/'choices'/name(case))
        out=outcomes[case];prefrow=prefixes[case];pref=prefix(units,r,b)
        assert json.loads(prefrow['selected_ids_json'])==pref
        totalA=math.fsum(float(x['area']) for x in units.values());totalT=math.fsum(float(x['target']) for x in units.values())
        close(totalA,manifest['domain']['area_m2']);close(totalT,manifest['domain']['target_m2'])
        measurements={}
        for label,ids in [('recovered',choice['selected_ids']),('prefix',pref)]:
            a,p=sums(units,ids,r);t=math.fsum(float(units[i]['target']) for i in ids)
            fields={'selected_count':len(ids),'actual_area_m2':a,'actual_area_fraction':a/totalA,
                'captured_target_m2':t,'capture_fraction':t/totalT,'missed_target_m2':totalT-t,
                'zero_annotation_area_m2':a-t,
                'selected_zero_target_unit_area_m2':math.fsum(float(units[i]['area']) for i in ids if float(units[i]['target'])==0.),
                'predicted_objective_m2':p}
            for k,v in fields.items():
                close(v,out[label+'_'+k])
                if label=='prefix':close(v,prefrow[k])
            measurements[label]=fields
        before=measurements['prefix'];after=measurements['recovered']
        for field,old in [('selected_count','k'),('actual_area_m2','actual_area'),('captured_target_m2','captured_target'),('capture_fraction','capture_fraction'),('zero_annotation_area_m2','zero_annotation_area')]:
            close(before[field],original[case][old])
        delta=100*(after['capture_fraction']-before['capture_fraction'])
        close(delta,out['delta_capture_pp']);close(after['predicted_objective_m2']-before['predicted_objective_m2'],out['delta_predicted_objective_m2'])
        close(b*totalA-after['actual_area_m2'],out['unused_budget_m2'])
        close(choice['recomputed_bound']['upper_m2'],out['final_predicted_upper_m2'])
        close(choice['recomputed_bound']['relative_gap'],out['final_predicted_relative_gap'])
        assert (out['optimality_proven']=='True')==choice['optimality_proven']
        assert (out['fallback_used']=='True')==choice['fallback_used']
        assert out['choice_sha256']==sha(directory/out['choice_file'])
        assert after['actual_area_m2']<=b*totalA
        checks.append({'M':m,'R':r,'budget':b,'recovered_A':after['actual_area_m2'],'recovered_T':after['captured_target_m2'],
            'recovered_P':after['predicted_objective_m2'],'prefix_A':before['actual_area_m2'],'prefix_T':before['captured_target_m2'],
            'prefix_P':before['predicted_objective_m2'],'delta_capture_pp':delta,'all_fields_recomputed':True})
    counts=[sum(x['delta_capture_pp']>1e-8 for x in checks),sum(x['delta_capture_pp']< -1e-8 for x in checks),sum(abs(x['delta_capture_pp'])<=1e-8 for x in checks)]
    assert counts==[summary[k] for k in ('capture_increased_vs_prefix','capture_decreased_vs_prefix','capture_unchanged_vs_prefix')]==[1,0,44]
    assert summary['fallback_count']==1 and summary['optimality_proven_within_tolerance']==32
    return {'directory':str(directory),'state':'passed','status_product_hashes_checked':len(status['files_sha256']),
        'resume_manifest_source_hashes_checked':len(newmanifest['resume_source_hashes']),
        'resume_source_sha256':newmanifest['resume_source_sha256'],'summary_sha256':sha(directory/'summary.json'),
        'outcomes_sha256':sha(directory/'all_45_policy_outcomes.csv'),'prefix_reference_sha256':sha(directory/'all_45_prefix_reference.csv'),
        'all_90_raw_choice_files_byte_identical_to_prior_seal':True,'all_45_final_and_prefix_ID_A_T_P_and_auxiliary_fields_recomputed':True,
        'all_45_original_prefix_reproduced':True,'capture_increase_decrease_unchanged':counts,
        'new_solver_calls_in_resume':0,'saved_case43_is_diagnostic_replay_not_asserted_original_unsaved_candidate_identity':True},checks

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--completed');args=parser.parse_args()
    out=PA/('policy_recovery_independent_review_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    manifest,seal,groups,audits,extra=audit_selections();equivalence=audit_r18_equivalence()
    write_csv(out/'all_45_selection_checks.csv',audits);write_csv(out/'r18_90_policy_equivalence.csv',equivalence)
    report={'state':'passed','stage':'sealed selection and R18 equivalence','sealed_directory':str(SEALED),
        'seal_sha256':sha(SEALED/'selection_seal.json'),'protocol_sha256':PROTOCOL_SHA,
        'executed_recovery_source_sha256':seal['implementation_sha256'],'independent_auditor_source_sha256':sha(__file__),
        'strict_feasible_final_cases':45,'fallback_count':1,'first_42_original_raw_byte_identity':True,
        'replay_43_byte_identity':True,'last_two_raw_origin':'new remaining cases only, recorded and source audited',
        'within_recorded_gap_cases':sum(x['optimality_proven'] for x in audits),
        'R18_equivalence_cases':90,'R18_fallback_count':0,'R18_exact_ID_A_P_upper_gap_proven_flags_equal':True,
        'independent_solver_calls':0,'independent_model_calls':0,'test_or_R6_accessed':False,
        'execution_output_failure':'Windows slash mismatch occurred after all choices sealed; selected choices intact; annotation output pending',
        'limits':['Raw solver upper bounds are checked for arithmetic consistency, not independently certified by solving.',
            'Seal-before-label ordering is supported by frozen source control flow and timestamps, not a separate OS access trace.',
            'This is one-scene validation development; it provides no field independence or application efficacy claim.'],**extra}
    if args.completed:
        completion,checks=audit_completed(args.completed,manifest,seal)
        report['completed_output_review']=completion;report['stage']='complete source, sealed selections, R18 equivalence and all 45 outcome tables'
        report['execution_output_failure']='Original output-only path-key failure preserved; zero-solver resume completed and independently verified'
        write_csv(out/'all_45_annotated_outcome_checks.csv',checks)
    write(out/'audit.json',report)
    (out/'review.md').write_text('# Independent numerical recovery audit\n\nPassed the 45 sealed choices and 90 previous R18 policy equivalence checks.\n\n'
        '- All 45 final sets satisfy the unchanged original budget using math.fsum and zero tolerance.\n'
        '- Original 42 raw results and replay 43 are byte-identical. Only cases 44/45 are recorded new solves; this audit performs no solve.\n'
        '- Case M4/R4/B0.3 uses the original 33-unit prefix: A=429.58867848863775 m², P=28.231939789952396 m², upper=28.242141703576955 m²; gap=0.00036136070353161764. It is feasible_fallback_not_proven.\n'
        '- Passing all 90 previous R18 raw dictionaries through the frozen pure recovery function preserves IDs, area, primal, upper, gap and proven flags exactly; fallback count is zero.\n'
        '- The original output failure is a Windows path-key mismatch after selection sealing. Its failed directory and source remain preserved.\n\n'
        'Raw bounds were not re-solved or independently mathematically certified. Label ordering was checked in source and timestamps. No test/R6, model or GPU access.\n','utf-8')
    if args.completed:
        with (out/'review.md').open('a',encoding='utf8') as f:
            f.write('\n## Completed output-only resume\n\n'+
                'All 45 final and prefix ID/A/T/P records and auxiliary fields were independently recomputed; all original prefixes reproduced. All 90 raw/choice files and original selection seal are byte-identical. Resume source, input hashes and every final product hash passed. The resume prohibits solver calls and never reselects final IDs.\n\n'+
                'Complete budget results: target capture improved in 1 case, declined in 0, and was unchanged in 44. All 45 cases are retained. Recorded optimality within tolerance is 32/45; the one fallback remains not proven. Case 43 is explicitly a saved diagnostic replay, not proof of identity to the unsaved original solver candidate.\n\n'+
                'Final directory: '+str(Path(args.completed).resolve())+'\n')
    print(json.dumps({'output':str(out),'report':report},ensure_ascii=True))

if __name__=='__main__':main()
