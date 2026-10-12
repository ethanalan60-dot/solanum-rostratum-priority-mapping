"""Focused artificial order checks; no actual confirmation audit or stage access."""
from pathlib import Path
import ast,copy,datetime,difflib,hashlib,json,runpy

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'review_confirmation_local_outputs.py'
NEW=ROOT/'review_confirmation_local_outputs_v2.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert sha(OLD)=='2dbf4e09430aae4ecef8ef60d1ba65d70da9080369039777ec8cf630cc38f410'
    texts=[p.read_text(encoding='utf-8-sig') for p in (OLD,NEW)]
    trees=[ast.parse(t) for t in texts]
    functions=[{n.name:n for n in t.body if isinstance(n,ast.FunctionDef)} for t in trees]
    changed=[n for n in functions[0] if ast.dump(functions[0][n],include_attributes=False)!=ast.dump(functions[1][n],include_attributes=False)]
    assert changed==['verify_models','gate','execute']
    checks=['original_frozen_auditor_exact_hash','only_order_view_and_provenance_functions_changed']
    subject=runpy.run_path(str(NEW),run_name='synthetic_local_audit_v2')
    # Compile the exact changed per-M view statements, stopping before numeric work.
    for_m=next(n for n in ast.walk(functions[1]['verify_models']) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='m')
    view=[]
    for statement in for_m.body:
        view.append(copy.deepcopy(statement))
        if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call) and any(isinstance(x,ast.Constant) and x.value=='ASCII input order' for x in statement.value.args):break
    assert len(view)==7
    def canonical(allowed,A):
        ns=dict(subject,allowed=allowed,unit={1:(A,{i:0. for i in A})},m=1)
        exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(view),type_ignores=[])),'exact_view_from_verify_models','exec'),ns)
        return ns['data'],ns['ids']
    def rejected(call):
        try:call()
        except (ValueError,AssertionError):return True
        return False
    values={'M1_E1_N0':.25,'M1_E2_N0':.10,'M1_E10_N0':.50}
    A={i:1. for i in values}
    rows=[{'M':'1','id':i,'area':'1',**{f'predicted_R{r}':str(p) for r in (1,2,4,6)}} for i,p in values.items()]
    original=copy.deepcopy(rows);data,ids=canonical(rows,A)
    assert ids==['M1_E10_N0','M1_E1_N0','M1_E2_N0'] and rows==original
    checks.append('numeric_coordinate_order_canonicalized_in_memory_without_mutating_rows')
    assert canonical(list(reversed(rows)),A)[1]==ids and canonical(data,A)[1]==ids
    checks.append('reversed_and_already_ASCII_rows_match')
    for name,bad in [('duplicate',[rows[0],rows[0],rows[2]]),('missing',rows[:2]),('unknown',[rows[0],rows[1],dict(rows[2],id='M1_E99_N0')])]:
        assert rejected(lambda:canonical(bad,A));checks.append(name+'_ID_rejected')
    P={x['id']:float(x['predicted_R1']) for x in data};selected=[ids[0]];total=sum(P.values());final=P[ids[0]];budget=.5
    raw=dict(solver_candidate_ids=selected,candidate_integrality_error=0.,candidate_area_m2=1.,candidate_predicted_m2=final,
             candidate_feasible_in_original_units=True,selected_ids=selected,raw_solver_dual_upper_m2=final,solver_status=0)
    choice=dict(selected_ids=selected,fallback_used=False,selected_area_m2=1.,predicted_primal_m2=final,
        strict_original_unit_feasible=True,original_area_tolerance_m2=0.,budget_area_m2=1.5,total_predicted_m2=total,total_area_m2=3.,
        recomputed_bound=dict(upper_m2=final,primal_m2=final,absolute_gap_m2=0.,relative_gap=0.,bound_anomaly=None),optimality_proven=True,
        raw_candidate_audit=dict(candidate_ids=selected,accepted_for_passthrough=True,recorded_feasible_flag=True,raw_solver_status=0,raw_solver_upper_m2=final,raw_selected_ids=selected))
    backend=dict(actual_solver_calls_for_case=1,persisted_before_selector_audit=True,solver_options_passed=dict(threads=1,mip_rel_gap=1e-8),
                 result=dict(status=0,mip_dual_bound=-final/total,x=[1.,0.,0.]))
    values_out=subject['verify_raw_choice'](raw,choice,backend,ids,A,P,budget,subject['Checks']())
    assert values_out==(final,final,True,False);checks.append('unchanged_backend_reconstruction_accepts_ASCII_solver_columns')
    assert rejected(lambda:subject['verify_raw_choice'](raw,choice,backend,[x['id'] for x in rows],A,P,budget,subject['Checks']()))
    checks.append('merely_deleting_old_assertion_without_reordering_would_still_misalign_backend')
    assert ast.dump(functions[0]['verify_references'],include_attributes=False)==ast.dump(functions[1]['verify_references'],include_attributes=False)
    checks.append('reference_checks_and_ASCII_reference_semantics_unchanged')
    assert not any(n in __import__('sys').modules for n in ('torch','numpy','scipy','rasterio'))
    checks.append('no_scientific_runtime_imported')
    out=ROOT/'provenance'/('local_output_auditor_v2_preparation_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    (out/'source.diff').write_text(''.join(difflib.unified_diff(texts[0].splitlines(keepends=True),texts[1].splitlines(keepends=True),fromfile=OLD.name,tofile=NEW.name)),encoding='utf-8')
    (out/'synthetic_fixture.json').write_text(json.dumps({'allowed_numeric_order':rows,'canonical_ids':ids,'raw':raw,'choice':choice,'backend':backend},indent=2)+'\n',encoding='utf-8')
    receipt={'passed':True,'scope':'Focused synthetic preparation of versioned local output reviewer; no real execution authorization',
        'source':str(NEW),'implementation_sha256':sha(NEW),'base_source':str(OLD),'base_sha256':sha(OLD),'checker_sha256':sha(Path(__file__)),
        'checks':checks,'check_count':len(checks),'changed_functions':changed,
        'unchanged_function_count':len(functions[0])-len(changed),'numeric_functions_and_tolerances_unchanged':True,
        'actual_confirmation_data_read':False,'actual_audit_run':False,'stage_claim_or_freeze_modified':False,'model_calls':0,'solver_calls':0,
        'requires_separate_root_review_and_manual_recovery':True,
        'artifacts':{p.name:sha(p) for p in out.iterdir()}}
    (out/'review.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'source_sha256':sha(NEW),'checks':len(checks),'receipt':str(out/'review.json'),'receipt_sha256':sha(out/'review.json')},ensure_ascii=False))

if __name__=='__main__':main()
