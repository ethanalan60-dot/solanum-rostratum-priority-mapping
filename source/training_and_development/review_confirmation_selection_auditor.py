"""Independent synthetic review; no real confirmation files or solvers."""
from pathlib import Path
from unittest.mock import patch
import copy, datetime, hashlib, importlib.util, json, math, shutil, sys

ROOT=Path(__file__).resolve().parent
CONF=ROOT/'confirmation_v1'
sys.path.insert(0,str(CONF))
import core

def main():
    out=ROOT/'provenance'/datetime.datetime.now().strftime('confirmation_selection_auditor_independent_%Y%m%d_%H%M%S_%f');out.mkdir()
    snapshot=out/'audit_under_review.py';shutil.copyfile(CONF/'audit_selections.py',snapshot)
    spec=importlib.util.spec_from_file_location('independent_selection_subject',snapshot);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    checks=[];gaps=[]
    def check(name,value):
        assert value,name;checks.append({'name':name,'passed':True})
    def rejects(fn):
        try:fn()
        except (ValueError,AssertionError,KeyError,TypeError,OSError):return True
        return False
    check('no_scientific_import', 'numpy' not in sys.modules and 'torch' not in sys.modules and 'scipy' not in sys.modules)
    check('missing_execution_grant_rejected',rejects(lambda:a.execute(out/'absent.json')))
    check('no_science_after_gate_refusal','numpy' not in sys.modules)
    check('zero_prediction_bound_defined',a.independent_bound(0.,0.,0.,0.,0.,True,False)['updated_relative_gap']==0.)
    check('nonfinite_retained_downgraded',a.independent_bound(.1,.2,1.,float('nan'),float('inf'),True,False)['updated_proven_flag'] is False)
    check('adverse_feasible_prefix_blocks_certification',not a.independent_bound(.1,.3,1.,.1,.1,True,False)['updated_proven_flag'])
    check('duplicate_ids_rejected',rejects(lambda:a.subset(['a','a'],{'a':(1.,.5)})))
    check('unknown_ids_rejected',rejects(lambda:a.subset(['b'],{'a':(1.,.5)})))
    check('nonfinite_arithmetic_rejected',rejects(lambda:a.close(float('nan'),1.)))
    file=out/'watch.txt';file.write_text('first');w=a.Watch();w.add(file);file.write_text('other')
    check('begin_end_drift_rejected',rejects(w.finish))
    source=ROOT/'provenance/confirmation_selection_auditor_checks_20260916_111244_277339/synthetic/stages/local_select_01'
    assert source.exists()
    fake=out/'fixture';(fake/'stages/local_geometry').mkdir(parents=True);(fake/'stages/predictions_seal').mkdir()
    for p in ('FREEZE.json','stages/local_geometry/metadata.json','stages/predictions_seal/all_predictions.json'):core.write(fake/p,{'SYNTHETIC_ONLY':True})
    shutil.copyfile(CONF/'local_management.py',fake/'local_management.py')
    target=fake/'stages/local_select_01';shutil.copytree(source,target)
    context=dict(run_id='SYNTHETIC_ONLY',freeze_sha256=core.sha(fake/'FREEZE.json'),all_predictions_sha256=core.sha(fake/'stages/predictions_seal/all_predictions.json'),geometry_metadata_sha256=core.sha(fake/'stages/local_geometry/metadata.json'),checkpoint_sha256='c'*64)
    allowed=a.rows(target/'policy/selection_inputs_without_annotations.csv')
    ap={(int(x['M']),x['id']):[float(x['area']),*[float(x[f'predicted_R{r}']) for r in a.RS]] for x in allowed}
    mapindex=out/'synthetic_map_index.json';core.write(mapindex,[]);pred={'map_index':core.binding(mapindex)}
    def reseal():
        inner=core.read(target/'policy/selection_seal.json');inner['context']=context
        for cat in ('raw','choices','solver_raw'):
            inner[cat+'_sha256']={name:core.sha(target/'policy'/name) for name in inner[cat+'_sha256']}
        core.write(target/'policy/selection_seal.json',inner)
        bound=core.read(target/'bound_audit.json')
        for row in bound:row['choice']=core.binding(target/'policy/choices'/Path(row['choice']['path']).name)
        core.write(target/'bound_audit.json',bound)
        seal=core.read(target/'selection_seal.json');seal.update(context);seal['implementation_sha256']=core.sha(fake/'local_management.py');seal['files']={n:core.binding(target/n) for n in seal['files']};seal['A_P_only_inputs']=core.binding(target/'policy/selection_inputs_without_annotations.csv');core.write(target/'selection_seal.json',seal)
        core.write(target/'artifact_manifest.json',core.stage_manifest(target))
    def audit():
        with patch.object(a,'HERE',fake),patch.object(a,'local_integrals',return_value=ap):
            return a.audit_local({'run_id':context['run_id'],'checkpoint_sha256':context['checkpoint_sha256']},1,pred,{'area_m2':10.},a.Watch())
    reseal();check('complete120_local_original_accepted',audit()['case_count']==120)
    name='M1_R1_B010.json';solverfile=target/'policy/solver_raw'/name;choicefile=target/'policy/choices'/name;boundfile=target/'bound_audit.json'
    originals={p:p.read_bytes() for p in (solverfile,choicefile,boundfile,target/'rankings.json')}
    def restore():
        for p,b in originals.items():p.write_bytes(b)
        reseal()
    def gap(name,modify):
        restore();modify();reseal();accepted=not rejects(audit)
        evidence={'name':name,'incorrect_self_consistently_resealed_fixture_accepted':accepted}
        (out/(name+'.json')).write_text(json.dumps(evidence,indent=2),encoding='utf-8')
        if accepted:gaps.append(evidence)
        else:checks.append({'name':name+'_rejected','passed':True})
    def change_solver_x():
        s=core.read(solverfile);s['result']['x']=[0.]*len(s['result']['x']);core.write(solverfile,s)
    gap('solver_x_not_matching_saved_candidate',change_solver_x)
    def change_solver_dual():
        s=core.read(solverfile);s['result']['mip_dual_bound']=-123.;s['result']['fun']=-456.;s['result']['status']=4;core.write(solverfile,s)
    gap('solver_dual_status_not_matching_selector',change_solver_dual)
    def change_choice_binding():
        c=core.read(choicefile);c.update(raw_sha256='0'*64,selection_input_sha256='1'*64,recovery_source_sha256='2'*64,raw_file='raw/foreign.json');core.write(choicefile,c)
    gap('choice_raw_source_input_hashes_inconsistent',change_choice_binding)
    def change_final_ids():
        c=core.read(choicefile);c.update(selected_ids=[],selected_area_m2=0.,predicted_primal_m2=0.);core.write(choicefile,c)
        b=core.read(boundfile);b[0]['selected_ids']=[];b[0].update(a.independent_bound(0.,.9,2.,.9,.9,True,False));core.write(boundfile,b)
    gap('feasible_solver_candidate_changed_to_empty',change_final_ids)
    restore();rank=core.read(target/'rankings.json');rank.pop();core.write(target/'rankings.json',rank);reseal()
    check('missing_rank_rejected_after_consistent_reseal',rejects(audit));restore()
    with patch.object(a,'HERE',fake),patch.object(a,'local_integrals',return_value=ap):
        check('wrong_checkpoint_identity_rejected',rejects(lambda:a.audit_local({'run_id':context['run_id'],'checkpoint_sha256':'9'*64},1,pred,{'area_m2':10.},a.Watch())))
    if hasattr(a,'solver_to_raw'):
        units={'a':(1.,.9),'b':(3.,.6),'c':(6.,.5)}
        for tag,x,candidate,err,fall in [
            ('no_incumbent',None,[],None,True),
            ('nonfinite_incumbent',[{'nonfinite_float':'NaN'},0.,0.],[],None,True),
            ('fractional_incumbent',[.5,0.,0.],[],.5,True),
            ('strict_overbudget',[0.,1.,0.],['b'],0.,True),
            ('near_integer',[1.+1e-7,0.,0.],['a'],1e-7,False),
            ('valid_empty',[0.,0.,0.],[],0.,False)]:
            backend={'x':x,'status':1,'message':'ARTIFICIAL_NO_SOLVER','mip_dual_bound':-1.,'fun':None,'mip_gap':None,'mip_node_count':0}
            raw=a.solver_to_raw(units,.1,backend)
            check(tag+'_candidate',raw['solver_candidate_ids']==candidate)
            check(tag+'_integrality',raw['candidate_integrality_error'] is None if err is None else math.isclose(raw['candidate_integrality_error'],err,rel_tol=1e-8,abs_tol=1e-14))
            recovered=a.recover_raw(units,.1,raw)
            check(tag+'_fallback',recovered['fallback_used']==fall and recovered['selected_ids']==(['a'] if fall else candidate) and recovered['optimality_proven'] is False)
        check('malformed_solver_vector_rejected',rejects(lambda:a.solver_to_raw(units,.1,{'x':[1.],'status':1,'message':'bad'})))
        check('malformed_IEEE_marker_rejected',rejects(lambda:a.decode({'nonfinite_float':'undefined'})))
    import numpy as np
    p=np.zeros((256,256),np.float32);exact={}
    for c,position in enumerate((1,63,64,4095)):
        cell=np.full((64,64),np.float32(2**-55));cell.flat[0]=.5;cell.flat[position]=.5;p[:64,c*64:(c+1)*64]=cell
        exact[f'f:g04:r00:c{c:02d}']=math.fsum(map(float,cell.ravel()))/65536
    actual=a.grid_values(p,'f',4)
    orders={'expected':sorted(exact,key=lambda k:(-exact[k],k)),'observed':sorted(exact,key=lambda k:(-actual[k],k)),'exact':exact,'numpy':{k:actual[k] for k in exact}}
    core.write(out/'floating_tie_evidence.json',orders)
    if orders['expected']!=orders['observed']:gaps.append({'name':'numpy_reduction_false_rejects_valid_fsum_tie','details':'floating_tie_evidence.json'})
    else:check('extreme_exact_ties_match_producer',True)
    check('no_torch_or_scipy_imports','torch' not in sys.modules and 'scipy' not in sys.modules)
    receipt={'passed':not gaps,'scope':'confirmation_selection_auditor_independent_preparation','implementation_sha256':core.sha(snapshot),'live_source_unchanged_since_copy':core.sha(CONF/'audit_selections.py')==core.sha(snapshot),'checker_sha256':core.sha(__file__),'checks':checks,'blocking_gaps':gaps,'raw_fixture_source':str(source),'synthetic_only':True,'actual_reserved_arrays_opened':0,'model_calls':0,'solver_calls':0,'limitations':['Local file semantics tested with independently supplied synthetic A/P, no actual geometry integration.','Existing synthetic solver returns copied; no backend or selector invoked.']}
    core.write(out/'review.json',receipt)
    core.write(out/'artifact_manifest.json',core.stage_manifest(out))
    print(json.dumps({'directory':str(out),'passed':receipt['passed'],'gap_count':len(gaps),'checks':len(checks),'receipt_sha256':core.sha(out/'review.json')},ensure_ascii=False))
if __name__=='__main__':main()
