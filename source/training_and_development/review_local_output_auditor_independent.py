"""Independent reviewer of the local saved-output auditor; artificial artifacts only."""
from pathlib import Path
from fractions import Fraction
from unittest.mock import patch
import datetime,importlib.util,json,math,shutil,sys
ROOT=Path(__file__).resolve().parent;CONF=ROOT/'confirmation_v1';sys.path.insert(0,str(CONF))
import core

def main():
    out=ROOT/'provenance'/datetime.datetime.now().strftime('confirmation_local_auditor_independent_%Y%m%d_%H%M%S_%f');out.mkdir()
    source=ROOT/'review_confirmation_local_outputs.py';snapshot=out/'subject.py';shutil.copyfile(source,snapshot)
    spec=importlib.util.spec_from_file_location('local_output_subject',snapshot);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    checks=[];gaps=[]
    def yes(tag,v=True):assert v,tag;checks.append(tag)
    def reject(fn):
        try:fn()
        except (ValueError,AssertionError,KeyError,OSError,TypeError):return True
        return False
    yes('no_model_solver_or_arrays_on_import',not any(x in sys.modules for x in ('numpy','torch','scipy')))
    with patch.object(r,'HOME',out/'missing'):
        yes('absent_grant_refuses_before_array_loading',reject(lambda:r.gate(out/'no_grant',out/'no_choices')))
    yes('rejected_gate_has_no_science_import',not any(x in sys.modules for x in ('numpy','torch','scipy')))
    A={'a':.125,'b':.375,'c':.5};T={'a':.025,'b':.225,'c':.05};P={'a':.0875,'b':.15,'c':.15}
    selected,trim,rank=r.prefix(A,P,.3);yes('nonuniform_strict_prefix',rank==['a','b','c'] and selected==['a'] and trim==0)
    a,t,s,c=r.curve(A,T,rank)
    exact_full=float(Fraction(3,8)*Fraction(1,12)+Fraction(1,2)*Fraction(5,6));exact_half=float(Fraction(3,8)*Fraction(1,12)*2)
    yes('independent_fraction_rectangle_full',math.isclose(r.auc(s,c,1.),exact_full,abs_tol=1e-14))
    yes('independent_fraction_rectangle_half',math.isclose(r.auc(s,c,.5),exact_half,abs_tol=1e-14))
    val=r.selection_values(A,T,['a','b'],.5)
    yes('independent_capture_and_non_target',math.isclose(val['capture_fraction'],5/6) and math.isclose(val['selected_non_target_area_m2'],.25))
    zeros={i:0. for i in A};_,_,s,z=r.curve(A,zeros,rank)
    yes('zero_target_NA_every_prefix',z==[None]*4 and r.auc(s,z,.5) is None and r.auc(s,z,1.) is None)
    for name,fn in [('strict_overbudget',lambda:r.selection_values(A,T,['a','c'],.5)),('unknown_id',lambda:r.selection_values(A,T,['other'],1.)),('duplicate_id',lambda:r.selection_values(A,T,['a','a'],1.)),('invalid_zero_for_NA',lambda:r.Checks().same(0.,None,'NA'))]:yes(name+'_rejected',reject(fn))
    yes('safe_bound_for_adverse_prefix',r.independent_bound(.2,.6,1.,.2,.2,True,False)['updated_upper_m2']==1.)
    yes('no_upgrade_from_unproven',not r.independent_bound(.5,.5,1.,.5,.5,False,False)['updated_proven_flag'])
    yes('fallback_cannot_be_certified',not r.independent_bound(.5,.5,1.,.5,.5,True,True)['updated_proven_flag'])
    receipt=ROOT/'provenance/confirmation_outcomes_checks_20260916_102007_531527/focused_checks.json'
    core.bound_file(receipt,'b273ec11d48419b807f0c169a3e2431254c1d82faec48c662e5b200ae9b1dd19')
    old=receipt.parent/'h/stages/local_outcomes';core.completed_stage(old)
    area=receipt.parent/'h/stages/local_geometry/area_units.csv';check=r.Checks();unit=r.verify_units(r.rows(old/'unit_truth.csv'),r.rows(area),check)
    target=out/'fixture';target.mkdir();shutil.copytree(old/'shared_reference',target/'shared_reference');ref=target/'shared_reference'
    # Relocate only the copied artificial fixture's absolute source references.
    labels=core.read(ref/'label_MILP_budgets.json')
    for label in labels:
        name=f"M{label['M']}_B{int(label['budget']*100):03d}"
        label['raw_return']=core.binding(ref/'label'/(name+'_adapted_return.json'))
        core.write(ref/'label'/(name+'_audited.json'),label)
    core.write(ref/'label_MILP_budgets.json',labels)
    seal=core.read(ref/'reference_seal.json');seal['files']={name:core.sha(ref/name) for name in seal['files']};core.write(ref/'reference_seal.json',seal)
    summary=r.verify_references(unit,target,check,core);yes('original_shared_15_3000_15000_artificial_reference',summary['label_cases']==15 and summary['random_full_curves']==3000 and summary['random_budget_rows']==15000)
    path=ref/'label/M1_B010_solver_raw.json';original=path.read_bytes();sealpath=ref/'reference_seal.json';originalseal=sealpath.read_bytes()
    b=core.read(path);b['result']['x']=[.5 for x in b['result']['x']];core.write(path,b)
    seal=core.read(sealpath);seal['files'][path.relative_to(ref).as_posix()]=core.sha(path);core.write(sealpath,seal)
    accepted=not reject(lambda:r.verify_references(unit,target,r.Checks(),core))
    if accepted:gaps.append({'name':'fractional_label_backend_with_claimed_feasible_raw_accepted','file':str(path),'why':'Label reference is produced after pretruth selection acceptance, so this cannot rely on that earlier audit.'})
    else:yes('fractional_label_backend_inconsistency_rejected')
    path.write_bytes(original);sealpath.write_bytes(originalseal)
    curvefile=ref/'label_prefix_curves.json';oldcurve=curvefile.read_bytes();lc=core.read(curvefile);lc.append(dict(lc[0],M=99));core.write(curvefile,lc)
    seal=core.read(sealpath);seal['files']['label_prefix_curves.json']=core.sha(curvefile);core.write(sealpath,seal)
    accepted=not reject(lambda:r.verify_references(unit,target,r.Checks(),core))
    if accepted:gaps.append({'name':'undeclared_extra_M99_label_curve_accepted','file':str(curvefile)})
    else:yes('undeclared_label_curve_inventory_rejected')
    curvefile.write_bytes(oldcurve);sealpath.write_bytes(originalseal)
    yes('no_torch_scipy_or_mask_decoder',not any(x in sys.modules for x in ('torch','scipy','PIL')))
    report={'passed':not gaps,'implementation_sha256':core.sha(snapshot),'source_path':str(source),'live_source_unchanged':core.sha(source)==core.sha(snapshot),'reviewer_sha256':core.sha(__file__),'checks':checks,'check_count':len(checks),'blocking_gaps':gaps,'actual_reserved_arrays_opened':0,'model_calls':0,'solver_calls':0,'original_fixture_receipt':core.binding(receipt),'scope':'Independent static plus synthetic arithmetic and consistently resealed adversarial reference fixtures; no actual execution.'}
    core.write(out/'review.json',report);core.write(out/'artifact_manifest.json',core.stage_manifest(out));print(json.dumps({'directory':str(out),'passed':report['passed'],'gaps':len(gaps),'sha256':core.sha(out/'review.json')},ensure_ascii=False))
if __name__=='__main__':main()
