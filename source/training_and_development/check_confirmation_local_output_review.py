"""Artificial/preparation-only checks for the saved local result auditor."""
from pathlib import Path
from unittest.mock import patch
import copy,datetime,hashlib,json,sys,traceback
import review_confirmation_local_outputs as r
ROOT=Path(__file__).resolve().parent
HOME=ROOT/'confirmation_v1'
sys.path.insert(0,str(HOME))
import core
RECEIPT=ROOT/'provenance/confirmation_outcomes_checks_20260916_102007_531527/focused_checks.json'
RECEIPT_SHA='b273ec11d48419b807f0c169a3e2431254c1d82faec48c662e5b200ae9b1dd19'
FIXTURE=RECEIPT.parent/'h/stages/local_outcomes'
def main():
    out=ROOT/'provenance'/('confirmation_local_output_auditor_checks_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    checks=[];c=r.Checks()
    def yes(name,ok=True):assert ok,name;checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,AssertionError,KeyError,FileNotFoundError):checks.append(name);return
        raise AssertionError('Failed refusal '+name)
    try:
        yes('Import has no numpy torch or scipy','numpy' not in sys.modules and 'torch' not in sys.modules and 'scipy' not in sys.modules)
        fake=out/'absent_home';fake.mkdir()
        review=out/'mock_grant.json';choice=out/'mock_choices.json';core.write(review,{});core.write(choice,{})
        with patch.object(r,'HOME',fake):
            reject('Missing formal FREEZE blocks before reading dataset',lambda:r.gate(review,choice))
        yes('Denied gate did not import scientific dependencies','numpy' not in sys.modules)
        core.bound_file(RECEIPT,RECEIPT_SHA);yes('Original saved all24 fixture attested',core.read(RECEIPT)['passed'] is True)
        require=lambda condition,message:r.require(condition,message)
        require(FIXTURE.is_relative_to(ROOT/'provenance'),'Not a preparation fixture');core.completed_stage(FIXTURE)
        population=core.read(RECEIPT.parent/'h/stages/local_selections_seal/all_local_selections.json')
        yes('Only artificial run identities',[x['run_id'] for x in population['models']]==[f'SYNTHETIC_{i:02d}' for i in range(1,25)])
        unit=r.verify_units(r.rows(FIXTURE/'unit_truth.csv'),r.rows(RECEIPT.parent/'h/stages/local_geometry/area_units.csv'),c)
        models=r.verify_models(population,unit,FIXTURE,c,core)
        yes('All24x120 saved artificial outcomes audited',models['fixed_budget_rows']==2880 and models['raw_P_cases']==1440)
        yes('All24x12 fullprefix AUC rows audited',models['auc_rows']==288 and models['prefix_curves']==288)
        yes('No production arithmetic imported',all(name not in sys.modules for name in ('local_outcomes','local_management','audit_common_bound_consistency','scipy','torch')))
        refs=r.verify_references(unit,FIXTURE,c,core)
        yes('One shared15 label reference set only',refs['label_cases']==15 and refs['actual_saved_label_solver_calls']==15)
        yes('All3000 saved random permutations and fullcurves',refs['random_full_curves']==3000 and refs['random_budget_rows']==15000 and refs['random_auc_rows']==3000)
        base_backend=core.read(FIXTURE/'shared_reference/label/M1_B010_solver_raw.json')
        base_raw=core.read(FIXTURE/'shared_reference/label/M1_B010_adapted_return.json')
        A0,T0=unit[1];ids0=sorted(A0)
        r.verify_label_backend(base_backend,base_raw,ids0,A0,T0,.1,c)
        for label,x in [('fractional',[.5]),('nonfinite',[{'nonfinite_float':'NaN'}]),('missing',None),('overbudget',[1.])]:
            bad=copy.deepcopy(base_backend);bad['result']['x']=x
            reject('Label backend '+label+' cannot retain raw feasible flag',lambda bad=bad:r.verify_label_backend(bad,base_raw,ids0,A0,T0,.1,r.Checks()))
        bad=copy.deepcopy(base_backend);bad['options']['time_limit']=60.
        reject('Changed label full options rejected',lambda:r.verify_label_backend(bad,base_raw,ids0,A0,T0,.1,r.Checks()))
        # Actual rejected candidates must be accepted when every adapted field
        # faithfully records the rejection and empty selected set.
        for label,x,error,violation in [('fractional',[.5],.5,None),('nonfinite',[{'nonfinite_float':'NaN'}],None,None),('missing',None,None,None),('overbudget',[1.],0.,.8)]:
            backend=copy.deepcopy(base_backend);backend['result'].update(x=x,fun=None,mip_dual_bound=-1.,mip_gap=None)
            raw=copy.deepcopy(base_raw);raw.update(status='solver_result_not_proven_optimal',candidate_feasible=False,candidate_integrality_error=error,candidate_budget_violation=violation,
              raw_solver_minimization_primal=None,raw_solver_minimization_dual=-1.,raw_solver_capture_upper=1.,raw_solver_relative_gap=None,
              dual_capture_bound=1.,dual_target_bound=1.,relative_gap=1e30,absolute_target_gap=1.,optimality_proven=False)
            r.verify_label_backend(backend,raw,ids0,A0,T0,.1,c);yes('Faithful rejected label '+label+' empty candidate accepted')
        # Nondegenerate direct tests independent of the tiny saved full24 fixture.
        A={'a':.2,'b':.3,'c':.5};T={'a':.1,'b':0.,'c':.4};P={'a':.1,'b':.15,'c':.05}
        pre,trim,order=r.prefix(A,P,.5)
        yes('ASCII tie ranking and strict actual budget',order==['a','b','c'] and pre==['a','b'] and trim==0)
        aa,tt,ss,cc=r.curve(A,T,order);yes('Independent full/half rectangle areas',abs(r.auc(ss,cc,1.)-.16)<1e-14 and abs(r.auc(ss,cc,.5)-.12)<1e-14)
        # total target zero: every ratio and both AUCs must stay undefined.
        zero=dict.fromkeys(A,0.);a,t,s,v=r.curve(A,zero,order)
        yes('ZeroT curve and AUC NA including endpoints',all(x is None for x in v) and r.auc(s,v,1.) is None and r.auc(s,v,.5) is None)
        yes('ZeroT selected fraction NA and absolute area retained',r.selection_values(A,zero,['a'],.5)['capture_fraction'] is None and r.selection_values(A,zero,['a'],.5)['actual_area_m2']==.2)
        reject('False zero in undefined result rejected',lambda:c.same(0.,None,'NA'))
        reject('Nonfinite saved result rejected',lambda:c.same(float('nan'),0.,'NaN'))
        reject('Overbudget selection rejected',lambda:r.selection_values(A,T,['c','a'],.5))
        reject('Unknown selected ID rejected',lambda:r.selection_values(A,T,['absent'],1.))
        reject('Duplicate selected ID rejected',lambda:r.selection_values(A,T,['a','a'],1.))
        z=r.independent_bound(.6,.8,1.,.6,.6,True,False)
        yes('Raw contradiction safe total upper and downgrade',z['updated_upper_m2']==1. and z['bound_anomaly'] and not z['updated_proven_flag'])
        z=r.independent_bound(.6,.6+5e-11,1.,.6,.6,True,False)
        yes('Only within-tolerance inversion is disclosed clamp',z['roundoff_clamp_m2']>0 and not z['bound_anomaly'])
        yes('Fallback is never certified',not r.independent_bound(1.,1.,1.,1.,1.,True,True)['updated_proven_flag'])
        yes('Originally false certification never upgraded',not r.independent_bound(1.,1.,1.,1.,1.,False,False)['updated_proven_flag'])
        yes('Missing retained bound safe total',r.independent_bound(.2,.1,1.,.3,None,False,False)['updated_upper_m2']==1.)
        bad=r.rows(FIXTURE/'unit_truth.csv');bad[0]['target_m2']='.7'
        reject('Native nested T drift rejected',lambda:r.verify_units(bad,r.rows(RECEIPT.parent/'h/stages/local_geometry/area_units.csv'),r.Checks()))
        bad=r.rows(FIXTURE/'unit_truth.csv');bad[0]['area_m2']='.8'
        reject('Truth domain A drift rejected',lambda:r.verify_units(bad,r.rows(RECEIPT.parent/'h/stages/local_geometry/area_units.csv'),r.Checks()))
        # Metadata-only 338-window source-proof fixture. Files are deliberately
        # plain ASCII sentinels, not labels or CSR arrays, and are only hashed.
        proofhome=out/'proof_fixture';proofout=proofhome/'stages/local_outcomes'
        geofolder=proofhome/'stages/local_geometry';truthfolder=proofhome/'stages/truth_local'
        for path in (proofout,geofolder,truthfolder):path.mkdir(parents=True)
        mask=proofhome/'NOT_A_MASK.txt';mask.write_text('ARTIFICIAL IDENTITY ONLY',encoding='ascii')
        csr=geofolder/'NOT_A_CSR.txt';csr.write_text('ARTIFICIAL IDENTITY ONLY',encoding='ascii')
        summary=proofhome/'summary.json';core.write(summary,{'sources':[dict(source_id='ARTIFICIAL',mask=str(mask),mask_sha256=core.sha(mask))]})
        windows=[];truthrecords=[];proofs=[]
        for i in range(338):
            ident=f'ARTIFICIAL_{i:03d}';x=i*512
            windows.append(dict(id=ident,source_id='ARTIFICIAL',x=x,y=0,file=csr.name,sha256=core.sha(csr),raw_area_m2=1.,native_valid_pixels=4))
            truthrecords.append(dict(id=ident,native_reference=dict(source_id='ARTIFICIAL',native_window_xywh=[x,0,512,512],mask_source=str(mask))))
            proofs.append(dict(id=ident,source_id='ARTIFICIAL',mask_sha256=core.sha(mask),geometry_sha256=core.sha(csr),native_valid_pixels=4,native_target_pixels=1,area_m2=1.,target_m2=.25,independent_native_target_m2=.25,target_relative_error=0.))
        geometry=dict(windows=windows,area_m2=338.);proof=dict(windows=proofs,area_m2=338.,target_m2=84.5,native_valid_pixels=1352,native_target_pixels=338,validation_T_reused=False,target_source='native binary mask>0 intersect native alpha; native CSR integration')
        core.write(geofolder/'metadata.json',geometry);core.write(truthfolder/'truth_manifest.json',dict(records=truthrecords))
        core.write(proofout/'native_truth_audit.json',proof)
        (proofout/'unit_truth.csv').write_text('ARTIFICIAL SEALED UNIT IDENTITY',encoding='ascii')
        watched={str(mask):core.sha(mask)}
        core.write(proofout/'begin_binding.json',dict(watched_sha256=watched))
        core.write(proofout/'end_binding.json',dict(watched_sha256=watched,all_rechecked=True,artifact_truth_sha256=core.sha(proofout/'unit_truth.csv')))
        contract=dict(datasets=dict(local=dict(source_summary_path=str(summary),source_summary_sha256=core.sha(summary))))
        units={1:({'a':338.},{'a':84.5})}
        result=r.verify_native_proof(proofout,proofhome,contract,units,c,core)
        yes('Full338 metadata native source proof accepts exact identities',result['windows']==338 and result['native_masks_decoded']==0 and result['native_T_reintegrated'] is False)
        wrong=copy.deepcopy(proof);wrong['windows'][0]['geometry_sha256']='0'*64;core.write(proofout/'native_truth_audit.json',wrong)
        reject('Native proof CSR hash drift rejected',lambda:r.verify_native_proof(proofout,proofhome,contract,units,r.Checks(),core))
        wrong=copy.deepcopy(proof);wrong['windows'][0]['independent_native_target_m2']=.3;core.write(proofout/'native_truth_audit.json',wrong)
        reject('Native conservation proof inconsistency rejected',lambda:r.verify_native_proof(proofout,proofhome,contract,units,r.Checks(),core))
        core.write(proofout/'native_truth_audit.json',proof)
        reject('Native proof totals versus saved unit T drift rejected',lambda:r.verify_native_proof(proofout,proofhome,contract,{1:({'a':338.},{'a':84.4})},r.Checks(),core))
        core.write(proofout/'end_binding.json',dict(watched_sha256={},all_rechecked=True,artifact_truth_sha256=core.sha(proofout/'unit_truth.csv')))
        reject('Native producer begin/end binding drift rejected',lambda:r.verify_native_proof(proofout,proofhome,contract,units,r.Checks(),core))
        yes('Explicit nonfinite backend encoding stays unavailable',r.decode_backend_number({'nonfinite_float':'NaN'}) is None)
        reject('Arbitrary raw backend dict is not a nonfinite marker',lambda:r.decode_backend_number({'unrecognized':'invalid'}))
        yes('No real solver or model module loaded','torch' not in sys.modules and 'scipy' not in sys.modules)
        report=dict(passed=True,scope='preparation_artificial_only',implementation_sha256=core.sha(r.__file__),checker_sha256=core.sha(__file__),fixture_receipt=core.binding(RECEIPT),checks=checks,check_count=len(checks),artificial_counts=dict(**models,**refs),numeric_comparisons=c.comparisons,maximum_numeric_residual=c.maximum_residual,actual_reserved_arrays_read=0,actual_model_calls=0,actual_solver_calls=0,actual_R6_forward=0,formal_FREEZE_created=False,limits=['Real execution gate reviewed statically and missing-FREEZE refusal tested; full production grant not exercised.','Native provenance proof uses338 artificial metadata records plus hash-only ASCII sentinels, not decoded arrays.','No new native-mask integration: real review scope is native source proof binding plus saved-unit arithmetic.'])
        core.write(out/'focused_checks.json',report);print(json.dumps({'passed':True,'out':str(out),'checks':len(checks),'sha256':core.sha(out/'focused_checks.json')}))
    except BaseException as error:
        core.write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),implementation_sha256=core.sha(r.__file__),checks_completed=checks));raise
if __name__=='__main__':main()
