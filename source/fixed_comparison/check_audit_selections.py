"""Synthetic and existing artificial fixture checks. Never use actual reserved data."""
from pathlib import Path
from unittest.mock import patch
import copy,datetime,json,math,shutil,sys
import audit_selections as a
import core
ROOT=a.HERE.parent

def main():
    out=ROOT/'provenance'/('confirmation_selection_auditor_checks_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();checks=[]
    def ok(name,value=True):
        assert value,name;checks.append({'name':name,'passed':True})
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,AssertionError,OSError,TypeError):ok(name);return
        raise AssertionError('Accepted: '+name)
    ok('no_science_import_before_gate','numpy' not in sys.modules and 'torch' not in sys.modules)
    reject('no_freeze_or_grant_no_actual_stage',lambda:a.execute(out/'absent.json'))
    ok('no_science_after_rejected_gate','numpy' not in sys.modules)
    toy={'c':(3.,2.7),'a':(2.,1.9),'b':(3.,2.7)}
    order,ids,trim=a.rank_prefix(toy,.75);ok('unequal_cost_prefix_stops_before_unaffordable',order==['a','b','c'] and ids==['a','b'])
    order,ids,trim=a.rank_prefix({f'x{i}':(.1,.05) for i in range(10)},.6);ok('strict_fsum_tail_trim',len(ids)==5 and trim==1)
    reject('duplicate_choice',lambda:a.subset(['a','a'],toy));reject('unknown_choice',lambda:a.subset(['absent'],toy))
    b=a.independent_bound(.6,.8,1.,.6,.6,True,False);ok('bound_contradiction_safe_upper',b['updated_upper_m2']==1 and b['bound_anomaly'] and not b['updated_proven_flag'])
    b=a.independent_bound(.6,.6+5e-11,1.,.6,.6,True,False);ok('within_tolerance_clamp',b['roundoff_clamp_m2']>0 and b['updated_proven_flag'])
    ok('fallback_never_proven',not a.independent_bound(1.,1.,1.,1.,1.,True,True)['updated_proven_flag'])
    ok('no_upgrade',not a.independent_bound(.6,.6,1.,.6,.6,False,False)['updated_proven_flag'])
    ok('missing_bound_safe_upper',a.independent_bound(.2,.3,1.,None,None,False,False)['updated_upper_m2']==1)
    watched=out/'watched.txt';watched.write_text('one');w=a.Watch();w.add(watched);watched.write_text('two');reject('input_drift_rejected',w.finish)
    import numpy as np
    import external_arithmetic as e
    fake=out/'synthetic';fake.mkdir();(fake/'stages/predictions_seal').mkdir(parents=True);(fake/'stages/local_geometry').mkdir()
    for path in ('FREEZE.json','stages/predictions_seal/all_predictions.json','stages/local_geometry/metadata.json'):core.write(fake/path,{'SYNTHETIC':True})
    shutil.copy2(a.HERE/'local_management.py',fake/'local_management.py')
    # Independent bincount integration, manually calculable native cells and 4 supports.
    gp=out/'g.npz';mp=out/'p.npz';plane=np.zeros((4,256,256),dtype=np.float32)
    for j in range(4):plane[j,0,:3]=np.array([.25,.5,.75],np.float32)/(j+1)
    np.savez(mp,probabilities=plane,valid=np.ones((256,256),np.float32),R=np.array(a.RS),id=np.array('toy'))
    np.savez(gp,cells=np.array([[-1,3],[0,4]]),area=np.array([2.,3.]),indices=np.array([0,1,1,2]),weights=np.array([1.,1.,1.,2.]),indptr=np.array([0,2,4]))
    geom={'windows':[{'id':'toy','file':'g.npz','sha256':core.sha(gp)}]};maps=[{'id':'toy','path':str(mp),'sha256':core.sha(mp)}]
    integrated=a.local_integrals(maps,geom,out,a.Watch())
    for m in a.MS:
        for j in range(4):a.close(integrated[m,f'M{m}_E{-1//m}_N{3//m}'][j+1],float(plane[j,0,0])+float(plane[j,0,1]))
    ok('bincount_weighted_P_with_negative_floor_coordinates');ok('all_three_M_area_conserved',all(math.fsum(v[0] for (mm,i),v in integrated.items() if mm==m)==5 for m in a.MS))
    # Existing producer toy contains 60 artificial solver returns; no new solver call.
    original=ROOT/'provenance/confirmation_management_checks_20260916_095229_278351/ok'
    if not original.exists():
        candidates=[p/'ok' for p in sorted((ROOT/'provenance').glob('confirmation_management_checks_*')) if (p/'ok/selection_seal.json').exists()]
        assert candidates;original=candidates[-1]
    target=fake/'stages/local_select_01';shutil.copytree(original,target)
    context=dict(run_id='SYNTHETIC_ONLY',freeze_sha256=core.sha(fake/'FREEZE.json'),all_predictions_sha256=core.sha(fake/'stages/predictions_seal/all_predictions.json'),geometry_metadata_sha256=core.sha(fake/'stages/local_geometry/metadata.json'),checkpoint_sha256='c'*64)
    bound=core.read(target/'bound_audit.json')
    for x in bound:x['choice']=core.binding(target/'policy/choices'/Path(x['choice']['path']).name)
    core.write(target/'bound_audit.json',bound);inner=core.read(target/'policy/selection_seal.json');inner['context']=context;core.write(target/'policy/selection_seal.json',inner)
    seal=core.read(target/'selection_seal.json');seal.update(context);seal['implementation_sha256']=core.sha(fake/'local_management.py');seal['files']={n:core.binding(target/n) for n in seal['files']};seal['A_P_only_inputs']=core.binding(target/'policy/selection_inputs_without_annotations.csv');core.write(target/'selection_seal.json',seal)
    core.write(target/'status.json',{'state':'completed'});core.write(target/'artifact_manifest.json',core.stage_manifest(target))
    allowed=a.rows(target/'policy/selection_inputs_without_annotations.csv');ap={(int(x['M']),x['id']):[float(x['area']),*[float(x[f'predicted_R{r}']) for r in a.RS]] for x in allowed}
    index=out/'maps.json';core.write(index,[]);pred={'map_index':core.binding(index)};entry={'run_id':context['run_id'],'checkpoint_sha256':context['checkpoint_sha256']}
    with patch.object(a,'HERE',fake),patch.object(a,'local_integrals',return_value=ap):result=a.audit_local(entry,1,pred,{'area_m2':10},a.Watch())
    ok('all120_local_choices_60_raw_bounds_reconstructed',result['case_count']==120 and result['independently_checked_P_bounds']==60)
    # All26 frames, 4R, 3 grids, nominal/matched aliases; independent NumPy sums vs fsum producer.
    ext=fake/'stages/select_toy';ext.mkdir();maps=[];frames=[];rankings=[];choices=[];units=[]
    ent=dict(run_id='toy',variant='hard_seg',seed=17,best_epoch=7,checkpoint_sha256='c'*64,config_sha256='d'*64)
    meta=dict(run_id='toy',variant='hard_seg',seed=17,original_selected_epoch=7,checkpoint_sha256='c'*64,config_sha256='d'*64,export_sha256='e'*64,source_manifest_sha256='f'*64)
    for fi in range(26):
        frame={'id':f'f{fi:02d}','acquisition_date':'d'+str(fi%4)};frames.append(frame);p=np.empty((4,256,256),np.float32)
        for j in range(4):p[j]=((np.arange(65536).reshape(256,256)*(fi+1)+j*37)%257).astype(np.float32)/256
        path=out/f'f{fi:02d}.npz';np.savez(path,probabilities=p,valid=np.ones((256,256),np.float32),R=np.array(a.RS),id=np.array(frame['id']));maps.append({'id':frame['id'],**core.binding(path)})
        for ri,r in enumerate(a.RS):
            for g in (4,8,16):
                rank,ch=e.ranking_record(meta,frame,r,g,p[ri]);rankings.append(rank);choices.extend(ch);units.extend({'run_id':'toy','R':r,'frame_id':frame['id'],'grid_g':g,**u} for u in rank['units'])
    for name,data in [('rankings.jsonl',rankings),('choices.jsonl',choices)]:
        with (ext/name).open('w',encoding='utf-8') as f:
            for x in data:f.write(json.dumps(x)+'\n')
    core.table(ext/'prediction_units.csv',units);core.write(ext/'input_binding.json',{'SYNTHETIC':True});core.write(index,maps)
    core.write(ext/'selection_seal.json',dict(scope='confirmation_external_label_free_selection',run_id='toy',dataset='weeds',freeze_sha256=context['freeze_sha256'],all_predictions_sha256=context['all_predictions_sha256'],R=list(a.RS),grids=[4,8,16],case_count=3120,rankings=312,unique_selections=2080,prediction_unit_rows=34944,truth_used_for_choices=False,files_sha256={n:core.sha(ext/n) for n in ('rankings.jsonl','choices.jsonl','prediction_units.csv','input_binding.json')}))
    core.write(ext/'status.json',{'state':'completed'});core.write(ext/'artifact_manifest.json',core.stage_manifest(ext))
    prediction={'map_index':core.binding(index),'artifact_manifest':{'sha256':'e'*64}};inp={'records':frames,'source_manifest':{'sha256':'f'*64}}
    with patch.object(a,'HERE',fake):result=a.audit_external(ent,prediction,inp,a.Watch())
    ok('all3120_external_cases_34944_units_2080_aliases',result['case_count']==3120 and result['prediction_unit_rows']==34944 and result['unique_selections']==2080)
    saved=(ext/'rankings.jsonl').read_bytes();(ext/'rankings.jsonl').write_bytes(saved+b' ')
    with patch.object(a,'HERE',fake):reject('tampered_saved_ranking_before_arithmetic',lambda:a.audit_external(ent,prediction,inp,a.Watch()))
    (ext/'rankings.jsonl').write_bytes(saved)
    ok('no_model_module_imported','torch' not in sys.modules)
    core.write(out/'focused_checks.json',dict(passed=True,implementation_sha256=core.sha(a.__file__),checker_sha256=core.sha(__file__),checks=checks,count=len(checks),test_or_R6_accessed=False,actual_model_calls=0,actual_solver_calls=0,synthetic_source=str(original)))
    print(json.dumps({'out':str(out),'checks':len(checks),'passed':True}))
if __name__=='__main__':main()
