"""Synthetic arithmetic and full26-frame selection lifecycle; no reserved input reads."""
from pathlib import Path
from contextlib import contextmanager
import datetime,hashlib,json,math,shutil,sys,traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE))
import core,external_spatial as ext
def main():
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output=ROOT/'provenance'/('confirmation_external_readiness_'+stamp);output.mkdir()
    fake=ROOT/'csyn'/stamp;fake.mkdir(parents=True)
    checks=[]
    def check(name,value):
        assert value,name
        checks.append({'name':name,'passed':True})
    def rejects(name,fn):
        try:fn()
        except (ValueError,AssertionError,FileNotFoundError,FileExistsError,KeyError):
            check(name,True);return
        raise AssertionError(name+' unexpectedly passed')
    sources={str(p):core.sha(p) for p in [HERE/'external_spatial.py',HERE/'external_arithmetic.py',Path(__file__)]}
    try:
        check('real_confirmation_not_frozen',not (HERE/'FREEZE.json').exists())
        rejects('real_select_requires_freeze_before_input',lambda:ext.select_run('not_a_real_run',fake/'absent_review.json'))
        rejects('real_evaluate_requires_freeze_before_input',lambda:ext.evaluate_all(fake/'absent_choices',fake/'absent_review'))
        import numpy as np
        import external_arithmetic as a
        extraction=core.read(HERE/'external_arithmetic_extraction.json')
        check('original_arithmetic_source_sha',core.sha(extraction['original_path'])==extraction['original_sha256'])
        check('extracted_arithmetic_sha',core.sha(HERE/'external_arithmetic.py')==extraction['output_sha256'])
        for g in (4,8,16):
            budgets=a.budget_rows(g)
            check('integer_budget_feasibility_g'+str(g),all(x['k']/x['N']<=x['effective_budget_fraction'] for x in budgets))
            check('two_budget_bases_g'+str(g),len(budgets)==10)
            same={x['k'] for x in budgets}
            check('aliases_g'+str(g),len(same)=={4:5,8:7,16:8}[g])
        reject_units=[{'id':'a','A':.5,'P':.1,'T':.2},{'id':'b','A':.5,'P':.2,'T':.1}]
        rejects('truth_key_rejected_by_selector',lambda:a.rank_units(reject_units))
        check('ascii_tie_order',[x['id'] for x in a.rank_units([{'id':'b','A':.5,'P':.2},{'id':'a','A':.5,'P':.2}])]==['a','b'])
        check('zero_ratio_missing',a.ratio(0,0,'zero')==(None,'zero'))
        plane=np.arange(256*256,dtype=np.float64).reshape(256,256)/65536
        for g in (4,8,16):
            vals=a.aggregate_plane(plane,'synthetic',g);side=256//g
            independent={f'synthetic:g{g:02d}:r{r:02d}:c{c:02d}':sum(float(plane[y,x]) for y in range(r*side,(r+1)*side) for x in range(c*side,(c+1)*side))/65536 for r in range(g) for c in range(g)}
            check('independent_cell_sums_g'+str(g),max(abs(vals[k]-v) for k,v in independent.items())<1e-14)
        meta={'run_id':'synthetic_hard','variant':'hard_seg','seed':17,'original_selected_epoch':1,'checkpoint_sha256':'0'*64,
              'config_sha256':'1'*64,'export_sha256':'2'*64,'source_manifest_sha256':'3'*64}
        frames=[{'id':'s0','acquisition_date':'d0'},{'id':'s1','acquisition_date':'d0'}]
        truth={}
        for frame in frames:
            for g in (4,8,16):
                vals=a.aggregate_plane(plane,frame['id'],g)
                truth[(frame['id'],g)]={i:{'A':1/g**2,'W':(value*.5 if frame['id']=='s0' else 0),'C':(1/g**2-value)*.3,
                                        'B':1/g**2-(value*.5 if frame['id']=='s0' else 0)-(1/g**2-value)*.3} for i,value in vals.items()}
        nested=a.check_nested_truth_cells(truth,[x['id'] for x in frames]);check('nested_three_class_synthetic',nested['scalar_comparisons']==480)
        for g in (4,8,16):
            cs=[]
            for frame in frames:
                ranking,choices=a.ranking_record(meta,frame,6,g,plane.astype(np.float32))
                curve=a.make_frame_curve(ranking,truth);cs.append(curve)
                check('all_curve_points_'+frame['id']+str(g),len(curve)==g*g+1)
                if frame['id']=='s1':
                    check('zero_target_curve_missing_g'+str(g),all(x['weed_capture_ratio'] is None for x in curve) and a.step_auc(curve,1) is None)
                else:
                    expected=sum(curve[k]['weed_capture_ratio']/(g*g) for k in range(g*g))
                    check('independent_full_step_auc_g'+str(g),abs(a.step_auc(curve,1)-expected)<1e-14)
                    check('strict_prefix_all_choices_g'+str(g),all(x['selected_ids']==ranking['ordered_ids'][:x['k']] for x in choices))
            pooled=a.group_curves(cs,'pooled','all2')
            check('pooled_weighted_target_g'+str(g),pooled[-1]['weed_capture_ratio']==1 and pooled[1]['frame_n']==2)
            rr=ext.reference_rankings(frames,truth,g,'random_expectation','3'*64)
            random_curves=[a.make_frame_curve(r,truth,'random_expectation',True) for r in rr]
            random_pooled=a.group_curves(random_curves,'pooled','all2')
            ar=ext.auc_row(random_pooled)
            check('random_step_auc_g'+str(g),abs(ar['auc_0_1']-(g*g-1)/(2*g*g))<1e-12)
            oracles=ext.reference_rankings(frames,truth,g,'label_oracle','3'*64)
            oracle_curve=a.make_frame_curve(oracles[0],truth,'label_oracle')
            check('oracle_equal_cost_sorted_g'+str(g),all(oracle_curve[k]['selected_weed_FAE']+1e-14>=cs[0][k]['selected_weed_FAE'] for k in range(g*g+1)))
        # Full production selection API under an explicitly synthetic, isolated gate.
        import pipeline,data_io
        ext.HERE=fake
        for name in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json'):core.write(fake/name,{'synthetic_fixture_only':True})
        shutil.copy2(HERE/'external_arithmetic.py',fake/'external_arithmetic.py')
        predhome=fake/'stages/predictions_seal';predhome.mkdir(parents=True);core.write(predhome/'all_predictions.json',{'synthetic_only':True})
        inputhome=fake/'stages/inputs_weeds';inputhome.mkdir();core.write(inputhome/'input_manifest.json',{'synthetic_only':True})
        (fake/'maps').mkdir();fakeframes=[];mapindex=[]
        stacked=np.stack([np.clip(plane*.9+(r*.001),0,1).astype(np.float32) for r in (1,2,4,6)])
        for i in range(26):
            identifier='F_'+str(i).zfill(3);frame={'id':identifier,'acquisition_date':'synthetic_date'}
            fakeframes.append(frame);path=fake/'maps'/(identifier+'.npz')
            np.savez_compressed(path,probabilities=stacked,valid=np.ones((256,256),np.float32),R=np.array([1,2,4,6]),id=np.asarray(identifier))
            mapindex.append({'id':identifier,'path':str(path),'sha256':core.sha(path)})
        core.write(fake/'map_index.json',mapindex)
        entry={'run_id':'synthetic_hard','dataset':'weeds','variant':'hard_seg','seed':17,'best_epoch':1,'checkpoint_sha256':'0'*64,'config_sha256':'1'*64}
        global_predictions={'models':[{'run_id':entry['run_id'],'map_index':core.binding(fake/'map_index.json'),'artifact_manifest':{'sha256':'2'*64}}]}
        core.write(predhome/'all_predictions.json',global_predictions)
        ext.validate_freeze=lambda:({},[entry],{})
        ext.validate_source_lock=lambda:None
        pipeline.validate_global_predictions=lambda:global_predictions
        data_io.validate_input_manifest=lambda dataset:{'records':fakeframes,'source_manifest':{'sha256':'3'*64}}
        ext.attempt=lambda name:core.attempt(name,home=fake)
        review=fake/'review.json';core.write(review,{'passed':True,'scope':'confirmation_external_selection_execution',
              'implementation_sha256':core.sha(HERE/'external_spatial.py'),'freeze_sha256':core.sha(fake/'FREEZE.json'),
              'arithmetic_sha256':core.sha(fake/'external_arithmetic.py')})
        out=ext.select_run(entry['run_id'],review);manifest=core.completed_stage(out)
        check('synthetic_selection_completed_manifest',bool(manifest['files']))
        _,seal=ext.validate_selection(entry)
        check('synthetic_selection_3120_cases',seal['case_count']==3120 and seal['rankings']==312 and seal['prediction_unit_rows']==34944 and seal['unique_selections']==2080)
        check('synthetic_selection_no_truth_cache',not (fake/'stages/truth_weeds').exists() and not seal['truth_used_for_choices'])
        rejects('permanent_completed_claim_denies_retry',lambda:ext.select_run(entry['run_id'],review))
        chosen=out/'choices.jsonl';original=chosen.read_bytes();chosen.write_bytes(original+b' ')
        rejects('choice_bytes_tamper_rejected',lambda:ext.validate_selection(entry))
        chosen.write_bytes(original)
        # A late map failure leaves one permanent failed stage and no completed seal.
        bad=dict(entry,run_id='synthetic_bad')
        ext.validate_freeze=lambda:({},[bad],{})
        global_predictions['models'][0]['run_id']=bad['run_id']
        first=Path(mapindex[0]['path']);saved=first.read_bytes();first.write_bytes(b'invalid')
        rejects('bad_map_hash_stops_claimed_selection',lambda:ext.select_run(bad['run_id'],review))
        first.write_bytes(saved)
        badout=fake/'stages'/('select_'+bad['run_id'])
        check('claimed_map_failure_preserved',core.read(badout/'status.json')['state']=='failed' and (fake/'claims'/('select_'+bad['run_id']+'.json')).exists() and not (badout/'selection_seal.json').exists())
        rejects('failed_claim_denies_retry',lambda:ext.select_run(bad['run_id'],review))
        check('root_sources_unchanged',all(core.sha(p)==h for p,h in sources.items()))
        report={'passed':True,'check_count':len(checks),'checks':checks,'sources_sha256':sources,'synthetic_fixture':str(fake),
                'reserved_data_read':False,'real_model_forward':False,'real_R6_forward':False,'real_solver':False,
                'arithmetic_scope':'R label6 used only for synthetic planes and arithmetic; no actual model or test data',
                'limitations':['Synthetic full26 selection and pure outcome arithmetic; actual confirmation outputs still require independent acceptance','No claim of independent author self-check']}
        core.write(output/'focused_checks.json',report);print(json.dumps({'output':str(output),'checks':len(checks),'passed':True}))
    except BaseException as error:
        core.write(output/'focused_checks.json',{'passed':False,'checks':checks,'error':repr(error),'traceback':traceback.format_exc(),'sources_sha256':sources})
        raise
if __name__=='__main__':main()
