"""Development extension for predeclared completed common segmentation models.

Reuses frozen v1 geometry/metric code and the audited label-free policy selector.
No inference, training, test or R6 access. Original result directories are untouched.
"""
from pathlib import Path
import argparse,datetime,json,math,shutil,subprocess,sys,time,traceback
import evaluate_management as ev
import diagnose_predicted_knapsack as policy
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
PA=ROOT/'precision_agriculture'
PROTOCOL=PA/'RM_common_extension_protocol_v1.json'
PROTOCOL_SHA='e73597bbfd233f82ecbd1fc04fbefda00143eb642fafa4b1d45039bf48a79e2d'
REVIEWED_EXPORT_SHA='56319f148fadc69b0cdc88c3c44f043ba020e812f2f4f919480304fe155dbecc'

def validate(maps_dir):
    assert ev.sha(PROTOCOL)==PROTOCOL_SHA
    extension=ev.read(PROTOCOL)
    assert ev.sha(ROOT/'common_queue_plan_v2.json')==extension['training_plan_sha256']
    for key,path in [('evaluate_management','evaluate_management.py'),('policy_selector','diagnose_predicted_knapsack.py'),('geometry','geo_management.py')]:
        assert ev.sha(ROOT/path)==extension['code_sha256'][key], f'Changed dependency: {path}'
    maps=ev.read(maps_dir/'metadata.json'); ms=ev.read(maps_dir/'status.json')
    assert ms['status']=='completed' and ms['verification']['all_checks_passed']
    assert ev.sha(maps_dir/'metadata.json')==ms['metadata_sha256']
    assert maps['schema']=='local_validation_probability_maps_v1' and maps['factors']==[1,2,4]
    assert not maps['test_accessed'] and not maps['R6_accessed']
    declared={x['run_id']:x for x in extension['declared_runs']}
    assert maps['run_id'] in declared
    run=(ROOT/'runs'/maps['run_id']).resolve()
    assert maps_dir.parent==run
    assert Path(maps['config']).resolve()==run/'config.json'
    assert Path(maps['checkpoint']).resolve()==run/'best.pt'
    assert ev.sha(run/'config.json')==maps['config_sha256']
    assert ev.sha(run/'best.pt')==maps['checkpoint_sha256']
    config=ev.read(run/'config.json'); state=ev.read(run/'status.json')
    plan=ev.read(ROOT/'common_queue_plan_v2.json')
    assert maps['model_family']=='common_v2' and maps['encoder']==config['encoder'] and maps['variant']==config['variant']
    assert maps['spatial_prediction_available'] is True
    assert config['manifest_sha256']==plan['manifest_sha256']
    assert state['stage']=='development' and config['test_accessed'] is False
    assert not (run/'USE_RESTRICTION.json').exists() and not (run/'failure.json').exists()
    assert state['status']=='completed' and state['executed_epochs']==24 and not state['test_accessed']
    assert state['best_epoch']==maps['checkpoint_epoch']
    assert config['run_id']==maps['run_id'] and config['stage']=='development'
    assert config['variant'] in ('hard_seg','soft_seg')
    assert config['n_val']==357 and config['n_train']==985 and config['epochs']==24
    assert not any(config.get(k,False) for k in ('max_train','max_val','smoke','diagnostic','diagnostic_epochs','no_pretrained'))
    assert config['decision_metrics_used_for_selection'] is False
    assert config['validation_R']==[1,2,4] and config['fixed_output_grid']==256
    assert len(config['selected_train_ids'])==len(set(config['selected_train_ids']))==985
    for k in ('encoder','variant','lr','label_grid','epochs','seed','batch_size','accumulation_steps','sampler','weight_decay','amp','scale_augmentation','freeze_batchnorm'):
        assert config[k]==declared[maps['run_id']][k], f'Configuration differs from declared plan: {k}'
    source_names=('train_common.py','models_common.py','conservative_resample.py')
    assert set(config['source_sha256'])==set(source_names)
    assert maps['model_dependency_sha256']==config['source_sha256']
    assert maps['model_source_sha256']==config['source_sha256']['train_common.py']
    assert set(maps['source_copies'])==set(source_names)|{'export_common_validation.py'}
    for k in source_names:
        h=config['source_sha256'][k]
        assert h==plan['source_sha256'][k]==ev.sha(ROOT/k)==ev.sha(run/'source'/k)
        assert Path(maps['source_copies'][k]).resolve()==maps_dir/'source'/k
        assert ev.sha(maps_dir/'source'/k)==h
    assert Path(maps['source_copies']['export_common_validation.py']).resolve()==maps_dir/'source'/'export_common_validation.py'
    assert ev.sha(maps_dir/'source'/'export_common_validation.py')==maps['export_source_sha256']==REVIEWED_EXPORT_SHA
    expected_export_inputs=[run/'config.json',run/'status.json',run/'metrics.jsonl',run/'best_validation.json',run/'best.pt',
        ROOT/'local_data_v2'/'val.json',ROOT/'local_data_v2'/'DATASET_LOCK.json',ROOT/'common_queue_plan_v2.json',
        ROOT/'local_data_v2'/'manifest_summary.json']+[ROOT/n for n in source_names]
    assert {Path(p).resolve() for p in maps['input_hashes']}=={p.resolve() for p in expected_export_inputs}
    for p,h in maps['input_hashes'].items():assert ev.sha(p)==h
    lock=ev.read(ROOT/'local_data_v2'/'DATASET_LOCK.json')
    assert ev.sha(ROOT/'local_data_v2'/'DATASET_LOCK.json')==config['manifest_sha256']['DATASET_LOCK.json']
    for split in ('train.json','val.json'):assert lock['manifest_sha256'][split]==config['manifest_sha256'][split]
    history=[json.loads(x) for x in (run/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    assert [x['epoch'] for x in history]==list(range(1,25))
    assert all(math.isfinite(x['selection_balanced_mae']) for x in history)
    for epoch in history:
        assert set(epoch['scales'])=={'1','2','4'}
        for scores in epoch['scales'].values():
            assert scores['n']==357 and scores['n_positive']==27
            assert math.isclose(scores['balanced_mae'],(scores['mae_zero']+scores['mae_positive'])/2,rel_tol=1e-12)
        assert math.isclose(epoch['selection_balanced_mae'],sum(x['balanced_mae'] for x in epoch['scales'].values())/3,rel_tol=1e-12)
    best=min(history,key=lambda x:(x['selection_balanced_mae'],x['epoch']))
    assert ev.read(run/'best_validation.json')==best and best['epoch']==state['best_epoch']
    assert best['selection_balanced_mae']==state['best_validation_balanced_mae']
    assert ev.sha(maps_dir/'predictions.csv')==maps['scalar_csv_sha256']
    assert ev.sha(maps_dir/'summary.json')==maps['scalar_summary_sha256']
    assert ev.read(maps_dir/'summary.json')['original_amp_validation']==best
    verification=ev.read(maps_dir/'verification.json')
    assert verification==ms['verification'] and verification['checked_pairs']==1071 and verification['expected_windows']==357
    assert verification['all_npz_reopened']==357 and verification['original_selection_audited_24_epochs'] and verification['original_epoch_preserved']
    geometry_dir=PA/'geometry_20260916_v1'
    geometry=ev.read(geometry_dir/'metadata.json')
    assert geometry['state']=='completed' and ev.read(geometry_dir/'status.json')['state']=='completed'
    assert not geometry['subset_diagnostic'] and not geometry['test_accessed'] and geometry['expected_windows']==357
    assert geometry['native_valid_pixels']==91061049 and geometry['native_target_pixels']==885226
    geometry_protocol=PA/extension['parent_geometry_protocol']
    assert ev.sha(geometry_protocol)==extension['parent_geometry_protocol_sha256']==geometry['protocol_sha256']
    original=ev.read(geometry_protocol)
    assert original['image_degradation_factors']==extension['R']==list(ev.R_VALUES)
    assert original['management_grid_m']==extension['M_m']==list(ev.M_VALUES)
    assert original['budgets']==extension['budgets']==list(ev.BUDGETS)
    assert geometry['crs']==original['grid_crs']=='EPSG:32649' and geometry['base_grid_m']==1
    assert ev.sha(ROOT/'geo_management.py')==geometry['script_sha256']==ev.sha(geometry_dir/'geo_management_source.py')
    manifest_path=ROOT/'local_data_v2'/'val.json'; manifest=ev.read(manifest_path)
    mh=ev.sha(manifest_path)
    assert mh==original['manifest_sha256']==geometry['manifest_sha256']==maps['manifest_sha256']==config['manifest_sha256']['val.json']
    assert len(manifest)==357 and config['selected_val_ids']==[x['id'] for x in manifest]
    for sid,source in geometry['sources'].items():
        assert source['tif_sha256']==maps['sources'][sid]['tif_sha256_from_dataset_summary']
        assert source['mask_sha256']==maps['sources'][sid]['mask_sha256_from_dataset_summary']
        assert np.allclose(source['transform'][:6],maps['sources'][sid]['affine'],rtol=0,atol=1e-15)
    ownership=ev.check_ownership(manifest,maps)
    paths=[PROTOCOL,geometry_protocol,ROOT/'common_queue_plan_v2.json',Path(__file__),ROOT/'evaluate_management.py',
           ROOT/'diagnose_predicted_knapsack.py',ROOT/'geo_management.py',manifest_path,run/'config.json',run/'status.json',
           run/'best.pt',maps_dir/'metadata.json',maps_dir/'status.json',maps_dir/'verification.json',
           geometry_dir/'metadata.json',geometry_dir/'status.json']
    for p in expected_export_inputs:
        if p not in paths:paths.append(p)
    paths += [maps_dir/'source'/n for n in (*source_names,'export_common_validation.py')]
    return extension,original,maps,geometry,manifest,ownership,{str(p):ev.sha(p) for p in paths}

def run_policy(out):
    """All 45 label-free selections precede annotation evaluation."""
    dest=out/'predicted_area_policy';dest.mkdir()
    allowed=policy.allowed_selection_rows(policy.read_csv(out/'unit_predictions.csv'))
    solutions=[];data={}
    for m in (1,2,4):
        rows=sorted([x for x in allowed if int(x['M'])==m],key=lambda x:x['id'])
        ids=[x['id'] for x in rows];area=np.array([float(x['area']) for x in rows])
        for r in (1,2,4):
            pred=np.array([float(x[f'predicted_R{r}']) for x in rows]);data[(m,r)]=(ids,area,pred)
            for b in ev.BUDGETS:
                s=policy.select_predicted_area(ids,area,pred,b)
                assert s['candidate_feasible_in_original_units']
                s.update(M=m,R=r,budget=b)
                name=f'M{m}_R{r}_B{int(b*100):03d}.json'
                ev.write_json(dest/name,s);solutions.append((name,s))
    assert len(solutions)==45
    truth={(int(x['M']),x['id']):float(x['target']) for x in policy.read_csv(out/'unit_predictions.csv')}
    original={(int(x['M']),int(x['R']),float(x['budget'])):x for x in policy.read_csv(out/'fixed_budgets.csv') if x['method']=='model_prefix'}
    results=[]
    for name,s in solutions:
        m,r,b=s['M'],s['R'],s['budget'];ids,area,pred=data[(m,r)]
        target=np.array([truth[(m,i)] for i in ids]);pids=policy.prefix_ids(ids,area,pred,b)
        current=policy.evaluate_selected(ids,s['selected_ids'],area,pred,target)
        baseline=policy.evaluate_selected(ids,pids,area,pred,target)
        prior=original[(m,r,b)]
        assert math.isclose(baseline['capture_fraction'],float(prior['capture_fraction']),rel_tol=1e-10,abs_tol=1e-10)
        assert math.isclose(baseline['actual_area_m2'],float(prior['actual_area']),rel_tol=1e-10,abs_tol=1e-9)
        assert current['actual_area_m2']<=s['budget_area_m2']
        delta=current['predicted_objective_m2']-baseline['predicted_objective_m2']
        if s['optimality_proven_within_tolerance']:assert delta>=-max(1e-10,1e-8*s['total_predicted_m2'])
        result={'M':m,'R':r,'budget':b,'solution_file':name,'optimality_proven':s['optimality_proven_within_tolerance'],
          'predicted_relative_gap':s['predicted_relative_gap'],'predicted_upper_m2':s['predicted_dual_upper_m2'],
          **{'pmilp_'+k:v for k,v in current.items()},**{'prefix_'+k:v for k,v in baseline.items()},
          'delta_capture_pp':100*(current['capture_fraction']-baseline['capture_fraction']),
          'delta_predicted_objective_m2':delta,'unused_budget_m2':s['budget_area_m2']-current['actual_area_m2']}
        results.append(result)
        ev.write_json(dest/(Path(name).stem+'_evaluation.json'),{'metrics':result,'prefix_selected_ids':pids})
    ev.write_csv(dest/'all_45_deltas.csv',results)
    return {'combinations':45,'optimal_within_tolerance':sum(x['optimality_proven'] for x in results),
      'improved':sum(x['delta_capture_pp']>1e-8 for x in results),'worsened':sum(x['delta_capture_pp']< -1e-8 for x in results),
      'unchanged':sum(abs(x['delta_capture_pp'])<=1e-8 for x in results),'all_solutions_feasible':True,
      'no_target_annotations_in_selection':True,'no_auc_from_sparse_budget_points':True}

def main(maps_dir):
    started=time.time();maps_dir=Path(maps_dir).resolve()
    extension,original,maps,geometry,manifest,ownership,hashes=validate(maps_dir)
    prefix='common_RM_'+maps['run_id']+'_'
    for old in PA.glob(prefix+'*'):
        if (old/'status.json').exists() and ev.read(old/'status.json').get('state')=='completed':
            meta=ev.read(old/'inputs.json')
            if meta['checkpoint_sha256']==maps['checkpoint_sha256'] and meta['extension_protocol_sha256']==PROTOCOL_SHA:
                print(json.dumps({'state':'already_completed','output':str(old)}));return 0
    out=PA/(prefix+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();(out/'source').mkdir()
    sp=out/'status.json';state={'state':'running','stage':'integration','model_run':maps['run_id'],'test_accessed':False,'R6_accessed':False}
    ev.write_json(sp,state);print(json.dumps({'output':str(out),'state':'running'}),flush=True)
    try:
        with threadpool_limits(limits=2):
            for name in ('evaluate_management.py','diagnose_predicted_knapsack.py','geo_management.py',Path(__file__).name):shutil.copyfile(ROOT/name,out/'source'/name)
            for p in (PROTOCOL,PA/extension['parent_geometry_protocol']):shutil.copyfile(p,out/'source'/p.name)
            toy_source=PA/'RM_evaluation_20260916_020003_962711'/'toy_checks.json'
            assert ev.read(toy_source)['passed']
            ev.write_json(out/'toy_checks.json',{'reused_checks':ev.read(toy_source),'source':str(toy_source),
              'sha256':ev.sha(toy_source),'note':'Previously passed checks of unchanged pinned geometry/metric functions; not rerun here.'})
            cells,values,integration,windows=ev.integrate(geometry,maps,manifest,sp,state)
            ev.write_json(out/'inputs.json',{'file_sha256':hashes,'extension_protocol_sha256':PROTOCOL_SHA,'extension_protocol':extension,
              'geometry_protocol':original,'checkpoint_sha256':maps['checkpoint_sha256'],'maps_directory':str(maps_dir),'windows':windows,
              'ownership':ownership,'no_inference':True,'no_test_or_R6':True})
            ev.write_json(out/'integration_checks.json',integration)
            result=ev.evaluate_tables(cells,values,original,out,sp,state)
            # Separate process preserves the frozen selector's single-thread HiGHS
            # settings; the parent label solver uses two threads.
            job=subprocess.run([sys.executable,'-X','utf8','-B','-s',str(Path(__file__).resolve()),'--policy-for',str(out)],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
            (out/'policy_stdout.log').write_text(job.stdout,encoding='utf-8')
            (out/'policy_stderr.log').write_text(job.stderr,encoding='utf-8')
            assert job.returncode==0,f'Policy process failed; see {out / "policy_stderr.log"}'
            policy_summary=ev.read(out/'predicted_area_policy'/'summary.json')
            summary={'state':'completed','created_at':ev.stamp(),'model_run':maps['run_id'],'checkpoint_epoch':maps['checkpoint_epoch'],
              'domain':{k:geometry[k] for k in ('native_valid_pixels','native_target_pixels','area_m2','target_m2','origin_utm_m','crs')},
              'results':result,'predicted_area_policy':policy_summary,'integration':{k:v for k,v in integration.items() if k!='per_window'},
              'extension_protocol_sha256':PROTOCOL_SHA,'ownership':ownership,'source_sha256':hashes,'versions':{'python':sys.version,'numpy':np.__version__,'scipy':ev.scipy.__version__},
              'model_tuned':False,'test_accessed':False,'R6_accessed':False,'elapsed_seconds':time.time()-started}
            for p,h in hashes.items():assert ev.sha(p)==h,f'Changed input: {p}'
            ev.write_json(out/'summary.json',summary);ev.make_report(summary,out)
            state.update(state='completed',stage='complete',completed_at=ev.stamp(),policy_summary=policy_summary,
              files_sha256={str(p.relative_to(out)):ev.sha(p) for p in out.rglob('*') if p.is_file() and p!=sp})
            ev.write_json(sp,state);print(json.dumps({'state':'completed','output':str(out),'policy_summary':policy_summary}),flush=True)
    except BaseException as err:
        state.update(state='failed',error=repr(err),traceback=traceback.format_exc());ev.write_json(sp,state);raise
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--maps-dir');group.add_argument('--policy-for',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.policy_for:
        out=Path(args.policy_for).resolve()
        assert out.parent==PA.resolve() and out.name.startswith('common_RM_dev_common_v2_')
        assert ev.sha(PROTOCOL)==PROTOCOL_SHA
        assert ev.read(out/'inputs.json')['extension_protocol_sha256']==PROTOCOL_SHA
        assert ev.read(out/'status.json')['state']=='running'
        for name,h in ev.read(out/'inputs.json')['file_sha256'].items():assert ev.sha(name)==h
        with threadpool_limits(limits=2):answer=run_policy(out)
        ev.write_json(out/'predicted_area_policy'/'summary.json',answer)
        print(json.dumps(answer))
    else:sys.exit(main(args.maps_dir))
