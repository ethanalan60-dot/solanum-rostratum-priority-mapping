"""Independent synthetic selection queue guards/lifecycle; no actual child or solver."""
from pathlib import Path
import datetime,importlib.util,json,sys,types
ROOT=Path(__file__).resolve().parent;HERE=ROOT/'confirmation_v1';sys.path.insert(0,str(HERE));import core

def main():
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=ROOT/'provenance'/('confirmation_selection_queue_independent_'+stamp);(out/'source').mkdir(parents=True)
    source=HERE/'run_selection_queue.py';snapshot=out/'source'/source.name;snapshot.write_bytes(source.read_bytes())
    (out/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    fixture=ROOT/'csq'/stamp;fixture.mkdir(parents=True)
    models=core.read(HERE/'contract.json')['roster'];checks=[];sequence=0;scenarios=[]
    sys.modules['pipeline']=types.SimpleNamespace(validate_global_predictions=lambda:None)
    def yes(name,condition):
        core.require(condition,name);checks.append({'name':name,'passed':True})
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,FileExistsError,FileNotFoundError):yes(name,True);return
        raise AssertionError(name)
    def setup():
        nonlocal sequence
        sequence+=1;fake=fixture/str(sequence);fake.mkdir()
        spec=importlib.util.spec_from_file_location('sqreview_'+str(sequence),snapshot);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
        q.HERE=fake;q.validate_freeze=lambda:({},models,{})
        for name in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json'):core.write(fake/name,{'synthetic':True})
        for name in ('local_management.py','external_spatial.py','external_arithmetic.py'):(fake/name).write_text('# synthetic source only\n')
        for stage,file in (('predictions_seal','all_predictions.json'),('local_geometry','metadata.json')):
            folder=fake/'stages'/stage;folder.mkdir(parents=True)
            core.write(folder/file,{'synthetic':True});core.write(folder/'status.json',{'state':'completed'});core.write(folder/'artifact_manifest.json',core.stage_manifest(folder))
        fh=core.sha(fake/'FREEZE.json');ph=core.sha(fake/'stages/predictions_seal/all_predictions.json');gh=core.sha(fake/'stages/local_geometry/metadata.json')
        reviews={
            'predictions_review':{'passed':True,'scope':'confirmation_predictions_independent','all_predictions_sha256':ph},
            'geometry_review':{'passed':True,'scope':'confirmation_geometry_independent','metadata_sha256':gh},
            'local_review':{'passed':True,'scope':'confirmation_local_management','implementation_sha256':core.sha(fake/'local_management.py'),'freeze_sha256':fh,'all_predictions_sha256':ph,'geometry_metadata_sha256':gh,'run_ids':[m['run_id'] for m in models if m['dataset']=='local' and m['variant']!='regression'],'case_count_per_model':120},
            'external_review':{'passed':True,'scope':'confirmation_external_selection_execution','implementation_sha256':core.sha(fake/'external_spatial.py'),'freeze_sha256':fh,'arithmetic_sha256':core.sha(fake/'external_arithmetic.py')}}
        review=fake/'review.json'
        grant={'passed':True,'scope':'confirmation_selection_queue','implementation_sha256':core.sha(snapshot),'freeze_sha256':fh,'run_ids':[m['run_id'] for m in models if m['variant']!='regression'],'R':[1,2,4,6],'label_free_selection_authorized':True,'truth_materialization_authorized':False,'automatic_retry_authorized':False}
        for key,value in reviews.items():core.write(fake/(key+'.json'),value);grant[key]=core.binding(fake/(key+'.json'))
        core.write(review,grant)
        return q,fake,review
    q,fake,review=setup();_,plan=q.validate(review)
    yes('valid_synthetic_contract_accepted',len(plan)==31)
    yes('exact_local24_then_seal_then_external6',[x[0] for x in plan]==[f'local_select_{i:02d}' for i in range(1,25)]+['local_selections_seal']+['select_'+m['run_id'] for m in models if m['dataset']=='weeds' and m['variant']!='regression'])
    yes('no_regression_commands',all('regression' not in ' '.join(c) for _,c in plan))
    yes('no_truth_or_scoring_commands',all(not set(c)&{'truth','materialize-truth','evaluate','--evaluate','--choice-review'} for _,c in plan))
    yes('all_commands_python_cpu_entry',all(c[:5]==[sys.executable,'-X','utf8','-B','-s'] for _,c in plan))
    yes('fixed_case_arithmetic',24*4*3*5*2==2880 and 6*26*4*3*5*2==18720)
    production=[HERE/'stages'/stage/name for stage,_ in plan for name in ('artifact_manifest.json.tmp','selection_seal.json.tmp','all_local_selections.json.tmp','queue_status.json.tmp')]
    yes('planned_stage_paths_under240',max(len(str(p)) for p in production)<240)
    mutations=[(None,'passed',False),(None,'scope','other'),(None,'implementation_sha256','0'*64),(None,'freeze_sha256','0'*64),(None,'run_ids',[]),(None,'R',[1,2,4]),(None,'label_free_selection_authorized',False),(None,'truth_materialization_authorized',True),(None,'automatic_retry_authorized',True),
      ('predictions_review','passed',False),('predictions_review','all_predictions_sha256','0'*64),('geometry_review','passed',False),('geometry_review','metadata_sha256','0'*64),('local_review','implementation_sha256','0'*64),('local_review','geometry_metadata_sha256','0'*64),('local_review','all_predictions_sha256','0'*64),('local_review','run_ids',[]),('local_review','case_count_per_model',90),('external_review','implementation_sha256','0'*64),('external_review','arithmetic_sha256','0'*64),('external_review','freeze_sha256','0'*64)]
    for group,key,value in mutations:
        q,fake,review=setup();grant=core.read(review)
        if group:
            target=Path(grant[group]['path']);obj=core.read(target);obj[key]=value;core.write(target,obj);grant[group]=core.binding(target)
        else:grant[key]=value
        core.write(review,grant);reject('reject_'+str(group)+'_'+key,lambda:q.validate(review))
    for scenario in ('success','review_drift_during_validate','child_failure_cleanup_raises','input_seal_drift','existing_stage'):
        q,fake,review=setup();q.validate_source_lock=lambda:None;q.attempt=lambda name:core.attempt(name,home=fake)
        plan=q.commands(models,core.read(review)['local_review']['path'],core.read(review)['external_review']['path']);launches=[]
        original_validate=q.validate
        if scenario=='review_drift_during_validate':
            def validate_and_drift(path):
                result=original_validate(path);obj=core.read(path);obj['passed']=False;core.write(path,obj);return result
            q.validate=validate_and_drift
        class Child:
            pid=70000002
            def __init__(self,code):self.code=code
            def wait(self,timeout=None):return self.code
            def poll(self):return self.code
            def terminate(self):raise AssertionError('completed synthetic child')
            def kill(self):raise AssertionError('completed synthetic child')
        def stop(records):
            if scenario=='child_failure_cleanup_raises':raise RuntimeError('synthetic cleanup error')
            return []
        q.ownership_helpers=lambda:types.SimpleNamespace(child_identity=lambda c:{'pid':c.pid,'created':123.},owned_descendants=lambda c:[],stop_recorded_descendants=stop)
        def launch(command,stdout,stderr):
            index=len(launches);stage=plan[index][0];launches.append(stage)
            Path(stdout).write_text('synthetic only');Path(stderr).write_text('')
            folder=fake/'stages'/stage;folder.mkdir()
            code=7 if scenario=='child_failure_cleanup_raises' and index==2 else 0
            core.write(folder/'status.json',{'state':'completed' if not code else 'failed'})
            name='all_local_selections.json' if stage=='local_selections_seal' else 'selection_seal.json'
            core.write(folder/name,{'synthetic':True});core.write(folder/'artifact_manifest.json',core.stage_manifest(folder))
            if scenario=='input_seal_drift':core.write(fake/'stages/local_geometry/metadata.json',{'synthetic':True,'drift':True})
            return Child(code)
        q.launch_child=launch
        if scenario=='existing_stage':(fake/'stages/local_select_01').mkdir()
        try:q.execute(review);error=None
        except BaseException as e:error=repr(e)
        folder=fake/'stages/selection_queue';state=core.read(folder/'queue_status.json') if (folder/'queue_status.json').exists() else None
        if scenario=='success':
            yes('success_all31_completed',len(launches)==31 and state['state']=='completed' and error is None)
            yes('success_still_pending_independent_selection_acceptance',core.read(folder/'summary.json')['independent_selection_output_review']=='pending')
            yes('success_truth_remains_unexecuted',core.read(folder/'summary.json')['truth_materialized'] is False)
        elif scenario=='review_drift_during_validate':yes('review_drift_rejected_before_child',len(launches)==0 and error is not None)
        elif scenario=='child_failure_cleanup_raises':
            yes('failed_third_child_stops',len(launches)==3 and error is not None and 'exit 7' in error)
            yes('primary_error_and_cleanup_both_recorded',state['state']=='failed' and 'exit 7' in state['error'] and 'descendants_error' in state['child_cleanup'])
            yes('partial_progress_and_claim_retained',len(state['completed'])==2 and (fake/'claims/selection_queue.json').exists())
        elif scenario=='input_seal_drift':yes('geometry_drift_stops_after_first_child',len(launches)==1 and state['state']=='failed' and error is not None)
        else:yes('preexisting_stage_rejected_before_child',len(launches)==0 and error is not None)
        before=len(launches);reject(scenario+'_retry_rejected',lambda:q.execute(review));yes(scenario+'_no_extra_child',len(launches)==before)
        scenarios.append({'scenario':scenario,'fixture':str(fake),'launches':len(launches),'error':error,'queue_state':state})
    yes('actual_source_unchanged_during_review',core.sha(source)==core.sha(snapshot))
    report={'passed':True,'check_count':len(checks),'checks':checks,'implementation':core.binding(source),'reviewer':core.binding(__file__),
      'prediction_queue_dependency':core.binding(HERE/'run_prediction_queue.py'),'ownership_dependency':core.binding(q.OWNER_SOURCE),'scenarios':scenarios,
      'actual_processes':0,'solver_calls':0,'model_forward_calls':0,'reserved_test_metadata_or_pixels_read':0,'actual_freeze_created':False,
      'longest_planned_stage_path_chars':max(len(str(p)) for p in production),'limits':'Stage consumer scientific/source checks are separately reviewed. Fixtures never execute those modules.'}
    core.write(out/'review.json',report)
    (out/'review.md').write_text('# Selection queue independent preparation review\n\nPASS: complete31 command inventory, root/source/freeze/accepted-input binding refusals, full mocked31-step lifecycle, permanent attempts, drift stop, and preservation of original child error despite cleanup failure. No actual process, test read, model or solver. Final source lock must include this module and the imported prediction queue. Actual selection output acceptance remains pending after queue completion.\n',encoding='utf-8')
    print(json.dumps({'out':str(out),'checks':len(checks),'source_sha256':core.sha(source),'review_sha256':core.sha(out/'review.json')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
