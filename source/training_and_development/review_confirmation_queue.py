"""Independent queue lifecycle mocks; never creates a subprocess or reads reserved data."""
from pathlib import Path
import argparse, datetime, importlib.util, json, sys, types

ROOT = Path(__file__).resolve().parent
HERE = ROOT / 'confirmation_v1'
sys.path.insert(0, str(HERE))
import core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=HERE/'run_prediction_queue.py')
    args = parser.parse_args()
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = ROOT/'provenance'/('confirmation_queue_independent_'+stamp)
    (out/'source').mkdir(parents=True)
    snapshot = out/'source'/'run_prediction_queue.py'
    snapshot.write_bytes(args.source.read_bytes())
    (out/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    fixture = ROOT/'cqi'/stamp
    fixture.mkdir(parents=True)
    models = core.read(HERE/'contract.json')['roster']
    checks = []

    def check(name, condition):
        checks.append({'name':name, 'passed':bool(condition)})

    def load():
        spec = importlib.util.spec_from_file_location('qreview_'+str(len(checks)), snapshot)
        q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(q)
        return q

    q=load(); plan=q.commands(models)
    check('fixed_40_commands', len(plan)==40)
    check('exact_37_ordered_models', [x[0] for x in plan[2:-1]]==['predict_'+m['run_id'] for m in models])
    check('two_inputs_then_global_seal', [x[0] for x in plan[:2]]==['inputs_local','inputs_weeds'] and plan[-1][0]=='predictions_seal')
    check('only_prediction_phase', all(cmd[6:] in (['prepare-inputs','--dataset','local'],['prepare-inputs','--dataset','weeds'],['seal-predictions']) or cmd[6:8]==['predict','--run-id'] for _,cmd in plan))
    check('same_interpreter_flags', all(cmd[:5]==[sys.executable,'-X','utf8','-B','-s'] for _,cmd in plan))
    check('no_selection_or_retry_flags', all(not set(cmd)&{'--epoch','--checkpoint','--test','--R','--retry'} for _,cmd in plan))
    check('helper_source_bound', core.sha(q.OWNER_SOURCE)==q.OWNER_SHA)
    # The original roster is metadata only. No checkpoint/test data are loaded here.
    production_paths=[HERE/'stages'/stage/name for stage,_ in plan for name in ('artifact_manifest.json.tmp','prediction_input_seal.json.tmp','maps/source_long_window_00000.npz')]
    check('declared_production_path_examples_below_240',max(len(str(x)) for x in production_paths)<240)
    scenarios=[]
    for number,scenario in enumerate(('success','validation_watch_drift','cleanup_error','source_drift','existing_child_stage'),1):
        q=load(); fake=fixture/str(number);fake.mkdir()
        for name in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json'):
            core.write(fake/name,{'synthetic':True})
        q.HERE=fake
        q.validate_freeze=lambda:({},models,{})
        review=fake/'review.json'
        core.write(review,{'passed':True,'scope':'confirmation_prediction_queue',
            'implementation_sha256':core.sha(snapshot),'freeze_sha256':core.sha(fake/'FREEZE.json'),
            'run_ids':[m['run_id'] for m in models],'R':[1,2,4,6],
            'test_inference_authorized':True,'truth_materialization_authorized':False,'automatic_retry_authorized':False})
        original_validate=q.validate
        if scenario=='validation_watch_drift':
            def drift(path):
                result=original_validate(path)
                value=core.read(path);value['passed']=False;value['test_inference_authorized']=False
                core.write(path,value)
                return result
            q.validate=drift
        q.validate_source_lock=lambda:None
        q.attempt=lambda name:core.attempt(name,home=fake)
        launches=[]; cleanup_calls=[]
        class Child:
            pid=70000001
            def __init__(self,code):self.code=code
            def wait(self,timeout=None):return self.code
            def poll(self):return self.code
            def terminate(self):raise AssertionError('No live fake parent')
            def kill(self):raise AssertionError('No live fake parent')
        def stop(records):
            cleanup_calls.append(records)
            if scenario=='cleanup_error':raise RuntimeError('MOCK cleanup helper failed')
            return []
        q.ownership_helpers=lambda:types.SimpleNamespace(child_identity=lambda child:{'pid':child.pid,'created':123.0},owned_descendants=lambda child:[],stop_recorded_descendants=stop)
        fakeplan=q.commands(models)
        def launch(command,stdout,stderr):
            stage=fakeplan[len(launches)][0];launches.append(stage)
            Path(stdout).write_text('mock only',encoding='utf-8');Path(stderr).write_text('',encoding='utf-8')
            folder=fake/'stages'/stage;folder.mkdir()
            code=7 if scenario=='cleanup_error' else 0
            core.write(folder/'status.json',{'state':'completed' if not code else 'failed'})
            if stage=='predictions_seal':core.write(folder/'all_predictions.json',{'scalar_rows':38792,'map_count':8268})
            core.write(folder/'artifact_manifest.json',core.stage_manifest(folder))
            if scenario=='source_drift':core.write(fake/'SOURCE_LOCK.json',{'synthetic':True,'modified':True})
            return Child(code)
        q.launch_child=launch
        if scenario=='existing_child_stage':(fake/'stages'/'inputs_local').mkdir(parents=True)
        try:
            q.execute(review);error=None
        except BaseException as e:error=repr(e)
        folder=fake/'stages'/'prediction_queue'
        state=core.read(folder/'status.json') if (folder/'status.json').exists() else None
        queue=core.read(folder/'queue_status.json') if (folder/'queue_status.json').exists() else None
        if scenario=='success':
            check('success_exactly_40',len(launches)==40 and error is None and state['state']=='completed')
            check('success_no_truth_or_reselection',core.read(folder/'summary.json')['truth_materialized'] is False and core.read(folder/'summary.json')['checkpoint_reselected'] is False)
        elif scenario=='validation_watch_drift':
            check('review_drift_before_watch_rejected_before_child',len(launches)==0 and error is not None)
        elif scenario=='cleanup_error':
            check('primary_child_error_preserved',error is not None and 'Stage failed:' in error and 'exit 7' in error)
            check('cleanup_failure_queue_state_preserved',queue is not None and queue['state']=='failed')
            check('core_failure_retained',state is not None and state['state']=='failed')
        elif scenario=='source_drift':
            check('source_drift_stops_after_one_child',len(launches)==1 and error is not None and state['state']=='failed')
        else:
            check('existing_downstream_stage_stops_before_child',len(launches)==0 and error is not None)
        before=len(launches)
        try:q.execute(review);second_error=None
        except BaseException as e:second_error=repr(e)
        check(scenario+'_no_new_child_on_second_call',len(launches)==before and second_error is not None)
        scenarios.append({'scenario':scenario,'fixture':str(fake),'launch_count':len(launches),'exception':error,'core_state':state,'queue_state':queue,'cleanup_calls':cleanup_calls})
    report={'passed':all(c['passed'] for c in checks),'implementation':core.binding(args.source),'snapshot':core.binding(snapshot),
        'reviewer':core.binding(__file__),'helper':core.binding(q.OWNER_SOURCE),'check_count':len(checks),'checks':checks,'scenarios':scenarios,
        'limits':{'actual_subprocesses':0,'checkpoint_loads':0,'model_forwards':0,'reserved_test_metadata_or_pixels_read':0,'real_freeze_created':False},
        'longest_declared_production_example_chars':max(len(str(p)) for p in production_paths),
        'path_limit_note':'Actual member-ID paths must be checked once authorized metadata is materialized; no test metadata read to infer them here.'}
    core.write(out/'review.json',report)
    failures=[x['name'] for x in checks if not x['passed']]
    (out/'review.md').write_text('# Independent confirmation prediction-queue review\n\n'+('PASS for preparation.' if report['passed'] else 'Fix required before execution: '+', '.join(failures))+ '\n\nAll lifecycle children were mocks. No test metadata, data arrays, checkpoint loads, inference or actual Popen calls. Original development checkpoint selection is retained through frozen pipeline gates. Windows parent log handles close after Popen returns; child handles are inherited independently.\n',encoding='utf-8')
    print(json.dumps({'out':str(out),'passed':report['passed'],'checks':len(checks),'failures':failures,'source_sha256':core.sha(args.source),'review_sha256':core.sha(out/'review.json')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
