"""Bounded mocked subprocess lifecycle; no actual pipeline commands or reserved reads."""
from pathlib import Path
import datetime,json,sys,types,shutil
import core,run_prediction_queue as q
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def main():
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');out=ROOT/'provenance'/('confirmation_queue_readiness_'+stamp);out.mkdir()
    checks=[]
    def yes(name,condition):assert condition,name;checks.append({'name':name,'passed':True})
    def reject(name,fn):
        try:fn()
        except (ValueError,FileNotFoundError,FileExistsError,AssertionError):yes(name,True);return
        raise AssertionError(name)
    models=core.read(HERE/'contract.json')['roster']
    reject('actual_missing_freeze_blocks_queue',lambda:q.validate(out/'absent_review'))
    plan=q.commands(models)
    yes('fixed40_in_original_order',[x[0] for x in plan]==['inputs_local','inputs_weeds']+['predict_'+x['run_id'] for x in models]+['predictions_seal'])
    yes('no_truth_or_selection_command',all('truth' not in ' '.join(cmd) and 'select' not in ' '.join(cmd) for _,cmd in plan))
    paths=[HERE/'stages'/stage/name for stage,_ in plan for name in ('artifact_manifest.json','prediction_input_seal.json','map_index.json','predicted_scalars.csv','maps/source_long_window_00000.npz')]
    yes('production_longest_paths_below240',max(len(str(x)) for x in paths)<240)
    for scenario in ('success','child_failure','review_drift'):
        fake=ROOT/'cqueue'/stamp/scenario;fake.mkdir(parents=True)
        for name in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json'):core.write(fake/name,{'synthetic_only':True})
        review=fake/'review.json';core.write(review,{'synthetic_only':True})
        q.HERE=fake;plan=q.commands(models);count=[];stopped=[]
        q.validate=lambda p:(models,plan)
        q.validate_source_lock=lambda:None
        q.attempt=lambda name:core.attempt(name,home=fake)
        class Child:
            pid=1234567
            def __init__(self,code):self.returncode=code;self.terminated=False
            def wait(self,timeout=None):return self.returncode
            def poll(self):return self.returncode
            def terminate(self):self.terminated=True
            def kill(self):self.terminated=True
        q.ownership_helpers=lambda:types.SimpleNamespace(child_identity=lambda c:{'pid':c.pid,'created':1},owned_descendants=lambda c:[],stop_recorded_descendants=lambda records:stopped.append(records) or [])
        def launch(command,stdout,stderr):
            index=len(count);stage=plan[index][0];count.append(stage)
            Path(stdout).write_text('synthetic',encoding='utf-8');Path(stderr).write_text('',encoding='utf-8')
            folder=fake/'stages'/stage;folder.mkdir()
            code=1 if scenario=='child_failure' and index==5 else 0
            core.write(folder/'status.json',{'state':'failed' if code else 'completed'})
            if stage=='predictions_seal':core.write(folder/'all_predictions.json',{'scalar_rows':38792,'map_count':8268})
            core.write(folder/'artifact_manifest.json',core.stage_manifest(folder))
            if scenario=='review_drift' and index==3:core.write(review,{'synthetic_only':True,'changed':True})
            return Child(code)
        q.launch_child=launch
        if scenario=='success':
            produced=q.execute(review);core.completed_stage(produced)
            status=core.read(produced/'queue_status.json')
            yes('all40_serial_commands_completed',len(count)==40 and len(status['completed'])==40 and status['state']=='completed')
            yes('sealed_prediction_only_summary',core.read(produced/'summary.json')['truth_materialized'] is False)
        else:
            reject(scenario+'_stops_queue',lambda:q.execute(review))
            produced=fake/'stages/prediction_queue';status=core.read(produced/'queue_status.json')
            yes(scenario+'_partial_progress_preserved',len(count)==(6 if scenario=='child_failure' else 4) and status['state']=='failed')
            yes(scenario+'_failed_stage_retained',core.read(produced/'status.json')['state']=='failed' and (fake/'claims/prediction_queue.json').exists())
            if scenario=='child_failure':yes('owned_child_cleanup_only',stopped==[[]])
        before=len(count);reject(scenario+'_permanent_attempt_no_retry',lambda:q.execute(review));yes(scenario+'_no_new_child_on_retry',len(count)==before)
    review={'passed':True,'check_count':len(checks),'checks':checks,'source_sha256':core.sha(HERE/'run_prediction_queue.py'),'checker_sha256':core.sha(__file__),
       'ownership_dependency_sha256':core.sha(q.OWNER_SOURCE),'real_child_processes':0,'real_test_or_R6_forward':0,'reserved_inputs_read':0,'real_freeze_created':False,
       'longest_planned_path_characters':max(len(str(x)) for x in paths),'author_self_check':True}
    core.write(out/'focused_checks.json',review);print(json.dumps({'passed':True,'checks':len(checks),'out':str(out)}))
if __name__=='__main__':main()

