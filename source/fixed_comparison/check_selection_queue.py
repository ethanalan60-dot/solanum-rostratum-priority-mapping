"""Mocked31-child selection lifecycle; no actual process/data/model/solver."""
from pathlib import Path
import datetime,json,types
import core,run_selection_queue as q
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def main():
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');out=ROOT/'provenance'/('confirmation_selection_queue_checks_'+stamp);out.mkdir()
    checks=[]
    def yes(name,ok):assert ok,name;checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,FileNotFoundError,FileExistsError):yes(name,True);return
        raise AssertionError(name)
    models=core.read(HERE/'contract.json')['roster'];plan=q.commands(models,'local.json','external.json')
    yes('all31 in original roster order',[s for s,_ in plan]==[f'local_select_{i:02d}' for i in range(1,25)]+['local_selections_seal']+['select_'+r['run_id'] for r in models if r['dataset']=='weeds' and r['variant']!='regression'])
    yes('no truth or evaluation command',all('materialize-truth' not in cmd and 'evaluate' not in cmd for _,cmd in plan))
    reject('absent formal freeze rejects actual validation',lambda:q.validate(out/'absent.json'))
    original_capture=q.capture_bindings
    for scenario in ('success','child_failure','review_drift','validation_drift','cleanup_failure'):
        fake=ROOT/'csq'/stamp/scenario;fake.mkdir(parents=True)
        for name in ('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json','local_review.json','external_review.json','predictions_review.json','geometry_review.json'):core.write(fake/name,{'synthetic':True})
        review=fake/'review.json';core.write(review,{k:core.binding(fake/(k+'.json')) for k in ('local_review','external_review','predictions_review','geometry_review')})
        for name in ('predictions_seal','local_geometry'):
            f=fake/'stages'/name;f.mkdir(parents=True)
            core.write(f/('all_predictions.json' if name=='predictions_seal' else 'metadata.json'),{'synthetic':True});core.write(f/'artifact_manifest.json',{'synthetic':True})
        q.HERE=fake;plan=q.commands(models,fake/'local_review.json',fake/'external_review.json');count=[];stopped=[]
        q.validate_source_lock=lambda:None;q.attempt=lambda name:core.attempt(name,home=fake)
        def validate(path):
            if scenario=='validation_drift':core.write(review,{'changed':True})
            return models,plan
        q.validate=validate
        class Child:
            pid=111111
            def __init__(self,code):self.returncode=code
            def wait(self,timeout=None):return self.returncode
            def poll(self):return self.returncode
            def terminate(self):raise AssertionError('finished fake child should not be terminated')
            def kill(self):raise AssertionError('finished fake child should not be killed')
        def cleanup(records):
            stopped.append(records)
            if scenario=='cleanup_failure':raise RuntimeError('synthetic descendant cleanup failure')
            return []
        q.ownership_helpers=lambda:types.SimpleNamespace(child_identity=lambda c:{'pid':c.pid,'created':1},owned_descendants=lambda c:[],stop_recorded_descendants=cleanup)
        def launch(command,stdout,stderr):
            index=len(count);stage=plan[index][0];count.append(stage)
            Path(stdout).write_text('synthetic',encoding='utf-8');Path(stderr).write_text('',encoding='utf-8')
            f=fake/'stages'/stage;f.mkdir();code=7 if scenario in ('child_failure','cleanup_failure') and index==4 else 0
            core.write(f/'status.json',{'state':'failed' if code else 'completed'})
            core.write(f/('all_local_selections.json' if stage=='local_selections_seal' else 'selection_seal.json'),{'synthetic':True})
            core.write(f/'artifact_manifest.json',core.stage_manifest(f))
            if scenario=='review_drift' and index==3:core.write(fake/'geometry_review.json',{'changed':True})
            return Child(code)
        q.launch_child=launch
        if scenario=='success':
            result=q.execute(review);core.completed_stage(result);state=core.read(result/'queue_status.json')
            yes('all31 completed once',len(count)==31 and state['state']=='completed' and len(state['completed'])==31)
            yes('summary retains six external seals',len(core.read(result/'summary.json')['external_selection_seals'])==6)
            yes('summary does not authorize truth',core.read(result/'summary.json')['truth_materialized'] is False)
        else:
            reject(scenario+' rejected',lambda:q.execute(review))
            if scenario=='validation_drift':yes('review prevalidation drift zero child',len(count)==0);continue
            f=fake/'stages/selection_queue';state=core.read(f/'queue_status.json')
            yes(scenario+' failed preserved',state['state']=='failed' and core.read(f/'status.json')['state']=='failed' and (fake/'claims/selection_queue.json').exists())
            yes(scenario+' no later child',len(count)==(4 if scenario=='review_drift' else 5))
            if scenario=='cleanup_failure':yes('cleanup error retains original exit7','exit 7' in state['error'] and 'descendants_error' in state['child_cleanup'])
        before=len(count);reject(scenario+' no retry',lambda:q.execute(review));yes(scenario+' unchanged child count',len(count)==before)
    core.write(out/'focused_checks.json',{'passed':True,'checks':checks,'source_sha256':core.sha(HERE/'run_selection_queue.py'),'checker_sha256':core.sha(__file__),
        'real_children':0,'test_or_R6_accessed':False,'actual_freeze_created':False,'production_execution_authorized':False})
    print(json.dumps({'passed':True,'checks':len(checks),'out':str(out)}))
if __name__=='__main__':main()
