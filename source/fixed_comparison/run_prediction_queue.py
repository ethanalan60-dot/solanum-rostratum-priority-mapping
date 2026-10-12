"""One frozen40-command input/prediction queue. Truth, geometry selection and scoring are separate."""
from pathlib import Path
import argparse,importlib.util,json,os,subprocess,sys
from core import HERE,attempt,binding,bound_file,completed_stage,configure_cpu,read,require,sha,validate_freeze,validate_source_lock,write
OWNER_SOURCE=HERE.parent/'run_local_seed_postprocess_queue_v2.py'
OWNER_SHA='be53702860cd35418c832dd77f5ff7b736566589373ae868c4a523b318f9bfa7'

def ownership_helpers():
    # This frozen standard-library supervisor only defines functions on import.
    bound_file(OWNER_SOURCE,OWNER_SHA)
    spec=importlib.util.spec_from_file_location('confirmation_owned_process_helpers',OWNER_SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def commands(models):
    prefix=[sys.executable,'-X','utf8','-B','-s',str(HERE/'pipeline.py')]
    result=[('inputs_local',prefix+['prepare-inputs','--dataset','local']),('inputs_weeds',prefix+['prepare-inputs','--dataset','weeds'])]
    result += [('predict_'+x['run_id'],prefix+['predict','--run-id',x['run_id']]) for x in models]
    result.append(('predictions_seal',prefix+['seal-predictions']))
    require(len(result)==40 and len({x[0] for x in result})==40,'Fixed complete queue required')
    return result

def validate(review_path):
    contract,models,freeze=validate_freeze();review=read(review_path)
    require(review.get('passed') is True and review['scope']=='confirmation_prediction_queue','Root prediction-queue grant required')
    require(review['implementation_sha256']==sha(__file__) and review['freeze_sha256']==sha(HERE/'FREEZE.json'),'Queue source/freeze changed')
    require(review['run_ids']==[x['run_id'] for x in models] and review['R']==[1,2,4,6],'Queue population differs')
    require(review['test_inference_authorized'] is True and review['truth_materialization_authorized'] is False,'Wrong phase authorization')
    require(review['automatic_retry_authorized'] is False,'No automatic retry')
    bound_file(OWNER_SOURCE,OWNER_SHA)
    return models,commands(models)

def launch_child(command,stdout,stderr):
    with Path(stdout).open('x',encoding='utf-8') as so,Path(stderr).open('x',encoding='utf-8') as se:
        return subprocess.Popen(command,cwd=HERE,stdout=so,stderr=se,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))

def execute(review_path):
    watched={str(Path(__file__).resolve()):sha(__file__),str(Path(review_path).resolve()):sha(review_path),
             str(HERE/'FREEZE.json'):sha(HERE/'FREEZE.json'),str(HERE/'SOURCE_LOCK.json'):sha(HERE/'SOURCE_LOCK.json'),
             str(OWNER_SOURCE):OWNER_SHA}
    models,plan=validate(review_path)
    for path,digest in watched.items():bound_file(path,digest)
    for stage,_ in plan:
        require(not (HERE/'claims'/(stage+'.json')).exists() and not (HERE/'stages'/stage).exists(),'Existing downstream stage: '+stage)
    with attempt('prediction_queue') as out:
        configure_cpu(out);helpers=ownership_helpers();(out/'logs').mkdir()
        state={'state':'running','owner_pid':os.getpid(),'completed':[],'models':len(models),'steps':40,
               'truth_materialization_authorized':False,'freeze':binding(HERE/'FREEZE.json'),'root_review':binding(review_path)}
        write(out/'queue_status.json',state);child=None;command=None
        try:
            for index,(stage,command) in enumerate(plan,1):
                for path,digest in watched.items():bound_file(path,digest)
                validate_source_lock()
                state.update(stage=stage,step=index,command=command,child_pid=None)
                write(out/'queue_status.json',state)
                child=launch_child(command,out/'logs'/f'{index:02d}.stdout.log',out/'logs'/f'{index:02d}.stderr.log')
                child.owned_identity=helpers.child_identity(child)
                state.update(child_pid=child.pid,child_identity=child.owned_identity);write(out/'queue_status.json',state)
                code=child.wait();require(code==0,'Stage failed: '+stage+' exit '+str(code));child=None
                folder=HERE/'stages'/stage;completed_stage(folder)
                for path,digest in watched.items():bound_file(path,digest)
                validate_source_lock();state['completed'].append({'stage':stage,'artifact_manifest':binding(folder/'artifact_manifest.json')})
                write(out/'queue_status.json',state)
            require(len(state['completed'])==40,'Incomplete fixed prediction queue')
            seal=read(HERE/'stages/predictions_seal/all_predictions.json')
            require(seal['scalar_rows']==38792 and seal['map_count']==8268,'Global result counts differ')
            state.update(state='completed',stage=None,child_pid=None)
            write(out/'queue_status.json',state)
            write(out/'summary.json',{'all37_models_exported':True,'commands_completed':40,'all_predictions':binding(HERE/'stages/predictions_seal/all_predictions.json'),
                   'truth_materialized':False,'spatial_selection_executed':False,'checkpoint_reselected':False,'watched':watched})
        except BaseException as error:
            cleanup={}
            if child is not None:
                descendants=[]
                try:descendants=helpers.owned_descendants(child)
                except BaseException as e:cleanup['inventory_error']=repr(e)
                try:
                    if child.poll() is None:child.terminate()
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=20)
                except BaseException as e:cleanup['parent_error']=repr(e)
                try:cleanup['descendants']=helpers.stop_recorded_descendants(descendants)
                except BaseException as e:cleanup['descendants_error']=repr(e)
            state.update(state='failed',error=repr(error),child_cleanup=cleanup,no_retry=True)
            try:write(out/'queue_status.json',state)
            except BaseException as secondary:print(json.dumps({'queue_error':repr(error),'status_error':repr(secondary)}),file=sys.stderr,flush=True)
            raise
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root-review',required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--validate-only',action='store_true');g.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.validate_only:
        models,plan=validate(a.root_review);print(json.dumps({'passed':True,'models':len(models),'commands':len(plan),'test_files_opened':0,'models_loaded':0}))
    else:print(execute(a.root_review))
if __name__=='__main__':main()
