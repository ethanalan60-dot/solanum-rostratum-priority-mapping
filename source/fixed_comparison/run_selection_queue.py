"""Fixed31-command label-free selection queue; requires audited prediction/geometry stages."""
from pathlib import Path
import argparse,json,os,subprocess,sys
from core import HERE,attempt,binding,bound_file,completed_stage,configure_cpu,read,require,sha,validate_freeze,validate_source_lock,write
from run_prediction_queue import ownership_helpers,launch_child,OWNER_SOURCE,OWNER_SHA

def commands(models,local_review,external_review):
    local=[x for x in models if x['dataset']=='local' and x['variant']!='regression']
    external=[x for x in models if x['dataset']=='weeds' and x['variant']!='regression']
    require(len(local)==24 and len(external)==6,'Wrong selection population')
    python=[sys.executable,'-X','utf8','-B','-s']
    result=[(f'local_select_{i:02d}',python+[str(HERE/'local_management.py'),'--root-review',str(local_review),'--select-run',x['run_id']]) for i,x in enumerate(local,1)]
    result.append(('local_selections_seal',python+[str(HERE/'local_management.py'),'--root-review',str(local_review),'--seal-population']))
    result.extend(('select_'+x['run_id'],python+[str(HERE/'external_spatial.py'),'select','--run-id',x['run_id'],'--root-review',str(external_review)]) for x in external)
    require(len(result)==31 and len({x[0] for x in result})==31,'Wrong selection command inventory')
    return result

def capture_bindings(review_path):
    watched={str(Path(__file__).resolve()):sha(__file__),str(Path(review_path).resolve()):sha(review_path),
      str(HERE/'FREEZE.json'):sha(HERE/'FREEZE.json'),str(HERE/'SOURCE_LOCK.json'):sha(HERE/'SOURCE_LOCK.json'),str(OWNER_SOURCE):OWNER_SHA}
    review=read(review_path)
    for key in ('local_review','external_review','predictions_review','geometry_review'):
        item=review[key];bound_file(item['path'],item['sha256']);watched[str(Path(item['path']).resolve())]=item['sha256']
    for name in ('stages/predictions_seal/all_predictions.json','stages/predictions_seal/artifact_manifest.json',
                 'stages/local_geometry/metadata.json','stages/local_geometry/artifact_manifest.json'):
        watched[str(HERE/name)]=sha(HERE/name)
    return watched

def validate(review_path):
    contract,models,_=validate_freeze();review=read(review_path)
    require(review.get('passed') is True and review['scope']=='confirmation_selection_queue','Root selection queue grant required')
    require(review['implementation_sha256']==sha(__file__) and review['freeze_sha256']==sha(HERE/'FREEZE.json'),'Queue source/freeze changed')
    expected=[x['run_id'] for x in models if x['variant']!='regression']
    require(review['run_ids']==expected and review['R']==[1,2,4,6],'Selection population changed')
    require(review['label_free_selection_authorized'] is True and review['truth_materialization_authorized'] is False and review['automatic_retry_authorized'] is False,'Wrong phase')
    pred=HERE/'stages/predictions_seal';geo=HERE/'stages/local_geometry'
    completed_stage(pred);completed_stage(geo)
    from pipeline import validate_global_predictions
    validate_global_predictions()
    for key in ('local_review','external_review','predictions_review','geometry_review'):bound_file(review[key]['path'],review[key]['sha256'])
    pa=read(review['predictions_review']['path']);ga=read(review['geometry_review']['path'])
    require(pa.get('passed') is True and pa['scope']=='confirmation_predictions_independent' and pa['all_predictions_sha256']==sha(pred/'all_predictions.json'),'Prediction acceptance missing')
    require(ga.get('passed') is True and ga['scope']=='confirmation_geometry_independent' and ga['metadata_sha256']==sha(geo/'metadata.json'),'Geometry acceptance missing')
    lr=read(review['local_review']['path']);er=read(review['external_review']['path'])
    require(lr.get('passed') is True and lr['scope']=='confirmation_local_management' and lr['implementation_sha256']==sha(HERE/'local_management.py'),'Local execution review missing')
    require(lr['freeze_sha256']==sha(HERE/'FREEZE.json') and lr['all_predictions_sha256']==sha(pred/'all_predictions.json') and lr['geometry_metadata_sha256']==sha(geo/'metadata.json'),'Local review inputs changed')
    require(lr['run_ids']==[x['run_id'] for x in models if x['dataset']=='local' and x['variant']!='regression'] and lr['case_count_per_model']==120,'Local review cases changed')
    require(er.get('passed') is True and er['scope']=='confirmation_external_selection_execution' and er['implementation_sha256']==sha(HERE/'external_spatial.py'),'External execution review missing')
    require(er['freeze_sha256']==sha(HERE/'FREEZE.json') and er['arithmetic_sha256']==sha(HERE/'external_arithmetic.py'),'External review inputs changed')
    return models,commands(models,review['local_review']['path'],review['external_review']['path'])

def execute(review_path):
    watched=capture_bindings(review_path);models,plan=validate(review_path)
    for path,digest in watched.items():bound_file(path,digest)
    for stage,_ in plan:require(not (HERE/'claims'/(stage+'.json')).exists() and not (HERE/'stages'/stage).exists(),'Existing downstream stage: '+stage)
    with attempt('selection_queue') as out:
        configure_cpu(out);helpers=ownership_helpers();(out/'logs').mkdir()
        state={'state':'running','owner_pid':os.getpid(),'completed':[],'models':30,'steps':31,'truth_materialization_authorized':False,
               'freeze':binding(HERE/'FREEZE.json'),'root_review':binding(review_path)}
        write(out/'queue_status.json',state);child=None
        try:
            for index,(stage,command) in enumerate(plan,1):
                for path,digest in watched.items():bound_file(path,digest)
                validate_source_lock();state.update(stage=stage,step=index,command=command,child_pid=None);write(out/'queue_status.json',state)
                child=launch_child(command,out/'logs'/f'{index:02d}.stdout.log',out/'logs'/f'{index:02d}.stderr.log')
                child.owned_identity=helpers.child_identity(child);state.update(child_pid=child.pid,child_identity=child.owned_identity);write(out/'queue_status.json',state)
                code=child.wait();require(code==0,'Stage failed: '+stage+' exit '+str(code));child=None
                folder=HERE/'stages'/stage;completed_stage(folder)
                for path,digest in watched.items():bound_file(path,digest)
                validate_source_lock();state['completed'].append({'stage':stage,'artifact_manifest':binding(folder/'artifact_manifest.json')});write(out/'queue_status.json',state)
            require(len(state['completed'])==31,'Incomplete selection queue')
            state.update(state='completed',stage=None,child_pid=None);write(out/'queue_status.json',state)
            write(out/'summary.json',{'commands_completed':31,'segmentation_models':30,'local_policy_cases':2880,'external_policy_cases':18720,
              'local_population':binding(HERE/'stages/local_selections_seal/all_local_selections.json'),
              'external_selection_seals':[binding(HERE/'stages'/('select_'+x['run_id'])/'selection_seal.json') for x in models if x['dataset']=='weeds' and x['variant']!='regression'],
              'truth_materialized':False,'independent_selection_output_review':'pending','watched':watched})
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
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--validate-only',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.validate_only:
        models,plan=validate(a.root_review);print(json.dumps({'passed':True,'models':30,'commands':len(plan),'truth_accessed':False}))
    else:print(execute(a.root_review))
if __name__=='__main__':main()
