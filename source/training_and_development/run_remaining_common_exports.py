"""Bounded sequential CPU export/audit driver for common-v2 plan entries 7..12.

No training, model selection, test/R6, geometry or management evaluation.
Writes only new exports and this batch's provenance records.
"""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, time

ROOT=Path(__file__).resolve().parent
BATCH=ROOT/'provenance/remaining_common_exports_20260916_0432'
EXPORT_SHA='56319f148fadc69b0cdc88c3c44f043ba020e812f2f4f919480304fe155dbecc'
PLAN_SHA='faefb03b14e4c44360ded4d065ffbe873d4e8512139a68d56e15ee90a2185a95'
def read(p): return json.loads(Path(p).read_text('utf8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.datetime.now().astimezone().isoformat()
def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf8');tmp.replace(p)
def emit(x): print(json.dumps(x,ensure_ascii=True),flush=True)
def existing_exports(run):
    found=[]
    for p in run.glob('common_validation_export_*'):
        if not (p/'status.json').exists(): raise RuntimeError('Untracked existing export: '+str(p))
        found.append((p,read(p/'status.json')))
    return found
def assert_no_export_process():
    cmd="Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object { $_.CommandLine -like '*export_common_validation.py*' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    q=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True,check=True)
    assert not q.stdout.strip(),'Another export process exists: '+q.stdout

def main():
    BATCH.mkdir(exist_ok=True,parents=True)
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONNOUSERSITE='1')
    temp=ROOT/'cpu_export_tmp';temp.mkdir(exist_ok=True);env['TEMP']=env['TMP']=str(temp)
    assert sha(ROOT/'export_common_validation.py')==EXPORT_SHA and sha(ROOT/'common_queue_plan_v2.json')==PLAN_SHA
    audit_source=ROOT/'audit_remaining_common_exports.py';audit_sha=sha(audit_source)
    plan=read(ROOT/'common_queue_plan_v2.json');entries=[e for e in plan['runs'] if 7<=e['index']<=12]
    assert len(entries)==6
    driver={'state':'running','pid':os.getpid(),'created_at':now(),'driver_sha256':sha(__file__),'exporter_sha256':EXPORT_SHA,
        'auditor_sha256':audit_sha,'plan_sha256':PLAN_SHA,'device':'cpu','threads':2,'temp':str(temp),'runs':[],
        'test_accessed':False,'R6_accessed':False,'training_started':False}
    write(BATCH/'driver_status.json',driver)
    try:
        for e in entries:
            rid=e['run_id'];idx=e['index'];run=ROOT/'runs'/rid;start=time.monotonic();last=None
            while True:
                state=read(run/'status.json')
                if state['status']=='completed': break
                assert state['status']=='running','Run not healthy: '+rid
                if time.monotonic()-start>1800: raise RuntimeError('Full training completion still pending after bounded wait: '+rid)
                epoch=state.get('epoch')
                driver.update(stage='waiting_for_full_training',waiting_run=rid,waiting_epoch=epoch,updated_at=now())
                write(BATCH/'driver_status.json',driver)
                if epoch!=last: emit({'waiting_full_training':rid,'epoch':epoch});last=epoch
                time.sleep(10)
            cfg=read(run/'config.json')
            assert state['executed_epochs']==24 and cfg['n_train']==985 and cfg['n_val']==357
            assert cfg['run_id']==rid and not cfg['test_accessed'] and not state['test_accessed']
            assert sha(ROOT/'export_common_validation.py')==EXPORT_SHA and sha(audit_source)==audit_sha
            before=existing_exports(run);assert not before,'Already exported or attempted; refuse duplicate: '+rid
            assert_no_export_process()
            command=[sys.executable,'-X','utf8','-B','-s',str(ROOT/'export_common_validation.py'),'--run-id',rid]
            prefix=BATCH/('index_%02d'%idx)
            record={'index':idx,'run_id':rid,'state':'export_running','command':command,'started_at':now(),
                'run_status_sha256_at_launch':sha(run/'status.json'),'original_best_epoch':state['best_epoch'],
                'export_source_sha256':EXPORT_SHA,'device':'cpu','threads':2}
            with prefix.with_name(prefix.name+'_stdout.log').open('w',encoding='utf8') as out, prefix.with_name(prefix.name+'_stderr.log').open('w',encoding='utf8') as err:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err)
                record['pid']=child.pid;driver['runs'].append(record);driver.update(stage='export',active_run=rid,active_pid=child.pid,updated_at=now())
                write(BATCH/'driver_status.json',driver);write(prefix.with_name(prefix.name+'_launch.json'),record);emit(record)
                code=child.wait()
            assert code==0,'Exporter failed '+rid+' code='+str(code)
            found=existing_exports(run);assert len(found)==1
            directory,export_state=found[0]
            assert export_state['status']=='completed' and export_state['pid']==child.pid
            assert export_state['completed_windows']==357 and export_state['completed_factor_predictions']==1071
            record.update(state='audit_running',directory=str(directory),export_status=export_state,audit_started_at=now())
            driver.update(stage='saved_output_audit',active_pid=None,updated_at=now());write(BATCH/'driver_status.json',driver)
            audit_out=prefix.with_name(prefix.name+'_audit.json')
            audit_command=[sys.executable,'-X','utf8','-B','-s',str(audit_source),'--export',str(directory),'--output',str(audit_out)]
            with prefix.with_name(prefix.name+'_audit_stdout.log').open('w',encoding='utf8') as out,prefix.with_name(prefix.name+'_audit_stderr.log').open('w',encoding='utf8') as err:
                audited=subprocess.run(audit_command,cwd=ROOT,env=env,stdout=out,stderr=err)
            assert audited.returncode==0,'Saved output audit failed: '+rid
            audit=read(audit_out);assert audit['all_checks_passed'] and len(audit['runs'])==1
            record.update(state='completed',completed_at=now(),audit=str(audit_out),audit_sha256=sha(audit_out),
                metadata_sha256=sha(directory/'metadata.json'),checkpoint_epoch=audit['runs'][0]['checkpoint_epoch'],
                map_scalar_max_difference=audit['runs'][0]['independent_map_scalar_max_difference'])
            write(prefix.with_name(prefix.name+'_completed.json'),record);driver.update(updated_at=now());write(BATCH/'driver_status.json',driver)
            emit({'completed':rid,'path':str(directory),'audit':str(audit_out),'best_epoch':record['checkpoint_epoch'],'max_map_scalar_difference':record['map_scalar_max_difference']})
        driver.update(state='completed',stage='completed',completed_at=now(),active_run=None,active_pid=None)
        write(BATCH/'driver_status.json',driver);emit({'batch_completed':str(BATCH),'runs':len(driver['runs'])})
    except BaseException as err:
        driver.update(state='failed',error=repr(err),failed_at=now());write(BATCH/'driver_status.json',driver);raise

if __name__=='__main__':main()
