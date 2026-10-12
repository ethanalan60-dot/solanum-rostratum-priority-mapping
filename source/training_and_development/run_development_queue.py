"""Run the recorded development comparisons serially on a shared GPU."""
from pathlib import Path
import datetime, json, os, subprocess, sys, time, msvcrt
ROOT=Path(__file__).resolve().parent

def write_status(value):
    path=ROOT/'queue_status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)

def main():
    lock=(ROOT/'development_queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Another development queue holds the lock.')
    plan=json.loads((ROOT/'development_plan.json').read_text(encoding='utf-8'))
    logdir=ROOT/'queue_logs';logdir.mkdir(exist_ok=True)
    state={'pid':os.getpid(),'status':'running','started_at':datetime.datetime.now().astimezone().isoformat(),'finished':[],'failed':[],'active_run':None}
    env=os.environ.copy();env['PYTHONUTF8']='1'
    for item in plan['runs']:
        run_id=item['run_id'];runpath=ROOT/'runs'/run_id
        if (runpath/'status.json').exists():
            status=json.loads((runpath/'status.json').read_text())
            if status.get('status')=='completed':state['finished'].append(run_id);continue
        if (runpath/'config.json').exists():
            state['failed'].append({'run':run_id,'reason':'Prior incomplete or failed run exists; retained for review.'});continue
        cmd=[sys.executable,str(ROOT/'train_local.py'),'--run-id',run_id,'--variant',item['variant'],
             '--epochs',str(item['epochs']),'--seed',str(item['seed']),'--lr',str(item['lr']),
             '--batch-size',str(plan['batch_size']),'--gpu-memory-fraction',str(plan['gpu_memory_fraction']),
             '--coverage-weight',str(item['coverage_weight']),'--consistency-weight',str(item['consistency_weight'])]
        if item['scale_augmentation']:cmd.append('--scale-augmentation')
        state['active_run']=run_id;state['command']=cmd;write_status(state)
        print(json.dumps({'start':run_id,'at':datetime.datetime.now().isoformat()}),flush=True)
        with (logdir/f'{run_id}.log').open('w',encoding='utf-8') as out:
            proc=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=out,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            state['child_pid']=proc.pid;write_status(state)
            code=proc.wait()
        if code==0:state['finished'].append(run_id)
        else:state['failed'].append({'run':run_id,'exit_code':code})
        print(json.dumps({'end':run_id,'exit_code':code}),flush=True)
        state['active_run']=None;state.pop('child_pid',None);write_status(state)
        # Do not repeatedly crash or disturb concurrent research processes.
        if len(state['failed'])>=2:state['status']='needs_diagnosis';break
    else:state['status']='completed' if not state['failed'] else 'completed_with_failures'
    state['ended_at']=datetime.datetime.now().astimezone().isoformat();state['active_run']=None;write_status(state)

if __name__=='__main__':main()
