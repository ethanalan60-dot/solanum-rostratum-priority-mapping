"""Bounded replay of the three v1 failures: full train/val, 4 of 24-scheduled epochs."""
from pathlib import Path
import json
import subprocess
import sys
import uuid
import train_common as tc

if __name__ == '__main__':
    root=tc.ROOT
    destination=root/'provenance'/'amp_fix_20260916'
    summary_path=destination/'diagnostic_runs_v2.json'
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary={'started_utc':tc.utc_now(),'status':'running','test_accessed':False,'runs':[]}
    tc.write_json(summary_path,summary)
    for encoder,variant in [('resnet18','hard_seg'),('resnet18','soft_seg'),('convnextv2_tiny','regression')]:
        run_id=f'diag_common_amp_v2_{encoder}_{variant}_{uuid.uuid4().hex[:8]}'
        command=[sys.executable,'-X','utf8','-B','-s',str(root/'train_common.py'),'--run-id',run_id,
                 '--encoder',encoder,'--variant',variant,'--label-grid','256','--epochs','24',
                 '--diagnostic','--diagnostic-epochs','4','--lr','1e-4','--seed','17',
                 '--batch-size','4','--accumulation-steps','2','--sampler','balanced',
                 '--gpu-memory-fraction','.16','--step-pause','.05']
        log_path=destination/(run_id+'.log')
        print(json.dumps({'starting':run_id}),flush=True)
        with log_path.open('x',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        state=json.loads((root/'runs'/run_id/'status.json').read_text(encoding='utf-8'))
        summary['runs'].append({'run_id':run_id,'command':command,'returncode':result.returncode,
                                'status':state,'log_path':str(log_path)})
        tc.write_json(summary_path,summary)
        print(json.dumps({'finished':run_id,'returncode':result.returncode,'status':state}),flush=True)
        if result.returncode:
            summary.update(status='failed',completed_utc=tc.utc_now())
            tc.write_json(summary_path,summary)
            raise RuntimeError('Diagnostic failure retained; do not start production queue')
    summary.update(status='completed',completed_utc=tc.utc_now())
    tc.write_json(summary_path,summary)
