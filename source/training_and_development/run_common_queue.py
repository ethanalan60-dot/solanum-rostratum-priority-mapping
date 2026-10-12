"""Run a hash-pinned development plan serially, with a project-scoped lock.

This driver has separate status/log files from the legacy development queue.
Completed/failed run directories are never rewritten. Three consecutive failed
entries stop the queue. No sealed-test evaluation or R6 path exists here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

import psutil

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT/'common_queue.lock.json'
STATUS_PATH = ROOT/'common_queue_status.json'
SCRIPT_NAMES = {'train_common.py', 'run_common_queue.py', 'run_amp_diagnostics.py',
                'check_amp_step.py', 'check_train_common.py'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def project_processes(exclude_pids=()):
    """Read actual command lines; inspect script arguments, not arbitrary text."""
    found = []
    excluded = set(exclude_pids)
    for process in psutil.process_iter(['pid', 'name', 'exe', 'cmdline', 'create_time']):
        if process.pid in excluded:
            continue
        info = process.info
        if 'python' not in (info.get('name') or '').lower():
            continue
        for argument in info.get('cmdline') or []:
            if Path(argument).name.lower() not in SCRIPT_NAMES:
                continue
            candidate = Path(argument)
            try:
                candidate = candidate if candidate.is_absolute() else Path(process.cwd())/candidate
                matches = candidate.resolve().parent == ROOT.resolve()
            except (OSError, psutil.Error):
                # A named matching script with unreadable cwd cannot be assumed unrelated.
                matches = True
            if matches:
                found.append({key: info.get(key) for key in ('pid', 'exe', 'cmdline', 'create_time')})
                break
    return found


def verify_plan(path, expected_sha):
    if path.resolve().parent != ROOT or sha256(path) != expected_sha:
        raise RuntimeError('Queue plan path or SHA256 mismatch')
    plan = json.loads(path.read_text(encoding='utf-8'))
    if plan.get('version') not in ('common_queue_v1', 'common_queue_v2'):
        raise RuntimeError('Unknown queue plan version')
    run_prefix = 'dev_common_v2_' if plan['version'] == 'common_queue_v2' else 'dev_common_v1_'
    if Path(plan['python_executable']).resolve() != Path(sys.executable).resolve():
        raise RuntimeError('Queue is running under a different Python executable')
    if plan.get('test_accessed') is not False or plan.get('validation_R') != [1, 2, 4]:
        raise RuntimeError('Unexpected test or validation-scale policy')
    for name, digest in plan['source_sha256'].items():
        if Path(name).name != name or sha256(ROOT/name) != digest:
            raise RuntimeError(f'Queue source changed: {name}')
    for name, digest in plan['manifest_sha256'].items():
        if name not in ('train.json', 'val.json', 'DATASET_LOCK.json') or sha256(ROOT/'local_data_v2'/name) != digest:
            raise RuntimeError(f'Development manifest or lock changed: {name}')
    if plan['version'] == 'common_queue_v2':
        for item in plan['required_evidence']:
            evidence = (ROOT/item['path']).resolve()
            if not evidence.is_relative_to(ROOT.resolve()) or sha256(evidence) != item['sha256']:
                raise RuntimeError('AMP v2 prerequisite evidence changed')
        diagnostic = json.loads((ROOT/plan['diagnostic_summary']).read_text(encoding='utf-8'))
        if diagnostic.get('status') != 'completed' or len(diagnostic.get('runs', [])) != 3 or any(
                run['status'].get('status') != 'completed' or run['status'].get('executed_epochs') != 4
                for run in diagnostic['runs']):
            raise RuntimeError('AMP v2 full-manifest diagnostics are incomplete')
    expected_order = [(lr, encoder, variant) for lr in (1e-4, 3e-4)
                      for encoder in ('resnet18', 'convnextv2_tiny')
                      for variant in ('regression', 'hard_seg', 'soft_seg')]
    if len(plan['runs']) != 12 or len({item['run_id'] for item in plan['runs']}) != 12:
        raise RuntimeError('Expected 12 unique entries')
    actual_order = [(item['lr'], item['encoder'], item['variant']) for item in plan['runs']]
    if actual_order != expected_order:
        raise RuntimeError('Unexpected model/LR order')
    fixed = {'label_grid': 256, 'epochs': 24, 'seed': 17, 'batch_size': 4,
             'accumulation_steps': 2, 'sampler': 'balanced', 'gpu_memory_fraction': .16,
             'step_pause': .05, 'scale_augmentation': True, 'freeze_batchnorm': False,
             'weight_decay': 1e-4, 'threads': 2, 'amp': True}
    for item in plan['runs']:
        if not item['run_id'].startswith(run_prefix) or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for character in item['run_id']) or '..' in item['run_id']:
            raise RuntimeError('Invalid run ID')
        if any(item.get(key) != value for key, value in fixed.items()):
            raise RuntimeError('Run differs from fixed fair budget')
    return plan


def command_for(item, python_executable):
    command = [python_executable, '-X', 'utf8', '-B', '-s', str(ROOT/'train_common.py')]
    for key in ('run_id', 'encoder', 'variant', 'label_grid', 'epochs', 'seed', 'batch_size',
                'accumulation_steps', 'lr', 'sampler', 'gpu_memory_fraction', 'step_pause',
                'weight_decay', 'threads'):
        command.extend(['--'+key.replace('_', '-'), str(item[key])])
    command += ['--device', 'cuda', '--scale-augmentation', '--amp']
    return command


def run_queue(args):
    plan_path = Path(args.plan).resolve()
    plan = verify_plan(plan_path, args.plan_sha256)
    existing = project_processes(exclude_pids=[os.getpid()])
    if existing:
        raise RuntimeError('An existing project common queue/trainer is active: '+json.dumps(existing))
    token = uuid.uuid4().hex
    owner = {'pid': os.getpid(), 'process_create_time': psutil.Process().create_time(), 'token': token,
             'python': sys.executable, 'plan_path': str(plan_path), 'plan_sha256': args.plan_sha256,
             'started_utc': utc_now(), 'commandline': psutil.Process().cmdline()}
    # Never auto-steal even a stale lock; inspect owner identity before recovery.
    with LOCK_PATH.open('x', encoding='utf-8') as stream:
        json.dump(owner, stream, ensure_ascii=False, indent=2)
    attempt = None
    state = None
    active_child = None
    try:
        conflicting = project_processes(exclude_pids=[os.getpid()])
        if conflicting:
            raise RuntimeError('Common process appeared during lock acquisition: '+json.dumps(conflicting))
        attempt_id = datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+token[:8]
        attempt = ROOT/'common_queue_runs'/attempt_id
        attempt.mkdir(parents=True, exist_ok=False)
        (attempt/'source').mkdir()
        shutil.copy2(plan_path, attempt/plan_path.name)
        for name in plan['source_sha256']:
            shutil.copy2(ROOT/name, attempt/'source'/name)
        state = {'status': 'running', 'owner': owner, 'attempt_directory': str(attempt),
                 'plan_sha256': args.plan_sha256, 'results': [], 'active_run_id': None,
                 'active_child_pid': None, 'consecutive_failures': 0, 'test_accessed': False,
                 'total': len(plan['runs']), 'updated_utc': utc_now()}

        def save():
            state['updated_utc'] = utc_now()
            write_json(attempt/'status.json', state)
            write_json(STATUS_PATH, state)

        save()
        for index, item in enumerate(plan['runs'], 1):
            # Refuse source/plan drift between runs rather than silently mix versions.
            verify_plan(plan_path, args.plan_sha256)
            current_processes = project_processes(exclude_pids=[os.getpid()])
            if current_processes:
                raise RuntimeError('Another common process was detected before launch: '+json.dumps(current_processes))
            run_dir = ROOT/'runs'/item['run_id']
            state['active_run_id'] = item['run_id']
            state['index'] = index
            command = command_for(item, plan['python_executable'])
            if run_dir.exists():
                status_path = run_dir/'status.json'
                prior = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
                if prior.get('status') not in ('completed', 'failed'):
                    raise RuntimeError(f'Existing unresolved run directory is protected: {run_dir}')
                if prior['status'] == 'completed':
                    config = json.loads((run_dir/'config.json').read_text(encoding='utf-8'))
                    if any(config.get(key) != value for key, value in item.items() if key != 'index'):
                        raise RuntimeError(f'Existing completed run has different settings: {run_dir}')
                    if any(config.get('source_sha256', {}).get(name) != digest for name, digest in plan['source_sha256'].items()
                           if name in ('train_common.py', 'models_common.py', 'conservative_resample.py')):
                        raise RuntimeError(f'Existing completed run has different source: {run_dir}')
                failed = prior['status'] == 'failed'
                result = {'run_id': item['run_id'], 'status': 'skipped_existing_'+prior['status'],
                          'existing_status_path': str(status_path), 'timestamp_utc': utc_now()}
            else:
                log_path = attempt/(f'{index:02d}_'+item['run_id']+'.log')
                entry_path = attempt/(f'{index:02d}_'+item['run_id']+'.json')
                entry = {'run_id': item['run_id'], 'command': command, 'source_sha256': plan['source_sha256'],
                         'plan_sha256': args.plan_sha256, 'started_utc': utc_now(), 'log_path': str(log_path)}
                write_json(entry_path, entry)
                print(json.dumps({'starting': item['run_id'], 'index': index, 'log_path': str(log_path)}), flush=True)
                with log_path.open('x', encoding='utf-8') as log:
                    active_child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    state['active_child_pid'] = active_child.pid
                    try:
                        state['active_child_create_time'] = psutil.Process(active_child.pid).create_time()
                    except psutil.Error:
                        # A failed import may exit before psutil sees the PID.
                        # Continue to collect its exit code and count the failure.
                        state['active_child_create_time'] = None
                    entry['pid'] = active_child.pid
                    write_json(entry_path, entry)
                    save()
                    while active_child.poll() is None:
                        time.sleep(5)
                        if active_child.poll() is None:
                            status_path = run_dir/'status.json'
                            if status_path.exists():
                                try:
                                    state['active_training_status'] = json.loads(status_path.read_text(encoding='utf-8'))
                                except (OSError, json.JSONDecodeError):
                                    pass
                            save()
                    returncode = active_child.returncode
                status_path = run_dir/'status.json'
                final = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
                failed = returncode != 0 or final.get('status') != 'completed'
                result = {'run_id': item['run_id'], 'status': 'failed' if failed else 'completed',
                          'returncode': returncode, 'training_status': final, 'entry_path': str(entry_path),
                          'log_path': str(log_path), 'completed_utc': utc_now()}
                entry.update(result)
                write_json(entry_path, entry)
                active_child = None
            state['results'].append(result)
            state['consecutive_failures'] = state['consecutive_failures']+1 if failed else 0
            state['active_child_pid'] = None
            state.pop('active_child_create_time', None)
            state.pop('active_training_status', None)
            save()
            print(json.dumps({'finished': item['run_id'], 'status': result['status'],
                              'consecutive_failures': state['consecutive_failures']}), flush=True)
            if state['consecutive_failures'] >= 3:
                state['status'] = 'stopped_three_consecutive_failures'
                break
        else:
            state['status'] = 'completed_with_failures' if any('failed' in r['status'] for r in state['results']) else 'completed'
        state.update(active_run_id=None, completed_utc=utc_now())
        save()
    except BaseException as error:
        failure = {'status': 'queue_failed', 'error': str(error), 'traceback': traceback.format_exc(),
                   'failed_utc': utc_now(), 'owner': owner, 'test_accessed': False}
        if state is not None:
            state.update(failure)
            write_json(attempt/'status.json', state)
            write_json(STATUS_PATH, state)
        elif attempt is not None:
            write_json(attempt/'failure.json', failure)
        raise
    finally:
        # If the queue is interrupted while its owned child still runs, retain
        # the lock for inspection; do not kill unrelated or even active training.
        child_running = active_child is not None and active_child.poll() is None
        if not child_running:
            try:
                current = json.loads(LOCK_PATH.read_text(encoding='utf-8'))
                if current.get('token') == token:
                    LOCK_PATH.unlink()
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    run_queue(parser.parse_args())
