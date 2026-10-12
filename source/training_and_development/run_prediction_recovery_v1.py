"""Explicit one-time recovery of a pre-forward Windows CUDA visibility failure.

No frozen scientific file is edited. The old failed queue stays in place; only
the exact failed, empty prediction slot is archived with its claim and lineage.
Completed input stages are reused. This is not a retry loop.
"""
from pathlib import Path
import argparse, datetime, importlib.util, json, os, shutil, subprocess, sys, traceback

ROOT = Path(__file__).resolve().parent
CONF = ROOT / 'confirmation_v1'
sys.path.insert(0, str(CONF))
import core

FIRST = 'predict_dev_common_v2_lr1e4_resnet18_regression_L256_s17'
EXPECTED_FREEZE = '24f8d1605c58df6bd6169b8434a451975abca0c3f8176e9c55c252abf8f3e732'
OWNER_SHA = 'be53702860cd35418c832dd77f5ff7b736566589373ae868c4a523b318f9bfa7'


def probe():
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    env['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
    code = ("import os,json;os.environ['CUDA_VISIBLE_DEVICES']='';import torch;"
            "x=torch.zeros(1,device='cpu');print(json.dumps({'nvml':os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'],"
            "'visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'available':torch.cuda.is_available(),"
            "'count':torch.cuda.device_count(),'device':str(x.device),'version':str(torch.__version__)}))")
    result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-s', '-c', code],
                            env=env, capture_output=True, text=True,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    core.require(result.returncode == 0, 'Environment probe failed: ' + result.stderr)
    value = json.loads(result.stdout)
    core.require(value['available'] is False and value['count'] == 0 and value['device'] == 'cpu', 'CPU environment not established')
    core.require(value['visible'] == '' and value['nvml'] == '1', 'Probe did not exercise frozen visibility setting')
    return value, env


def preflight(review_path):
    watched = {str(Path(__file__).resolve()): core.sha(__file__), str(Path(review_path).resolve()): core.sha(review_path)}
    review = core.read(review_path)
    core.require(review.get('passed') is True and review['scope'] == 'manual_pre_forward_prediction_recovery_v1', 'Explicit recovery grant required')
    core.require(review['implementation_sha256'] == core.sha(__file__) and review['freeze_sha256'] == EXPECTED_FREEZE, 'Wrong recovery source/freeze')
    core.require(review['automatic_retry'] is False and review['scientific_source_changes'] is False and review['truth_access'] is False, 'Wrong recovery scope')
    core.bound_file(CONF / 'FREEZE.json', EXPECTED_FREEZE)
    _, models, _ = core.validate_freeze()
    core.require(review['run_ids'] == [m['run_id'] for m in models], 'Wrong recovery population')
    old_queue = CONF / 'stages/prediction_queue'
    queue_status = core.read(old_queue / 'queue_status.json')
    core.require(queue_status['state'] == 'failed' and queue_status['stage'] == FIRST and queue_status['step'] == 3, 'Unexpected previous queue failure')
    core.require([r['stage'] for r in queue_status['completed']] == ['inputs_local', 'inputs_weeds'], 'Unexpected completed steps')
    expected_files = {'failure.json', 'input_seal.json', 'status.json'}
    failed = CONF / 'stages' / FIRST
    manifest = core.read(failed / 'artifact_manifest.json')
    core.require(set(manifest['files']) == expected_files, 'Failed prediction contains unexpected artifacts')
    core.require({p.relative_to(failed).as_posix() for p in failed.rglob('*') if p.is_file()} == expected_files | {'artifact_manifest.json'}, 'Unrecorded prediction artifacts')
    failure = core.read(failed / 'failure.json')
    core.require(failure['state'] == 'failed' and failure['stage'] == FIRST and failure['error'] == "ValueError('GPU must be hidden')", 'Wrong failure point')
    core.require('pipeline.py' in failure['traceback'] and 'line 30, in export_run' in failure['traceback'], 'Failure was not the frozen pre-load assertion')
    for folder in (old_queue, failed):
        man = core.read(folder / 'artifact_manifest.json')
        core.require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'} == set(man['files']), 'Failed-stage inventory changed')
        for name, digest in man['files'].items():
            p = core.resolve_under(folder, name); core.bound_file(p, digest); watched[str(p)] = digest
        watched[str(folder / 'artifact_manifest.json')] = core.sha(folder / 'artifact_manifest.json')
    for name in ('inputs_local', 'inputs_weeds'):
        folder = CONF / 'stages' / name
        core.completed_stage(folder)
        watched[str(folder / 'artifact_manifest.json')] = core.sha(folder / 'artifact_manifest.json')
        watched[str(folder / 'input_manifest.json')] = core.sha(folder / 'input_manifest.json')
    for name in ('prediction_queue', FIRST):
        path = CONF / 'claims' / (name + '.json'); watched[str(path)] = core.sha(path)
    core.require(not (CONF / 'stages/truth_local').exists() and not (CONF / 'stages/truth_weeds').exists(), 'Truth phase already started')
    prefix = [sys.executable, '-X', 'utf8', '-B', '-s', str(CONF / 'pipeline.py')]
    plan = [('predict_' + m['run_id'], prefix + ['predict', '--run-id', m['run_id']]) for m in models]
    plan.append(('predictions_seal', prefix + ['seal-predictions']))
    core.require(len(plan) == 38 and plan[0][0] == FIRST, 'Wrong recovery plan')
    for name, _ in plan[1:]:
        core.require(not (CONF / 'stages' / name).exists() and not (CONF / 'claims' / (name + '.json')).exists(), 'Downstream stage already attempted')
    for name in ('SOURCE_LOCK.json', 'FREEZE.json', 'contract.json', 'checkpoints.json'):
        watched[str(CONF / name)] = core.sha(CONF / name)
    for path, digest in review['evidence_sha256'].items():
        core.bound_file(path, digest); watched[str(Path(path).resolve())] = digest
    return models, plan, watched


def execute(review_path):
    models, plan, watched = preflight(review_path)
    probe_result, env = probe()
    control = ROOT / 'confirmation_runtime'
    out = ROOT / 'provenance' / ('prediction_recovery_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    claim = control / 'PREDICTION_RECOVERY_V1_CLAIM.json'
    with claim.open('x', encoding='utf-8') as stream:
        json.dump({'created_at': core.now(), 'output': str(out), 'root_review': core.binding(review_path), 'source': core.binding(__file__)}, stream, ensure_ascii=False, indent=2)
    child = None; helpers = None
    state = {'state': 'initializing', 'owner_pid': os.getpid(), 'completed': [], 'models': len(models), 'steps': 38, 'automatic_retry': False, 'truth_access': False}
    try:
        out.mkdir(); (out / 'logs').mkdir(); shutil.copy2(__file__, out / 'recovery_source.py')
        core.write(out / 'environment_probe.json', probe_result)
        for path, digest in watched.items(): core.bound_file(path, digest)
        helper_path = ROOT / 'run_local_seed_postprocess_queue_v2.py'; core.bound_file(helper_path, OWNER_SHA)
        spec = importlib.util.spec_from_file_location('recovery_owned_helpers', helper_path)
        helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
        old_stage = CONF / 'stages' / FIRST; old_claim = CONF / 'claims' / (FIRST + '.json')
        core.require(old_stage.resolve().is_relative_to((CONF / 'stages').resolve()) and old_claim.resolve().is_relative_to((CONF / 'claims').resolve()) and out.resolve().is_relative_to((ROOT / 'provenance').resolve()), 'Archive path escaped workspace')
        archived_stage = out / 'original_failed_prediction'; archived_claim = out / 'original_failed_claim.json'
        old_stage.rename(archived_stage); old_claim.rename(archived_claim)
        moved = {}
        for path, digest in watched.items():
            p = Path(path)
            if p.is_relative_to(old_stage): p = archived_stage / p.relative_to(old_stage)
            elif p == old_claim: p = archived_claim
            core.bound_file(p, digest); moved[str(p)] = digest
        watched = moved
        lineage = {'scope': 'explicit_environment_only_recovery', 'created_at': core.now(), 'freeze_sha256': EXPECTED_FREEZE,
                   'original_failed_stage': str(archived_stage), 'original_failed_claim': core.binding(archived_claim),
                   'original_queue_preserved_at': str(CONF / 'stages/prediction_queue'), 'failure_before_checkpoint_load_or_forward': True,
                   'failed_prediction_artifacts': ['failure.json', 'input_seal.json', 'status.json'],
                   'runtime_change': {'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}, 'scientific_source_changes': False,
                   'completed_inputs_reused': True, 'root_review': core.binding(review_path), 'watched': watched}
        core.write(out / 'supersession.json', lineage)
        state.update(state='running', freeze_sha256=EXPECTED_FREEZE, lineage=core.binding(out / 'supersession.json'))
        core.write(out / 'queue_status.json', state)
        for index, (stage, command) in enumerate(plan, 1):
            for path, digest in watched.items(): core.bound_file(path, digest)
            core.validate_source_lock()
            state.update(stage=stage, step=index, child_pid=None, command=command); core.write(out / 'queue_status.json', state)
            with (out / 'logs' / f'{index:02d}.stdout.log').open('x', encoding='utf-8') as so, (out / 'logs' / f'{index:02d}.stderr.log').open('x', encoding='utf-8') as se:
                child = subprocess.Popen(command, cwd=CONF, env=env, stdout=so, stderr=se, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            child.owned_identity = helpers.child_identity(child)
            state.update(child_pid=child.pid, child_identity=child.owned_identity); core.write(out / 'queue_status.json', state)
            code = child.wait(); core.require(code == 0, 'Recovery stage failed: ' + stage + ' exit ' + str(code)); child = None
            folder = CONF / 'stages' / stage; core.completed_stage(folder)
            for path, digest in watched.items(): core.bound_file(path, digest)
            state['completed'].append({'stage': stage, 'artifact_manifest': core.binding(folder / 'artifact_manifest.json')})
            core.write(out / 'queue_status.json', state)
        seal = core.read(CONF / 'stages/predictions_seal/all_predictions.json')
        core.require(seal['scalar_rows'] == 38792 and seal['map_count'] == 8268, 'Incomplete recovery output')
        state.update(state='completed', stage=None, child_pid=None); core.write(out / 'queue_status.json', state)
        core.write(out / 'summary.json', {'passed': True, 'commands_completed': 38, 'inputs_reused': 2, 'all37_models_exported': True, 'all_predictions': core.binding(CONF / 'stages/predictions_seal/all_predictions.json'), 'truth_materialized': False, 'source_and_checkpoint_changes': False})
        core.write(out / 'artifact_manifest.json', core.stage_manifest(out))
        print(out)
    except BaseException as error:
        cleanup = {}
        if child is not None:
            descendants = []
            try: descendants = helpers.owned_descendants(child) if helpers else []
            except BaseException as e: cleanup['inventory_error'] = repr(e)
            try:
                if child.poll() is None: child.terminate()
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=20)
            except BaseException as e: cleanup['parent_error'] = repr(e)
            try:
                if helpers: cleanup['descendants'] = helpers.stop_recorded_descendants(descendants)
            except BaseException as e: cleanup['descendants_error'] = repr(e)
        out.mkdir(exist_ok=True)
        state.update(state='failed', error=repr(error), traceback=traceback.format_exc(), cleanup=cleanup, no_automatic_retry=True)
        core.write(out / 'queue_status.json', state); core.write(out / 'failure.json', state)
        core.write(out / 'artifact_manifest.json', core.stage_manifest(out))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-review', type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--probe-only', action='store_true'); group.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.probe_only: print(json.dumps(probe()[0]))
    else:
        core.require(args.root_review is not None, 'Root review required')
        execute(args.root_review.resolve())
