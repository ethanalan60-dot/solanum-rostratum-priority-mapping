"""Read-only artifact/provenance review; writes its own JSON snapshot only."""
from pathlib import Path
import argparse
import json
import math
import os

import run_common_queue as queue

ROOT = Path(__file__).resolve().parent
DEST = ROOT/'provenance'/'amp_fix_20260916'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(output):
    old_plan = read(ROOT/'common_queue_plan_v1.json')
    report = {'test_accessed':False, 'R6_accessed':False, 'gpu_used':False,
              'cache_samples_accessed':False, 'reviewed_utc':queue.utc_now(),
              'trainer_sha256':queue.sha256(ROOT/'train_common.py'),
              'queue_sha256':queue.sha256(ROOT/'run_common_queue.py')}
    archived = ROOT/'common_queue_runs'/'20260916_015407_9d388198'/'source'
    assert all(queue.sha256(archived/name) == digest for name,digest in old_plan['source_sha256'].items())
    assert queue.sha256(ROOT/'common_queue_plan_v1.json') == '7020be998af00400c8e2442953fc422af166b274bb0ae6e9b2ce03214c7410ae'
    report['v1_archive_preserved'] = True
    targeted = read(DEST/'targeted_amp_checks_v2.json')
    own_cpu = read(DEST/'independent_cpu_accumulation_review.json')
    for item in (targeted, own_cpu):
        assert item['passed']
        assert item['source_sha256']['train_common.py'] == report['trainer_sha256']
    report['targeted_real_cuda_and_independent_cpu_match_current_source'] = True
    diagnostic = read(DEST/'diagnostic_runs_v2.json')
    report['diagnostic_status'] = diagnostic['status']
    report['diagnostic_runs'] = []
    for item in diagnostic['runs']:
        run_dir = ROOT/'runs'/item['run_id']
        config = read(run_dir/'config.json')
        status = read(run_dir/'status.json')
        events = [json.loads(line) for line in (run_dir/'optimizer_events.jsonl').read_text().splitlines()]
        metrics = [json.loads(line) for line in (run_dir/'metrics.jsonl').read_text().splitlines()]
        assert status['status'] == 'completed' and item['returncode'] == 0
        assert status['stage'] == 'diagnostic' and status['executed_epochs'] == 4
        assert config['n_train'] == 985 and config['n_val'] == 357
        assert config['validation_R'] == [1,2,4] and config['test_accessed'] is False
        assert config['source_sha256']['train_common.py'] == report['trainer_sha256']
        assert queue.sha256(run_dir/'source'/'train_common.py') == report['trainer_sha256']
        assert len(events) == 496 and [e['attempt'] for e in events] == list(range(1,497))
        assert len(metrics) == 4
        actual = skipped = 0
        for i,event in enumerate(events):
            assert event['epoch'] == i//124+1 and event['group'] == i%124+1
            assert event['group_samples'] == (1 if i%124 == 123 else 8)
            assert math.isfinite(event['group_loss'])
            assert event['group_denominator'] > 0 and math.isfinite(event['group_denominator'])
            assert event['status'] in ('updated','overflow_skipped')
            if event['status'] == 'updated':
                actual += 1
                assert event['optimizer_step_executed'] and event['all_unscaled_gradients_finite']
                assert math.isfinite(event['unscaled_gradient_norm'])
            else:
                skipped += 1
                assert not event['optimizer_step_executed'] and not event['all_unscaled_gradients_finite']
                assert event['scale_after'] == event['scale_before']/2
            assert event['actual_updates_total'] == actual and event['skipped_updates_total'] == skipped
        assert status['optimizer_updates_total'] == actual and status['optimizer_skips_total'] == skipped
        for i,metric in enumerate(metrics):
            subset=events[i*124:(i+1)*124]
            assert metric['optimizer_update_attempts'] == 124
            assert metric['optimizer_updates'] == sum(e['optimizer_step_executed'] for e in subset)
            assert metric['optimizer_skipped_overflows'] == sum(e['skipped'] for e in subset)
            expected_lr = .0001*(1+math.cos(math.pi*i/24))/2
            assert math.isclose(metric['lr'],expected_lr,rel_tol=1e-12,abs_tol=1e-15)
            assert sum(metric['train_factor_sample_counts'].values()) == 985
        report['diagnostic_runs'].append({'run_id':item['run_id'],'attempted':len(events),
                                         'actual':actual,'skipped':skipped,
                                         'overflow_events':[{k:e[k] for k in ('attempt','epoch','group','group_samples','scale_before','scale_after')}
                                                            for e in events if e['skipped']],
                                         'all_event_counters_consistent':True,'scheduler_fixed_24_epoch_verified':True})
    report['diagnostic_complete'] = diagnostic['status'] == 'completed' and len(diagnostic['runs']) == 3
    path=ROOT/'common_queue_plan_v2.json'
    report['new_plan_exists'] = path.exists()
    if path.exists():
        digest=queue.sha256(path)
        plan=queue.verify_plan(path,digest)
        assert plan['version'] == 'common_queue_v2'
        assert plan['manifest_sha256'] == old_plan['manifest_sha256']
        assert len(plan['runs']) == 12
        fresh_ids=[]
        for old,new in zip(old_plan['runs'],plan['runs']):
            assert new['run_id'].replace('dev_common_v2_','dev_common_v1_',1) == old['run_id']
            assert {k:v for k,v in old.items() if k not in ('run_id','index')} == {k:v for k,v in new.items() if k not in ('run_id','index')}
            fresh_ids.append({'id':new['run_id'],'directory_already_exists':(ROOT/'runs'/new['run_id']).exists()})
        report['plan']={'sha256':digest,'version':plan['version'],'runs':fresh_ids,
                        'same_12_settings_except_versioned_ids':True,'source_and_development_manifest_locks_valid':True,
                        'all_run_ids_new_at_snapshot':all(not e['directory_already_exists'] for e in fresh_ids)}
    report['active_project_processes'] = queue.project_processes(exclude_pids=[os.getpid()])
    report['review_passed'] = report['diagnostic_complete'] and path.exists()
    report['launch_ready_at_snapshot'] = bool(report['review_passed'] and report['plan']['all_run_ids_new_at_snapshot'] and not report['active_project_processes'])
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('diagnostic_status','diagnostic_complete','new_plan_exists','review_passed','launch_ready_at_snapshot')}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
