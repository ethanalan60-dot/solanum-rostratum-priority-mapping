"""Independent bounded entry/interface audit; no inference or training.

validate() tests replace only serialized metadata reads/hashes in memory;
original project inputs are not mutated. This is not a real-export certificate.
"""
from pathlib import Path
import copy, csv, datetime, hashlib, itertools, json, math
from unittest.mock import patch
import numpy as np
import evaluate_common_management as cm

ROOT = Path(__file__).resolve().parent
PA = ROOT/'precision_agriculture'


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    out = PA/('common_management_entry_review_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    original_read, original_sha = cm.ev.read, cm.ev.sha
    code_hash = original_sha(cm.__file__)
    protocol = original_read(cm.PROTOCOL)
    plan = original_read(ROOT/'common_queue_plan_v2.json')
    selected_run = protocol['declared_runs'][0]['run_id']
    run = (ROOT/'runs'/selected_run).resolve()
    maps_dir = run/'independent_in_memory_maps_fixture'
    config = original_read(run/'config.json')
    status = original_read(run/'status.json')
    old = original_read(PA/'RM_evaluation_20260916_020003_962711/inputs.json')
    maps = original_read(Path(old['maps_directory'])/'metadata.json')
    checkpoint_hash = original_sha(run/'best.pt')
    maps.update(run_id=selected_run, model_family='common_v2', encoder=config['encoder'], variant=config['variant'],
                config=str(run/'config.json'), config_sha256=json_hash(config), checkpoint=str(run/'best.pt'),
                checkpoint_sha256=checkpoint_hash, checkpoint_epoch=status['best_epoch'],
                model_source_sha256=config['source_sha256']['train_common.py'],
                model_dependency_sha256=config['source_sha256'], export_source_sha256=original_sha(ROOT/'export_common_validation.py'),
                source_copies={n: str(run/'source'/n) for n in config['source_sha256']},
                input_hashes={}, spatial_prediction_available=True)
    maps['source_copies']['export_common_validation.py'] = str(ROOT/'export_common_validation.py')
    verification = {'all_checks_passed': True, 'expected_windows': 357, 'checked_pairs': 1071, 'expected_pairs': 1071,
                    'all_npz_reopened': 357, 'strict_weights_only_checkpoint_loaded': True,
                    'original_selection_audited_24_epochs': True, 'original_epoch_preserved': True,
                    'manifest_order_preserved': True, 'test_accessed': False, 'R6_accessed': False}

    def case(name, mutate):
        m, c, s, v = copy.deepcopy((maps, config, status, verification))
        mutate(m, c, s, v)
        m['config_sha256'] = json_hash(c)
        ms = {'status': 'completed', 'metadata_sha256': json_hash(m), 'verification': v}
        values = {str((maps_dir/'metadata.json').resolve()): m,
                  str((maps_dir/'status.json').resolve()): ms,
                  str((maps_dir/'verification.json').resolve()): v,
                  str((run/'config.json').resolve()): c,
                  str((run/'status.json').resolve()): s}
        def fake_read(path):
            key = str(Path(path).resolve())
            return copy.deepcopy(values[key]) if key in values else original_read(path)
        def fake_sha(path):
            key = str(Path(path).resolve())
            return json_hash(values[key]) if key in values else original_sha(path)
        error = None
        try:
            with patch.object(cm.ev, 'read', side_effect=fake_read), patch.object(cm.ev, 'sha', side_effect=fake_sha):
                cm.validate(maps_dir)
            accepted = True
        except Exception as exc:
            accepted = False; error = repr(exc)
        return {'name': name, 'accepted': accepted, 'error': error}

    cases = [('baseline_valid_metadata', lambda m,c,s,v: None),
             ('incomplete_epoch23', lambda m,c,s,v: s.update(executed_epochs=23)),
             ('regression_config', lambda m,c,s,v: c.update(variant='regression')),
             ('wrong_declared_lr', lambda m,c,s,v: c.update(lr=.002)),
             ('subset_validation', lambda m,c,s,v: c.update(max_val=10)),
             ('reserved_R6', lambda m,c,s,v: m.update(factors=[1,2,6])),
             ('test_accessed', lambda m,c,s,v: m.update(test_accessed=True)),
             ('wrong_checkpoint_epoch', lambda m,c,s,v: m.update(checkpoint_epoch=999)),
             ('empty_config_source_hashes', lambda m,c,s,v: c.update(source_sha256={})),
             ('wrong_map_model_source', lambda m,c,s,v: m.update(model_source_sha256='0'*64)),
             ('wrong_map_dependencies', lambda m,c,s,v: m.update(model_dependency_sha256={})),
             ('wrong_export_source', lambda m,c,s,v: m.update(export_source_sha256='0'*64)),
             ('empty_source_copies', lambda m,c,s,v: m.update(source_copies={})),
             ('unpretrained', lambda m,c,s,v: c.update(no_pretrained=True)),
             ('decision_selected_checkpoint', lambda m,c,s,v: c.update(decision_metrics_used_for_selection=True)),
             ('wrong_train_manifest_hash', lambda m,c,s,v: c['manifest_sha256'].update({'train.json':'0'*64})),
             ('wrong_dataset_lock_hash', lambda m,c,s,v: c['manifest_sha256'].update({'DATASET_LOCK.json':'0'*64})),
             ('wrong_output_grid', lambda m,c,s,v: c.update(fixed_output_grid=128)),
             ('wrong_validation_R', lambda m,c,s,v: c.update(validation_R=[1,2,6])),
             ('unverified_checkpoint_selection', lambda m,c,s,v: v.update(original_selection_audited_24_epochs=False)),
             ('map_wrong_encoder_tag', lambda m,c,s,v: m.update(encoder='unrelated_encoder'))]
    entry_results = [case(name, fn) for name, fn in cases]
    assert entry_results[0]['accepted'], entry_results[0]
    for item in entry_results[1:8]:
        assert not item['accepted'], item

    # Small synthetic all-45 interface test with exact eight-subset enumeration.
    trial = out/'synthetic_policy_interface'; trial.mkdir()
    unit = []; fixed = []; costs = [2., 3., 3.]; p = [1.9, 2.7, 2.7]; truth = [2., 0., 0.]
    for M in (1,2,4):
        ids = [f'synthetic_m{M}_{i}' for i in ('a','b','c')]
        for i in range(3):
            unit.append({'M': M, 'id': ids[i], 'area': costs[i], 'target': truth[i],
                         'predicted_R1':p[i], 'predicted_R2':p[i], 'predicted_R4':p[i]})
        for R in (1,2,4):
            for b in cm.ev.BUDGETS:
                chosen = cm.policy.prefix_ids(ids, np.array(costs), np.array(p), b)
                indices = [ids.index(i) for i in chosen]
                fixed.append({'M':M, 'method':'model_prefix', 'R':R, 'budget':b,
                              'actual_area':sum(costs[i] for i in indices), 'capture_fraction':sum(truth[i] for i in indices)/2})
    cm.ev.write_csv(trial/'unit_predictions.csv', unit)
    cm.ev.write_csv(trial/'fixed_budgets.csv', fixed)
    answer = cm.run_policy(trial)
    assert answer['combinations'] == 45 and answer['all_solutions_feasible']
    with (trial/'predicted_area_policy/all_45_deltas.csv').open(encoding='utf-8-sig') as stream:
        results = list(csv.DictReader(stream))
    for row in results:
        budget = float(row['budget'])*sum(costs)
        best = max(sum(p[i] for i in range(3) if mask[i])
                   for mask in itertools.product((0,1), repeat=3)
                   if sum(costs[i] for i in range(3) if mask[i]) <= budget)
        assert math.isclose(float(row['pmilp_predicted_objective_m2']), best, rel_tol=0, abs_tol=1e-10)
    report = {'created_local': datetime.datetime.now().astimezone().isoformat(),
              'implementation_sha256': code_hash, 'implementation_unchanged_during_check': original_sha(cm.__file__) == code_hash,
              'extension_protocol_sha256': original_sha(cm.PROTOCOL), 'script_sha256': original_sha(__file__),
              'entry_fixture': 'In-memory metadata substitutions; originals unchanged, no real export certification',
              'entry_checks': entry_results, 'synthetic_policy_45_cases': answer,
              'all_45_predicted_objectives_match_independent_exhaustive_enumeration': True,
              'no_training_inference_map_export_or_test_access': True}
    cm.ev.write_json(out/'entry_and_interface_checks.json', report)
    print(json.dumps(report, ensure_ascii=True), flush=True)
    print('OUTPUT',str(out))


if __name__ == '__main__':
    with cm.threadpool_limits(limits=2):
        main()
