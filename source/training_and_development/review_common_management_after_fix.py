"""Post-fix real-entry and in-memory rejection checks; no map inference/integration."""
from pathlib import Path
import copy, datetime, hashlib, json
from unittest.mock import patch
import evaluate_common_management as cm

ROOT = Path(__file__).resolve().parent
MAPS = [ROOT/'runs/dev_common_v2_lr1e4_resnet18_hard_seg_L256_s17/common_validation_export_20260916_030715_466122',
        ROOT/'runs/dev_common_v2_lr1e4_resnet18_soft_seg_L256_s17/common_validation_export_20260916_030913_433508']


def jsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    out = cm.PA/('common_management_after_fix_review_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    read, sha, exists = cm.ev.read, cm.ev.sha, Path.exists
    before = sha(cm.__file__)
    real = []
    for directory in MAPS:
        result = cm.validate(directory.resolve())
        real.append({'maps_directory':str(directory.resolve()), 'accepted':True,
                     'model_run':result[2]['run_id'], 'checkpoint_epoch':result[2]['checkpoint_epoch'],
                     'windows':len(result[4]), 'watched_files':len(result[-1]),
                     'metadata_sha256':sha(directory/'metadata.json')})
    maps_dir = MAPS[0].resolve(); run = maps_dir.parent
    base = (read(maps_dir/'metadata.json'), read(run/'config.json'), read(run/'status.json'), read(maps_dir/'verification.json'))
    cases = [('baseline_valid_metadata', lambda m,c,s,v:None),
             ('incomplete_epoch23',lambda m,c,s,v:s.update(executed_epochs=23)),
             ('regression_config',lambda m,c,s,v:c.update(variant='regression')),
             ('wrong_declared_lr',lambda m,c,s,v:c.update(lr=.002)),
             ('subset_validation',lambda m,c,s,v:c.update(max_val=10)),
             ('reserved_R6',lambda m,c,s,v:m.update(factors=[1,2,6])),
             ('test_accessed',lambda m,c,s,v:m.update(test_accessed=True)),
             ('wrong_checkpoint_epoch',lambda m,c,s,v:m.update(checkpoint_epoch=999)),
             ('empty_config_source_hashes',lambda m,c,s,v:c.update(source_sha256={})),
             ('wrong_map_model_source',lambda m,c,s,v:m.update(model_source_sha256='0'*64)),
             ('wrong_map_dependencies',lambda m,c,s,v:m.update(model_dependency_sha256={})),
             ('wrong_export_source',lambda m,c,s,v:m.update(export_source_sha256='0'*64)),
             ('empty_source_copies',lambda m,c,s,v:m.update(source_copies={})),
             ('unpretrained',lambda m,c,s,v:c.update(no_pretrained=True)),
             ('decision_selected_checkpoint',lambda m,c,s,v:c.update(decision_metrics_used_for_selection=True)),
             ('wrong_train_manifest_hash',lambda m,c,s,v:c['manifest_sha256'].update({'train.json':'0'*64})),
             ('wrong_dataset_lock_hash',lambda m,c,s,v:c['manifest_sha256'].update({'DATASET_LOCK.json':'0'*64})),
             ('wrong_output_grid',lambda m,c,s,v:c.update(fixed_output_grid=128)),
             ('wrong_validation_R',lambda m,c,s,v:c.update(validation_R=[1,2,6])),
             ('unverified_checkpoint_selection',lambda m,c,s,v:v.update(original_selection_audited_24_epochs=False)),
             ('map_wrong_encoder_tag',lambda m,c,s,v:m.update(encoder='unrelated_encoder')),
             ('empty_export_inputs',lambda m,c,s,v:m.update(input_hashes={})),
             ('wrong_family_tag',lambda m,c,s,v:m.update(model_family='unrelated')),
             ('no_spatial_map',lambda m,c,s,v:m.update(spatial_prediction_available=False)),
             ('incomplete_map_pairs',lambda m,c,s,v:v.update(checked_pairs=1000)),
             ('incomplete_npz_reopens',lambda m,c,s,v:v.update(all_npz_reopened=356)),
             ('wrong_scalar_hash',lambda m,c,s,v:m.update(scalar_csv_sha256='0'*64)),
             ('wrong_scalar_summary_hash',lambda m,c,s,v:m.update(scalar_summary_sha256='0'*64))]
    checks = []
    for name, mutate in cases:
        m,c,s,v = copy.deepcopy(base); mutate(m,c,s,v)
        m['config_sha256'] = jsha(c)
        for path,obj in [(run/'config.json',c),(run/'status.json',s)]:
            if str(path) in m['input_hashes']:
                m['input_hashes'][str(path)] = jsha(obj)
        ms = {'status':'completed','metadata_sha256':jsha(m),'verification':v}
        replacements = {str(maps_dir/'metadata.json'):m, str(maps_dir/'status.json'):ms,
                        str(maps_dir/'verification.json'):v, str(run/'config.json'):c, str(run/'status.json'):s}
        def fake_read(path):
            key = str(Path(path).resolve())
            return copy.deepcopy(replacements[key]) if key in replacements else read(path)
        def fake_sha(path):
            key = str(Path(path).resolve())
            return jsha(replacements[key]) if key in replacements else sha(path)
        try:
            with patch.object(cm.ev,'read',side_effect=fake_read), patch.object(cm.ev,'sha',side_effect=fake_sha):
                cm.validate(maps_dir)
            accepted,error = True,None
        except Exception as exc:
            accepted,error = False,repr(exc)
        checks.append({'name':name,'accepted':accepted,'error':error})
        assert accepted == (name == 'baseline_valid_metadata'), checks[-1]
    for forbidden in ('USE_RESTRICTION.json','failure.json'):
        def fake_exists(path):
            return True if path.resolve() == (run/forbidden).resolve() else exists(path)
        try:
            with patch.object(Path,'exists',fake_exists):
                cm.validate(maps_dir)
            accepted,error = True,None
        except Exception as exc:
            accepted,error = False,repr(exc)
        assert not accepted
        checks.append({'name':'existing_'+forbidden,'accepted':accepted,'error':error})
    after = sha(cm.__file__)
    assert before == after
    record = {'created_local':datetime.datetime.now().astimezone().isoformat(),'passed':True,
              'wrapper_sha256':before,'wrapper_unchanged_during_check':True,
              'extension_protocol_sha256':sha(cm.PROTOCOL),'reviewed_exporter_sha256':cm.REVIEWED_EXPORT_SHA,
              'script_sha256':sha(__file__),'actual_completed_exports':real,'rejection_checks':checks,
              'rejected_invalid_cases':sum(not r['accepted'] for r in checks),
              'original_inputs_mutated':False,'training_inference_integration_or_test_accessed':False,
              'previous_findings_retained':'common_management_entry_review_20260916_030929_419282/entry_and_interface_checks.json'}
    cm.ev.write_json(out/'post_fix_checks.json',record)
    print(json.dumps(record,ensure_ascii=True),flush=True)
    print('OUTPUT',str(out))


if __name__ == '__main__':
    with cm.threadpool_limits(limits=2):
        main()
