"""Independent preparation review: source/metadata and isolated synthetic arrays only.

No real image/cache/label/checkpoint tensor is opened. No learned model is
instantiated or forwarded, at R6 or any other R. Mock model forwards test the
export function's numerical wiring without evaluating a trained system.
"""
from pathlib import Path
from unittest.mock import patch
import ast
import copy
import datetime
import hashlib
import json
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parent
HOME = ROOT/'confirmation_v1'
sys.path.insert(0, str(HOME))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import core
import data_io
import pipeline


def main():
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = ROOT/'provenance'/('confirmation_component_independent_'+stamp)
    out.mkdir(exist_ok=False)
    mock = Path('./temporary_checks/confirmation_independent')/stamp
    mock.mkdir(parents=True, exist_ok=False)
    for key in ('TMP', 'TEMP'):
        os.environ[key] = str(mock)
    checks = []

    def accept(name, **facts):
        checks.append({'name': name, 'passed': True, **facts})

    def reject(name, call):
        try:
            call()
        except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, FloatingPointError):
            accept(name)
        else:
            raise AssertionError('Missing rejection: '+name)

    def fixture(name):
        home = mock/name; home.mkdir()
        for filename in ('FREEZE.json', 'SOURCE_LOCK.json', 'contract.json', 'checkpoints.json'):
            core.write(home/filename, {'synthetic': True})
        return home

    try:
        assert not (HOME/'FREEZE.json').exists(), 'Review is preparation-only'
        lock = core.validate_source_lock()
        source_hashes = {name: core.sha(HOME/name) for name in lock['files']}
        readiness = core.read(lock['readiness']['path'])
        core.bound_file(lock['readiness']['path'], lock['readiness']['sha256'])
        assert readiness['passed']
        for name in ('core.py', 'data_io.py', 'pipeline.py', 'frozen_nn.py'):
            assert readiness['source_sha256'][name] == source_hashes[name]
        accept('final_component_source_lock_and_preparer_receipt')
        assert 'torch' not in sys.modules
        accept('metadata_gate_imports_no_torch')
        reject('missing_freeze_blocks_input_preparation', lambda: data_io.prepare_inputs('local'))
        reject('missing_freeze_blocks_export', lambda: pipeline.export_run('synthetic_run'))
        reject('missing_freeze_blocks_global_prediction_seal', pipeline.seal_predictions)
        reject('missing_freeze_blocks_reference_materialization', lambda: data_io.materialize_truth('local', 'missing'))
        assert not (HOME/'stages').exists() and not (HOME/'claims').exists()
        accept('real_stage_and_claim_roots_remain_absent')

        # Independent AST equivalence to the trained calculation, without importing
        # the training module, reading data or creating a model.
        parent_source = (ROOT/'train_common.py').read_text(encoding='utf-8-sig')
        parent_nodes = {n.name: ast.get_source_segment(parent_source, n) for n in ast.parse(parent_source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        frozen_source = (HOME/'frozen_nn.py').read_text(encoding='utf-8-sig')
        frozen_nodes = {n.name: ast.get_source_segment(frozen_source, n) for n in ast.parse(frozen_source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in ('degrade_rgb', 'coverage', 'require_finite_forward', 'CommonRegressionModel', 'make_model'):
            assert parent_nodes[name] == frozen_nodes[name]
        external_source = (ROOT/'external_weeds_v1/frozen_training_core.py').read_text(encoding='utf-8-sig')
        external_nodes = {n.name: ast.get_source_segment(external_source, n) for n in ast.parse(external_source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in ('degrade_rgb', 'coverage', 'CommonRegressionModel', 'make_model'):
            assert external_nodes[name] == parent_nodes[name]
        assert core.sha(HOME/'frozen_models.py') == core.sha(ROOT/'models_common.py')
        assert core.sha(HOME/'frozen_resample.py') == core.sha(ROOT/'conservative_resample.py')
        accept('local_and_external_original_inference_AST_and_model_resampler_identity')

        # Metadata compatibility for all declared entries. Incomplete training is
        # recorded as pending, never promoted to a complete 37-checkpoint freeze.
        contract = core.read(HOME/'contract.json'); core.validate_contract(contract)
        model_snapshot = []
        for entry in contract['roster']:
            folder = Path(entry['run_directory'])
            status_path = folder/'status.json'
            item = {'run_id': entry['run_id'], 'dataset': entry['dataset']}
            if not status_path.exists():
                item.update(phase='pending_not_created', metadata_schema_verified=False)
                model_snapshot.append(item); continue
            paths = [status_path, folder/'config.json', folder/'best_validation.json', folder/'metrics.jsonl']
            # Existing running files may legitimately advance; this is a timestamped
            # metadata snapshot and not a final source-lock assertion for them.
            status, config, best = [core.read(p) for p in paths[:3]]
            history = [json.loads(line) for line in paths[3].read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            assert all(config[k] == entry[k] for k in ('run_id', 'encoder', 'variant', 'seed', 'lr'))
            assert config['epochs'] == 24
            key = 'selection_balanced_mae' if entry['dataset'] == 'local' else 'selection_mae'
            assert all(key in x for x in history)
            if status['status'] == 'completed':
                assert status['executed_epochs'] == 24 and [x['epoch'] for x in history] == list(range(1, 25))
                earliest = min(history, key=lambda x: x[key])
                assert earliest == best and best['epoch'] == status['best_epoch']
                item.update(phase='completed', original_best_epoch=best['epoch'], metadata_schema_verified=True,
                            metadata_sha256={p.name: core.sha(p) for p in paths})
            else:
                item.update(phase='pending_'+status['status'], observed_epochs=len(history), metadata_schema_verified=True)
            model_snapshot.append(item)
        assert len(model_snapshot) == 37
        accept('all37_declared_configs_status_compatibility_snapshot', completed=sum(x['phase']=='completed' for x in model_snapshot), pending=sum(x['phase']!='completed' for x in model_snapshot))

        # Existing validation metadata establishes the edge-window convention.
        # No cache or image is loaded here.
        allowed = core.read(ROOT/'local_data_v2/val.json')
        registered = {x['source_id']: x for x in core.read(ROOT/'local_sampling_completeness.json')['sources']}
        padded = [x for x in allowed if x['x']+512 > registered[x['source_id']]['width'] or x['y']+512 > registered[x['source_id']]['height']]
        assert len(padded) == 7
        for row in padded:
            source = registered[row['source_id']]
            data_io.validate_native_window(row['x'], row['y'], source['width'], source['height'])
        reject('origin_at_source_right_edge_rejected', lambda: data_io.validate_native_window(520, 0, 520, 512))
        accept('all7_allowed_validation_padded_windows_compatible', example=padded[0]['id'])

        # Every post-claim failure remains inspectable, including serialization.
        p = fixture('claim_failure')
        with patch.object(core.json, 'dump', side_effect=RuntimeError('synthetic claim write failure')):
            reject('claim_serialization_failure_propagates', lambda: core.attempt('job', p).__enter__())
        assert (p/'claims/job.json').exists() and core.read(p/'stages/job/status.json')['state'] == 'failed'
        reject('failed_claim_never_reused', lambda: core.attempt('job', p).__enter__())
        accept('claim_serialization_failure_retained')
        p = fixture('ordinary_failure')
        try:
            with core.attempt('job', p):
                raise RuntimeError('synthetic source failure')
        except RuntimeError:
            pass
        assert core.read(p/'stages/job/status.json')['no_retry']
        accept('body_failure_retained')
        p = fixture('success')
        with core.attempt('job', p) as folder:
            core.write(folder/'data.json', {'x': 1})
        core.completed_stage(folder)
        core.write(folder/'data.json', {'x': 2})
        reject('completed_stage_rejects_mutation', lambda: core.completed_stage(folder))

        # Global prediction consumer uses a purely synthetic, mocked artifact
        # reader; malformed complete-count seals must still be rejected.
        expected_entries = []
        for model in contract['roster']:
            count = 338 if model['dataset'] == 'local' else 26
            item = {'run_id': model['run_id'], 'dataset': model['dataset'], 'variant': model['variant'],
                    'stage': str(HOME/'stages'/('predict_'+model['run_id'])), 'scalar_rows': count*4,
                    'map_count': count if model['variant'] != 'regression' else 0}
            for key in ('artifact_manifest', 'metadata', 'map_index', 'predicted_scalars'):
                filename = key+('.csv' if key == 'predicted_scalars' else '.json')
                item[key] = {'path': str(Path(item['stage'])/filename), 'sha256': 'a'*64}
            expected_entries.append(item)
        seal = {'freeze_sha256': 'b'*64, 'run_ids': [x['run_id'] for x in contract['roster']], 'models': expected_entries,
                'all_declared_models_completed': True, 'scalar_rows': 38792, 'map_count': 8268}
        def seal_read(path):
            return contract if Path(path).name == 'contract.json' else seal
        with patch.object(pipeline, 'completed_stage', return_value={}), patch.object(pipeline, 'bound_file', return_value=None), patch.object(pipeline, 'sha', return_value='b'*64), patch.object(pipeline, 'read', side_effect=seal_read):
            pipeline.validate_global_predictions(); accept('complete_synthetic_global_prediction_inventory')
            for name, models in [('empty', []), ('missing', expected_entries[:-1]), ('duplicate', expected_entries[:-1]+[expected_entries[0]])]:
                seal['models'] = models
                reject('global_seal_rejects_'+name+'_model_entries', pipeline.validate_global_predictions)
            seal['models'] = copy.deepcopy(expected_entries); seal['models'][0]['scalar_rows'] -= 1
            reject('global_seal_rejects_per_model_count', pipeline.validate_global_predictions)
            seal['models'] = copy.deepcopy(expected_entries); seal['models'][0]['stage'] = 'synthetic_wrong_stage'
            reject('global_seal_rejects_wrong_stage', pipeline.validate_global_predictions)
        events = []
        def stop_choices(_):
            events.append('choices'); raise ValueError('synthetic incomplete all-model choices')
        with patch.object(data_io, 'validate_freeze', side_effect=lambda: (events.append('freeze') or (contract, [], {}))), patch.object(pipeline, 'validate_global_predictions', side_effect=lambda: events.append('predictions')), patch.object(pipeline, 'validate_choices', side_effect=stop_choices), patch.object(data_io, 'validate_input_manifest', side_effect=AssertionError('Truth/input reached before choices')):
            reject('truth_waits_for_all_choices', lambda: data_io.materialize_truth('local', 'synthetic'))
        assert events == ['freeze', 'predictions', 'choices']
        accept('truth_gate_order_freeze_predictions_all_choices_before_truth')
        segmentation = [x for x in contract['roster'] if x['variant'] != 'regression']
        choice_models = [{'run_id': x['run_id'], 'case_count': 120 if x['dataset']=='local' else 3120,
                          'selection_seal': {'path': 'synthetic_selection', 'sha256': 'c'*64},
                          'independent_review': {'path': 'synthetic_independent', 'sha256': 'd'*64}}
                         for x in segmentation]
        choice = {'passed': True, 'scope': 'confirmation_all_label_free_selections',
                  'freeze_sha256': 'b'*64, 'all_predictions_sha256': 'b'*64,
                  'models': choice_models, 'truth_used_for_choices': False, 'complete_case_inventory_verified': True}
        choice_audit = {'passed': True, 'models': [{'run_id': x['run_id'], 'selection_seal_sha256': 'c'*64} for x in segmentation]}
        def choice_read(path):
            if Path(path).name == 'contract.json':
                return contract
            return choice_audit if str(path) == 'synthetic_independent' else choice
        with patch.object(pipeline, 'read', side_effect=choice_read), patch.object(pipeline, 'sha', return_value='b'*64), patch.object(pipeline, 'bound_file', return_value=None):
            pipeline.validate_choices('synthetic_choice'); accept('all30_synthetic_selections_and_audit_bindings')
            choice['models'] = choice_models[:-1]
            reject('choices_reject_missing_segmentation_run', lambda: pipeline.validate_choices('synthetic_choice'))
            choice['models'] = copy.deepcopy(choice_models); choice['models'][0]['case_count'] = 119
            reject('choices_reject_missing_case', lambda: pipeline.validate_choices('synthetic_choice'))
            choice['models'] = choice_models; choice['truth_used_for_choices'] = True
            reject('choices_reject_truth_used', lambda: pipeline.validate_choices('synthetic_choice'))
            choice['truth_used_for_choices'] = False; choice_audit['models'][0]['selection_seal_sha256'] = '0'*64
            reject('choices_reject_independent_audit_seal_mismatch', lambda: pipeline.validate_choices('synthetic_choice'))

        import numpy as np
        import torch
        from torch.nn import functional as F
        import frozen_nn
        import frozen_resample
        torch.set_num_threads(2)
        original = {'torch': torch, 'F': F, 'normalized_area_resize': frozen_resample.normalized_area_resize,
                    'VALIDATION_R': (1, 2, 4)}
        for name in ('degrade_rgb', 'coverage'):
            exec(compile(parent_nodes[name], '<verified_original_'+name+'>', 'exec'), original)
        generator = torch.Generator().manual_seed(934)
        x = torch.rand((2, 3, 256, 256), generator=generator)
        valid = torch.ones((2, 1, 256, 256)); valid[0, :, :77, :42] = 0; valid[1, :, 10:90, 40:80] = .25
        for factor in (1, 2, 4):
            a = original['degrade_rgb'](x, factor, valid); b = frozen_nn.degrade_rgb(x, factor, valid)
            assert torch.equal(a, b)
            accept('exact_original_degradation_R'+str(factor), max_abs_error=0.)
        # Independent area-overlap weights for the noninteger 256->42 reduction.
        n, m = 256, 42
        s = np.arange(n+1, dtype=np.float64); t = np.arange(m+1, dtype=np.float64)*(n/m)
        weights = np.maximum(0, np.minimum(t[1:, None], s[None, 1:])-np.maximum(t[:-1, None], s[None, :-1]))/(n/m)
        xx, vv = x.double().numpy(), valid.double().numpy()
        num = np.matmul(np.matmul(weights, xx*vv), weights.T)
        den = np.matmul(np.matmul(weights, vv), weights.T)
        small = np.divide(num, den, out=np.zeros_like(num), where=den>0)
        restored_num = F.interpolate(torch.from_numpy(small*den), size=(256, 256), mode='bilinear', align_corners=False)
        restored_den = F.interpolate(torch.from_numpy(den), size=(256, 256), mode='bilinear', align_corners=False)
        restored = torch.where(restored_den>0, restored_num/torch.where(restored_den>0, restored_den, torch.ones_like(restored_den)), torch.zeros_like(restored_num))
        restored = torch.where(valid>0, restored, torch.zeros_like(restored))
        expected = (restored-restored.new_tensor([.485, .456, .406])[None, :, None, None])/restored.new_tensor([.229, .224, .225])[None, :, None, None]
        actual = frozen_nn.degrade_rgb(x.double(), 6, valid.double())
        error = float((actual-expected).abs().max())
        assert error < 1e-12
        accept('synthetic_R6_256_to42_independent_overlap_equivalence', max_abs_error=error, actual_axial_ratio=256/42)
        class MockModel(torch.nn.Module):
            def __init__(self, regression):
                super().__init__(); self.regression = regression
            def forward(self, inputs, weights):
                return inputs.mean((1, 2, 3)) if self.regression else inputs.mean(1, keepdim=True)
        for variant in ('regression', 'hard_seg', 'soft_seg'):
            model = MockModel(variant == 'regression')
            for factor in (1, 2, 4, 6):
                p, q = pipeline.prediction_arrays(model, x, valid, variant, factor)
                z = frozen_nn.degrade_rgb(x, factor, valid)
                logits = z.mean((1, 2, 3)) if variant == 'regression' else z.mean(1, keepdim=True)
                refp = logits.sigmoid()
                refq = refp if variant == 'regression' else original['coverage'](refp, valid)
                assert torch.equal(p, refp) and torch.equal(q, refq)
        accept('all12_variant_R_mock_export_paths_exact', real_model_created=False)
        class BadMock(MockModel):
            def forward(self, inputs, weights):
                return super().forward(inputs, weights)*float('inf')
        reject('nonfinite_logits_rejected_before_sigmoid', lambda: pipeline.prediction_arrays(BadMock(False), x, valid, 'soft_seg', 1))
        for dtype in (np.float32, np.uint16):
            reject('local_nonuint8_RGB_rejected_'+str(dtype), lambda: data_io.local_input(np.zeros((256, 256, 3), dtype), np.ones((256, 256), np.float32)))
        rgb, v = data_io.weeds_input([np.full((600, 600), 65535, dtype=np.uint16)]*3)
        assert np.array_equal(rgb, np.ones((3, 256, 256), np.float32)) and np.array_equal(v, np.ones((256, 256), np.float32))
        accept('external_uint16_white_remains_one_no_255_scaling')
        reject('external_uint8_RGB_rejected', lambda: data_io.weeds_input([np.ones((600, 600), np.uint8)]*3))
        labels = np.arange(600*600, dtype=np.uint32).reshape(600, 600) % 6
        truth = data_io.weeds_truth(labels.astype(np.uint8))
        assert np.array_equal(truth['native_class_counts'], np.array([60000, 60000, 240000], np.uint32))
        assert np.max(abs(truth['soft3'].sum(axis=0)-1)) < 1e-6
        accept('external_background_crop_pooledweed_mutually_exclusive', counts=[60000, 60000, 240000])
        assert source_hashes == {name: core.sha(HOME/name) for name in lock['files']}
        accept('reviewed_component_sources_unchanged')
        core.write(out/'model_metadata_snapshot.json', model_snapshot)
        report = {'passed': True, 'scope': 'independent source and synthetic preparation review', 'checks': checks,
                  'check_count': len(checks), 'source_sha256': source_hashes,
                  'source_lock_sha256': core.sha(HOME/'SOURCE_LOCK.json'), 'preparer_receipt': lock['readiness'],
                  'reviewer_source_sha256': core.sha(__file__), 'mock_root': str(mock),
                  'model_metadata_snapshot_sha256': core.sha(out/'model_metadata_snapshot.json'),
                  'actual_test_manifest_or_arrays_read': False, 'real_model_instantiated_or_forwarded': False,
                  'actual_R6_model_forward': False, 'GPU': False,
                  'pending_production_requirements': ['All 37 training runs and original selected epochs must be completed and accepted',
                      'Root-populated checkpoint table and immutable FREEZE', 'All spatial implementations and their separate root reviews',
                      'Independent acceptance of actual exports/selections/truth after authorized production'],
                  'limitations': ['Mock model paths test inference wiring, not trained checkpoint scores',
                      'Completed metadata checked as a snapshot; no checkpoint tensors loaded',
                      'No source archive decoded or actual reserved support geometry checked']}
        core.write(out/'review.json', report)
        (out/'review.md').write_text('# Confirmation component preparation: independent review\n\n'
            'Passed source and synthetic checks. This is not production authorization or reserved-data validation. '
            'Two raised issues were repaired before signing: nominal padded source-edge windows and complete global prediction model inventory.\n\n'
            f'{len(checks)} checks passed. All37 declared source entries were inspected for metadata availability/schema; '
            'pending fits remain pending and block a production freeze. No real model, test cache, label or R6 forward was read/run.\n\n'
            'See review.json and model_metadata_snapshot.json for exact scope, source bindings and unfinished production prerequisites.\n', encoding='utf-8')
        core.write(out/'artifact_manifest.json', core.stage_manifest(out))
        print(json.dumps({'passed': True, 'checks': len(checks), 'output': str(out), 'review_sha256': core.sha(out/'review.json')}))
    except BaseException as error:
        core.write(out/'failure.json', {'error': repr(error), 'traceback': traceback.format_exc(), 'passed_checks': checks,
                                     'test_accessed': False, 'real_model_forward': False})
        print(str(out), file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
