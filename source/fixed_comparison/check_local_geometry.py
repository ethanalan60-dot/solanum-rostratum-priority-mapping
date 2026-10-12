"""Preparation checks: synthetic arrays and two explicitly allowed validation windows.

Never opens the reserved member manifest, reserved cache, mask or model. Legacy
validation metadata may contain target fields; only geometry/valid fields and
archived area-weight arrays are consumed by the checks.
"""
from pathlib import Path
from types import SimpleNamespace
import datetime
import json
import os
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
sys.path.insert(0, str(HERE))
import local_geometry as lg
import core


def main():
    out = ROOT/'provenance'/('confirmation_geometry_readiness_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    temp = out/'temp'; temp.mkdir()
    for key in ('TMP', 'TEMP'):
        os.environ[key] = str(temp)
    checks = []

    def passed(name, **facts):
        checks.append({'name': name, 'passed': True, **facts})

    def rejected(name, call):
        try:
            call()
        except (ValueError, AssertionError, FileNotFoundError):
            passed(name)
        else:
            raise AssertionError('Expected rejection: '+name)

    try:
        # With no root FREEZE, failure precedes any access to the supplied root-review
        # and to the production input-manifest path.
        if not (HERE/'FREEZE.json').exists():
            rejected('production_gate_closed_before_input_manifest', lambda: lg.execute(out/'nonexistent_review.json'))
        else:
            raise AssertionError('Preparation checker must not run after production freeze')
        geo = lg.dependencies(); np = geo.np
        identity = SimpleNamespace(transform=lambda x, y: (x, y))
        affine = geo.rasterio.Affine(.004, .00013, .37, -.00009, .004, .19)
        lattice = geo.projected_lattice(identity, affine, 0, 0, 512, 512, [0., 0.])
        valid = np.ones((512, 512), dtype=bool)
        valid[50:72, 97:301] = False
        valid[470:, 400:] = False
        result = lg.area_only_overlap(geo, lattice, valid)
        passed('synthetic_native_support_integrals', **result['audit'])
        direct = geo.overlap_native(lattice, valid, np.zeros_like(valid), support=2)
        assert np.array_equal(result['cells'], direct['cells'])
        residual = result['weight']-direct['weight']
        error = float(abs(residual.data).max()) if residual.nnz else 0.
        assert error < 1e-12
        passed('native_to_support_matches_frozen_support2', max_abs_weight_error_m2=error)
        # Exact native label integration is possible using already persisted native
        # weights. These patterns are synthetic and do not read any dataset truth.
        target = (np.indices(valid.shape).sum(axis=0) % 7 == 0) & valid
        target_result = geo.overlap_native(lattice, valid, target, support=2)
        from_native = result['native_weight'] @ target.ravel().astype(float)
        assert np.allclose(from_native, target_result['target'], rtol=1e-12, atol=1e-12)
        passed('native_weights_reproduce_synthetic_exact_target', max_abs_error_m2=float(abs(from_native-target_result['target']).max()))
        midpoint = lg.all_edge_midpoint_error(geo, identity, affine, 0, 0, [0., 0.], lattice)
        assert midpoint['edge_midpoints_checked'] == 2*512*513
        passed('all_native_edge_midpoints', **midpoint)
        curve = SimpleNamespace(transform=lambda x, y: (x + np.asarray(x)**2*.1, y))
        curved_lattice = geo.projected_lattice(curve, affine, 0, 0, 512, 512, [0., 0.])
        rejected('curved_edge_error_rejected', lambda: lg.all_edge_midpoint_error(geo, curve, affine, 0, 0, [0., 0.], curved_lattice))
        rejected('empty_valid_window_rejected', lambda: lg.area_only_overlap(geo, lattice, np.zeros_like(valid)))
        cells = np.asarray([[-3, -1], [-2, 0], [0, 0], [1, 1], [2, 0], [3, 3], [4, 2]])
        _, nested = lg.nested_area_checks(geo, cells, np.asarray([.1, .2, .3, .4, .5, .6, .7]))
        passed('nested_direct_vs_child_with_negative_indices', **nested)
        sources = [{'source_id': 'a', 'width': 512, 'height': 512, 'global_offset': [0, 0],
                    'transform': [1., 0., 0., 0., 1., 0.]}]
        corners = lg.source_corner_alignment(geo, identity, sources)
        assert len(corners['sources'][0]['four_corner_error_m']) == 4
        wrong = dict(sources[0], source_id='b', transform=[1.-1e-6, 1e-6, 0., 0., 1., 0.])
        # This shear agrees on (0,0) and (512,512), so the former diagonal-only
        # check would miss it. Off-diagonal corners must reject it.
        rejected('off_diagonal_corner_error_rejected', lambda: lg.source_corner_alignment(geo, identity, sources+[wrong]))
        dataset = SimpleNamespace(width=512, height=512, transform=geo.rasterio.Affine.identity(),
                                  crs=geo.rasterio.crs.CRS.from_epsg(4326), count=4,
                                  colorinterp=[None, None, None, SimpleNamespace(name='alpha')])
        registered = dict(sources[0], crs='EPSG:4326')
        lg.registered_headers(geo, [registered], {'a': dataset}); passed('actual_header_match')
        wrong_header = dict(registered, transform=[1., 0., 1e-12, 0., 1., 0.])
        rejected('registered_actual_affine_mismatch', lambda: lg.registered_headers(geo, [wrong_header], {'a': dataset}))
        rejected('registered_actual_CRS_mismatch', lambda: lg.registered_headers(geo, [dict(registered, crs='EPSG:3857')], {'a': dataset}))
        own_sources = [dict(sources[0], width=520, height=512)]
        own_windows = [dict(id='x', source_id='a', x=512, y=0, global_x=512, global_y=0),
                       dict(id='y', source_id='a', x=0, y=0, global_x=0, global_y=0)]
        lg.ownership_checks(own_sources, own_windows); passed('source_clipped_padding_ownership')
        bad_window = dict(own_windows[0], x=500, global_x=500)
        rejected('overlap_ownership_rejected', lambda: lg.ownership_checks(own_sources, [bad_window, own_windows[1]]))
        cache = out/'synthetic_inputs.npz'
        # Object sentinel fails if the image member is ever decoded with
        # allow_pickle=False. The validity-only check must leave it untouched.
        expected_valid = valid.reshape(256, 2, 256, 2).mean(axis=(1, 3)).astype(np.float32)
        np.savez(cache, image=np.array([object()], dtype=object), valid=expected_valid)
        lg.check_cached_valid(geo, valid, {'input_cache': str(cache), 'input_cache_sha256': lg.sha(cache)})
        passed('valid_only_NPZ_member_access')
        altered = expected_valid.copy(); altered[0, 0] = .25
        bad_cache = out/'wrong_valid.npz'; np.savez(bad_cache, image=np.zeros((3, 256, 256), np.float32), valid=altered)
        rejected('native_cache_valid_mismatch', lambda: lg.check_cached_valid(geo, valid, {'input_cache': str(bad_cache), 'input_cache_sha256': lg.sha(bad_cache)}))

        # Bounded real-data parity. Read only known validation members, native
        # TIFF alpha, their cached valid keys, and archived geometry area arrays.
        manifest_path = ROOT/'local_data_v2'/'val.json'
        assert lg.sha(manifest_path) == 'bd1545bdc91499861a4b83c57d70b5024790bac6981ab3358c53268cc9d30851'
        old_rows = core.read(manifest_path)
        assert len(old_rows) == 357 and all(row['split'] == 'val' for row in old_rows)
        old_by_id = {row['id']: row for row in old_rows}
        audit = core.read(ROOT/'local_sampling_completeness.json')
        summary = core.read(ROOT/'local_data_v2'/'manifest_summary.json')
        hashes = {row['source_id']: row['tif_sha256'] for row in summary['sources']}
        used_sources = {row['source_id'] for row in old_rows}
        sources = [{**{k: source[k] for k in ('source_id', 'tif', 'width', 'height', 'crs', 'global_offset')},
                    'transform': source['transform'][:6], 'tif_sha256': hashes[source['source_id']], 'valid_band': 4}
                   for source in audit['sources'] if source['source_id'] in used_sources]
        selected_ids = ['4_7_10_10', '4_7_11_6']
        records = [{'id': row['id']} for row in old_rows]
        records_by_id = {x['id']: x for x in records}
        for identifier in selected_ids:
            row = old_by_id[identifier]
            assert lg.sha(row['cache']) == row['cache_sha256']
            with np.load(row['cache'], allow_pickle=False) as data:
                prepared_valid = data['valid'].astype(np.float32)
            path = out/(identifier+'_input_fixture.npz')
            # Image is synthetic and never used by geometry. No RGB is decoded.
            np.savez_compressed(path, image=np.zeros((3, 256, 256), np.float32), valid=prepared_valid)
            records_by_id[identifier].update(input_cache=str(path), input_cache_sha256=lg.sha(path))
        windows = [{**{k: row[k] for k in ('id', 'source_id', 'x', 'y', 'global_x', 'global_y')},
                    'native_size': 512} for row in old_rows]
        fixture = {'split': 'val', 'expected_count': 357, 'records': records,
                   'geometry_sources': sources, 'geometry_windows': windows}
        archived = core.read(ROOT/'precision_agriculture/geometry_20260916_v1/metadata.json')
        validation_out = out/'allowed_validation_subset'; validation_out.mkdir()
        result = lg.build_geometry(fixture, validation_out, 357, 'val', core.write,
                                   selected_ids=selected_ids, reference_origin=archived['origin_utm_m'])
        comparisons = []
        old_windows = {x['id']: x for x in archived['windows']}
        for record in result['windows']:
            old = old_windows[record['id']]
            assert lg.sha(old['file']) == old['sha256']
            with np.load(old['file'], allow_pickle=False) as original, np.load(validation_out/record['file'], allow_pickle=False) as current:
                assert np.array_equal(original['cells'], current['cells'])
                W0 = geo.sparse.csr_matrix((original['weights'], original['indices'], original['indptr']), shape=original['shape'])
                W1 = geo.sparse.csr_matrix((current['weights'], current['indices'], current['indptr']), shape=current['shape'])
                delta = W0-W1
                max_weight = float(abs(delta.data).max()) if delta.nnz else 0.
                max_area = float(abs(original['area']-current['area']).max())
                assert max_weight < 1e-12 and max_area < 1e-12
                assert record['native_valid_pixels'] == old['native_valid_pixels']
                comparisons.append({'id': record['id'], 'native_valid_pixels': record['native_valid_pixels'],
                                    'max_weight_abs_error_m2': max_weight, 'max_area_abs_error_m2': max_area,
                                    'original_geometry_sha256': old['sha256']})
        passed('allowed_validation_full_and_partial_alpha_parity', windows=comparisons,
               origin_utm_m=result['origin_utm_m'], expected_full_val_ownership=357,
               actual_alpha_windows_read=len(selected_ids))
        assert lg.sha(manifest_path) == 'bd1545bdc91499861a4b83c57d70b5024790bac6981ab3358c53268cc9d30851'
        passed('original_validation_manifest_unchanged')
        receipt = {'passed': True, 'stage': 'preparation_only', 'checks': checks, 'check_count': len(checks),
                   'implementation_sha256': lg.sha(lg.__file__), 'checker_sha256': lg.sha(__file__),
                   'core_sha256': lg.sha(core.__file__), 'frozen_lowlevel_sha256': lg.GEO_SHA,
                   'test_manifest_accessed': False, 'test_pixels_or_cache_accessed': False,
                   'source_masks_decoded': 0, 'learned_predictions_read': 0, 'R6_forward_calls': 0,
                   'allowed_validation_windows': selected_ids, 'validation_cache_array_keys_read': ['valid'],
                   'validation_legacy_metadata_label_fields': 'May be parsed by normal JSON, not passed to geometry',
                   'production_gate': 'No FREEZE exists; no production run authorized or attempted past gate'}
        core.write(out/'focused_checks.json', receipt)
        core.write(out/'artifact_manifest.json', core.stage_manifest(out))
        print(json.dumps({'passed': True, 'output': str(out), 'checks': len(checks)}))
    except BaseException as error:
        core.write(out/'failure.json', {'error': repr(error), 'traceback': traceback.format_exc(), 'checks': checks})
        print(str(out), file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
