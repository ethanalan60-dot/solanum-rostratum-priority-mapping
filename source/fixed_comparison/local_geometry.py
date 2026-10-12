"""Native-alpha geometry for reserved local support; production requires root freeze.

No mask, soft label, target count or prediction is loaded. Frozen development
low-level geometry is reused with a synthetic all-zero target argument. Its
target-related return values are discarded, not interpreted as observations.
"""
from pathlib import Path
from contextlib import ExitStack
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GEO_PATH = ROOT / 'geo_management.py'
GEO_SHA = '16c6fb38e1c2c1c9c5ea9f5f25db317d0fa021505631ddef9f07dea528549d8f'
SOURCE_KEYS = frozenset(('source_id', 'tif', 'tif_sha256', 'width', 'height',
                         'transform', 'crs', 'global_offset', 'valid_band'))
WINDOW_KEYS = frozenset(('id', 'source_id', 'x', 'y', 'global_x', 'global_y', 'native_size'))
AREA_RTOL = 1e-9
NESTED_RTOL = 1e-10
POSITION_TOL_M = 1e-7


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dependencies():
    """Import only after the caller's access gate; never call the old build()."""
    require(sha(GEO_PATH) == GEO_SHA, 'Frozen low-level geometry source changed')
    spec = importlib.util.spec_from_file_location('confirmation_original_geometry', GEO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def integer(value, name):
    require(type(value) is int, 'Expected integer: ' + name)
    return value


def validate_whitelist(manifest, expected_count, expected_split):
    """Only geometry and record identities enter this projection."""
    require(manifest['split'] == expected_split, 'Wrong geometry split')
    require(manifest['expected_count'] == expected_count, 'Wrong expected member count')
    sources = manifest['geometry_sources']
    windows = manifest['geometry_windows']
    require(len(windows) == expected_count and bool(sources), 'Incomplete geometry membership')
    require(all(set(x) == SOURCE_KEYS for x in sources), 'Source whitelist differs')
    require(all(set(x) == WINDOW_KEYS for x in windows), 'Window whitelist differs')
    require(len({x['source_id'] for x in sources}) == len(sources), 'Duplicate source ID')
    require(len({x['id'] for x in windows}) == len(windows), 'Duplicate window ID')
    require([x['id'] for x in manifest['records']] == [x['id'] for x in windows],
            'Prediction/geometry member order differs')
    by_source = {x['source_id']: x for x in sources}
    for source in sources:
        require(source['valid_band'] == 4, 'Native validity must be original band 4 alpha')
        require(integer(source['width'], 'width') > 0 and integer(source['height'], 'height') > 0,
                'Empty source')
        require(len(source['transform']) == 6 and len(source['global_offset']) == 2,
                'Wrong affine/offset shape')
        require(all(math.isfinite(float(x)) for x in source['transform']), 'Nonfinite affine')
        for value in source['global_offset']:
            integer(value, 'global offset')
    for window in windows:
        require(window['source_id'] in by_source, 'Unknown source')
        require(window['native_size'] == 512, 'Native support changed')
        for key in ('x', 'y', 'global_x', 'global_y'):
            integer(window[key], key)
        source = by_source[window['source_id']]
        require(0 <= window['x'] < source['width'] and 0 <= window['y'] < source['height'],
                'Window starts outside source')
        require([window['global_x'] - window['x'], window['global_y'] - window['y']]
                == source['global_offset'], 'Window/global source offset differs')
        require(all(c.isascii() and (c.isalnum() or c in '_-') for c in window['id'])
                and bool(window['id']), 'Unsafe window ID')
    return sources, windows


def ownership_checks(sources, windows):
    """Padded source-exterior pixels own no ground; reject overlapping real supports."""
    by_source = {x['source_id']: x for x in sources}
    rectangles = []
    comparisons = 0
    for window in windows:
        source = by_source[window['source_id']]
        x, y = window['global_x'], window['global_y']
        w = min(512, source['width'] - window['x'])
        h = min(512, source['height'] - window['y'])
        rect = (x, y, x + w, y + h)
        for other in rectangles:
            comparisons += 1
            require(min(rect[2], other[2]) <= max(rect[0], other[0]) or
                    min(rect[3], other[3]) <= max(rect[1], other[1]),
                    'Overlapping source-clipped native ownership')
        rectangles.append(rect)
    return {'passed': True, 'windows': len(windows), 'pair_checks': comparisons,
            'rule': 'Disjoint source-clipped rectangles in one aligned native grid; padded pixels excluded'}


def registered_headers(geo, sources, datasets):
    """Check actual TIFF headers against registration before using any affine."""
    checked = []
    for source in sources:
        dataset = datasets[source['source_id']]
        require(dataset.width == source['width'] and dataset.height == source['height'],
                'Actual source dimensions differ')
        require(tuple(dataset.transform)[:6] == tuple(source['transform']), 'Actual source affine differs')
        require(dataset.crs == geo.rasterio.crs.CRS.from_user_input(source['crs']), 'Actual source CRS differs')
        require(dataset.crs == geo.rasterio.crs.CRS.from_epsg(4326), 'Frozen native CRS is EPSG:4326')
        require(dataset.count >= 4 and dataset.colorinterp[3].name == 'alpha', 'Band 4 is not alpha')
        checked.append({'source_id': source['source_id'], 'transform': list(dataset.transform)[:6],
                        'crs': str(dataset.crs), 'width': dataset.width, 'height': dataset.height,
                        'alpha_band': 4})
    return checked


def source_corner_alignment(geo, transformer, sources):
    np = geo.np
    reference = sources[0]
    reference_affine = geo.rasterio.Affine(*reference['transform'])
    maximum = 0.0
    residuals = []
    for source in sources:
        offset = np.asarray(source['global_offset']) - np.asarray(reference['global_offset'])
        expected = reference_affine * geo.rasterio.Affine.translation(*offset)
        actual = geo.rasterio.Affine(*source['transform'])
        # All four corners, not the old diagonal-only test.
        x = np.asarray([0., source['width'], source['width'], 0.])
        y = np.asarray([0., 0., source['height'], source['height']])
        ex, ey = transformer.transform(expected.a*x + expected.b*y + expected.c,
                                       expected.d*x + expected.e*y + expected.f)
        ax, ay = transformer.transform(actual.a*x + actual.b*y + actual.c,
                                       actual.d*x + actual.e*y + actual.f)
        errors = np.hypot(np.asarray(ex)-ax, np.asarray(ey)-ay)
        require(np.isfinite(errors).all(), 'Nonfinite corner transform')
        maximum = max(maximum, float(errors.max()))
        residuals.append({'source_id': source['source_id'], 'four_corner_error_m': errors.tolist()})
    require(maximum < POSITION_TOL_M, 'Source corner alignment exceeded 1e-7 m')
    return {'maximum_error_m': maximum, 'tolerance_exclusive_m': POSITION_TOL_M,
            'reference_source': reference['source_id'], 'corners_per_source': 4, 'sources': residuals}


def geometric_origin(geo, transformer, sources, windows):
    """Same origin rule as development: all nominal window corners, floor to 4 m.

    Padded corners can affect only the grid origin, never area/ownership. No alpha,
    annotation or model score is used to choose the origin.
    """
    np = geo.np
    by_source = {x['source_id']: x for x in sources}
    bounds = []
    for window in windows:
        affine = geo.rasterio.Affine(*by_source[window['source_id']]['transform'])
        x = np.asarray([window['x'], window['x']+512, window['x']+512, window['x']], dtype=float)
        y = np.asarray([window['y'], window['y'], window['y']+512, window['y']+512], dtype=float)
        east, north = transformer.transform(affine.a*x + affine.b*y + affine.c,
                                            affine.d*x + affine.e*y + affine.f)
        bounds.append(np.column_stack((east, north)))
    all_bounds = np.concatenate(bounds)
    require(np.isfinite(all_bounds).all(), 'Nonfinite geometric origin')
    return (np.floor(all_bounds.min(axis=0)/4)*4).tolist()


def all_edge_midpoint_error(geo, transformer, affine, x0, y0, origin, lattice):
    """Check every horizontal and vertical native edge midpoint (including boundary)."""
    np = geo.np
    h, w = lattice.shape[0]-1, lattice.shape[1]-1
    maximum = 0.0
    count = 0
    for horizontal in (True, False):
        xx, yy = np.meshgrid(np.arange(w if horizontal else w+1, dtype=float),
                             np.arange(h+1 if horizontal else h, dtype=float))
        x = xx+x0+(0.5 if horizontal else 0.)
        y = yy+y0+(0. if horizontal else 0.5)
        east, north = transformer.transform(affine.a*x+affine.b*y+affine.c,
                                            affine.d*x+affine.e*y+affine.f)
        exact = np.stack((east-origin[0], north-origin[1]), axis=-1)
        chord = ((lattice[:, :-1]+lattice[:, 1:]) if horizontal else
                 (lattice[:-1]+lattice[1:]))*.5
        errors = np.linalg.norm(exact-chord, axis=-1)
        require(np.isfinite(errors).all(), 'Nonfinite edge midpoint projection')
        maximum = max(maximum, float(errors.max()))
        count += int(errors.size)
    require(maximum < POSITION_TOL_M, 'Native edge midpoint chord error exceeded 1e-7 m')
    return {'maximum_error_m': maximum, 'edge_midpoints_checked': count,
            'tolerance_exclusive_m': POSITION_TOL_M}


def area_only_overlap(geo, lattice, valid):
    """Reuse accepted overlap implementation without any observed target array."""
    np = geo.np
    require(valid.dtype == np.bool_ and valid.shape == (512, 512), 'Wrong native alpha shape/type')
    require(bool(valid.any()), 'Empty valid window: cannot silently drop locked member')
    # Keep native columns for a later exact label join. Collapsing label fractions
    # into 2x2 means before the metre-grid overlap would lose boundary information.
    result = geo.overlap_native(lattice, valid, np.zeros(valid.shape, dtype=np.bool_), support=1)
    allowed_audit = ('native_valid_pixels', 'raw_area_m2', 'area_relative_error',
                     'max_pixel_conservation_abs_m2', 'max_support_area_abs_error_m2',
                     'crossing_native_pixels', 'base_cells', 'sparse_entries')
    native = result['weight']
    coo = native.tocoo()
    support_column = (coo.col//512//2)*256+(coo.col % 512//2)
    support_weight = geo.sparse.coo_matrix((coo.data, (coo.row, support_column)),
                                          shape=(native.shape[0], 65536)).tocsr()
    raw_native = result['raw_valid_by_support']
    support_raw = raw_native.reshape(256, 2, 256, 2).sum(axis=(1, 3)).ravel()
    output = {'cells': result['cells'], 'area': result['area'], 'weight': support_weight,
              'raw_valid_by_support': support_raw, 'native_weight': native,
              'raw_valid_by_native': raw_native}
    output['audit'] = {key: result['audit'][key] for key in allowed_audit}
    A, weight, raw = output['area'], output['weight'], output['raw_valid_by_support']
    native_total = output['audit']['raw_area_m2']
    require(np.isfinite(A).all() and (A > 0).all(), 'Nonpositive/nonfinite management area')
    require(np.isfinite(weight.data).all() and (weight.data > 0).all(), 'Invalid sparse overlap')
    require(abs(float(A.sum())-native_total)/native_total < AREA_RTOL, 'Native/grid area failure')
    column_error = float(np.max(np.abs(np.asarray(weight.sum(axis=0)).ravel()-raw)))
    require(column_error/native_total < AREA_RTOL, 'Native/prediction support area failure')
    row_error = float(np.max(np.abs(np.asarray(weight.sum(axis=1)).ravel()-A)))
    require(row_error/native_total < AREA_RTOL, 'Sparse row sum/area failure')
    # A fixed synthetic bounded field checks the actual future W @ p data flow.
    # This is not a learned map, a label or an R6 model forward.
    p = ((np.arange(65536, dtype=np.float64)*37) % 257)/256.
    expected = float(np.dot(raw, p))
    got = float(np.asarray(weight @ p).sum())
    relative = abs(got-expected)/max(expected, 1e-30)
    require(relative < AREA_RTOL, 'Synthetic prediction-area integration failure')
    output['audit'].update(sparse_row_area_max_abs_m2=row_error,
                           max_native_column_abs_error_m2=result['audit']['max_support_area_abs_error_m2'],
                           max_support_area_abs_error_m2=column_error,
                           synthetic_prediction_integral_relative_error=relative,
                           native_sparse_entries=int(native.nnz), support_sparse_entries=int(support_weight.nnz))
    return output


def nested_area_checks(geo, cells, areas):
    np = geo.np
    total = float(areas.sum())
    require(total > 0 and np.isfinite(total), 'Empty/nonfinite domain')
    outputs = {}
    errors = {}
    for size in (1, 2, 4):
        parents, A = geo.aggregate_nested(cells, areas, size)
        A = A[:, 0]
        relative = abs(float(A.sum())-total)/total
        require(relative < NESTED_RTOL, 'Nested total area failure')
        outputs[size] = (parents, A)
        errors[str(size)] = relative
    # 1->4 direct must equal 1->2->4 cell by cell, not only the grand total.
    via_keys, via_A = geo.aggregate_nested(outputs[2][0], outputs[2][1], 2)
    require(np.array_equal(via_keys, outputs[4][0]), 'Nested parent IDs differ')
    residual = np.abs(via_A[:, 0]-outputs[4][1])
    require(np.all(residual/outputs[4][1] < NESTED_RTOL), 'Nested direct/child area failure')
    return outputs, {'total_relative_errors': errors, 'parent_cell_checks': len(via_keys),
                     'max_parent_relative_error': float((residual/outputs[4][1]).max()),
                     'max_parent_absolute_error_m2': float(residual.max()), 'relative_tolerance': NESTED_RTOL}


def check_cached_valid(geo, native_valid, record):
    """Access just the valid NPZ member, never image/soft/mask members."""
    np = geo.np
    path = record['input_cache']
    require(sha(path) == record['input_cache_sha256'], 'Prepared input cache changed')
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == {'image', 'valid'}, 'Input cache includes unexpected arrays')
        valid = data['valid']
    require(sha(path) == record['input_cache_sha256'], 'Prepared input cache changed while reading valid')
    expected = native_valid.reshape(256, 2, 256, 2).mean(axis=(1, 3)).astype(np.float32)
    require(valid.dtype == np.float32 and valid.shape == (256, 256), 'Wrong prepared validity type/shape')
    require(np.array_equal(valid, expected), 'Prepared validity differs from native alpha 2x2 mean')
    return {'input_cache_sha256': record['input_cache_sha256'], 'array_accessed': 'valid',
            'max_absolute_error': 0.0, 'comparison': 'exact; fractions 0/.25/.5/.75/1'}


def build_geometry(manifest, out, expected_count, expected_split, write_json,
                   selected_ids=None, reference_origin=None):
    """Internal builder; preparation checks may use explicit allowed validation IDs.

    Production calls never supply selected_ids/reference_origin and process all
    frozen members. Source TIFF hashes are checked before and after all reads.
    """
    geo = dependencies()
    np = geo.np
    sources, all_windows = validate_whitelist(manifest, expected_count, expected_split)
    record_by_id = {x['id']: x for x in manifest['records']}
    ownership = ownership_checks(sources, all_windows)
    windows = all_windows if selected_ids is None else [x for x in all_windows if x['id'] in selected_ids]
    require(bool(windows) and (selected_ids is None or len(windows) == len(set(selected_ids))),
            'Diagnostic member selection incomplete')
    out = Path(out)
    (out/'tiles').mkdir(exist_ok=False)
    transformer = geo.Transformer.from_crs('EPSG:4326', 'EPSG:32649', always_xy=True)
    hashes = {}
    for source in sources:
        require(sha(source['tif']) == source['tif_sha256'], 'Source TIFF hash differs')
        hashes[source['tif']] = source['tif_sha256']
    all_cells, all_A, records = [], [], []
    with ExitStack() as stack:
        datasets = {x['source_id']: stack.enter_context(geo.rasterio.open(x['tif'])) for x in sources}
        headers = registered_headers(geo, sources, datasets)
        alignment = source_corner_alignment(geo, transformer, sources)
        origin = geometric_origin(geo, transformer, sources, all_windows)
        if reference_origin is not None:
            require(origin == reference_origin, 'Allowed validation origin disagrees with archived geometry')
        for index, window in enumerate(windows):
            dataset = datasets[window['source_id']]
            valid = dataset.read(4, window=geo.Window(window['x'], window['y'], 512, 512),
                                 boundless=True, fill_value=0) > 0
            # Boundless padded pixels must remain unowned even if driver behavior changes.
            h, w = min(512, dataset.height-window['y']), min(512, dataset.width-window['x'])
            require(not valid[h:].any() and not valid[:, w:].any(), 'Nonzero source-exterior alpha')
            cached = check_cached_valid(geo, valid, record_by_id[window['id']])
            lattice = geo.projected_lattice(transformer, dataset.transform, window['x'], window['y'],
                                            512, 512, origin)
            midpoint = all_edge_midpoint_error(geo, transformer, dataset.transform, window['x'], window['y'], origin, lattice)
            result = area_only_overlap(geo, lattice, valid)
            weight = result['weight']
            native = result['native_weight']
            path = out/'tiles'/(window['id']+'.npz')
            np.savez_compressed(path, cells=result['cells'], area=result['area'], weights=weight.data,
                                indices=weight.indices, indptr=weight.indptr, shape=np.array(weight.shape),
                                raw_valid_by_support=result['raw_valid_by_support'],
                                native_weights=native.data, native_indices=native.indices,
                                native_indptr=native.indptr, native_shape=np.array(native.shape),
                                raw_valid_by_native=result['raw_valid_by_native'])
            records.append({**window, **result['audit'], 'midpoint': midpoint,
                            'prepared_validity_check': cached,
                            'file': path.relative_to(out).as_posix(), 'sha256': sha(path)})
            all_cells.append(result['cells']); all_A.append(result['area'])
            write_json(out/'progress.json', {'completed_windows': index+1, 'expected_windows': len(windows)})
    cells, inverse = np.unique(np.concatenate(all_cells), axis=0, return_inverse=True)
    area = np.bincount(inverse, weights=np.concatenate(all_A), minlength=len(cells))
    nested, nested_checks = nested_area_checks(geo, cells, area)
    rows = []
    for size, (keys, areas) in nested.items():
        for (ix, iy), A in zip(keys, areas):
            rows.append({'M': size, 'id': f'M{size}_E{int(ix)}_N{int(iy)}',
                         'ix': int(ix), 'iy': int(iy), 'area_m2': float(A)})
    with (out/'area_units.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    total = math.fsum(row['raw_area_m2'] for row in records)
    require(abs(float(area.sum())-total)/total < AREA_RTOL, 'Cross-window area aggregation failure')
    for path, digest in hashes.items():
        require(sha(path) == digest, 'Source changed during geometry integration')
    require(sha(GEO_PATH) == GEO_SHA, 'Low-level source changed during geometry integration')
    metadata = {'state': 'completed', 'geometry_kind': 'native_alpha_only', 'split': expected_split,
                'expected_windows': len(windows), 'complete_declared_domain': selected_ids is None,
                'native_valid_pixels': sum(x['native_valid_pixels'] for x in records),
                'area_m2': total, 'origin_utm_m': origin, 'crs': 'EPSG:32649', 'base_grid_m': 1,
                'prediction_support': [256, 256], 'native_pixels_per_prediction_cell': [2, 2],
                'origin_rule': 'floor(min of all nominal native window projected corners / 4) * 4',
                'annotation_arrays_opened': 0, 'prediction_arrays_opened': 0,
                'synthetic_zero_argument_only': True, 'mask_paths_consumed': False,
                'source_tif_sha256': hashes, 'source_headers': headers,
                'ownership': ownership, 'source_four_corner_alignment': alignment,
                'nested_area_checks': nested_checks, 'windows': records,
                'source_sha256': {str(Path(__file__).resolve()): sha(__file__), str(GEO_PATH): GEO_SHA},
                'versions': {'numpy': np.__version__, 'scipy': geo.scipy.__version__,
                             'shapely': geo.shapely.__version__, 'pyproj': geo.pyproj_version,
                             'rasterio': geo.rasterio.__version__},
                'subsequent_truth_rule': 'Reconstruct native_weight from native_weights/native_indices/native_indptr/'
                                        'native_shape and multiply by an allowed native 512x512 truth vector after '
                                        'all model choices are sealed. Do not multiply support weights by coarse label means.'}
    write_json(out/'metadata.json', metadata)
    return metadata


def capture_execution_bindings(root_review):
    # Preserve the versions that validators will parse, not later replacements.
    import core
    watched={str(Path(p).resolve()):sha(p) for p in (HERE/'FREEZE.json',HERE/'SOURCE_LOCK.json',HERE/'contract.json',HERE/'checkpoints.json',
      Path(__file__),HERE/'core.py',Path(root_review),GEO_PATH)}
    manifest_path=HERE/'stages/inputs_local/input_manifest.json'
    for path in (manifest_path,manifest_path.parent/'artifact_manifest.json'):watched[str(path.resolve())]=sha(path)
    return watched

def execute(root_review):
    import core
    watched=capture_execution_bindings(root_review)
    # First action involving data gates: absent FREEZE fails before any manifest read.
    contract, _, freeze = core.validate_freeze(check_models=False)
    require(contract['datasets']['local']['count'] == 338, 'Wrong frozen local count')
    review = core.read(root_review)
    manifest_path = HERE/'stages'/'inputs_local'/'input_manifest.json'
    require(review.get('passed') is True and review.get('scope') == 'confirmation_local_geometry',
            'Independent readiness/root execution review required')
    require(review['implementation_sha256'] == sha(__file__) and review['freeze_sha256'] == sha(HERE/'FREEZE.json'),
            'Review does not bind this implementation/freeze')
    require(review['input_manifest_sha256'] == sha(manifest_path), 'Review does not bind prepared geometry inputs')
    core.completed_stage(manifest_path.parent)
    manifest = core.read(manifest_path)
    source_manifest = manifest['source_manifest']
    require(source_manifest['sha256'] == contract['datasets']['local']['manifest_sha256'], 'Membership digest differs')
    require(Path(source_manifest['path']).resolve() == Path(contract['datasets']['local']['manifest_path']).resolve(),
            'Membership source path differs')
    core.bound_file(source_manifest['path'], source_manifest['sha256'])
    for path,digest in watched.items():core.bound_file(path,digest)
    with core.attempt('local_geometry') as out:
        core.configure_cpu(out)
        source = out/'source'; source.mkdir()
        for path in (Path(__file__), GEO_PATH, HERE/'core.py', Path(root_review), HERE/'FREEZE.json'):
            shutil.copy2(path, source/path.name)
        core.write(out/'input_binding.json', {'files_sha256': watched, 'membership': source_manifest,
                                            'implementation': core.binding(__file__)})
        started = time.time()
        metadata = build_geometry(manifest, out, 338, 'test_sealed', core.write)
        metadata['seconds'] = time.time()-started
        core.write(out/'metadata.json', metadata)
        for path, digest in watched.items():
            core.bound_file(path, digest)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--root-review', required=True)
    args = parser.parse_args()
    print(execute(args.root_review))


if __name__ == '__main__':
    main()
