"""Independent CPU geometry checks. Reads validation only; never uses test/predictions.

The oracle constructs and clips every native polygon independently, including
interior polygons. It does not call implementation area/overlap helpers.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import geo_management as gm
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(lattice, valid, target, probabilities, size=1, support=2):
    h, w = valid.shape
    result = {}
    columns = np.zeros((h // support) * (w // support), np.float64)
    for y, x in zip(*np.nonzero(valid)):
        poly = Polygon([lattice[y, x], lattice[y, x+1],
                        lattice[y+1, x+1], lattice[y+1, x]])
        assert poly.is_valid and poly.area > 0
        col = (y // support) * (w // support) + x // support
        columns[col] += poly.area
        xmin, ymin, xmax, ymax = poly.bounds
        # Include zero-area candidates deliberately: no implementation nextafter.
        for ix in range(int(np.floor(xmin / size)), int(np.floor(xmax / size))+1):
            for iy in range(int(np.floor(ymin / size)), int(np.floor(ymax / size))+1):
                area = poly.intersection(box(ix*size, iy*size,
                                             (ix+1)*size, (iy+1)*size)).area
                if area <= 0:
                    continue
                row = result.setdefault((ix, iy), np.zeros(2+len(probabilities)))
                row[0] += area
                row[1] += area * bool(target[y, x])
                row[2:] += area * probabilities[:, col]
    return result, columns


def dictionary(cells, values):
    return {tuple(map(int, c)): np.asarray(v) for c, v in zip(cells, values)}


def compare(actual, expected, atol=2e-10, rtol=1e-8):
    keys = sorted(set(actual) | set(expected))
    ncols = len(next(iter(actual.values()), next(iter(expected.values()), [0, 0, 0])))
    a = np.array([actual.get(k, np.zeros(ncols)) for k in keys]).reshape(-1, ncols)
    b = np.array([expected.get(k, np.zeros(ncols)) for k in keys]).reshape(-1, ncols)
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
    return {"cells": len(keys), "max_abs_m2": float(np.max(np.abs(a-b))) if a.size else 0.,
            "l1_by_quantity_m2": np.abs(a-b).sum(axis=0).tolist()}


def check_case(name, lattice, valid, target, sizes=(1, 2, 4)):
    start = time.time()
    h, w = valid.shape
    n = h*w//4
    rng = np.random.default_rng(73021)
    # Constant + unique support-index ramp + random values detect transposition.
    p = np.vstack([np.full(n, .37), np.arange(n)/max(1, n-1), rng.random(n)])
    got = gm.overlap_native(lattice, valid, target, support=2)
    values = np.column_stack([got['area'], got['target'], got['weight'].dot(p.T)])
    assert np.all(got['weight'].data >= 0)
    np.testing.assert_allclose(values[:, 2], .37*values[:, 0], atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(got['weight'].sum(axis=1)).ravel(),
                               got['area'], atol=1e-11, rtol=1e-10)
    result = {"name": name, "shape": list(valid.shape), "grids": {}}
    refs = {}
    for size in sizes:
        want, column_area = reference(lattice, valid, target, p, size=size)
        refs[size] = want
        parents, aggregate = gm.aggregate_nested(got['cells'], values, size)
        result['grids'][str(size)] = compare(dictionary(parents, aggregate), want)
        np.testing.assert_allclose(got['raw_valid_by_support'], column_area,
                                   atol=1e-11, rtol=1e-9)
    if 2 in refs and 4 in refs:
        p2, v2 = gm.aggregate_nested(got['cells'], values, 2)
        # p2 are 2 m indices: grouping by two gives original 4 m indices.
        p4, v4 = gm.aggregate_nested(p2, v2, 2)
        result['via_2_to_4'] = compare(dictionary(p4, v4), refs[4])
    result['audit'] = got['audit']
    result['seconds'] = time.time()-start
    print(json.dumps({"passed": name, "seconds": round(result['seconds'], 3)}, ensure_ascii=True), flush=True)
    return result


def affine_lattice(h, w, matrix, offset):
    x, y = np.meshgrid(np.arange(w+1), np.arange(h+1))
    return np.stack([x, y], axis=-1) @ np.asarray(matrix).T + np.asarray(offset)


def synthetic():
    results = []
    rng = np.random.default_rng(4567)
    h, w = 6, 10
    v = rng.random((h, w)) > .3
    t = (rng.random((h, w)) > .55) & v
    cases = [
        ('rotated_sheared_irregular', [[.19, -.11], [.08, .23]], [-.45, -.31]),
        ('exact_positive_negative_grid_edges', [[.25, 0], [0, -.25]], [-1., 1.]),
        ('large_absolute_utm_small_pixels', [[.0041, -.0012], [.0008, .0039]], [500000.992, 4000000.992]),
    ]
    for name, matrix, offset in cases:
        results.append(check_case(name, affine_lattice(h, w, matrix, offset), v, t))
    # Outer tile corners share one cell; alpha contains an interior hole.
    lattice = affine_lattice(8, 8, [[.07, 0], [0, .07]], [.2, .2])
    hole = np.ones((8, 8), bool); hole[2:6, 2:6] = False
    target = np.indices((8, 8)).sum(axis=0) % 3 == 0
    results.append(check_case('all_outer_corners_same_cell_with_alpha_hole', lattice, hole, target & hole))
    # Partial valid support straddles x=1; native locations must be retained.
    lattice = affine_lattice(2, 2, [[.6, 0], [0, .4]], [.4, .1])
    valid = np.array([[1, 0], [1, 0]], bool)
    target = np.array([[1, 0], [0, 0]], bool)
    results.append(check_case('crossing_2x2_only_left_valid', lattice, valid, target))
    got = gm.overlap_native(lattice, valid, target)
    assert got['cells'].tolist() == [[0, 0]]
    np.testing.assert_allclose([got['area'][0], got['target'][0]], [.48, .24], atol=1e-12)
    # A smooth nonaffine vertex mesh detects replacing native quads by coarse quads.
    lattice = affine_lattice(6, 10, [[.18, -.06], [.07, .18]], [-.3, -.2])
    x, y = np.meshgrid(np.arange(11), np.arange(7))
    lattice[..., 0] += .003*y*y
    lattice[..., 1] += .002*x*x
    results.append(check_case('curved_shared_lattice', lattice, v, t))
    results.append(check_case('empty_valid', lattice, np.zeros_like(v), np.zeros_like(t)))
    results.append(check_case('empty_target', lattice, v, np.zeros_like(t)))
    results.append(check_case('full_target', lattice, v, v))
    single = np.zeros_like(v); single[2, 4] = True
    results.append(check_case('single_valid_pixel', lattice, single, single))
    # Equivalent common origins separated by integer 4 m multiples.
    near = affine_lattice(6, 10, [[.0041, -.0012], [.0008, .0039]], [.992, .992])
    shift = np.array([500000., 4000000.])
    a = gm.overlap_native(near, v, t)
    b = gm.overlap_native(near+shift, v, t)
    np.testing.assert_array_equal(a['cells'], b['cells']-shift.astype(np.int64))
    np.testing.assert_allclose(a['area'], b['area'], rtol=2e-7, atol=1e-10)
    np.testing.assert_allclose(a['target'], b['target'], rtol=2e-7, atol=1e-10)
    results.append({"name": "large_coordinate_translation", "max_area_difference_m2": float(np.max(np.abs(a['area']-b['area']))), "tolerance_note": "Accounts for representable-coordinate precision before polygon clipping."})
    # All support columns, not only sampled predictions, against an identity oracle.
    lattice = affine_lattice(6, 10, [[.19, -.11], [.08, .23]], [-.45, -.31])
    got = gm.overlap_native(lattice, v, t)
    identity = np.eye(15)
    expected, _ = reference(lattice, v, t, identity)
    quantities = np.column_stack([got['area'], got['target'], got['weight'].toarray()])
    result = compare(dictionary(got['cells'], quantities), expected)
    result['name'] = 'all_sparse_support_columns_identity_oracle'
    results.append(result)
    # Shared-edge source chunks merge to the same map when ownership is disjoint.
    p = rng.random((3, 5))
    whole = np.column_stack([got['area'], got['target'], got['weight'].dot(p.ravel())])
    merged = {}
    for y0, y1 in [(0, 2), (2, 6)]:
        for x0, x1 in [(0, 4), (4, 10)]:
            part = gm.overlap_native(lattice[y0:y1+1, x0:x1+1], v[y0:y1, x0:x1], t[y0:y1, x0:x1])
            pred = part['weight'].dot(p[y0//2:y1//2, x0//2:x1//2].ravel())
            for cell, quantities in zip(part['cells'], np.column_stack([part['area'], part['target'], pred])):
                key = tuple(map(int, cell))
                merged[key] = merged.get(key, np.zeros(3)) + quantities
    result = compare(merged, dictionary(got['cells'], whole))
    result['name'] = 'disjoint_shared_edge_retiling_invariance'
    results.append(result)
    return results


def real_validation():
    rows = gm.read(ROOT/'local_data_v2/val.json')
    assert len(rows) == 357 and all(r['split'] == 'val' for r in rows)
    # Fixed lexical first positive validation tile; uses annotation, never a score.
    row = sorted([r for r in rows if r['coverage'] > 0], key=lambda r: r['id'])[0]
    sources = {s['source_id']: s for s in gm.read(ROOT/'local_sampling_completeness.json')['sources']}
    source = sources[row['source_id']]
    with gm.rasterio.open(source['tif']) as ds:
        valid = ds.read(4, window=gm.Window(row['x'], row['y'], 512, 512), boundless=True, fill_value=0) > 0
        affine = ds.transform
        assert str(ds.crs) == 'EPSG:4326'
    with gm.Image.open(source['mask']) as im:
        target = np.asarray(im.crop((row['x'], row['y'], row['x']+512, row['y']+512))) > 0
    target &= valid
    transformer = gm.Transformer.from_crs('EPSG:4326', 'EPSG:32649', always_xy=True)
    corner = transformer.transform(*(affine*(row['x'], row['y'])))
    origin = np.floor(np.asarray(corner)/4)*4
    lattice = gm.projected_lattice(transformer, affine, row['x'], row['y'], 512, 512, origin)
    # Independently check coordinate convention at several interior vertices.
    for y, x in [(0, 0), (511, 287), (512, 512)]:
        east, north = transformer.transform(*(affine*(row['x']+x, row['y']+y)))
        np.testing.assert_allclose(lattice[y, x], np.asarray([east, north])-origin, atol=1e-12, rtol=0)
    result = check_case('real_val_'+row['id'], lattice, valid, target, sizes=(1,))
    result.update(tile_id=row['id'], source_id=row['source_id'], x=row['x'], y=row['y'],
                  origin_utm_m=origin.tolist(), manifest_sha256=digest(ROOT/'local_data_v2/val.json'))
    return result


def validate_source_metadata():
    rows = gm.read(ROOT/'local_data_v2/val.json')
    assert all(r['split'] == 'val' for r in rows)
    audit = {s['source_id']: s for s in gm.read(ROOT/'local_sampling_completeness.json')['sources']}
    ids = sorted({r['source_id'] for r in rows})
    assert ids == ['4_7', '4_8']
    actual = {}
    for sid in ids:
        with gm.rasterio.open(audit[sid]['tif']) as ds:
            actual[sid] = {'transform': ds.transform, 'crs': str(ds.crs),
                           'width': ds.width, 'height': ds.height}
    ref = ids[0]
    transformer = gm.Transformer.from_crs('EPSG:4326', 'EPSG:32649', always_xy=True)
    records = []
    for sid in ids:
        entry = actual[sid]; source = audit[sid]
        assert entry['crs'] == source['crs'] == 'EPSG:4326'
        assert entry['width'] == source['width'] and entry['height'] == source['height']
        np.testing.assert_array_equal(np.array(tuple(entry['transform'])), np.array(source['transform']))
        delta = np.asarray(source['global_offset'])-np.asarray(audit[ref]['global_offset'])
        expected = actual[ref]['transform'] * gm.rasterio.Affine.translation(*delta)
        cached = gm.rasterio.Affine(*source['transform'][:6])
        x = np.array([0., entry['width'], entry['width'], 0.])
        y = np.array([0., 0., entry['height'], entry['height']])
        def corners(transform):
            east, north = transformer.transform(transform.a*x+transform.b*y+transform.c,
                                                transform.d*x+transform.e*y+transform.f)
            return np.column_stack([east, north])
        real = corners(entry['transform'])
        relative_error = np.linalg.norm(real-corners(expected), axis=1)
        cached_error = np.linalg.norm(real-corners(cached), axis=1)
        assert max(relative_error) < 1e-7 and max(cached_error) < 1e-7
        for row in rows:
            if row['source_id'] == sid:
                assert row['global_x'] == row['x']+source['global_offset'][0]
                assert row['global_y'] == row['y']+source['global_offset'][1]
        records.append({'source_id': sid, 'crs': entry['crs'], 'width': entry['width'], 'height': entry['height'],
                        'actual_transform': list(entry['transform']), 'audit_transform_exact_match': True,
                        'actual_vs_audit_four_corner_errors_m': cached_error.tolist(),
                        'actual_vs_reference_offset_four_corner_errors_m': relative_error.tolist(),
                        'actual_four_corners_utm_m': real.tolist()})
    return {'name': 'validation_source_metadata_and_all_four_corner_alignment',
            'reference_source': ref, 'strict_threshold_m': 1e-7,
            'audit_sha256': digest(ROOT/'local_sampling_completeness.json'), 'sources': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-val', action='store_true')
    parser.add_argument('--output', default=str(ROOT/'precision_agriculture/geo_management_independent_checks.json'))
    args = parser.parse_args()
    before = digest(gm.__file__)
    results = synthetic()
    results.append(validate_source_metadata())
    if args.real_val:
        results.append(real_validation())
    after = digest(gm.__file__)
    record = {"passed": True, "implementation_sha256_at_start": before,
              "implementation_sha256_at_end": after, "source_changed_during_check": before != after,
              "check_script_sha256": digest(__file__), "test_accessed": False,
              "model_predictions_accessed": False, "versions": {"numpy": np.__version__, "shapely": gm.shapely.__version__, "pyproj": gm.pyproj_version},
              "cases": results}
    Path(args.output).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({"passed": True, "cases": len(results), "output": args.output, "implementation_changed": before != after}, ensure_ascii=True), flush=True)
