"""Export validation-only probability rasters with source geometry and scalar checks.

CPU only. R is fixed to (1, 2, 4); no training, sealed test access or R=6 path.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import sys
import time
import traceback

# Explicitly hide CUDA before importing torch. The external Python environment is unchanged.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import rasterio
from affine import Affine
import torch
from torch.utils.data import DataLoader
import train_local as training

ROOT = Path(__file__).resolve().parent
FACTORS = (1, 2, 4)
SCALAR_TOLERANCE = 1e-6


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def put_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def affine_six(transform):
    return list(transform)[:6]


@torch.inference_mode()
def main(args):
    if not 1 <= args.threads <= 4:
        raise ValueError('CPU thread count must be between 1 and 4')
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.time()
    run = ROOT/'runs'/args.run_id
    config_path = run/'config.json'
    config, run_status = read(config_path), read(run/'status.json')
    if run_status['status'] != 'completed' or config['stage'] != 'development':
        raise ValueError('A completed development run is required')
    if config['variant'] not in ('hard_seg', 'soft_seg', 'consistent_seg'):
        raise ValueError('Only segmentation models have spatial probability maps')
    if config.get('max_train', 0) or config.get('max_val', 0):
        raise ValueError('Subset runs are not accepted')
    if (run/'USE_RESTRICTION.json').exists():
        raise ValueError('Restricted run cannot be exported')
    data = Path(config['data_dir'])
    if not data.is_absolute():
        data = ROOT/data
    data = data.resolve()
    if data != (ROOT/'local_data_v2').resolve():
        raise ValueError('This exporter is limited to local_data_v2')
    manifest = data/'val.json'
    manifest_hash = sha(manifest)
    model_source_hash = sha(training.__file__)
    export_source_hash = sha(__file__)
    assert model_source_hash == config['script_sha256'], 'Matching model/preprocess source required'
    assert manifest_hash == config['val_manifest_sha256']
    assert sha(data/'DATASET_LOCK.json') == config['dataset_lock_sha256']
    lock = read(data/'DATASET_LOCK.json')
    assert lock['manifest_sha256']['val.json'] == manifest_hash
    records = read(manifest)
    assert len(records) == config['n_val'] == 357
    assert len({r['id'] for r in records}) == len(records)
    assert all(r['split'] == 'val' for r in records)
    assert all(re.fullmatch(r'[A-Za-z0-9_-]+', r['id']) for r in records)
    checkpoint_path = run/'best.pt'
    checkpoint_hash = sha(checkpoint_path)

    candidates = [Path(args.scalar_dir).resolve()] if args.scalar_dir else sorted(run.glob('validation_export_*'), reverse=True)
    scalar_dir = None
    for candidate in candidates:
        if not (candidate/'summary.json').exists() or not (candidate/'predictions.csv').exists():
            continue
        summary = read(candidate/'summary.json')
        if (summary.get('checkpoint_sha256') == checkpoint_hash and
                summary.get('manifest_sha256') == manifest_hash and summary.get('device') == 'cpu'):
            scalar_dir = candidate
            scalar_summary = summary
            break
    if scalar_dir is None:
        raise ValueError('No matching existing CPU scalar validation export')
    assert not scalar_summary['test_accessed']
    with (scalar_dir/'predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        scalar_rows = list(csv.DictReader(f))
    scalar_lookup = {(r['id'], int(r['scale_factor'])):r for r in scalar_rows}
    expected_keys = {(r['id'], f) for r in records for f in FACTORS}
    assert set(scalar_lookup) == expected_keys and len(scalar_lookup) == len(scalar_rows)
    assert all(r['split'] == 'val' for r in scalar_rows)

    out = run/('validation_maps_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'maps').mkdir()
    (out/'source').mkdir()
    export_copy = out/'source'/f'{export_source_hash}__export_validation_maps.py'
    model_copy = out/'source'/f'{model_source_hash}__train_local.py'
    export_copy.write_bytes(Path(__file__).read_bytes())
    model_copy.write_bytes(Path(training.__file__).read_bytes())
    assert sha(export_copy) == export_source_hash and sha(model_copy) == model_source_hash
    status = {'status':'running', 'run_id':args.run_id, 'output_directory':str(out),
              'started_at':stamp(), 'pid':os.getpid(), 'device':'cpu', 'threads':args.threads,
              'total_windows':len(records), 'completed_windows':0, 'factors':list(FACTORS),
              'checkpoint_sha256':checkpoint_hash, 'manifest_sha256':manifest_hash,
              'export_source_sha256':export_source_hash, 'test_accessed':False, 'R6_accessed':False}
    put_json(out/'status.json', status)
    print(json.dumps(status, ensure_ascii=True), flush=True)

    try:
        dataset_summary_path = data/'manifest_summary.json'
        source_refs = {s['source_id']:s for s in read(dataset_summary_path)['sources']}
        sources = {}
        for source_id in sorted({r['source_id'] for r in records}):
            source_ref = source_refs[source_id]
            # Read TIFF metadata only. Native alpha/mask integration is a separate task.
            with rasterio.open(source_ref['tif']) as ds:
                sources[source_id] = {
                    'source_id':source_id, 'tif':source_ref['tif'], 'mask':source_ref['mask'],
                    'tif_sha256_from_dataset_summary':source_ref['tif_sha256'],
                    'mask_sha256_from_dataset_summary':source_ref['mask_sha256'],
                    'width':ds.width, 'height':ds.height, 'affine':affine_six(ds.transform),
                    'crs':str(ds.crs), 'crs_wkt':ds.crs.to_wkt(),
                    'color_interpretation':[x.name for x in ds.colorinterp],
                    'source_pixel_hashes_recomputed':False,
                }

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert checkpoint['config']['script_sha256'] == config['script_sha256']
        assert checkpoint['config']['val_manifest_sha256'] == manifest_hash
        assert checkpoint['epoch'] == run_status['best_epoch'] == scalar_summary['checkpoint_epoch']
        model = training.Model(regression=False, pretrained=False).cpu().eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        loader = DataLoader(training.Tiles(records), batch_size=args.batch_size, shuffle=False, num_workers=0)
        window_rows, checks = [], []
        cursor = 0
        for image, soft, valid, hard, target in loader:
            assert image.device.type == valid.device.type == 'cpu'
            batch_prob, batch_q = [], []
            for factor in FACTORS:
                # Identical CPU-float32 preprocessing and model path as the scalar exporter.
                probability = model(training.degrade(image, factor, valid), valid).float().sigmoid()
                batch_q.append(training.coverage(probability, valid).numpy())
                batch_prob.append(probability[:, 0].numpy())
            probabilities = np.stack(batch_prob, axis=1).astype(np.float32, copy=False)
            validity = valid[:, 0].numpy().astype(np.float32, copy=False)
            for j in range(image.shape[0]):
                rec = records[cursor+j]
                source = sources[rec['source_id']]
                x, y = int(rec['x']), int(rec['y'])
                source_affine = Affine(*source['affine'])
                native_affine = source_affine*Affine.translation(x, y)
                output_affine = native_affine*Affine.scale(2, 2)
                map_path = out/'maps'/f"{rec['id']}.npz"
                np.savez_compressed(map_path, p=probabilities[j], valid=validity[j])
                # Reopen the actual output, validating dtype, geometry and weighted coverage.
                with np.load(map_path, allow_pickle=False) as stored:
                    p, v = stored['p'], stored['valid']
                    assert p.shape == (3,256,256) and p.dtype == np.float32
                    assert v.shape == (256,256) and v.dtype == np.float32
                    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                    assert np.isfinite(v).all() and ((v>=0)&(v<=1)).all()
                    assert np.array_equal(v, validity[j])
                    saved_q = (p.astype(np.float64)*v[None]).sum(axis=(1,2))/v.astype(np.float64).sum()
                for fi, factor in enumerate(FACTORS):
                    reference = scalar_lookup[(rec['id'], factor)]
                    assert reference['source_id'] == rec['source_id']
                    assert int(reference['x']) == x and int(reference['y']) == y
                    assert abs(float(reference['cache_valid_area'])-float(v.sum())) < 1e-6
                    old_q = float(reference['coverage_pred'])
                    check = {'id':rec['id'], 'scale_factor':factor, 'scalar_coverage':old_q,
                             'map_coverage_cpu_torch_float32':float(batch_q[fi][j]),
                             'map_coverage_saved_float64_sum':float(saved_q[fi]),
                             'absolute_difference_torch':abs(float(batch_q[fi][j])-old_q),
                             'absolute_difference_saved':abs(float(saved_q[fi])-old_q)}
                    assert max(check['absolute_difference_torch'], check['absolute_difference_saved']) <= SCALAR_TOLERANCE, check
                    checks.append(check)
                window_rows.append({
                    'id':rec['id'], 'source_id':rec['source_id'], 'scene_group':rec['scene_group'], 'split':'val',
                    'file':str(map_path), 'relative_file':f"maps/{rec['id']}.npz", 'file_sha256':sha(map_path),
                    'cache':rec['cache'], 'cache_sha256':rec['cache_sha256'],
                    'native_window_xywh':[x,y,512,512],
                    'native_bounds_xyxy':[x,y,x+512,y+512],
                    'native_source_intersection_xyxy':[x,y,min(x+512,source['width']),min(y+512,source['height'])],
                    'global_x':rec['global_x'], 'global_y':rec['global_y'],
                    'native_window_affine':affine_six(native_affine), 'output_affine':affine_six(output_affine),
                    'source_affine':source['affine'], 'crs':source['crs'],
                    'probability_shape':[3,256,256], 'valid_shape':[256,256], 'factors':list(FACTORS),
                    'native_pixels_per_output_pixel':[2,2],
                    'valid_sum_cache_pixels':float(v.sum()), 'native_valid_pixel_equivalent':float(v.sum())*4,
                    'coverage_true_from_manifest':rec['coverage'],
                    'manifest_sha256':manifest_hash, 'checkpoint_sha256':checkpoint_hash,
                })
            cursor += image.shape[0]
            status.update(completed_windows=cursor, completed_factor_maps=cursor*3,
                          updated_at=stamp(), elapsed_seconds=round(time.time()-started,2))
            put_json(out/'status.json', status)
            if cursor % (args.batch_size*8) == 0 or cursor == len(records):
                print(json.dumps({'completed_windows':cursor,'total':len(records),
                                  'elapsed_seconds':round(time.time()-started,1)}, ensure_ascii=True), flush=True)
        assert cursor == len(records) and len(checks) == 1071
        assert sha(manifest) == manifest_hash and sha(checkpoint_path) == checkpoint_hash
        assert sha(training.__file__) == model_source_hash and sha(__file__) == export_source_hash
        metadata = {
            'schema':'local_validation_probability_maps_v1', 'run_id':args.run_id, 'created_at':stamp(),
            'device':'cpu', 'precision':'float32; autocast disabled', 'threads':args.threads, 'batch_size':args.batch_size,
            'checkpoint':str(checkpoint_path), 'checkpoint_sha256':checkpoint_hash, 'checkpoint_epoch':checkpoint['epoch'],
            'config':str(config_path), 'config_sha256':sha(config_path),
            'manifest':str(manifest), 'manifest_sha256':manifest_hash,
            'dataset_summary':str(dataset_summary_path), 'dataset_summary_sha256':sha(dataset_summary_path),
            'model_source_sha256':model_source_hash, 'export_source_sha256':export_source_hash,
            'source_copies':{'model':str(model_copy),'exporter':str(export_copy)},
            'scalar_reference_directory':str(scalar_dir),
            'scalar_csv_sha256':sha(scalar_dir/'predictions.csv'), 'scalar_summary_sha256':sha(scalar_dir/'summary.json'),
            'factors':list(FACTORS), 'npz_fields':{'p':'float32 [3,256,256], factors ordered 1,2,4', 'valid':'float32 [256,256]'},
            'pixel_support':'p[f,v,u] has support at native [x+2u,x+2u+2) x [y+2v,y+2v+2); affine maps pixel corners',
            'valid_semantics':'fraction of original alpha-valid native pixels in each 2x2 block, from sealed val cache',
            'invalid_probability_values':'retained; downstream integration must apply native validity or valid fractions',
            'ground_truth_policy':'No cached soft/hard masks are exported as new ground truth; exact area uses original alpha and PNG in downstream integration',
            'sources':sources, 'windows':window_rows,
            'python':sys.version, 'torch':torch.__version__, 'numpy':np.__version__, 'argv':sys.argv,
            'test_accessed':False, 'R6_accessed':False, 'training_run_invoked':False,
        }
        put_json(out/'metadata.json', metadata)
        with (out/'scalar_validation.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(checks[0]));writer.writeheader();writer.writerows(checks)
        verification = {'checked_pairs':len(checks), 'expected_pairs':1071, 'all_npz_reopened':len(window_rows),
                        'scalar_tolerance':SCALAR_TOLERANCE,
                        'max_absolute_difference_torch':max(c['absolute_difference_torch'] for c in checks),
                        'max_absolute_difference_saved':max(c['absolute_difference_saved'] for c in checks),
                        'all_checks_passed':True, 'test_accessed':False, 'R6_accessed':False}
        put_json(out/'verification.json', verification)
        status.update(status='completed', completed_at=stamp(), elapsed_seconds=time.time()-started,
                      metadata=str(out/'metadata.json'), metadata_sha256=sha(out/'metadata.json'),
                      verification=verification, source_copies=metadata['source_copies'])
        put_json(out/'status.json', status)
        print(json.dumps(status, ensure_ascii=True), flush=True)
    except BaseException as exc:
        status.update(status='failed', failed_at=stamp(), error=repr(exc), traceback=traceback.format_exc(),
                      elapsed_seconds=time.time()-started)
        put_json(out/'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--scalar-dir')
    main(parser.parse_args())
