"""CPU-only export of completed, full common_v2 development checkpoints.

Regression emits source-linked scalars only. Segmentation emits p/valid NPZ in
the existing local_validation_probability_maps_v1 interface. No training,
test/R6, new checkpoint selection, raw image pixels or geometry integration.
"""
from pathlib import Path
import argparse
import csv
import datetime
import json
import os
import re
import shutil
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import rasterio
from affine import Affine
import torch
from torch.utils.data import DataLoader
import train_common as training

ROOT = Path(__file__).resolve().parent
FACTORS = (1, 2, 4)
PLAN_SHA256 = 'faefb03b14e4c44360ded4d065ffbe873d4e8512139a68d56e15ee90a2185a95'
SOURCE_NAMES = ('train_common.py', 'models_common.py', 'conservative_resample.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return training.sha256_file(path)


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def write_json(path, value):
    training.write_json(path, value)


def same_number(a, b, tolerance=1e-12):
    if not np.isfinite([a, b]).all() or abs(float(a)-float(b)) > tolerance:
        raise ValueError(f'Numeric provenance mismatch: {a} vs {b}')


def validate_run(run_id):
    if not re.fullmatch(r'dev_common_v2_[A-Za-z0-9_]+', run_id):
        raise ValueError('Only full dev_common_v2 run IDs are supported')
    run = ROOT/'runs'/run_id
    config_path, state_path = run/'config.json', run/'status.json'
    config, state = read(config_path), read(state_path)
    if config.get('run_id') != run_id or config.get('stage') != 'development' or state.get('stage') != 'development' or state.get('status') != 'completed':
        raise ValueError('A completed development run is required')
    if any(config.get(k) for k in ('smoke', 'diagnostic', 'diagnostic_epochs', 'max_train', 'max_val', 'no_pretrained')):
        raise ValueError('Smoke, diagnostic, subset or unpretrained runs are rejected')
    if config.get('test_accessed') is not False or state.get('test_accessed') is not False:
        raise ValueError('Unexpected test-access provenance')
    if (run/'USE_RESTRICTION.json').exists() or (run/'failure.json').exists():
        raise ValueError('Restricted or failed run is rejected')
    if config.get('epochs') != 24 or state.get('executed_epochs') != 24 or config.get('n_train') != 985 or config.get('n_val') != 357:
        raise ValueError('Full24-epoch/full985-train/357-validation run required')
    if len(config.get('selected_train_ids', [])) != 985 or len(set(config['selected_train_ids'])) != 985:
        raise ValueError('Training identity list is incomplete')
    if config.get('validation_R') != list(FACTORS) or config.get('fixed_output_grid') != 256 or config.get('label_grid') != 256:
        raise ValueError('Unexpected validation R/output or supervision support')
    if config.get('decision_metrics_used_for_selection') is not False:
        raise ValueError('Decision metrics must not select the checkpoint')

    plan_path = ROOT/'common_queue_plan_v2.json'
    if sha(plan_path) != PLAN_SHA256:
        raise ValueError('Unexpected common_v2 plan version')
    plan = read(plan_path)
    entries = [r for r in plan['runs'] if r['run_id'] == run_id]
    if len(entries) != 1 or any(config.get(k) != v for k, v in entries[0].items() if k != 'index'):
        raise ValueError('Run configuration differs from the fixed full development plan')
    if config.get('variant') not in ('regression', 'hard_seg', 'soft_seg') or config.get('encoder') not in ('resnet18', 'convnextv2_tiny'):
        raise ValueError('Unknown task or encoder')
    for name in SOURCE_NAMES:
        expected = config['source_sha256'][name]
        if expected != plan['source_sha256'][name] or sha(ROOT/name) != expected or sha(run/'source'/name) != expected:
            raise ValueError('Matching current and archived model/preprocessing source required: '+name)

    data = ROOT/'local_data_v2'
    manifest, lock_path = data/'val.json', data/'DATASET_LOCK.json'
    lock = read(lock_path)
    if sha(manifest) != config['manifest_sha256']['val.json'] or sha(lock_path) != config['manifest_sha256']['DATASET_LOCK.json']:
        raise ValueError('Validation manifest/lock differs from training')
    if lock['manifest_sha256']['val.json'] != sha(manifest) or lock['manifest_sha256']['train.json'] != config['manifest_sha256']['train.json']:
        raise ValueError('Manifest lock differs from recorded development sources')
    if sha(run/'source'/'val.json') != sha(manifest):
        raise ValueError('Archived validation manifest mismatch')
    records = read(manifest)  # No train or sealed-test manifest is opened.
    ids = [r['id'] for r in records]
    if len(records) != 357 or len(set(ids)) != 357 or ids != config['selected_val_ids']:
        raise ValueError('The complete ordered357 validation population is required')
    if sum(r['coverage'] > 0 for r in records) != 27:
        raise ValueError('Unexpected validation stratification')
    for rec in records:
        if rec.get('split') != 'val' or not re.fullmatch(r'[A-Za-z0-9_-]+', rec['id']):
            raise ValueError('Invalid validation record')
        if not Path(rec['cache']).resolve().is_relative_to((data/'cache').resolve()) or not rec.get('cache_sha256'):
            raise ValueError('Validation cache path/hash missing')
    if len({(r['source_id'], r['x'], r['y']) for r in records}) != 357:
        raise ValueError('Repeated validation source location')

    history = [json.loads(line) for line in (run/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if [m['epoch'] for m in history] != list(range(1, 25)):
        raise ValueError('Incomplete or repeated epoch history')
    for metric in history:
        if set(metric['scales']) != {'1', '2', '4'}:
            raise ValueError('Unexpected epoch evaluation scales')
        for result in metric['scales'].values():
            if result['n'] != 357 or result['n_positive'] != 27:
                raise ValueError('Partial epoch evaluation')
            same_number(result['balanced_mae'], (result['mae_zero']+result['mae_positive'])/2)
        same_number(metric['selection_balanced_mae'], sum(m['balanced_mae'] for m in metric['scales'].values())/3)
    selected = min(history, key=lambda m: m['selection_balanced_mae'])  # Python preserves earliest tie.
    best = read(run/'best_validation.json')
    if best != selected or selected['epoch'] != state['best_epoch']:
        raise ValueError('Original strict-MAE selected epoch does not match saved best/status')
    same_number(state['best_validation_balanced_mae'], selected['selection_balanced_mae'])
    checkpoint_path = run/'best.pt'
    checkpoint_hash_before_load = sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if sha(checkpoint_path) != checkpoint_hash_before_load:
        raise ValueError('Checkpoint changed while loading')
    if checkpoint['config'] != config or checkpoint['validation'] != selected or checkpoint['epoch'] != selected['epoch']:
        raise ValueError('Checkpoint config/validation/epoch mismatch')
    model = training.make_model(config['encoder'], config['variant'], pretrained=False).cpu().eval()
    alignment = model.load_state_dict(checkpoint['model'], strict=True)
    if alignment.missing_keys or alignment.unexpected_keys:
        raise ValueError('Strict checkpoint alignment failed')
    watched = [config_path, state_path, run/'metrics.jsonl', run/'best_validation.json', checkpoint_path,
               manifest, lock_path, plan_path] + [ROOT/name for name in SOURCE_NAMES]
    hashes = {str(path): sha(path) for path in watched}
    return run, config, state, records, selected, checkpoint_path, model, hashes


def source_metadata(records):
    summary_path = ROOT/'local_data_v2'/'manifest_summary.json'
    summary = read(summary_path)
    refs = {r['source_id']: r for r in summary['sources']}
    sources = {}
    for source_id in sorted({r['source_id'] for r in records}):
        reference = refs[source_id]
        with rasterio.open(reference['tif']) as dataset:
            if dataset.crs is None:
                raise ValueError('Source CRS missing')
            sources[source_id] = {'source_id':source_id,'tif':reference['tif'],'mask':reference['mask'],
                'tif_sha256_from_dataset_summary':reference['tif_sha256'],
                'mask_sha256_from_dataset_summary':reference['mask_sha256'],
                'width':dataset.width,'height':dataset.height,'affine':list(dataset.transform)[:6],
                'crs':str(dataset.crs),'crs_wkt':dataset.crs.to_wkt(),
                'color_interpretation':[item.name for item in dataset.colorinterp],
                'source_pixel_hashes_recomputed':False,'raw_source_pixels_read':False}
    return sources, summary_path


def window_metadata(rec, source, validity, manifest_hash, checkpoint_hash):
    x, y = int(rec['x']), int(rec['y'])
    if x < 0 or y < 0 or x >= source['width'] or y >= source['height']:
        raise ValueError('Native window has no source footprint')
    native_affine = Affine(*source['affine'])*Affine.translation(x, y)
    return {'id':rec['id'],'source_id':rec['source_id'],'scene_group':rec['scene_group'],'split':'val',
        'cache':rec['cache'],'cache_sha256':rec['cache_sha256'],
        'native_window_xywh':[x,y,512,512],'native_bounds_xyxy':[x,y,x+512,y+512],
        'native_source_intersection_xyxy':[x,y,min(x+512,source['width']),min(y+512,source['height'])],
        'global_x':rec['global_x'],'global_y':rec['global_y'],
        'native_window_affine':list(native_affine)[:6],'output_affine':list(native_affine*Affine.scale(2,2))[:6],
        'source_affine':source['affine'],'crs':source['crs'],'factors':list(FACTORS),
        'native_pixels_per_output_pixel':[2,2],'valid_shape':[256,256],
        'valid_sum_cache_pixels':float(validity.sum()),'native_valid_pixel_equivalent':float(validity.sum())*4,
        'coverage_true_from_manifest':rec['coverage'],'manifest_sha256':manifest_hash,'checkpoint_sha256':checkpoint_hash}


@torch.inference_mode()
def main(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    start = time.time()
    run, config, run_state, records, original, checkpoint_path, model, input_hashes = validate_run(args.run_id)
    regression = config['variant'] == 'regression'
    manifest = ROOT/'local_data_v2'/'val.json'
    manifest_hash, checkpoint_hash = sha(manifest), sha(checkpoint_path)
    export_hash = sha(__file__)
    out = run/('common_validation_export_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'source').mkdir()
    if not regression:
        (out/'maps').mkdir()
    state = {'status':'running','run_id':args.run_id,'output_directory':str(out),'started_at':stamp(),
             'device':'cpu','precision':'float32; autocast disabled','threads':2,'pid':os.getpid(),
             'total_windows':357,'completed_windows':0,'factors':list(FACTORS),
             'test_accessed':False,'R6_accessed':False,'checkpoint_sha256':checkpoint_hash,'manifest_sha256':manifest_hash}
    write_json(out/'status.json', state)
    print(json.dumps(state), flush=True)
    try:
        copies = {}
        for name in (*SOURCE_NAMES, Path(__file__).name):
            source = ROOT/name
            # Plain source names avoid legacy Windows MAX_PATH; metadata binds hashes.
            destination = out/'source'/name
            shutil.copy2(source, destination)
            if sha(source) != sha(destination):
                raise ValueError('Source copy mismatch')
            copies[name] = str(destination)
        for source in (run/'config.json', run/'best_validation.json', manifest, ROOT/'local_data_v2'/'DATASET_LOCK.json', ROOT/'common_queue_plan_v2.json'):
            shutil.copy2(source, out/'source'/source.name)
        sources, source_summary = source_metadata(records)
        input_hashes[str(source_summary)] = sha(source_summary)
        shutil.copy2(source_summary, out/'source'/source_summary.name)
        loader = DataLoader(training.DevelopmentTiles(records, augment=False), batch_size=config['batch_size'], shuffle=False, num_workers=0)
        rows, windows, checks = [], [], []
        cursor = 0
        for image, soft, valid, truth in loader:
            probabilities, estimates = [], []
            for factor in FACTORS:
                logits = model(training.degrade_rgb(image, factor, valid), valid)
                training.require_finite_forward(logits, model)
                p = logits.float().sigmoid()
                q = p if regression else training.coverage(p, valid)
                estimates.append(q.numpy())
                if not regression:
                    probabilities.append(p[:, 0].numpy())
            combined = np.stack(probabilities, axis=1) if not regression else None
            for offset in range(image.shape[0]):
                rec = records[cursor+offset]
                validity = valid[offset, 0].numpy().astype(np.float32, copy=False)
                window = window_metadata(rec, sources[rec['source_id']], validity, manifest_hash, checkpoint_hash)
                saved_q = None
                if not regression:
                    path = out/'maps'/(rec['id']+'.npz')
                    np.savez_compressed(path, p=combined[offset].astype(np.float32), valid=validity)
                    with np.load(path, allow_pickle=False) as stored:
                        saved_p, saved_v = stored['p'], stored['valid']
                        if set(stored.files) != {'p','valid'} or saved_p.shape != (3,256,256) or saved_p.dtype != np.float32 or saved_v.dtype != np.float32 or not np.array_equal(saved_v, validity):
                            raise ValueError('Stored probability-map interface mismatch')
                        if not np.isfinite(saved_p).all() or np.any((saved_p < 0) | (saved_p > 1)):
                            raise ValueError('Invalid stored probabilities')
                        saved_q = (saved_p.astype(np.float64)*saved_v[None]).sum((1,2))/saved_v.astype(np.float64).sum()
                    window.update(file=str(path),relative_file='maps/'+path.name,file_sha256=sha(path),probability_shape=[3,256,256])
                else:
                    window.update(spatial_prediction_available=False, prediction_support='whole native window valid domain only')
                for factor_index, factor in enumerate(FACTORS):
                    predicted = float(estimates[factor_index][offset])
                    target = float(truth[offset])
                    row = {'id':rec['id'],'source_id':rec['source_id'],'scene_group':rec['scene_group'],'split':'val',
                           'scale_factor':factor,'effective_input_pixels':256//factor,'coverage_true':target,
                           'coverage_true_manifest':rec['coverage'],'coverage_pred':predicted,
                           'absolute_error':abs(predicted-target),'signed_error':predicted-target,
                           'cache_valid_area':float(validity.sum()),'native_valid_pixel_equivalent':float(validity.sum())*4,
                           'x':rec['x'],'y':rec['y'],'global_x':rec['global_x'],'global_y':rec['global_y'],
                           'checkpoint_epoch':original['epoch'],'checkpoint_sha256':checkpoint_hash,
                           'manifest_sha256':manifest_hash,'cache_sha256':rec['cache_sha256']}
                    if not regression:
                        binary = torch.from_numpy(combined[offset, factor_index] > .5).float()[None]
                        row.update(tp_area=float((binary*soft[offset]*valid[offset]).sum()),
                                   fp_area=float((binary*(1-soft[offset])*valid[offset]).sum()),
                                   fn_area=float(((1-binary)*soft[offset]*valid[offset]).sum()))
                        difference = abs(float(saved_q[factor_index])-predicted)
                        if difference > 1e-6:
                            raise ValueError('Saved probability-map coverage differs from source-linked scalar')
                        checks.append({'id':rec['id'],'scale_factor':factor,'scalar_coverage':predicted,
                                       'map_coverage_saved_float64_sum':float(saved_q[factor_index]),'absolute_difference_saved':difference})
                    rows.append(row)
                windows.append(window)
            cursor += image.shape[0]
            state.update(completed_windows=cursor,completed_factor_predictions=cursor*3,updated_at=stamp(),elapsed_seconds=time.time()-start)
            write_json(out/'status.json', state)
            if cursor % 64 == 0 or cursor == 357:
                print(json.dumps({'windows':cursor,'elapsed_seconds':time.time()-start}), flush=True)
        expected = {(r['id'],factor) for r in records for factor in FACTORS}
        if cursor != 357 or len(rows) != 1071 or {(r['id'],r['scale_factor']) for r in rows} != expected:
            raise ValueError('Incomplete prediction population')
        with (out/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        write_json(out/'predictions.json', rows)
        with (out/'predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
            reopened = list(csv.DictReader(stream))
        if len(reopened) != 1071:
            raise ValueError('Saved scalar CSV incomplete')
        summaries, differences = {}, {}
        for factor in FACTORS:
            selected_rows = [r for r in reopened if int(r['scale_factor']) == factor]
            values = training.summarize([float(r['coverage_pred']) for r in selected_rows], [float(r['coverage_true']) for r in selected_rows])
            if not regression:
                tp, fp, fn = [sum(float(r[name]) for r in selected_rows) for name in ('tp_area','fp_area','fn_area')]
                values.update(target_iou=tp/max(tp+fp+fn,1),dice=2*tp/max(2*tp+fp+fn,1))
            summaries[str(factor)] = values
            differences[str(factor)] = {key:float(values[key])-float(original['scales'][str(factor)][key])
                for key in ('mae','bias','mae_zero','mae_positive','balanced_mae')}
        cpu_score = sum(r['balanced_mae'] for r in summaries.values())/3
        summary = {'run_id':args.run_id,'checkpoint_epoch':original['epoch'],'checkpoint_sha256':checkpoint_hash,
            'manifest_sha256':manifest_hash,'device':'cpu','precision':'float32; autocast disabled','threads':2,
            'created_at':stamp(),'scales':summaries,'selection_balanced_mae':cpu_score,
            'original_amp_validation':original,'cpu_minus_original_amp_by_R':differences,
            'cpu_minus_original_amp_selection_balanced_mae':cpu_score-original['selection_balanced_mae'],
            'original_selected_epoch_preserved':True,'checkpoint_reselected':False,'test_accessed':False,'R6_accessed':False,
            'area_units':'cache valid/TP/FP/FN are256-grid effective pixels, not m2; native equivalent is x4',
            'spatial_prediction_available':not regression,'export_script_sha256':export_hash,
            'training_configuration':config,'elapsed_seconds':time.time()-start}
        write_json(out/'summary.json', summary)
        metadata = {'schema':'local_validation_scalar_predictions_v1' if regression else 'local_validation_probability_maps_v1',
            'run_id':args.run_id,'model_family':'common_v2','encoder':config['encoder'],'variant':config['variant'],
            'created_at':stamp(),'device':'cpu','precision':'float32; autocast disabled','threads':2,'batch_size':config['batch_size'],
            'checkpoint':str(checkpoint_path),'checkpoint_sha256':checkpoint_hash,'checkpoint_epoch':original['epoch'],
            'config':str(run/'config.json'),'config_sha256':sha(run/'config.json'),
            'manifest':str(manifest),'manifest_sha256':manifest_hash,'dataset_summary':str(source_summary),'dataset_summary_sha256':sha(source_summary),
            'model_source_sha256':config['source_sha256']['train_common.py'],'model_dependency_sha256':config['source_sha256'],
            'export_source_sha256':export_hash,'source_copies':copies,'input_hashes':input_hashes,
            'scalar_reference_directory':str(out),'scalar_csv_sha256':sha(out/'predictions.csv'),'scalar_summary_sha256':sha(out/'summary.json'),
            'factors':list(FACTORS),'npz_fields':None if regression else {'p':'float32[3,256,256],R1/2/4','valid':'float32[256,256]'},
            'pixel_support':None if regression else 'p[f,v,u] supports native[x+2u,x+2u+2) by[y+2v,y+2v+2); affine maps corners',
            'valid_semantics':'fraction of alpha-valid native pixels in each2x2 block from sealed val cache',
            'invalid_probability_values':'retained; downstream must integrate native validity' if not regression else None,
            'ground_truth_policy':'No new ground truth export; exact area remains original alpha/PNG, not cache hard labels',
            'spatial_prediction_available':not regression,'regression_spatial_policy':'scalar-only; never uniform p raster' if regression else None,
            'sources':sources,'windows':windows,'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,
            'argv':sys.argv,'test_accessed':False,'R6_accessed':False,'training_run_invoked':False,'raw_geometry_recomputed':False}
        if not regression:
            with (out/'scalar_validation.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(checks[0]));writer.writeheader();writer.writerows(checks)
        verification = {'all_checks_passed':True,'expected_windows':357,'checked_pairs':len(rows),'expected_pairs':1071,
            'all_npz_reopened':0 if regression else len(windows),'scalar_tolerance':None if regression else 1e-6,
            'max_absolute_difference_saved':None if regression else max(r['absolute_difference_saved'] for r in checks),
            'strict_weights_only_checkpoint_loaded':True,'original_selection_audited_24_epochs':True,
            'original_epoch_preserved':True,'manifest_order_preserved':True,'test_accessed':False,'R6_accessed':False}
        for path, digest in input_hashes.items():
            if sha(path) != digest:
                raise ValueError('Input changed during export: '+path)
        if sha(__file__) != export_hash:
            raise ValueError('Exporter changed during execution')
        write_json(out/'metadata.json', metadata)
        write_json(out/'verification.json', verification)
        state.update(status='completed',completed_at=stamp(),elapsed_seconds=time.time()-start,
                     metadata=str(out/'metadata.json'),metadata_sha256=sha(out/'metadata.json'),verification=verification)
        write_json(out/'status.json',state)
        print(json.dumps({'output':str(out),'status':'completed','cpu_score':cpu_score,
                          'original_amp_score':original['selection_balanced_mae'],
                          'difference':summary['cpu_minus_original_amp_selection_balanced_mae'],'elapsed_seconds':state['elapsed_seconds']}),flush=True)
    except BaseException as error:
        state.update(status='failed',error=repr(error),traceback=traceback.format_exc(),failed_at=stamp())
        write_json(out/'status.json',state)
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    main(parser.parse_args())
