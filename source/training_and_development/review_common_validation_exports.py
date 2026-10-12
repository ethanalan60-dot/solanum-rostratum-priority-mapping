"""Independent CPU audit of fixed common-v2 validation exports, without export.

Reads only val caches, completed checkpoints and supplied derived artifacts.
Verifies all saved rasters/scalars and a few fixed CPU forward batches.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from affine import Affine
from torch.utils.data._utils.collate import default_collate
import train_common as tc

ROOT=Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return tc.sha256_file(path)


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


@torch.inference_mode()
def main(exports,output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    manifest_path=ROOT/'local_data_v2'/'val.json'
    manifest=read(manifest_path)
    manifest_hash=sha(manifest_path)
    records={r['id']:r for r in manifest}
    assert len(manifest)==len(records)==357 and all(r['split']=='val' for r in manifest)
    geometry=read(ROOT/'precision_agriculture'/'geometry_20260916_v1'/'metadata.json')
    assert geometry['state']=='completed' and geometry['manifest_sha256']==manifest_hash
    first_positive=next(i for i,r in enumerate(manifest) if r['coverage']>0)
    # Exact exporter batch boundaries; include a populated unit and one-sample tail.
    batch_starts=sorted({0,first_positive//4*4,356})
    report={'reviewed_utc':tc.utc_now(),'gpu_used':False,'threads':2,'test_accessed':False,
            'R6_accessed':False,'reexport_invoked':False,'model_reselected':False,
            'manifest_sha256':manifest_hash,'fixed_cpu_batch_starts':batch_starts,
            'audit_source_sha256':sha(__file__),'runs':[]}
    entry_checks=read(ROOT/'provenance'/'common_export_entry_checks.json')
    assert entry_checks['exporter_sha256']==sha(ROOT/'export_common_validation.py')
    assert all(e['rejected'] and e['no_output_created'] for e in entry_checks['checks'])
    assert not entry_checks['test_manifest_opened'] and not entry_checks['inference_run']
    report['entry_guard_checks_sha256']=sha(ROOT/'provenance'/'common_export_entry_checks.json')
    expected_variants={'regression','hard_seg','soft_seg'}
    variants=set()
    for directory in exports:
        directory=directory.resolve()
        metadata=read(directory/'metadata.json')
        status=read(directory/'status.json')
        summary=read(directory/'summary.json')
        verification=read(directory/'verification.json')
        rid=metadata['run_id'];variant=metadata['variant'];variants.add(variant)
        assert rid=='dev_common_v2_lr1e4_resnet18_'+variant+'_L256_s17'
        run=ROOT/'runs'/rid
        config=read(run/'config.json');state=read(run/'status.json')
        assert status['status']=='completed' and verification['all_checks_passed']
        assert status['verification']==verification
        assert status['completed_windows']==357 and status['completed_factor_predictions']==1071
        assert metadata['factors']==[1,2,4] and metadata['device']=='cpu' and metadata['threads']==2
        assert metadata['test_accessed'] is False and metadata['R6_accessed'] is False
        assert sha(directory/'metadata.json')==status['metadata_sha256']
        assert sha(directory/'predictions.csv')==metadata['scalar_csv_sha256']
        assert sha(directory/'summary.json')==metadata['scalar_summary_sha256']
        assert Path(metadata['config']).resolve()==(run/'config.json').resolve()
        assert sha(metadata['config'])==metadata['config_sha256']
        assert Path(metadata['checkpoint']).resolve()==(run/'best.pt').resolve()
        assert sha(run/'best.pt')==metadata['checkpoint_sha256']==summary['checkpoint_sha256']
        assert metadata['manifest_sha256']==summary['manifest_sha256']==manifest_hash
        assert [r['id'] for r in metadata['windows']]==[r['id'] for r in manifest]==config['selected_val_ids']
        assert config['n_val']==357 and state['executed_epochs']==24 and state['status']=='completed'
        assert config['stage']=='development' and not config['smoke'] and not config['diagnostic']
        for name,digest in config['source_sha256'].items():
            assert sha(ROOT/name)==digest==sha(run/'source'/name)
            assert sha(metadata['source_copies'][name])==digest
        assert metadata['model_dependency_sha256']==config['source_sha256']
        assert metadata['model_source_sha256']==config['source_sha256']['train_common.py']
        assert sha(metadata['source_copies']['export_common_validation.py'])==metadata['export_source_sha256']
        assert metadata['export_source_sha256']==entry_checks['exporter_sha256']
        for path,digest in metadata['input_hashes'].items():
            assert sha(path)==digest
        history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        chosen=min(history,key=lambda m:m['selection_balanced_mae'])
        checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
        assert checkpoint['epoch']==chosen['epoch']==state['best_epoch']==metadata['checkpoint_epoch']
        assert checkpoint['validation']==chosen==read(run/'best_validation.json')
        assert checkpoint['config']==config
        assert summary['original_amp_validation']==chosen and summary['original_selected_epoch_preserved']
        assert summary['checkpoint_reselected'] is False
        scalar=rows(directory/'predictions.csv')
        lookup={(r['id'],int(r['scale_factor'])):r for r in scalar}
        assert len(scalar)==len(lookup)==1071 and set(lookup)=={(r['id'],R) for r in manifest for R in (1,2,4)}
        max_map_difference=0.;max_truth_difference=0.
        native_pixels=0
        for window in metadata['windows']:
            rec=records[window['id']]
            assert window['source_id']==rec['source_id'] and window['split']=='val'
            assert window['factors']==[1,2,4] and window['native_pixels_per_output_pixel']==[2,2]
            assert window['native_window_xywh']==[rec['x'],rec['y'],512,512]
            assert (window['global_x'],window['global_y'])==(rec['global_x'],rec['global_y'])
            source=metadata['sources'][rec['source_id']]
            native_source=geometry['sources'][rec['source_id']]
            assert source['tif_sha256_from_dataset_summary']==native_source['tif_sha256']
            assert source['mask_sha256_from_dataset_summary']==native_source['mask_sha256']
            assert np.allclose(source['affine'],native_source['transform'][:6],rtol=0,atol=1e-15)
            expected_affine=Affine(*source['affine'])*Affine.translation(rec['x'],rec['y'])*Affine.scale(2,2)
            assert np.allclose(window['output_affine'],list(expected_affine)[:6],rtol=0,atol=1e-15)
            assert window['crs']==source['crs']
            assert sha(rec['cache'])==rec['cache_sha256']==window['cache_sha256']
            with np.load(rec['cache'],allow_pickle=False) as cache:
                valid=cache['valid'].astype(np.float32)
            native_pixels+=int(valid.sum()*4)
            if variant!='regression':
                assert metadata['schema']=='local_validation_probability_maps_v1' and metadata['spatial_prediction_available']
                assert Path(window['file']).resolve()==directory/'maps'/(rec['id']+'.npz')
                assert sha(window['file'])==window['file_sha256']
                with np.load(window['file'],allow_pickle=False) as stored:
                    assert set(stored.files)=={'p','valid'}
                    p=stored['p'];v=stored['valid']
                    assert p.shape==(3,256,256) and p.dtype==np.float32 and v.dtype==np.float32
                    assert np.array_equal(v,valid) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                    recomputed=(p.astype(np.float64)*v[None]).sum((1,2))/v.astype(np.float64).sum()
            else:
                assert metadata['schema']=='local_validation_scalar_predictions_v1' and not metadata['spatial_prediction_available']
                assert 'file' not in window and not (directory/'maps').exists()
            for index,R in enumerate((1,2,4)):
                row=lookup[(rec['id'],R)]
                assert row['split']=='val' and row['source_id']==rec['source_id']
                assert int(row['x'])==rec['x'] and int(row['y'])==rec['y']
                assert row['manifest_sha256']==manifest_hash and row['checkpoint_sha256']==metadata['checkpoint_sha256']
                assert float(row['cache_valid_area'])==float(valid.sum())
                assert float(row['coverage_true_manifest'])==rec['coverage']
                assert float(row['coverage_true'])==float(np.float32(rec['coverage']))
                assert math.isfinite(float(row['coverage_pred'])) and 0<=float(row['coverage_pred'])<=1
                max_truth_difference=max(max_truth_difference,abs(float(row['coverage_true'])-rec['coverage']))
                if variant!='regression':
                    max_map_difference=max(max_map_difference,abs(float(recomputed[index])-float(row['coverage_pred'])))
        assert native_pixels==91061049
        assert max_map_difference<=1e-6
        # Recompute all scalar summaries using independent NumPy operations.
        score=0.;max_summary_difference=0.
        for R in (1,2,4):
            subset=[lookup[(r['id'],R)] for r in manifest]
            q=np.array([float(r['coverage_pred']) for r in subset]);truth=np.array([float(r['coverage_true']) for r in subset])
            error=q-truth;positive=truth>0
            derived={'mae':float(np.abs(error).mean()),'bias':float(error.mean()),
                     'mae_zero':float(np.abs(error[~positive]).mean()),'mae_positive':float(np.abs(error[positive]).mean())}
            derived['balanced_mae']=(derived['mae_zero']+derived['mae_positive'])/2
            score+=derived['balanced_mae']/3
            for key,value in derived.items():
                max_summary_difference=max(max_summary_difference,abs(value-summary['scales'][str(R)][key]))
                delta=value-chosen['scales'][str(R)][key]
                assert abs(delta-summary['cpu_minus_original_amp_by_R'][str(R)][key])<=1e-12
        assert max_summary_difference<=1e-12 and abs(score-summary['selection_balanced_mae'])<=1e-12
        assert abs(score-chosen['selection_balanced_mae']-summary['cpu_minus_original_amp_selection_balanced_mae'])<=1e-12
        model=tc.make_model('resnet18',variant,pretrained=False).cpu().eval()
        model.load_state_dict(checkpoint['model'],strict=True)
        dataset=tc.DevelopmentTiles(manifest,augment=False)
        max_forward_scalar=max_forward_map=0.;forward_pairs=0
        for start in batch_starts:
            indices=list(range(start,min(start+4,len(manifest))))
            image,soft,valid,truth=default_collate([dataset[i] for i in indices])
            for R in (1,2,4):
                logits=model(tc.degrade_rgb(image,R,valid),valid)
                tc.require_finite_forward(logits,model)
                p=logits.float().sigmoid()
                q=p if variant=='regression' else tc.coverage(p,valid)
                for j,i in enumerate(indices):
                    key=manifest[i]['id'];forward_pairs+=1
                    max_forward_scalar=max(max_forward_scalar,abs(float(q[j])-float(lookup[(key,R)]['coverage_pred'])))
                    if variant!='regression':
                        with np.load(directory/'maps'/(key+'.npz'),allow_pickle=False) as stored:
                            max_forward_map=max(max_forward_map,float(np.max(np.abs(stored['p'][(1,2,4).index(R)]-p[j,0].numpy()))))
        assert max_forward_scalar<=1e-6 and max_forward_map<=1e-6
        report['runs'].append({'run_id':rid,'directory':str(directory),'checkpoint_epoch':checkpoint['epoch'],
            'checkpoint_sha256':metadata['checkpoint_sha256'],'metadata_sha256':sha(directory/'metadata.json'),
            'export_source_sha256':metadata['export_source_sha256'],'full_val_pairs':len(scalar),'npz_files':0 if variant=='regression' else 357,
            'native_valid_pixels':native_pixels,'independent_map_scalar_max_difference':max_map_difference,
            'independent_scalar_summary_max_difference':max_summary_difference,'truth_float32_max_difference':max_truth_difference,
            'fixed_cpu_forward_pairs':forward_pairs,'fixed_cpu_forward_scalar_max_difference':max_forward_scalar,
            'fixed_cpu_forward_map_max_difference':max_forward_map,
            'cpu_minus_original_amp_selection_balanced_mae':summary['cpu_minus_original_amp_selection_balanced_mae'],
            'checkpoint_optimizer_counts':checkpoint['optimizer_step_counts'],
            'final_optimizer_counts':{k:state[k] for k in ['optimizer_attempts_total','optimizer_updates_total','optimizer_skips_total']},
            'all_checks_passed':True})
        del model,checkpoint
    assert variants==expected_variants and len(report['runs'])==3
    report.update(all_checks_passed=True,completed_utc=tc.utc_now())
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'all_checks_passed':True,'runs':3,'gpu_used':False,'output':str(output)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',type=Path,action='append',required=True,dest='exports')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.exports,args.output)
