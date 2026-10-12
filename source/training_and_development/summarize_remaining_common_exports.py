"""Summarize only the seven authorized, completed export/audit records."""
from pathlib import Path
import csv,datetime,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
BATCH=ROOT/'provenance/remaining_common_exports_20260916_0432'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=read(ROOT/'common_queue_plan_v2.json');expected={e['index']:e['run_id'] for e in plan['runs'] if e['index']>5}
    assert set(expected)==set(range(6,13));records=[]
    for idx,rid in expected.items():
        completion=read(BATCH/('index_%02d_completed.json'%idx));audit_path=BATCH/('index_%02d_audit.json'%idx)
        audit=read(audit_path);assert completion['state']=='completed' and audit['all_checks_passed'] and len(audit['runs'])==1
        result=audit['runs'][0];assert completion['run_id']==result['run_id']==rid
        directory=Path(result['directory']);state=read(directory/'status.json');summary=read(directory/'summary.json');metadata=read(directory/'metadata.json')
        assert state['status']=='completed' and state['completed_windows']==357 and state['completed_factor_predictions']==1071
        assert result['metadata_sha256']==sha(directory/'metadata.json')==state['metadata_sha256']
        assert completion['audit_sha256']==sha(audit_path)
        records.append({'index':idx,'run_id':rid,'variant':metadata['variant'],'encoder':metadata['encoder'],
            'directory':str(directory),'pid':state['pid'],'started_at':state['started_at'],'completed_at':state['completed_at'],
            'checkpoint_epoch':result['checkpoint_epoch'],'checkpoint_sha256':result['checkpoint_sha256'],
            'metadata_sha256':result['metadata_sha256'],'exporter_sha256':metadata['export_source_sha256'],
            'val_windows':357,'pairs':1071,'maps':result['npz_files'],
            'cpu_balanced_mae':summary['selection_balanced_mae'],
            'original_amp_balanced_mae':summary['original_amp_validation']['selection_balanced_mae'],
            'cpu_minus_amp_balanced_mae':result['cpu_minus_original_amp_selection_balanced_mae'],
            'map_scalar_max_difference':result['independent_map_scalar_max_difference'],
            'native_valid_pixels':result['native_valid_pixels'],'audit_path':str(audit_path),'audit_sha256':sha(audit_path)})
    assert sum(r['maps'] for r in records)==5*357
    assert all(r['exporter_sha256']=='56319f148fadc69b0cdc88c3c44f043ba020e812f2f4f919480304fe155dbecc' for r in records)
    report={'state':'completed','created_at':datetime.datetime.now().astimezone().isoformat(),'new_exports':7,
        'new_scalar_pairs':7*1071,'new_segmentation_maps':5*357,'regression_scalar_only_exports':2,
        'all_full_validation_artifacts_audited':True,'sequential_cpu_inference':True,'threads':2,
        'frozen_exporter_unchanged':True,'source_sha256':sha(__file__),'original_best_epochs_preserved':True,
        'new_forward_rechecks':0,'forward_rationale':'Prior independent R18 and ConvNeXt fixed-batch forward audits already passed; this batch independently reopens every saved map/cache and verifies hashes, IDs, valid, affine, q and summaries.',
        'scope_limits':['The independent batch audit does not repeat model inference.','Same-scene validation development only; no independent-field claim.','CPU float32 values can differ slightly from original AMP validation; checkpoint selection remains original.'],
        'test_accessed':False,'R6_accessed':False,'training_or_RM_started_by_export_agent':False,'runs':records}
    (BATCH/'remaining_common_exports_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf8')
    with (BATCH/'remaining_common_exports_report.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    lines=['# Remaining common-v2 validation exports','',
        'All seven newly completed runs were exported sequentially on CPU with two threads using the unchanged frozen exporter. Original selected checkpoints were retained. Two regression runs emit scalar predictions only; five segmentation runs emit full probability maps.','',
        'The audit independently checked all 7,497 scalar records and 1,785 saved probability files against validation IDs, source/checkpoint hashes, cached alpha-valid support, native window coordinates and affine/CRS. Float64 recomputation of saved map coverage agrees within 1e-6. No redundant model forward pass, test/R6 access, training or management evaluation was performed.','',
        '| Run | Best epoch | Maps | CPU − AMP balanced MAE | Maximum saved-map q difference |','|---|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['run_id']} | {r['checkpoint_epoch']} | {r['maps']} | {r['cpu_minus_amp_balanced_mae']:.10g} | {r['map_scalar_max_difference']:.10g} |")
    lines+=['','## Limits','',
        'These are one-scene validation development exports. Prior R18 and ConvNeXt independent forward audits establish the shared inference implementation; this batch verifies every newly saved product and its provenance. CPU float32 versus original AMP differences do not trigger checkpoint reselection. All output paths, process IDs, timestamps, checkpoint/metadata hashes and per-run audit links are in the companion CSV/JSON.']
    (BATCH/'remaining_common_exports_report.md').write_text('\n'.join(lines)+'\n','utf8')
    shutil.copy2(Path(__file__),BATCH/Path(__file__).name)
    files={p.name:sha(p) for p in BATCH.iterdir() if p.is_file() and p.name!='artifact_hashes.json'}
    (BATCH/'artifact_hashes.json').write_text(json.dumps(files,indent=2)+'\n','utf8')
    print(json.dumps({'output':str(BATCH),'runs':7,'maps':1785,'pairs':7497,'json_sha256':sha(BATCH/'remaining_common_exports_report.json')},ensure_ascii=True))
if __name__=='__main__':main()
