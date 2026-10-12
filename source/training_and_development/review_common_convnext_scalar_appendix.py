"""Bounded fourth-export appendix: original identity and five CPU probe windows."""
from pathlib import Path
import argparse
import csv
import json
import os

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from torch.utils.data._utils.collate import default_collate
import train_common as tc

ROOT=Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@torch.inference_mode()
def main(directory,output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    directory=directory.resolve()
    metadata=read(directory/'metadata.json');status=read(directory/'status.json')
    summary=read(directory/'summary.json')
    rid='dev_common_v2_lr1e4_convnextv2_tiny_regression_L256_s17'
    run=ROOT/'runs'/rid;config=read(run/'config.json');run_status=read(run/'status.json')
    assert metadata['run_id']==rid and status['status']=='completed' and status['verification']['all_checks_passed']
    assert metadata['schema']=='local_validation_scalar_predictions_v1'
    assert not metadata['spatial_prediction_available'] and not (directory/'maps').exists()
    assert metadata['device']=='cpu' and metadata['threads']==2 and metadata['factors']==[1,2,4]
    assert not metadata['test_accessed'] and not metadata['R6_accessed']
    assert tc.sha256_file(directory/'metadata.json')==status['metadata_sha256']
    assert tc.sha256_file(directory/'predictions.csv')==metadata['scalar_csv_sha256']
    assert tc.sha256_file(directory/'summary.json')==metadata['scalar_summary_sha256']
    manifest=read(ROOT/'local_data_v2'/'val.json')
    assert len(manifest)==357 and [r['id'] for r in manifest]==config['selected_val_ids']==[r['id'] for r in metadata['windows']]
    assert all(r['split']=='val' for r in manifest)
    assert tc.sha256_file(ROOT/'local_data_v2'/'val.json')==metadata['manifest_sha256']==config['manifest_sha256']['val.json']
    assert run_status['status']=='completed' and run_status['executed_epochs']==24
    history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    selected=min(history,key=lambda m:m['selection_balanced_mae'])
    checkpoint_path=run/'best.pt';checkpoint_hash=tc.sha256_file(checkpoint_path)
    assert checkpoint_hash==metadata['checkpoint_sha256']=='f13fa1835f6d3c26c2feada3aeeb372ad18f40fb02a25fbb980d8cd8567e869a'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    assert checkpoint['epoch']==selected['epoch']==run_status['best_epoch']==metadata['checkpoint_epoch']==19
    assert checkpoint['validation']==selected==read(run/'best_validation.json')==summary['original_amp_validation']
    assert checkpoint['config']==config and not summary['checkpoint_reselected']
    for name,digest in config['source_sha256'].items():
        assert tc.sha256_file(ROOT/name)==digest==tc.sha256_file(run/'source'/name)==tc.sha256_file(metadata['source_copies'][name])
    with (directory/'predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
        scalar=list(csv.DictReader(stream))
    lookup={(r['id'],int(r['scale_factor'])):r for r in scalar}
    assert len(lookup)==len(scalar)==1071
    model=tc.make_model('convnextv2_tiny','regression',pretrained=False).cpu().eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    dataset=tc.DevelopmentTiles(manifest,augment=False)
    checks=[]
    for start in (0,356):
        indices=list(range(start,min(start+4,len(manifest))))
        image,soft,valid,truth=default_collate([dataset[i] for i in indices])
        for R in (1,2,4):
            logits=model(tc.degrade_rgb(image,R,valid),valid)
            tc.require_finite_forward(logits,model)
            predicted=logits.float().sigmoid()
            for j,i in enumerate(indices):
                row=lookup[(manifest[i]['id'],R)]
                assert row['split']=='val' and row['source_id']==manifest[i]['source_id']
                assert row['checkpoint_sha256']==checkpoint_hash
                difference=abs(float(predicted[j])-float(row['coverage_pred']))
                assert difference<=1e-6
                checks.append({'id':manifest[i]['id'],'R':R,'difference':difference})
    assert tc.sha256_file(checkpoint_path)==checkpoint_hash
    report={'reviewed_utc':tc.utc_now(),'run_id':rid,'directory':str(directory),
            'scope':'Fourth-export appendix; original checkpoint/full manifest identity and five fixed CPU probe windows, not full357 forward recomputation',
            'checkpoint_epoch':19,'checkpoint_sha256':checkpoint_hash,'metadata_sha256':tc.sha256_file(directory/'metadata.json'),
            'original_epoch_preserved':True,'scalar_only':True,'fixed_cpu_forward_pairs':len(checks),
            'maximum_cpu_scalar_difference':max(r['difference'] for r in checks),'checks':checks,
            'cpu_balanced_mae_from_export_summary':summary['selection_balanced_mae'],
            'original_amp_balanced_mae':selected['selection_balanced_mae'],
            'cpu_minus_original_amp_from_export_summary':summary['cpu_minus_original_amp_selection_balanced_mae'],
            'gpu_used':False,'threads':2,'test_accessed':False,'R6_accessed':False,'reexport_invoked':False,
            'audit_source_sha256':tc.sha256_file(__file__),'all_checks_passed':True}
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'all_checks_passed':True,'pairs':len(checks),'max_difference':report['maximum_cpu_scalar_difference'],'output':str(output)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.export,args.output)
