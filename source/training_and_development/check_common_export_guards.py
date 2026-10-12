"""Read-only entry rejection checks; no inference or training."""
import datetime
import json
import export_common_validation as export

results = []
for run_id in ('test', 'dev_common_v1_lr1e4_resnet18_regression_L256_s17',
               'smoke_common', 'dev_common_v2_lr1e4_convnextv2_tiny_hard_seg_L256_s17'):
    directory = export.ROOT/'runs'/run_id
    before = list(directory.glob('common_validation_export_*'))
    try:
        export.validate_run(run_id)
    except (ValueError, FileNotFoundError) as error:
        unchanged = before == list(directory.glob('common_validation_export_*'))
        assert unchanged
        results.append({'run_id':run_id,'rejected':True,'error':repr(error),'no_output_created':unchanged})
    else:
        raise AssertionError('Unexpected accepted run '+run_id)
out = export.ROOT/'provenance'/'common_export_entry_checks.json'
export.write_json(out, {'at':datetime.datetime.now().astimezone().isoformat(),
    'exporter_sha256':export.sha(export.__file__), 'checks':results,
    'test_manifest_opened':False,'inference_run':False,'training_invoked':False,
    'note':'Actual invalid/old/missing-or-active run IDs; not synthetic alteration of completed files.'})
print(json.dumps({'report':str(out),'checks':results},indent=2))
