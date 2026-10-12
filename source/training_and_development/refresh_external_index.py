"""Extend the accepted local evidence index with completed external development artifacts."""
from pathlib import Path
import datetime,hashlib,json,os
ROOT=Path(__file__).resolve().parent
PA=ROOT/'precision_agriculture'
TRAIN=ROOT/'external_weeds_training_reports/20260916_062524_429008'
EXPORT=ROOT/'external_exports/exw_20260916_063648_275359'
REVIEW=ROOT/'provenance/external_export_independent_20260916_064324_431899'
FIG=ROOT/'external_weeds_training_reports/figure_20260916_063029_278387'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def relative(p):return os.path.relpath(p,PA).replace('\\','/')
def main():
    index=read(PA/'latest_results.json')
    for row in index['files']:
        assert sha(PA/row['path'])==row['sha256'],('Existing indexed evidence changed',row['path'])
    assert sha(TRAIN/'audit.json')=='172bb72f0b31c761ccb584cba29b8817b4f58b71e9bcff33a6665d88cd2d75fc'
    assert sha(REVIEW/'audit.json')=='51b92fbe6740621776af063e6b8c91bda8a2cf3c516e9254bda0697c32c8ec47'
    assert read(REVIEW/'audit.json')['passed'] and read(EXPORT/'status.json')['state']=='completed'
    assert sha(REVIEW/'spatial_sanitized_acceptance.json')=='649ea0dd7e385967f9e09653d2acad2c4b5413be24483db69d0537818c591bc0'
    assert read(TRAIN/'audit.json')['run_count']==9 and read(EXPORT/'status.json')['scalar_rows']==702
    additions=[TRAIN/x for x in ('audit.json','review.md','artifact_manifest.json','all9_configs.csv','all3_hard_soft_seed_pairs.csv','all3_variant_seed_summary.csv','constant_baselines.csv')]
    additions += [EXPORT/x for x in ('status.json','summary.json','pre_execution_protocol.json','pre_execution_seal.json','artifact_manifest.json','map_index.json','all702_scalars.csv')]
    additions += [REVIEW/x for x in ('audit.json','review.md','artifact_manifest.json','spatial_sanitized_acceptance.json','all9_CPU_GPU_comparison.csv','all468_map_mean_checks.csv')]
    additions += [FIG/x for x in ('figure_manifest.json','delivery_manifest.json','root_png_review.json','weeds_all9_development_comparison.png','weeds_all9_development_comparison.pdf','weeds_all9_development_comparison.svg')]
    additions += [ROOT/'external_weeds_v1/spatial_v1'/x for x in ('protocol.json','freeze_receipt.json')]
    additions += [ROOT/'provenance/local_regression_diagnostic_design_20260916_061936_264785'/x for x in ('design.json','design.md','artifact_manifest.json')]
    additions += [PA/(x+y) for x in ('PA_external_training_draft_v1','PA_agricultural_implications_draft_v1') for y in ('.md','_evidence.json')]
    writing_review=ROOT/'provenance/PA_external_implications_review_20260916_064825_159442'
    additions += [writing_review/x for x in ('review.json','review.md','citation_supplement.md','WeedsGalore_official.bib')]
    additions += [ROOT/x for x in ('safe_load_external_checkpoint.py','export_external_weeds_validation.py','audit_external_weeds_export.py','freeze_external_spatial_protocol.py')]
    archives=sorted((ROOT/'code_versions').glob('manifest_*.json'));assert archives;additions.append(archives[-1])
    table={x['path']:x for x in index['files']}
    for p in additions:
        key=relative(p);table[key]={'path':key,'sha256':sha(p)}
    index.update(schema_version=4,updated_local=datetime.datetime.now().astimezone().isoformat(),
        scope='Completed local12 training/eight spatial models; completed external9 training and accepted CPU exports. External spatial evaluation, local normalization diagnosis, reserved confirmation and full submission package incomplete.',
        weedsgalore_training_snapshot=read(ROOT/'external_weeds_v1/current_queue.json'),
        weedsgalore_training_review=relative(TRAIN/'audit.json'),weedsgalore_CPU_export=relative(EXPORT),
        weedsgalore_CPU_export_review=relative(REVIEW/'audit.json'),weedsgalore_spatial_sanitized_acceptance=relative(REVIEW/'spatial_sanitized_acceptance.json'),
        weedsgalore_spatial_protocol='../external_weeds_v1/spatial_v1/protocol.json',weedsgalore_spatial_executed=False,
        weedsgalore_training_completed=9,weedsgalore_scalar_rows=702,weedsgalore_segmentation_map_files=156,
        local_regression_diagnostic_design='../provenance/local_regression_diagnostic_design_20260916_061936_264785/design.json',
        current_external_training_draft='PA_external_training_draft_v1.md',current_agricultural_discussion_draft='PA_agricultural_implications_draft_v1.md',
        code_archive_manifest=relative(archives[-1]),next_spatial_step='Review and execute frozen external relative-grid protocol after source readiness; root-review separate local normalization diagnostic; no test/R6 until final confirmation protocol.',
        test_accessed=False,R6_accessed=False,final_confirmation_complete=False,files=list(table.values()))
    save(PA/'latest_results.json',index)
    receipt={'passed':True,'updated_local':index['updated_local'],'indexed_files':len(table),'all_previous_indexed_hashes_unchanged':True,
        'all_indexed_hashes_match':all(sha(PA/x['path'])==x['sha256'] for x in index['files']),
        'latest_results_sha256':sha(PA/'latest_results.json'),'external_runs':9,'scalar_rows':702,'map_files':156,
        'spatial_external_executed':False,'test_accessed':False,'R6_accessed':False,'publication_objective_complete':False}
    save(PA/'latest_index_verification.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
