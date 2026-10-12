"""Create an exact internal delivery plan from completed reviews; no experiment calls."""
from pathlib import Path
import datetime as dt
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parent
PA = ROOT / 'precision_agriculture'
FINAL = PA / 'final_v5'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()

def pin(p):
    p = Path(p).resolve()
    assert p.is_relative_to(ROOT)
    return {'path': str(p), 'sha256': sha(p)}

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p, obj):
    with Path(p).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def main():
    builder = ROOT / 'build_internal_submission_review_bundle.py'
    assert sha(builder) == '39e0919f900f7dc45ad5339c4abd3e17fdffd49a8c664f2999e02a27de9f2257'
    plan = read(PA / 'submission_package_preparation_20260916_123023_154997/internal_bundle_plan.DRAFT.json')
    plan.update(created_at=dt.datetime.now().astimezone().isoformat(), finalization_complete=True,
                builder_sha256=sha(builder), pending=[], files=[])
    report = ROOT / 'provenance/completed_confirmation_report_independent_20260916_142830_724677/review.json'
    content = ROOT / 'provenance/results_text_independent_20260916_144015_927931/review.json'
    assert sha(report) == '3af89351cf4a813bae16c6c6ba82e7334e05be4aa048a49707ad4f521f57bed8'
    assert sha(content) == '2608e47f627cb3311ddc6a0b60bdb9b7373b927c0db56ad934a27bded5fe4269'
    assert sha(FINAL / 'PA_manuscript.md') == '07bca0bebd178e5071bf9b8e2e329cc55532c708724ab35f7b97fc08835ecff6'
    assert sha(FINAL / 'PA_supplementary.md') == '1343fede7c05aa70e781c7da8cea95f2d98a0eaac147aee4464eb604e8af18c6'
    plan['evidence'].update(final_report_review=pin(report), final_manuscript=pin(FINAL/'PA_manuscript.md'),
        final_content_review={**pin(content), 'manuscript_hash_field':['manuscript_sha256'],
                              'report_review_hash_field':['accepted_report','review_sha256']})
    list_paths = [
        ROOT/'provenance/final_failure_lineage_terminal_20260916_144408_019256/files.json',
        PA/'internal_bundle_additions_20260916_142426_690735/additional_files.json',
        ROOT/'provenance/final_v5_document_lineage_20260916_145851_378208/files.json',
        FINAL/'package_preparation/explicit_final_publication_files.json',
    ]
    seen = {}
    deduplicated = []
    def add(item):
        path = Path(item['path']).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == item['sha256']
        key = str(path).casefold()
        if key in seen:
            previous = seen[key]
            assert previous['sha256'] == item['sha256']
            if item.get('visual_acceptance'):
                assert previous.get('visual_acceptance'), 'Visual entry unexpectedly shadowed'
            deduplicated.append({'path':str(path),'retained':previous['logical_path'],'omitted_duplicate':item['logical_path']})
            return
        seen[key] = item
        plan['files'].append(item)
    for p in list_paths:
        for item in read(p)['files']:
            add(item)
    extra = list_paths + [
        ROOT/'provenance/final_failure_lineage_terminal_20260916_144408_019256/inventory.json',
        ROOT/'provenance/final_failure_lineage_terminal_20260916_144408_019256/files_eligibility_check.json',
        ROOT/'provenance/root_finalization_review_20260916_150408_328288/review.json',
        ROOT/'provenance/final_delivery_promoter_final_checks_20260916_151025_615654/readiness.json',
        ROOT/'provenance/final_citation_review_20260916_134713_7823288/review.json',
        ROOT/'provenance/final_citation_review_20260916_134713_7823288/findings.md',
        ROOT/'provenance/final_citation_review_20260916_134713_7823288/Wang2021_crossref.json',
        ROOT/'provenance/final_citation_review_20260916_134713_7823288/Elmachtoub2022_crossref.json',
        ROOT/'provenance/results_text_independent_20260916_144015_927931/supplement_integration_receipt.json',
        ROOT/'provenance/results_text_independent_20260916_144015_927931/integration_receipt.json',
        ROOT/'provenance/results_text_independent_20260916_144015_927931/computed_receipt.json',
        ROOT/'provenance/results_text_independent_20260916_144015_927931/review_state.json',
        ROOT/'provenance/results_text_independent_20260916_144015_927931/findings.md',
        PA/'PA_supplementary_v5_sourcepins.json', PA/'PA_supplementary_v5_prose_receipt.json',
        FINAL/'PA_cover_letter.md', FINAL/'农业意义与结果说明.md', FINAL/'closing_text_sourcepins.json',
        FINAL/'README_交付说明.md',
        FINAL/'package_preparation/explicit_final_publication_validation.json',
        ROOT/'promote_final_delivery_v5.py', Path(__file__),
    ]
    for p in extra:
        group = 'failure_recovery_lineage' if 'lineage' in str(p) else 'submission_text'
        kind = 'code' if p.suffix == '.py' else ('manuscript' if p.name in ['PA_cover_letter.md','农业意义与结果说明.md'] else 'audit')
        add({**pin(p), 'logical_path':'final_support/'+p.relative_to(ROOT).as_posix(),
             'classification':kind, 'group':group})
    plan['notes'] = [
        'Finalization means the computational manuscript, accepted evidence and visual assets are complete for internal author review, not author approval or submission authorization.',
        'No public release or journal submission is performed. Contributions, interests, funder roles, author approval and local-data authority remain factual author inputs.',
        'All historical manifests, complete full24 and accepted confirmation report inventories are expanded by the unchanged builder.',
        'Scientific failure lineage is bound through report acceptance; the separate document lineage extends it through Word/PDF QA. Final ZIP status is a later sidecar, not a fabricated included receipt.',
        'Raw images, arrays, masks, weights, private author source DOCX and reserved membership/truth manifests are excluded.',
        'The documented Word renderer is retained as a byte-exact .ps1.txt text archive with the original path and hash; it is not presented as an executable output.',
    ]
    spec = importlib.util.spec_from_file_location('delivery_packager', builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for item in plan['files']:
        p = Path(item['path'])
        module.eligible(p, item['logical_path'], item['classification'], bool(item.get('visual_acceptance')))
        if item.get('visual_acceptance'):
            qa = item['visual_acceptance']
            assert sha(Path(qa['path'])) == qa['sha256']
            module.validate_visual_acceptance(read(qa['path']), qa, item, plan['evidence']['final_manuscript']['sha256'])
    assert len({x['logical_path'] for x in plan['files']}) == len(plan['files'])
    out = ROOT/'provenance'/('final_internal_plan_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    write(out/'plan.json', plan)
    write(out/'preparation_receipt.json', {'passed':True, 'plan':pin(out/'plan.json'),
        'input_lists':[pin(p) for p in list_paths], 'explicit_files':len(plan['files']),
        'explicit_bytes':sum(Path(x['path']).stat().st_size for x in plan['files']),
        'duplicates_retained_once':deduplicated, 'scientific_execution':False,
        'actual_zip_created':False, 'full_builder_validation_remains_at_execution':True})
    print(json.dumps({'plan':pin(out/'plan.json'),'explicit_files':len(plan['files'])},ensure_ascii=False))

if __name__ == '__main__':
    main()
