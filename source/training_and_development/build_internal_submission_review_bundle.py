"""Assemble an explicitly pinned INTERNAL review ZIP; never execute scientific stages.

All final receipts are supplied by the caller. This script creates no approval,
FREEZE, data release, experiment, or submission. Import has no side effects.
"""
from pathlib import Path, PurePosixPath
import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent
TEXT = {'.py', '.json', '.jsonl', '.csv', '.md', '.txt', '.html', '.bib', '.yaml', '.yml', '.toml', '.cff'}
VISUAL = {'.png', '.svg', '.pdf', '.docx'}
KINDS = {'code', 'protocol', 'config', 'audit', 'aggregate_result', 'rights_notice', 'manuscript', 'figure', 'package_readme', 'historical_text'}
HEX = re.compile(r'^[0-9a-f]{64}$')
FORBIDDEN_NAMES = ('test_sealed', 'input_manifest', 'truth_manifest', 'native_truth', 'sample_truth')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def stamp():
    return dt.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def under_root(path):
    p = Path(path)
    require(p.is_absolute(), 'Every source path must be absolute')
    p = p.resolve()
    require(p.is_relative_to(ROOT.resolve()), 'Source must remain inside campaign root')
    require(p.is_file(), 'Source is not a regular file: ' + str(p))
    return p


def safe_name(value):
    require(isinstance(value, str) and value and '\\' not in value and ':' not in value, 'Unsafe logical/archive name')
    p = PurePosixPath(value)
    require(not p.is_absolute() and all(x not in ('', '.', '..') for x in p.parts), 'Unsafe logical/archive path')
    require(p.as_posix() == value, 'Noncanonical logical/archive path')
    return value


def field(data, tokens):
    require(isinstance(tokens, list) and tokens, 'Receipt field must use an explicit nonempty token list')
    for token in tokens:
        require(isinstance(token, (str, int)) and not isinstance(token, bool), 'Invalid receipt field token')
        data = data[token]
    return data


class Sources:
    def __init__(self):
        self.watched = {}

    def pin(self, pin):
        require(isinstance(pin, dict) and HEX.fullmatch(str(pin.get('sha256', ''))), 'Missing exact source hash')
        p = under_root(pin['path'])
        actual = sha(p)
        require(actual == pin['sha256'], 'Source hash mismatch: ' + str(p))
        require(str(p) not in self.watched or self.watched[str(p)] == actual, 'Conflicting source pin')
        self.watched[str(p)] = actual
        return p

    def json(self, pin):
        p = self.pin(pin)
        value = json.loads(p.read_text(encoding='utf-8-sig'))
        require(sha(p) == pin['sha256'], 'Source changed during JSON read')
        return value

    def unchanged(self):
        for p, expected in self.watched.items():
            require(sha(p) == expected, 'Source drift: ' + p)


def eligible(path, logical, kind, visual=False):
    safe_name(logical)
    require(kind in KINDS, 'Unknown file classification')
    require(not any(term in logical.lower() for term in FORBIDDEN_NAMES), 'Reserved membership/truth source excluded')
    require(not any(term in str(Path(path).resolve()).lower() for term in FORBIDDEN_NAMES), 'Reserved source path cannot be relabeled into the bundle')
    suffix = path.suffix.lower()
    text = suffix in TEXT or (not suffix and PurePosixPath(logical).name.upper() == 'LICENSE')
    require(text or (visual and suffix in VISUAL), 'Unpermitted file type: ' + str(path))
    if suffix in VISUAL:
        require(visual and kind in {'manuscript', 'figure'}, 'Visual files require exact visual acceptance')
    else:
        require(kind not in {'figure'}, 'Figure must have a supported visual type')
        # All admitted text has to decode; files containing binary payloads are refused.
        data = path.read_bytes()
        require(b'\x00' not in data, 'Binary payload masquerading as text')
        data.decode('utf-8-sig')


def validate_visual_acceptance(qa, qa_pin, item, manuscript_sha256):
    require(qa.get('passed') is True, 'Visual acceptance failed')
    require(field(qa, qa_pin['asset_hash_field']) == item['sha256'], 'QA does not bind this exact visual asset')
    if item['classification'] == 'manuscript':
        require('manuscript_hash_field' in qa_pin, 'Manuscript visual QA must bind its Markdown source')
        require(field(qa, qa_pin['manuscript_hash_field']) == manuscript_sha256, 'Visual manuscript is not derived from the accepted final Markdown')


def validate_plan(plan_pin):
    src = Sources()
    plan = src.json(plan_pin)
    require(plan.get('version') == 'internal_submission_review_bundle_v1', 'Wrong plan version')
    require(plan.get('internal_review_only') is True and plan.get('publication_authorized') is False, 'Internal-only boundary required')
    # Stop here for a draft, before opening historical objects, final outcomes, or visuals.
    require(plan.get('finalization_complete') is True and plan.get('pending') == [], 'Final evidence is still pending')
    require(plan.get('builder_sha256') == sha(__file__), 'Builder source hash mismatch')
    evidence = plan['evidence']
    freeze = src.json(evidence['freeze'])
    require(freeze.get('passed') is True and freeze.get('frozen') is True and freeze.get('scope') == 'confirmation_v1', 'Actual formal freeze required')
    require(len(freeze.get('run_ids', [])) == 37 and len(set(freeze['run_ids'])) == 37 and freeze.get('R') == [1, 2, 4, 6], 'Wrong confirmation population')
    for name in ('source_lock', 'contract', 'checkpoints'):
        require(evidence[name]['sha256'] == freeze[name + '_sha256'], 'Formal freeze metadata binding mismatch: ' + name)
        require(Path(evidence[name]['path']).resolve() == (ROOT / 'confirmation_v1' / (('SOURCE_LOCK' if name == 'source_lock' else name) + '.json')).resolve(), 'Wrong canonical freeze metadata source')
        src.pin(evidence[name])
    lock = src.json(evidence['source_lock'])
    report = src.json(evidence['final_report_review'])
    require(report.get('passed') is True and report.get('scope') == 'confirmation_report_tables_independent', 'Completed independent report review required')
    require(report.get('freeze_sha256') == evidence['freeze']['sha256'], 'Report/freeze identity mismatch')
    require(isinstance(report.get('table_rows'), dict) and report['table_rows'] and report.get('numeric_comparisons', 0) > 0, 'Report review lacks actual numerical work')
    out = Path(report['output_directory']).resolve()
    require(out == (ROOT / 'confirmation_v1/stages/report_tables').resolve(), 'Wrong completed report directory')
    for basename, key in [('artifact_manifest.json', 'audited_artifact_manifest_sha256'), ('summary.json', 'summary_sha256')]:
        src.pin({'path': str(out / basename), 'sha256': report[key]})
    content_pin = evidence['final_content_review']
    content = src.json(content_pin)
    require(content.get('passed') is True, 'Final manuscript content review has not passed')
    require(field(content, content_pin['manuscript_hash_field']) == evidence['final_manuscript']['sha256'], 'Final manuscript not bound by content review')
    require(field(content, content_pin['report_review_hash_field']) == evidence['final_report_review']['sha256'], 'Final content review must bind actual confirmation report review')
    final_md = src.pin(evidence['final_manuscript'])
    require(final_md.suffix.lower() == '.md', 'Final manuscript must have a reviewable Markdown source')
    dev = src.json(evidence['full24_review'])
    require(dev.get('passed') is True and dev.get('scope') == 'full24_seed_summary_independent', 'Full24 numerical review required')
    require(evidence['full24_manifest']['sha256'] == dev['audited_artifact_manifest_sha256'], 'Full24 manifest/review mismatch')
    src.pin(evidence['rights_register'])
    rows, identities = [], set()

    def add(p, logical, kind, expected, provenance, visual=False):
        require(logical not in identities, 'Duplicate archive logical identity: ' + logical)
        eligible(p, logical, kind, visual)
        require(sha(p) == expected, 'Source changed during admission')
        identities.add(logical)
        rows.append({'source_path': str(p), 'logical_path': logical, 'classification': kind,
                     'sha256': expected, 'bytes': p.stat().st_size, 'provenance': provenance,
                     'public_release_approved': False})

    bundles = plan.get('historical_bundles', [])
    require(len(bundles) == 2 and {x['label'] for x in bundles} == {'base_2714', 'failure_20'}, 'Both exact historical snapshots required')
    for entry in bundles:
        manifest = src.json(entry)
        require(manifest.get('publication_authorized') is False, 'Historical snapshot release boundary differs')
        expected_count = {'base_2714': 2714, 'failure_20': 20}[entry['label']]
        require(len(manifest['files']) == expected_count, 'Wrong historical snapshot count')
        base = Path(entry['path']).resolve().parent
        for item in manifest['files']:
            relative = safe_name(item['archive_path'])
            p = (base / relative).resolve()
            require(p.is_relative_to(base), 'Historical object escapes snapshot')
            src.pin({'path': str(p), 'sha256': item['sha256']})
            require(p.stat().st_size == item['bytes'], 'Historical object length mismatch')
            logical = entry['label'] + '/' + safe_name(item['logical_path'])
            add(p, logical, 'historical_text', item['sha256'], {'manifest': entry, 'original_classification': item['classification']})
        add(Path(entry['path']).resolve(), entry['label'] + '/historical_package_manifest.json', 'audit', entry['sha256'], {'historical_manifest': True})
    for key, folder in [('files', ROOT / 'confirmation_v1'), ('external_dependency_sha256', ROOT), ('root_output_auditor_sha256', ROOT)]:
        require(isinstance(lock.get(key), dict) and lock[key], 'Frozen source inventory missing')
        for relative, expected in lock[key].items():
            safe_name(relative)
            p = (folder / relative).resolve()
            require(p.is_relative_to(folder.resolve()), 'Frozen source escapes declared directory')
            src.pin({'path': str(p), 'sha256': expected})
            add(p, 'confirmation_source_and_freeze/' + key + '/' + relative, 'protocol' if p.suffix.lower() != '.py' else 'code', expected, {'source_lock': evidence['source_lock']})
    # Complete saved aggregate inventories come from the exact two accepted manifests,
    # never from a live directory search or a selected favorable subset.
    for label, pin, key, expected_summary in [
        ('full24_results', evidence['full24_manifest'], 'file_sha256', dev['summary_sha256']),
        ('confirmation_accepted_results', {'path': str(out / 'artifact_manifest.json'), 'sha256': report['audited_artifact_manifest_sha256']}, 'files', report['summary_sha256'])]:
        manifest = src.json(pin)
        saved = manifest[key]
        require(isinstance(saved, dict) and saved and saved.get('summary.json') == expected_summary, 'Accepted table manifest lacks bound summary')
        folder = Path(pin['path']).resolve().parent
        for relative, expected in saved.items():
            safe_name(relative)
            p = (folder / relative).resolve()
            require(p.is_relative_to(folder), 'Accepted table file escapes its output')
            src.pin({'path': str(p), 'sha256': expected})
            add(p, label + '/' + relative, 'aggregate_result', expected, {'accepted_manifest': pin})
        add(Path(pin['path']).resolve(), label + '/artifact_manifest.json', 'audit', pin['sha256'], {'accepted_manifest': True})
    # New files are individually enumerated by the final caller. No directory walk.
    files = plan.get('files', [])
    require(files, 'Explicit final file inventory is missing')
    required_groups = {'failure_recovery_lineage', 'submission_text'}
    require(required_groups <= {x.get('group') for x in files}, 'One or more mandatory final evidence groups are missing')
    for item in files:
        p = src.pin(item)
        qa_pin = item.get('visual_acceptance')
        if p.suffix.lower() in VISUAL:
            require(qa_pin is not None, 'Visual asset has no final QA receipt')
            qa = src.json(qa_pin)
            validate_visual_acceptance(qa, qa_pin, item, evidence['final_manuscript']['sha256'])
        add(p, safe_name(item['logical_path']), item['classification'], item['sha256'], {'group': item['group'], 'visual_acceptance': qa_pin}, visual=qa_pin is not None)
    paths = {r['source_path'] for r in rows}
    # Evidence used to admit the package is itself included, with field selectors intact in the plan.
    for name, pin in evidence.items():
        p = src.pin(pin)
        if str(p) not in paths:
            add(p, 'final_evidence/' + name + p.suffix.lower(), 'audit' if name != 'final_manuscript' else 'manuscript', pin['sha256'], {'evidence_role': name})
            paths.add(str(p))
    for item in files:
        pin = item.get('visual_acceptance')
        if pin:
            p = src.pin(pin)
            if str(p) not in paths:
                add(p, 'final_evidence/visual_' + pin['sha256'] + '.json', 'audit', pin['sha256'], {'visual_receipt': True})
                paths.add(str(p))
    require(any(r['classification'] == 'figure' for r in rows), 'No final accepted figures supplied')
    require(any(Path(r['source_path']).suffix.lower() == '.docx' and r['classification']=='manuscript' for r in rows), 'No final accepted editable Word supplied')
    src.unchanged()
    return plan, src, rows


def package(plan_pin):
    plan, src, rows = validate_plan(plan_pin)
    base = ROOT / 'internal_submission_review_bundles'
    base.mkdir(exist_ok=True)
    output = base / stamp()
    output.mkdir()  # A new immutable attempt; never resume/overwrite.
    try:
        shutil.copyfile(__file__, output / 'builder.py')
        shutil.copyfile(plan_pin['path'], output / 'plan.json')
        manifest = {'version': plan['version'], 'internal_review_only': True,
                    'publication_authorized': False, 'created_at': dt.datetime.now().astimezone().isoformat(),
                    'plan_sha256': plan_pin['sha256'], 'builder_sha256': sha(__file__), 'files': rows,
                    'not_a_runnable_checkout': True, 'historical_reference_only_entries_preserved_in_manifests': True,
                    'failure_lineage_completeness_automatically_proven': False,
                    'failure_lineage_scope': 'Only the explicitly listed, hash-verified records are included. The root must separately review the full history and post-FREEZE stages/recoveries for completeness.',
                    'secret_content_automatically_certified_absent': False}
        for n, row in enumerate(rows, 1):
            row['archive_path'] = 'objects/' + f'{n:05d}' + Path(row['source_path']).suffix.lower()
        write_json(output / 'package_manifest.json', manifest)
        zip_path = output / 'internal_review_only.zip'
        with zipfile.ZipFile(zip_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
            for row in rows:
                z.write(row['source_path'], row['archive_path'])
            for name in ('builder.py', 'plan.json', 'package_manifest.json'):
                z.write(output / name, name)
        expected_members = {r['archive_path'] for r in rows} | {'builder.py', 'plan.json', 'package_manifest.json'}
        with zipfile.ZipFile(zip_path) as z:
            require(len(z.namelist()) == len(expected_members) and set(z.namelist()) == expected_members, 'ZIP member inventory mismatch')
            require(z.testzip() is None, 'ZIP CRC failure')
            for row in rows:
                h = hashlib.sha256()
                with z.open(row['archive_path']) as stream:
                    for block in iter(lambda: stream.read(2 ** 20), b''):
                        h.update(block)
                require(h.hexdigest() == row['sha256'], 'ZIP object differs from admitted source')
        src.unchanged()
        require(sha(__file__) == plan['builder_sha256'], 'Builder source changed')
        status = {'status': 'completed_internal_review_only', 'passed': True,
                  'publication_authorized': False, 'zip_sha256': sha(zip_path), 'file_count': len(rows),
                  'zip_members': len(expected_members), 'manifest_sha256': sha(output / 'package_manifest.json'),
                  'source_hashes_before_after': src.watched, 'actual_source_sha256': sha(output / 'builder.py')}
        write_json(output / 'status.json', status)
    except BaseException as exc:
        failure = {'status': 'failed', 'passed': False, 'publication_authorized': False,
                   'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
        try:
            write_json(output / 'FAILED.json', failure)
        except BaseException as record_error:
            print(json.dumps({'original_failure': failure, 'failure_record_error': str(record_error)}, ensure_ascii=False))
        raise
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    require(args.execute, 'Explicit --execute is required; draft plans never package')
    print(package({'path': args.plan, 'sha256': args.plan_sha256}))


if __name__ == '__main__':
    main()
