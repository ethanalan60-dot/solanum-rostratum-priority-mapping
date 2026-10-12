"""Small synthetic safety smoke; never read actual final results or historical objects."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import shutil
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'build_internal_submission_review_bundle.py'


def run():
    spec = importlib.util.spec_from_file_location('bundle', SOURCE)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    out = ROOT / 'provenance' / ('submission_bundle_preparation_' + mod.stamp())
    out.mkdir()
    toy = out / 'synthetic_only'; toy.mkdir()
    results = []

    def check(name, condition):
        assert condition, name
        results.append({'name': name, 'passed': True})

    def reject(name, fn):
        try:
            fn()
        except (ValueError, KeyError, TypeError, UnicodeError):
            results.append({'name': name, 'passed': True})
        else:
            raise AssertionError(name)

    try:
        source_hash = mod.sha(SOURCE)
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        imports = {node.names[0].name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import)}
        check('stdlib-only/no network or experiment launch', not imports & {'torch', 'numpy', 'PIL', 'subprocess', 'requests', 'rasterio'})
        for value in ['../escape', '/absolute', 'a\\b', 'C:/escape', 'a//b', 'a/../b']:
            reject('unsafe archive name ' + value, lambda value=value: mod.safe_name(value))
        check('valid short archive path', mod.safe_name('objects/00001.py') == 'objects/00001.py')
        text = toy / 'example.md'; text.write_text('Synthetic manuscript data, not actual results.\n', encoding='utf-8')
        require_pin = {'path': str(text.resolve()), 'sha256': mod.sha(text)}
        with patch.object(mod, 'ROOT', toy):
            reject('path outside workspace', lambda: mod.under_root(SOURCE))
            src = mod.Sources(); src.pin(require_pin); src.unchanged()
            check('exact input pin', src.watched[str(text.resolve())] == require_pin['sha256'])
            reject('hash drift at read', lambda: mod.Sources().pin(dict(require_pin, sha256='0' * 64)))
            reject('unknown receipt field', lambda: mod.field({'passed': True}, ['missing']))
            check('nested exact QA field', mod.field({'asset': {'sha': 'ok'}}, ['asset', 'sha']) == 'ok')
            reject('forbidden membership table', lambda: mod.eligible(text, 'test_sealed.json', 'audit'))
            fake_weight = toy / 'model.pt'; fake_weight.write_bytes(b'not a tensor')
            reject('binary weight extension', lambda: mod.eligible(fake_weight, 'model.pt', 'code'))
            fake_image = toy / 'raw.png'; fake_image.write_bytes(b'not a PNG')
            reject('raw PNG without accepted figure role', lambda: mod.eligible(fake_image, 'raw.png', 'audit'))
            binary_text = toy / 'binary.txt'; binary_text.write_bytes(b'hello\x00data')
            reject('binary disguised as text', lambda: mod.eligible(binary_text, 'binary.txt', 'audit'))
            archived_license = toy / '00001'; archived_license.write_text('License notice', encoding='utf-8')
            mod.eligible(archived_license, 'base_2714/vendor/LICENSE', 'historical_text')
            check('extensionless historical license admitted by logical name', True)
            pending = toy / 'pending.json'
            pending.write_text(json.dumps({'version': 'internal_submission_review_bundle_v1', 'internal_review_only': True,
                                          'publication_authorized': False, 'finalization_complete': False,
                                          'pending': ['real final review']}), encoding='utf-8')
            pp = {'path': str(pending.resolve()), 'sha256': mod.sha(pending)}
            reject('pending plan refuses before evidence or object loading', lambda: mod.validate_plan(pp))
            check('pending refusal creates no package directory', not (toy / 'internal_submission_review_bundles').exists())
            # Archive plumbing only: bypass admission with one manually made toy text.
            toyplan = toy / 'toy_plan.json'; toyplan.write_text('{}', encoding='utf-8')
            pin = {'path': str(toyplan.resolve()), 'sha256': mod.sha(toyplan)}
            toy_src = mod.Sources(); toy_src.pin(pin); toy_src.pin(require_pin)
            plan = {'version': 'synthetic_only', 'builder_sha256': source_hash}
            row = {'source_path': str(text.resolve()), 'logical_path': 'synthetic.md', 'classification': 'package_readme',
                   'sha256': mod.sha(text), 'bytes': text.stat().st_size, 'provenance': 'synthetic', 'public_release_approved': False}
            with patch.object(mod, 'validate_plan', return_value=(plan, toy_src, [dict(row)])):
                bundle = mod.package(pin)
            status = json.loads((bundle / 'status.json').read_text(encoding='utf-8'))
            check('tiny ZIP only, exact four members', status['passed'] and status['zip_members'] == 4 and status['file_count'] == 1)
            with zipfile.ZipFile(bundle / 'internal_review_only.zip') as z:
                check('archive object content/CRC', z.testzip() is None and z.read('objects/00001.md') == text.read_bytes())
            with patch.object(mod, 'validate_plan', return_value=(plan, toy_src, [dict(row)])), patch.object(toy_src, 'unchanged', side_effect=ValueError('synthetic post-copy drift')):
                reject('copy failure remains failure', lambda: mod.package(pin))
            failures = list((toy / 'internal_submission_review_bundles').glob('*/FAILED.json'))
            check('failed attempt and partial ZIP preserved', len(failures) == 1 and (failures[0].parent / 'internal_review_only.zip').exists())
            failure = json.loads(failures[0].read_text(encoding='utf-8'))
            check('original failure recorded', failure['error'] == 'synthetic post-copy drift' and not failure['passed'])
        check('builder source unchanged throughout smoke', mod.sha(SOURCE) == source_hash)
        receipt = {'passed': True, 'scope': 'internal_bundle_synthetic_preparation', 'checks': results,
                   'check_count': len(results), 'builder_sha256': source_hash, 'checker_sha256': mod.sha(__file__),
                   'actual_campaign_zip_built': False, 'test_arrays_or_model_or_solver_accessed': False,
                   'admission_of_actual_final_artifacts_tested': False,
                   'limits': 'Small guard and archive-plumbing smoke; final receipts do not yet exist. No claim of an actual complete submission package.'}
        mod.write_json(out / 'checks.json', receipt)
        shutil.copyfile(SOURCE, out / SOURCE.name); shutil.copyfile(__file__, out / Path(__file__).name)
        print(json.dumps({'directory': str(out), 'receipt_sha256': mod.sha(out / 'checks.json'), 'checks': len(results)}, ensure_ascii=False))
    except BaseException as exc:
        mod.write_json(out / 'FAILED.json', {'passed': False, 'error': str(exc), 'completed_checks': results})
        raise


if __name__ == '__main__':
    run()
