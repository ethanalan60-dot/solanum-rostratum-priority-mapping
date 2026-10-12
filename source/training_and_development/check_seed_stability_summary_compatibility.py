"""Bounded preparation check: original8 saved tables and first2 new saved tables only."""
import importlib.util
from pathlib import Path
from datetime import datetime
import shutil

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('summary', ROOT / 'seed_stability_summary_v1.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
sources = s.Sources()
sources.current(ROOT / 'seed_stability_summary_v1.py')
sources.current(__file__)
oldb = ROOT / 'precision_agriculture/common_management_reports/full8_20260916_050927_560091/all_720_model_policy_budgets.csv'
olda = ROOT / 'precision_agriculture/common_prefix_auc_full8_20260916_052212_839933/all72_model_prefix_auc.csv'
sources.pin({'path': str(oldb), 'sha256': s.OLD_PINS['budget_csv']})
sources.pin({'path': str(olda), 'sha256': s.OLD_PINS['auc_csv']})
b = [s.normalize_original(r) for r in s.read_csv(oldb)]
s.require(len(b) == 720 and len({(r['run_id'], r['M'], r['R'], r['budget'], r['policy']) for r in b}) == 720, 'old grid')
models = {r['run_id']: s.model_fields(r) for r in b}
a = s.read_csv(olda)
s.require(len(a) == 72 and all(s.model_fields(r) == models[r['run_id']] for r in a), 'old AUC identity')
checks = ['original8 canonical720/AUC72 schema, unique IDs and checkpoint binding']
for variant in ('hard_seg', 'soft_seg'):
    rid = 'dev_seed_v1_lr1e4_resnet18_' + variant + '_L256_s29'
    rp = ROOT / 'local_seed_stability_v1/postprocess_v1/run_records' / (rid + '.completed.json')
    sources.current(rp)
    rec = s.load(rp)
    spatial = Path(rec['spatial_directory'])
    stp = sources.pin({'path': str(spatial / 'status.json'), 'sha256': rec['spatial_status_sha256']})
    sup = sources.pin({'path': str(spatial / 'summary.json'), 'sha256': rec['spatial_summary_sha256']})
    st, su = s.load(stp), s.load(sup)
    p = spatial / 'all_45_policy_outcomes.csv'
    sources.pin({'path': str(p), 'sha256': st['files_sha256'][p.name]})
    rows = [x for row in s.read_csv(p) for x in s.normalize_new(row, su, p, s.sha(p))]
    s.require(len(rows) == 90 and len({(r['M'], r['R'], r['budget'], r['policy']) for r in rows}) == 90, 'new grid')
    for r in rows:
        s.require(r['actual_area_m2'] <= r['budget_area_m2'], 'strict budget')
        s.close(r['capture_pct'], 100 * r['captured_target_m2'] / r['domain_target_m2'])
        s.close(r['actual_area_fraction'], r['actual_area_m2'] / r['domain_area_m2'])
    export = Path(rec['export_directory'])
    est = sources.pin({'path': str(export / 'status.json'), 'sha256': rec['export_status_sha256']})
    emp = sources.pin({'path': str(export / 'artifact_manifest.json'), 'sha256': rec['export_manifest_sha256']})
    esp = sources.pin({'path': str(export / 'summary.json'), 'sha256': s.load(emp)['file_sha256']['summary.json']})
    s.require(len(s.load_scalar_export(s.model_fields(su), est, esp, sources)) == 3, 'new scalar')
    checks.append(rid + ':90 normalized rows and3 saved CPU summary rows')
# One fixed original CPU export suffices to check its schema; production requires all8.
export = ROOT / 'runs/dev_common_v2_lr1e4_resnet18_hard_seg_L256_s17/common_validation_export_20260916_030715_466122'
esp, est = sources.current(export / 'summary.json'), sources.current(export / 'status.json')
s.require(len(s.load_scalar_export(models['dev_common_v2_lr1e4_resnet18_hard_seg_L256_s17'], est, esp, sources)) == 3, 'old scalar')
checks.append('original saved CPU summary schema')
sources.verify_unchanged()
out = ROOT / 'provenance' / ('seed_summary_schema_compatibility_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
out.mkdir()
shutil.copyfile(__file__, out / Path(__file__).name)
shutil.copyfile(ROOT / 'seed_stability_summary_v1.py', out / 'seed_stability_summary_v1.py')
s.dump(out / 'review.json', {'passed': True, 'scope': 'metadata/schema compatibility only, not full24 production or independent numerical audit',
    'checks': checks, 'sources': sources.files, 'new_model_tables_read': 2, 'old_model_tables_read': 8,
    'full24_aggregation_executed': False, 'raw_arrays_or_models_read': False, 'test_or_R6_accessed': False, 'model_calls': 0, 'solver_calls': 0})
print(out)
