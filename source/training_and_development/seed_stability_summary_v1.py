"""Full 24-model local development summary. Standard library; never runs inference.

Preparation: --self-check. Production: --execute --root-review PATH.
No discovery/globbing of runs, partial scientific report, test access, or solver.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PA = ROOT / 'precision_agriculture'
SCOPE = 'seed_stability_summary_v1_full24'
SEEDS = (17, 29, 43)
ENCODERS = ('resnet18', 'convnextv2_tiny')
LRS = (0.0001, 0.0003)
VARIANTS = ('hard_seg', 'soft_seg')
RS = MS = (1, 2, 4)
BUDGETS = (0.1, 0.2, 0.3, 0.5, 1.0)
POLICIES = ('prefix', 'predicted_area')
CONTRASTS = ((4, 1), (2, 1), (4, 2))
PP_EPS = 1e-8                 # same as accepted seed17 central report
AUC_EPS = 1e-12               # same as accepted seed17 AUC supplement
PLAN_SHA = 'b96d83d3f5e68addbf5461758f66674b900c8777e8a9f02c50ebcff7735c1168'
OLD_PINS = {
    'budget_csv': 'b7baa56fb153e924ca8f7ebf8f2a3a0f793806e993010fc7d9da76150086d71b',
    'auc_csv': 'da91a03e4e9e402d7999e178d8ce2c9c75fe34c05f0952b2aa00638cdfabf5ba',
    'budget_audit': '6225b0b64dbd15df24dc69952af78a5919e3a98b6a434d30682f04e5a29ff9af',
    'auc_audit': 'f189c95b62d7a9c226949db2893daa6327f35bfec6b4cec0fe75af89464e5e07',
}
MODEL_KEYS = ('run_id', 'encoder', 'variant', 'lr', 'seed', 'checkpoint_epoch', 'checkpoint_sha256')
BUDGET_METRICS = ('capture_pct', 'captured_target_m2', 'actual_area_m2', 'actual_area_fraction',
                  'missed_target_m2', 'zero_annotation_area_m2', 'selected_zero_target_unit_area_m2',
                  'predicted_objective_m2', 'unused_budget_m2')
AUC_METRICS = ('auc_0_1', 'auc_0_0p5_normalized')
SCALAR_METRICS = ('balanced_mae_pp', 'mae_pp', 'mae_zero_pp', 'mae_positive_pp', 'bias_pp')
FLAGS = ('fallback_used', 'optimality_proven', 'original_proven_flag', 'bound_anomaly', 'certification_downgraded')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    require(bool(rows), 'empty output table')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def number(value, optional=False):
    if value is None or value == '':
        require(optional, 'missing required numeric value')
        return None
    x = float(value)
    require(math.isfinite(x), 'nonfinite numeric value')
    return x


def boolean(value, optional=False):
    if value is None or value == '':
        require(optional, 'missing required boolean')
        return None
    require(value in (True, False, 'True', 'False', 'true', 'false'), 'invalid boolean')
    return value in (True, 'True', 'true')


def close(a, b, tolerance=1e-9):
    require(math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=tolerance), f'numeric disagreement: {a}, {b}')


def run_id(encoder, variant, lr, seed):
    tag = '1e4' if lr == 0.0001 else '3e4'
    prefix = 'dev_common_v2' if seed == 17 else 'dev_seed_v1'
    return f'{prefix}_lr{tag}_{encoder}_{variant}_L256_s{seed}'


def expected_models():
    return {run_id(e, v, lr, s): {'run_id': run_id(e, v, lr, s), 'encoder': e, 'variant': v, 'lr': lr, 'seed': s}
            for s, e, lr, v in itertools.product(SEEDS, ENCODERS, LRS, VARIANTS)}


def model_fields(row):
    result = {k: row[k] for k in MODEL_KEYS}
    for key in ('seed', 'checkpoint_epoch'):
        result[key] = int(result[key])
    result['lr'] = number(result['lr'])
    expect = expected_models().get(result['run_id'])
    require(expect is not None, 'unexpected model ID')
    require(all(result[k] == expect[k] for k in expect), 'model ID/config mismatch')
    require(1 <= result['checkpoint_epoch'] <= 24, 'invalid selected epoch')
    require(len(result['checkpoint_sha256']) == 64, 'invalid checkpoint hash')
    return result


class Sources:
    def __init__(self):
        self.files = {}

    def pin(self, spec):
        p = Path(spec['path']).resolve()
        require(p.is_relative_to(ROOT), 'source must be in campaign ROOT')
        require(p.suffix.lower() in ('.json', '.csv', '.py', '.md'), 'only metadata/tables/code allowed')
        require('test.json' != p.name.lower(), 'test manifest forbidden')
        digest = sha(p)
        require(digest == spec['sha256'], f'source hash mismatch: {p}')
        key = p.as_posix()
        require(key not in self.files or self.files[key] == digest, 'conflicting source binding')
        self.files[key] = digest
        return p

    def current(self, path):
        return self.pin({'path': str(path), 'sha256': sha(path)})

    def verify_unchanged(self):
        for p, h in self.files.items():
            require(sha(p) == h, f'source changed during summary: {p}')


def normalize_original(row):
    r = dict(row)  # preserve accepted canonical fields, not historical uncorrected flags
    r.update(model_fields(row))
    for k in ('M', 'R', 'selected_count'):
        r[k] = int(row[k])
    for k in ('budget', 'budget_area_m2', 'domain_area_m2', 'domain_target_m2') + BUDGET_METRICS:
        r[k] = number(row[k])
    for k in ('predicted_relative_gap', 'predicted_upper_m2', 'original_retained_upper_m2', 'original_retained_gap'):
        r[k] = number(row.get(k), optional=True)
    for k in FLAGS:
        r[k] = boolean(row.get(k), optional=True)
    r.pop('original_gpu_mae_pp', None)  # scalar errors are joined separately from CPU exports
    return r


def normalize_new(row, summary, source_path, source_hash):
    outputs = []
    bound = json.loads(row['bound_audit_json'])
    require(row['run_id'] == summary['run_id'], 'outcome run mismatch')
    for policy, prefix in (('prefix', 'prefix'), ('predicted_area', 'pmilp')):
        r = model_fields(summary)
        r.update(policy=policy, M=int(row['M']), R=int(row['R']), budget=number(row['budget']),
                 budget_area_m2=number(row['budget_area_m2']), selected_count=int(row[f'{prefix}_selected_count']),
                 domain_area_m2=number(summary['domain']['area_m2']), domain_target_m2=number(summary['domain']['target_m2']),
                 source_csv=str(source_path), source_csv_sha256=source_hash)
        for metric in BUDGET_METRICS:
            if metric == 'capture_pct':
                r[metric] = 100 * number(row[f'{prefix}_capture_fraction'])
            elif metric == 'unused_budget_m2':
                r[metric] = r['budget_area_m2'] - number(row[f'{prefix}_actual_area_m2'])
            else:
                r[metric] = number(row[f'{prefix}_{metric}'])
        r['selected_ids_json'] = row['prefix_ids_json' if policy == 'prefix' else 'selected_ids_json']
        r['fallback_used'] = False if policy == 'prefix' else boolean(row['fallback_used'])
        for flag in FLAGS[1:]:
            src = 'original_proven' if flag == 'original_proven_flag' else flag
            r[flag] = None if policy == 'prefix' else boolean(row[src])
        r['predicted_upper_m2'] = None if policy == 'prefix' else number(row['predicted_upper_m2'])
        r['predicted_relative_gap'] = None if policy == 'prefix' else number(row['predicted_relative_gap'])
        r['original_retained_upper_m2'] = None if policy == 'prefix' else number(bound['retained_upper_m2'])
        r['original_retained_gap'] = None  # not present in the new table; never reverse-engineered
        r['raw_solver_status'] = None if policy == 'prefix' else row['raw_solver_status']
        r['raw_solver_gap'] = None if policy == 'prefix' else number(row['raw_solver_gap'], optional=True)
        r['bound_audit_json'] = '' if policy == 'prefix' else row['bound_audit_json']
        outputs.append(r)
    return outputs


def validate_grids(budgets, auc, scalars, allow_na=False):
    models = expected_models()
    require(len(budgets) == 2160 and len(auc) == 216 and len(scalars) == 72, 'incomplete full24 table counts')
    require({r['run_id'] for r in budgets} == set(models), 'not exactly all24 model IDs')
    grids = ((budgets, ('run_id', 'M', 'R', 'budget', 'policy'),
              set(itertools.product(models, MS, RS, BUDGETS, POLICIES))),
             (auc, ('run_id', 'M', 'R'), set(itertools.product(models, MS, RS))),
             (scalars, ('run_id', 'R'), set(itertools.product(models, RS))))
    identity = {}
    for rows, keys, expected in grids:
        actual = [tuple(r[k] for k in keys) for r in rows]
        require(len(set(actual)) == len(actual) and set(actual) == expected, 'duplicate/missing grid cell')
        for r in rows:
            meta = model_fields(r)
            if r['run_id'] in identity:
                require(identity[r['run_id']] == meta, 'checkpoint/config mismatch across source tables')
            identity[r['run_id']] = meta
    for r in budgets:
        require(r['actual_area_m2'] <= r['budget_area_m2'], 'strict original area budget violated')
        close(r['budget_area_m2'], r['budget'] * r['domain_area_m2'])
        close(r['actual_area_fraction'], r['actual_area_m2'] / r['domain_area_m2'])
        close(r['unused_budget_m2'], r['budget_area_m2'] - r['actual_area_m2'])
        require(r['domain_target_m2'] > 0 and r['domain_area_m2'] > 0, 'invalid common domain')
        if r['capture_pct'] is not None:
            close(r['capture_pct'], 100 * r['captured_target_m2'] / r['domain_target_m2'])
        else:
            require(allow_na, 'missing production capture')
        if r['policy'] == 'predicted_area':
            require(not r['fallback_used'] or not r['optimality_proven'], 'fallback cannot be certified')
    domains = {(r['domain_area_m2'], r['domain_target_m2']) for r in budgets}
    require(len(domains) == 1, 'different spatial evaluation domains')
    for r in auc:
        for metric in AUC_METRICS:
            value = number(r[metric], optional=allow_na)
            require(value is None or -1e-12 <= value <= 1 + 1e-12, 'AUC outside fraction bounds')
    return identity


def load_scalar_export(model, status_path, summary_path, sources):
    status, summary = load(status_path), load(summary_path)
    require(status.get('status', status.get('state')) == 'completed', 'CPU export incomplete')
    require(status['run_id'] == model['run_id'] == summary['run_id'], 'CPU run identity mismatch')
    require(status['factors'] == [1, 2, 4] and status['completed_windows'] == 357 and status['completed_factor_predictions'] == 1071,
            'CPU export does not contain full357 R1/2/4')
    require(status['test_accessed'] is False and status['R6_accessed'] is False, 'forbidden export scope')
    require(summary['device'] == 'cpu' and summary['precision'] == 'float32; autocast disabled', 'CPU FP32 summary required')
    require(summary['checkpoint_epoch'] == model['checkpoint_epoch'] and summary['checkpoint_sha256'] == model['checkpoint_sha256'],
            'CPU checkpoint differs from original selected spatial checkpoint')
    require(summary['manifest_sha256'] == 'bd1545bdc91499861a4b83c57d70b5024790bac6981ab3358c53268cc9d30851', 'wrong validation source')
    require(set(summary['scales']) == {'1', '2', '4'}, 'wrong scalar scales')
    rows = []
    for R in RS:
        s = summary['scales'][str(R)]
        require(s['n'] == 357 and s['n_positive'] == 27, 'wrong validation strata')
        close(s['balanced_mae'], (number(s['mae_zero']) + number(s['mae_positive'])) / 2, 1e-12)
        r = dict(model, R=R, n=357, n_positive=27, device='cpu', precision='float32', summary_path=str(summary_path),
                 summary_sha256=sources.files[Path(summary_path).resolve().as_posix()])
        for metric in SCALAR_METRICS:
            r[metric] = 100 * number(s[metric.removesuffix('_pp')])
        rows.append(r)
    cpu = statistics.fmean(r['balanced_mae_pp'] for r in rows)
    close(cpu, 100 * number(summary['selection_balanced_mae']), 1e-10)
    amp = 100 * number(summary['original_amp_validation']['selection_balanced_mae'])
    for r in rows:
        r.update(cpu_three_R_balanced_mae_pp=cpu, original_amp_selection_mae_pp=amp,
                 cpu_minus_original_amp_selection_pp=cpu - amp, epoch_reselected=False)
    return rows


def check_review_header(review, implementation_sha):
    require(review.get('passed') is True and review.get('execution_scope') == SCOPE, 'explicit full24 root review required')
    require(review.get('implementation_sha256') == implementation_sha, 'root review source mismatch')
    require(review.get('all24_required') is True and review.get('test_or_R6_accessed') is False, 'wrong summary scope')
    require(len(review.get('new_models', [])) == 16, 'all16 new completion/audit bindings required')
    require(len(review.get('seed17_exports', [])) == 8, 'all8 original CPU exports required')


def load_inputs(review_path):
    sources = Sources()
    review_path = sources.current(review_path)
    sources.current(__file__)
    review = load(review_path)
    check_review_header(review, sha(__file__))
    plan_path = sources.pin(review['plan'])
    require(review['plan']['sha256'] == PLAN_SHA, 'fixed16 plan mismatch')
    plan = load(plan_path)['runs']
    require(len(plan) == 16, 'wrong fixed seed plan')
    require([r['run_id'] for r in review['new_models']] == [r['run_id'] for r in plan], 'new models must be exact plan order')
    for entry in plan:
        require(entry['run_id'] in expected_models() and entry['seed'] in (29, 43), 'plan scope mismatch')
        require(all(entry[k] == expected_models()[entry['run_id']][k] for k in ('encoder', 'variant', 'lr', 'seed')), 'plan/config mismatch')
        require(entry['paired_seed17_run_id'] == run_id(entry['encoder'], entry['variant'], entry['lr'], 17), 'paired seed17 mismatch')
    old = {}
    for key, expected_sha in OLD_PINS.items():
        require(review['seed17'][key]['sha256'] == expected_sha, 'fixed original8 source mismatch')
        old[key] = sources.pin(review['seed17'][key])
    require(load(old['budget_audit'])['numerical_passed'] is True and load(old['auc_audit'])['passed'] is True, 'original8 audits not passed')
    budgets = [normalize_original(r) for r in read_csv(old['budget_csv'])]
    auc = []
    for row in read_csv(old['auc_csv']):
        r = model_fields(row)
        r.update(M=int(row['M']), R=int(row['R']), curve_points=int(row['curve_points']),
                 source_csv=str(old['auc_csv']), source_csv_sha256=OLD_PINS['auc_csv'])
        r.update({metric: number(row[metric]) for metric in AUC_METRICS})
        auc.append(r)
    require(len(budgets) == 720 and len(auc) == 72, 'wrong seed17 row counts')
    scalars, receipts = [], []
    old_models = {r['run_id']: model_fields(r) for r in budgets}
    require(len(old_models) == 8, 'wrong original8 model count')
    require(len({e['run_id'] for e in review['seed17_exports']}) == 8 and
            {e['run_id'] for e in review['seed17_exports']} == set(old_models), 'wrong original CPU export IDs')
    for entry in review['seed17_exports']:
        st, su, md = (sources.pin(entry[k]) for k in ('status', 'summary', 'metadata'))
        require(st.parent == su.parent == md.parent, 'mixed original export directories')
        require(load(st)['metadata_sha256'] == entry['metadata']['sha256'], 'original metadata binding mismatch')
        scalars.extend(load_scalar_export(old_models[entry['run_id']], st, su, sources))
    for entry, cfg in zip(review['new_models'], plan):
        rp = sources.pin(entry['completion_receipt'])
        rec = load(rp)
        require(rec['run_id'] == entry['run_id'] and rec['original_epoch_preserved'] is True and
                rec['all_declared_cases_retained'] is True and rec['test_or_R6_accessed'] is False, 'completion receipt scope mismatch')
        audit_path = sources.pin(entry['numerical_audit'])
        audit = load(audit_path)
        require(audit.get('passed') is True and audit.get('test_or_R6_accessed') is False, 'new numerical audit not passed')
        found = [m for m in audit['models'] if m['run_id'] == entry['run_id']]
        require(len(found) == 1 and found[0]['receipt_sha256'] == entry['completion_receipt']['sha256'], 'audit does not bind exact completion receipt')
        am = found[0]
        for key, value in {'maps': 357, 'scalar_pairs': 1071, 'management_units': 2183, 'prefix_curve_points': 6558,
                           'fixed_prefix_rows': 45, 'P_policy_rows': 45, 'auc_rows': 9}.items():
            require(am[key] == value, f'incomplete numerical audit: {key}')
        require(am['test_or_R6_accessed'] is False, 'wrong model audit scope')
        spatial = Path(rec['spatial_directory']).resolve()
        require(spatial.parent == PA and spatial.name.startswith(('lsm_', 'lsm2_')), 'unexpected spatial artifact root')
        sp = sources.pin({'path': str(spatial / 'status.json'), 'sha256': rec['spatial_status_sha256']})
        su = sources.pin({'path': str(spatial / 'summary.json'), 'sha256': rec['spatial_summary_sha256']})
        status, summary = load(sp), load(su)
        require(status['state'] == summary['state'] == 'completed' and status['run_id'] == entry['run_id'], 'spatial output incomplete')
        require(status['test_accessed'] is False and status['R6_accessed'] is False, 'wrong spatial scope')
        model = model_fields(summary)
        require(all(model[k] == cfg[k] for k in ('run_id', 'encoder', 'variant', 'lr', 'seed')), 'spatial model config mismatch')
        tables = {}
        for name in ('all_45_policy_outcomes.csv', 'prefix_auc.csv'):
            tables[name] = sources.pin({'path': str(spatial / name), 'sha256': status['files_sha256'][name]})
        wide = read_csv(tables['all_45_policy_outcomes.csv'])
        require(len(wide) == 45, 'incomplete new policy rows')
        for row in wide:
            budgets.extend(normalize_new(row, summary, tables['all_45_policy_outcomes.csv'], status['files_sha256']['all_45_policy_outcomes.csv']))
        new_auc = read_csv(tables['prefix_auc.csv'])
        require(len(new_auc) == 9, 'incomplete new AUC rows')
        for row in new_auc:
            require(row['method'] == 'model_prefix', 'unexpected AUC policy')
            r = dict(model, M=int(row['M']), R=int(row['R']), curve_points=int(row['curve_points']),
                     source_csv=str(tables['prefix_auc.csv']), source_csv_sha256=status['files_sha256']['prefix_auc.csv'])
            r.update({metric: number(row[metric]) for metric in AUC_METRICS})
            auc.append(r)
        export = Path(rec['export_directory']).resolve()
        st = sources.pin({'path': str(export / 'status.json'), 'sha256': rec['export_status_sha256']})
        manifest_path = sources.pin({'path': str(export / 'artifact_manifest.json'), 'sha256': rec['export_manifest_sha256']})
        manifest = load(manifest_path)['file_sha256']
        su = sources.pin({'path': str(export / 'summary.json'), 'sha256': manifest['summary.json']})
        scalars.extend(load_scalar_export(model, st, su, sources))
        receipts.append({'run_id': entry['run_id'], 'receipt': str(rp), 'receipt_sha256': entry['completion_receipt']['sha256'],
                         'audit': str(audit_path), 'audit_sha256': entry['numerical_audit']['sha256'],
                         'spatial_directory': str(spatial), 'export_directory': str(export)})
    models = validate_grids(budgets, auc, scalars)
    sources.verify_unchanged()
    return budgets, auc, scalars, models, receipts, sources


def difference(a, b):
    return None if a is None or b is None else a - b


def direction(value, epsilon):
    return 'NA' if value is None else ('higher' if value > epsilon else 'lower' if value < -epsilon else 'tied')


def pair_hard_soft(rows, dims, metrics):
    keys = ('encoder', 'lr', 'seed') + dims
    grouped = {}
    for r in rows:
        key = tuple(r[k] for k in keys)
        group = grouped.setdefault(key, {})
        require(r['variant'] not in group, 'duplicate hard/soft pair member')
        group[r['variant']] = r
    result = []
    for key, group in sorted(grouped.items()):
        require(set(group) == set(VARIANTS), 'incomplete hard/soft pair')
        h, s = group['hard_seg'], group['soft_seg']
        out = dict(zip(keys, key), hard_run_id=h['run_id'], soft_run_id=s['run_id'],
                   hard_checkpoint_epoch=h['checkpoint_epoch'], soft_checkpoint_epoch=s['checkpoint_epoch'])
        for metric in metrics:
            out['hard_' + metric], out['soft_' + metric] = h[metric], s[metric]
            d = difference(s[metric], h[metric])
            out['soft_minus_hard_' + metric] = d
            if metric in ('capture_pct',) + AUC_METRICS + ('balanced_mae_pp',):
                out[metric + '_direction'] = direction(d, AUC_EPS if metric in AUC_METRICS else PP_EPS)
        result.append(out)
    return result


def contrasts(rows, axis, other_dims, metrics):
    keys = MODEL_KEYS + other_dims
    grouped = {}
    for r in rows:
        key = tuple(r[k] for k in keys)
        group = grouped.setdefault(key, {})
        require(r[axis] not in group, 'duplicate contrast member')
        group[r[axis]] = r
    result = []
    for key, group in sorted(grouped.items()):
        require(set(group) == {1, 2, 4}, 'incomplete axis contrast')
        for high, low in CONTRASTS:
            out = dict(zip(keys, key), contrast=f'{axis}{high}_minus_{axis}{low}', comparison=high, reference=low)
            for metric in metrics:
                out['comparison_' + metric], out['reference_' + metric] = group[high][metric], group[low][metric]
                out['difference_' + metric] = difference(group[high][metric], group[low][metric])
            result.append(out)
    return result


def seed_statistics(rows, group_keys, metrics):
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        group = groups.setdefault(key, {})
        require(r['seed'] not in group, 'duplicate seed in descriptive group')
        group[r['seed']] = r
    output = []
    for key, group in sorted(groups.items()):
        require(set(group) == set(SEEDS), 'all three seed rows are required, including NA observations')
        r = dict(zip(group_keys, key), seeds='17;29;43', n_expected=3, independent_fields=1)
        for metric in metrics:
            values = [group[s][metric] for s in SEEDS if group[s][metric] is not None]
            r[metric + '_n_valid'] = len(values)
            r[metric + '_mean'] = statistics.fmean(values) if values else None
            r[metric + '_sample_sd'] = statistics.stdev(values) if len(values) >= 2 else None
        output.append(r)
    return output


def direction_counts(rows, group_keys, metric, epsilon):
    grouped = {}
    for r in rows:
        grouped.setdefault(tuple(r[k] for k in group_keys), []).append(r[metric])
    output = []
    for key, values in sorted(grouped.items()):
        ds = [direction(v, epsilon) for v in values]
        r = dict(zip(group_keys, key), metric=metric, rows=len(values), tolerance=epsilon)
        r.update({name: ds.count(name) for name in ('higher', 'lower', 'tied', 'NA')})
        finite = [v for v in values if v is not None]
        r.update(minimum=min(finite) if finite else None, maximum=max(finite) if finite else None,
                 interpretation='descriptive cell counts; not independent field replicates')
        output.append(r)
    return output


def make_tables(budgets, auc, scalars):
    validate_grids(budgets, auc, scalars, allow_na=True)
    paired = pair_hard_soft(budgets, ('M', 'R', 'budget', 'policy'), BUDGET_METRICS)
    paired_auc = pair_hard_soft(auc, ('M', 'R'), AUC_METRICS)
    paired_scalar = pair_hard_soft(scalars, ('R',), SCALAR_METRICS)
    require(len(paired) == 1080 and len(paired_auc) == 108 and len(paired_scalar) == 36, 'pair table count mismatch')
    pairs = [dict(encoder=e, lr=lr, seed=s, hard_run_id=run_id(e, 'hard_seg', lr, s), soft_run_id=run_id(e, 'soft_seg', lr, s))
             for e, lr, s in itertools.product(ENCODERS, LRS, SEEDS)]
    tables = {'all2160_model_policy_budgets': budgets, 'all216_model_prefix_auc': auc, 'all72_cpu_scalar_by_R': scalars,
              'all12_hard_soft_model_pairs': pairs, 'all1080_hard_soft_budget_pairs': paired,
              'all108_hard_soft_auc_pairs': paired_auc, 'all36_hard_soft_cpu_scalar_pairs': paired_scalar}
    if all('cpu_three_R_balanced_mae_pp' in r for r in scalars):
        tables['all24_cpu_scalar_model_summary'] = [dict(model_fields(r),
            cpu_three_R_balanced_mae_pp=r['cpu_three_R_balanced_mae_pp'],
            original_amp_selection_mae_pp=r['original_amp_selection_mae_pp'],
            cpu_minus_original_amp_selection_pp=r['cpu_minus_original_amp_selection_pp'],
            epoch_reselected=False) for r in scalars if r['R'] == 1]
    bases = (('budget', budgets, BUDGET_METRICS, ('budget', 'policy')),
             ('auc', auc, AUC_METRICS, ()))
    for name, rows, metrics, extras in bases:
        tables[f'{name}_three_seed_mean_sample_sd'] = seed_statistics(rows, ('encoder', 'variant', 'lr', 'M', 'R') + extras, metrics)
        pr = paired if name == 'budget' else paired_auc
        tables[f'hard_soft_{name}_three_seed_mean_sample_sd'] = seed_statistics(pr, ('encoder', 'lr', 'M', 'R') + extras,
                                                                             tuple('soft_minus_hard_' + m for m in metrics))
        for axis, other in (('R', 'M'), ('M', 'R')):
            c = contrasts(rows, axis, (other,) + extras, metrics)
            tables[f'all_{name}_{axis}_contrasts'] = c
            tables[f'{name}_{axis}_contrast_three_seed_mean_sample_sd'] = seed_statistics(
                c, ('encoder', 'variant', 'lr', other) + extras + ('contrast',), tuple('difference_' + m for m in metrics))
    tables['cpu_scalar_three_seed_mean_sample_sd'] = seed_statistics(scalars, ('encoder', 'variant', 'lr', 'R'), SCALAR_METRICS)
    tables['cpu_scalar_R_contrasts'] = contrasts(scalars, 'R', (), SCALAR_METRICS)
    tables['hard_soft_cpu_scalar_three_seed_mean_sample_sd'] = seed_statistics(
        paired_scalar, ('encoder', 'lr', 'R'), tuple('soft_minus_hard_' + m for m in SCALAR_METRICS))
    tables['budget_hard_soft_direction_counts'] = direction_counts(paired, ('encoder', 'lr', 'seed', 'policy', 'M'), 'soft_minus_hard_capture_pct', PP_EPS)
    tables['auc_hard_soft_direction_counts'] = [r for m in AUC_METRICS for r in direction_counts(paired_auc, ('encoder', 'lr', 'seed', 'M'), 'soft_minus_hard_' + m, AUC_EPS)]
    return tables


def execute(review_path):
    # All gates and input completeness pass BEFORE the one-shot claim.
    budgets, auc, scalars, models, receipts, sources = load_inputs(review_path)
    home = ROOT / 'seed_stability_summary_v1'
    home.mkdir(exist_ok=True)
    claim = home / 'EXECUTION_CLAIM.json'
    handle = claim.open('x', encoding='utf-8')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = PA / ('seed_summary_v1_' + stamp)
    failure = {'state': 'failed', 'output_directory': str(out), 'claim': str(claim)}
    try:
        with handle:
            json.dump({'scope': SCOPE, 'pid': os.getpid(), 'root_review': str(Path(review_path).resolve()),
                       'root_review_sha256': sha(review_path), 'implementation_sha256': sha(__file__),
                       'output_directory': str(out), 'created_at': datetime.now(timezone.utc).isoformat()}, handle, indent=2)
        out.mkdir(exist_ok=False)
        archive = out / 'source'
        archive.mkdir()
        ledger = []
        for i, (p, h) in enumerate(sources.files.items()):
            dest = archive / f'{i:03d}{Path(p).suffix}'
            shutil.copyfile(p, dest)
            require(sha(dest) == h, 'source copy mismatch')
            ledger.append({'path': p, 'sha256': h, 'archived': dest.relative_to(out).as_posix()})
        tables = make_tables(budgets, auc, scalars)
        for name, rows in tables.items():
            write_csv(out / (name + '.csv'), rows)
        dump(out / 'models.json', list(models.values()))
        dump(out / 'source_ledger.json', ledger)
        dump(out / 'completion_audit_bindings.json', receipts)
        summary = {'state': 'completed', 'scope': SCOPE, 'models': 24, 'seeds': list(SEEDS),
                   'table_rows': {name: len(rows) for name, rows in tables.items()}, 'primary_M': 2,
                   'sensitivity_M': [1, 4], 'cpu_scalar_source': 'saved full357 CPU FP32 export summaries; no epoch reselection',
                   'standard_deviation': 'sample SD (ddof=1) across three training seeds only',
                   'direction_tolerance_capture_pp': PP_EPS, 'direction_tolerance_auc': AUC_EPS,
                   'field_replicates': 1, 'test_or_R6_accessed': False, 'new_model_calls': 0, 'new_solver_calls': 0,
                   'scope_limit': 'One-scene development seed sensitivity; cells, budgets, policies and R/M levels are repeated evaluations, not fields.',
                   'recorded_flags': {flag: sum(r[flag] is True for r in budgets if r['policy'] == 'predicted_area') for flag in FLAGS}}
        dump(out / 'summary.json', summary)
        (out / 'report.md').write_text('# Full local development seed summary\n\n'
            'All 24 declared segmentation models, 2,160 budget outcomes, 216 complete-prefix AUC rows, and 72 CPU scalar rows are retained. '
            'The 2 m management unit is primary; 1 m and 4 m remain full sensitivity tables. '
            'R4−R1, R2−R1 and R4−R2, and the analogous management-unit contrasts, are provided without selecting favorable cells.\n\n'
            'The three seeds describe training variability within one scene. Sample SD uses n−1; missing values remain blank with n_valid. '
            'No significance test, field-generalization claim, model selection, new inference, new optimization or test/R6 evaluation was performed. '
            'Equal nominal budgets can spend different actual areas; both are retained. AUC is copied from audited complete staircases, never interpolated from five budgets. '
            'Recorded bound qualifications are inherited numerical evidence and are not new proofs of optimality.\n\n'
            'See source_ledger.json and completion_audit_bindings.json for exact source bytes and all 16 completion/audit gates. '
            'The final descriptive tables require a separate numerical output review before manuscript use.\n', encoding='utf-8')
        sources.verify_unchanged()
        dump(out / 'artifact_manifest.json', {'file_sha256': {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
        dump(out / 'status.json', {'state': 'completed', 'scope': SCOPE, 'source_sha256': sha(__file__),
                                   'root_review_sha256': sha(review_path), 'artifact_manifest_sha256': sha(out / 'artifact_manifest.json'),
                                   'table_rows': summary['table_rows'], 'independent_output_review': 'pending'})
        print(out)
    except BaseException:
        failure['traceback'] = traceback.format_exc()
        # Claim survives even when its JSON serialization or output mkdir fails.
        dump(home / ('FAILED_' + stamp + '.json'), failure)
        if out.is_dir():
            dump(out / 'status.json', failure)
        raise


def synthetic_tables():
    budgets, auc, scalar = [], [], []
    for rid, meta in expected_models().items():
        m = dict(meta, checkpoint_epoch=7, checkpoint_sha256='a' * 64)
        offset = (SEEDS.index(m['seed']) - 1) * 0.01
        for M, R, b, policy in itertools.product(MS, RS, BUDGETS, POLICIES):
            capture = b + offset + (0.01 if m['variant'] == 'soft_seg' else 0)
            area = b * 100
            r = dict(m, M=M, R=R, budget=b, policy=policy, budget_area_m2=area, actual_area_m2=area,
                     actual_area_fraction=b, capture_pct=100 * capture, captured_target_m2=capture * 10,
                     domain_area_m2=100., domain_target_m2=10., missed_target_m2=(1-capture)*10,
                     zero_annotation_area_m2=0., selected_zero_target_unit_area_m2=0., predicted_objective_m2=capture*9,
                     unused_budget_m2=0., selected_count=1)
            r.update({f: False for f in FLAGS})
            budgets.append(r)
        for M, R in itertools.product(MS, RS):
            auc.append(dict(m, M=M, R=R, auc_0_1=0.4+offset, auc_0_0p5_normalized=0.2+offset, curve_points=5))
        for R in RS:
            r = dict(m, R=R)
            r.update({k: 1. + offset for k in SCALAR_METRICS})
            r.update(cpu_three_R_balanced_mae_pp=1.+offset, original_amp_selection_mae_pp=1.+offset,
                     cpu_minus_original_amp_selection_pp=0.)
            scalar.append(r)
    return budgets, auc, scalar


def self_check():
    passed = []
    def yes(name, test):
        require(test, name)
        passed.append(name)
    def rejects(name, fn):
        try:
            fn()
        except (ValueError, KeyError):
            passed.append(name)
        else:
            raise AssertionError(name)
    b, a, s = synthetic_tables()
    tables = make_tables(b, a, s)
    for name, n in {'all2160_model_policy_budgets': 2160, 'all216_model_prefix_auc': 216, 'all72_cpu_scalar_by_R': 72,
                    'all12_hard_soft_model_pairs': 12, 'all1080_hard_soft_budget_pairs': 1080, 'all108_hard_soft_auc_pairs': 108,
                    'all24_cpu_scalar_model_summary': 24,
                    'all_budget_R_contrasts': 2160, 'all_budget_M_contrasts': 2160,
                    'all_auc_R_contrasts': 216, 'all_auc_M_contrasts': 216,
                    'budget_three_seed_mean_sample_sd': 720, 'auc_three_seed_mean_sample_sd': 72}.items():
        yes(name, len(tables[name]) == n)
    yes('all paired signs positive at exact 1pp', all(math.isclose(x['soft_minus_hard_capture_pct'], 1., abs_tol=1e-12) for x in tables['all1080_hard_soft_budget_pairs']))
    yes('sample SD uses n-1', math.isclose(tables['budget_three_seed_mean_sample_sd'][0]['capture_pct_sample_sd'], 1., abs_tol=1e-12))
    toy = [{'seed': s0, 'group': 1, 'x': x} for s0, x in zip(SEEDS, (None, 2., 4.))]
    stat = seed_statistics(toy, ('group',), ('x',))[0]
    yes('NA not imputed', stat['x_n_valid'] == 2 and stat['x_mean'] == 3 and math.isclose(stat['x_sample_sd'], math.sqrt(2)))
    yes('NA difference propagates', difference(None, 3) is None and difference(2, None) is None)
    stat = seed_statistics([dict(seed=s0, group=1, x=None) for s0 in SEEDS], ('group',), ('x',))[0]
    yes('all NA mean SD remain NA', stat['x_mean'] is None and stat['x_sample_sd'] is None and stat['x_n_valid'] == 0)
    yes('one observation SD remains NA', seed_statistics([dict(seed=s0, group=1, x=1 if s0==17 else None) for s0 in SEEDS], ('group',), ('x',))[0]['x_sample_sd'] is None)
    dc = direction_counts([{'x': x} for x in (None, -2e-8, 0, 2e-8, PP_EPS)], (), 'x', PP_EPS)[0]
    yes('direction denominators and tolerance', [dc[k] for k in ('rows', 'higher', 'lower', 'tied', 'NA')] == [5,1,1,2,1])
    rejects('missing model/grid rejected', lambda: validate_grids(b[:-90], a, s))
    rejects('duplicate replaces missing rejected', lambda: validate_grids(b[:-1] + [b[0]], a, s))
    rejects('missing hard/soft member rejected', lambda: pair_hard_soft([r for r in a if r['variant']=='hard_seg'], ('M','R'), AUC_METRICS))
    rejects('missing seed not silently summarized', lambda: seed_statistics(toy[:2], ('group',), ('x',)))
    rejects('nonfinite metric rejected', lambda: number(float('nan')))
    bad = dict(b[0], actual_area_m2=b[0]['budget_area_m2']+1e-12)
    rejects('strict budget has no permissive tolerance', lambda: validate_grids([bad]+b[1:], a, s))
    rejects('no implicit production review', lambda: check_review_header({}, sha(__file__)))
    fake = dict(passed=True, execution_scope=SCOPE, implementation_sha256=sha(__file__), all24_required=True,
                test_or_R6_accessed=False, new_models=[{}]*15, seed17_exports=[{}]*8)
    rejects('15 completion bindings rejected before production', lambda: check_review_header(fake, sha(__file__)))
    bad_a = [dict(a[0], checkpoint_epoch=9)] + a[1:]
    rejects('checkpoint drift between scalar/space rejected', lambda: validate_grids(b, bad_a, s))
    c = contrasts([dict(model_fields(a[0]), R=R, x=float(R)) for R in RS], 'R', (), ('x',))
    yes('all signed R differences correct', {r['contrast']:r['difference_x'] for r in c} == {'R4_minus_R1':3.,'R2_minus_R1':1.,'R4_minus_R2':2.})
    out = ROOT / 'provenance' / ('seed_summary_preparation_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    shutil.copyfile(__file__, out / Path(__file__).name)
    dump(out / 'focused_checks.json', {'passed': True, 'stage': 'synthetic preparation only', 'checks': passed,
         'count': len(passed), 'source_sha256': sha(__file__), 'production_executed': False, 'empirical_outputs_read': False,
         'test_or_R6_accessed': False, 'model_calls':0, 'solver_calls':0, 'synthetic_table_counts':{k:len(v) for k,v in tables.items()}})
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-check', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--root-review', type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        require(args.root_review is not None, '--execute needs exact root-review')
        execute(args.root_review)
