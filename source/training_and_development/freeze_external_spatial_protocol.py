"""Freeze the reviewed metadata-only external spatial design before spatial outcomes."""
from pathlib import Path
import collections, datetime, hashlib, json, shutil

ROOT = Path(__file__).resolve().parent
DRAFT = ROOT/'external_weeds_v1/spatial_preparation_20260916_062041/spatial_protocol_draft.json'
OUT = ROOT/'external_weeds_v1/spatial_v1'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def main():
    assert sha(DRAFT) == '90ec5f204f41832b29d2992e6f0ccb9dc861e5b61a5f5996ba42348ee4e34d0e'
    p = read(DRAFT)
    for rel, digest in p['basis_source_sha256'].items():
        assert sha(ROOT/rel) == digest, rel
    val = read(ROOT/'weedsgalore_data_v1/val.json')
    assert p['dataset']['frame_ids_in_official_order'] == [x['id'] for x in val]
    assert len(val) == 26 and len({x['id'] for x in val}) == 26
    metadata = p['dataset']['frame_metadata']
    assert collections.Counter(x['acquisition_date'] for x in metadata) == p['dataset']['date_counts_from_metadata']
    assert { (x['variant'], x['seed']) for x in p['eligible_models']} == {(v,s) for v in ('hard_seg','soft_seg') for s in (17,29,43)}
    assert len(p['eligible_models']) == 6 and p['grids']['primary_g'] == 8
    for m, v in zip(metadata, val):
        assert (m['id'],m['cache'],m['cache_sha256']) == (v['id'],v['cache'],v['cache_sha256'])
    for g in p['grids']['definitions']:
        assert g['g'] in (4,8,16) and g['N'] == g['g']**2
        assert g['side_pixels']*g['g'] == 256 and g['cell_area_FAE'] == 1/g['N']
    plans = p['budget_plan']+p['matched_actual_budget_plan']
    assert len(plans) == 30
    for row in plans:
        n = row['grid_cells_per_axis']**2
        if row['budget_basis'] == 'nominal_primary':
            k = n*row['budget_numerator']//row['budget_denominator']
        else:
            assert row['anchor_denominator'] == 16
            assert row['anchor_numerator'] == 16*row['parent_nominal_numerator']//row['parent_nominal_denominator']
            assert n*row['anchor_numerator']%16 == 0
            k = n*row['anchor_numerator']//16
        assert k == row['selected_cell_count'] and k/n == row['actual_frame_area_equivalents']
    unique = {(r['grid_cells_per_axis'],r['selected_cell_count']) for r in plans}
    assert len(unique) == 20
    c = p['planned_counts_not_observed_results']
    expected = {'prediction_unit_rows':6*3*26*(16+64+256),
        'learned_complete_rankings':6*3*26*3,'learned_frame_budget_rows':6*3*26*30,
        'unique_learned_frame_model_R_grid_k_choices':6*3*26*20,
        'learned_pooled_budget_rows':6*3*30,'learned_date_budget_rows':6*3*4*30,
        'hard_soft_paired_pooled_budget_rows':3*3*30,
        'learned_frame_complete_prefix_points_including_zero':6*3*26*(17+65+257),
        'learned_pooled_complete_prefix_points_including_zero':6*3*(17+65+257),
        'learned_date_complete_prefix_points_including_zero':6*3*4*(17+65+257),
        'unique_truth_unit_rows_after_selection_seal':26*(16+64+256)}
    for key, value in expected.items():
        assert c[key] == value, key
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    p['version'] = 'external_weeds_relative_grid_spatial_v1'
    p['status'] = 'frozen_development_protocol_before_spatial_evaluation'
    p['frozen_utc'] = stamp
    p['derived_from'] = {'path':str(DRAFT),'sha256':sha(DRAFT),'draft_preserved':True}
    p['root_review'] = {'accepted':True,'reviewed_sections':'All eight narrative sections and JSON design; exact budgets, matched-area controls, selection isolation, truth, references, AUC and counts.',
        'scope':'Protocol frozen for implementation. Production source requires separate review and complete independently accepted CPU export before execution.',
        'timing':'All nine scalar training outcomes are known; no external spatial outcomes or prediction maps inspected by this freeze. Not preregistration of training.',
        'read_only_metadata_validation':True,'array_read':False,'inference':False,'spatial_selection':False,'test_accessed':False,'R6_accessed':False}
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT/'protocol.json',p)
    shutil.copy2(DRAFT,OUT/'reviewed_draft.json')
    shutil.copy2(DRAFT.with_suffix('.md'),OUT/'reviewed_draft.md')
    shutil.copy2(__file__,OUT/'freeze_protocol_source.py')
    receipt = {'created_utc':stamp,'passed':True,'protocol_sha256':sha(OUT/'protocol.json'),
        'metadata_checks':expected,'unique_grid_k_selections':20,
        'files_sha256':{x.name:sha(x) for x in OUT.iterdir() if x.is_file()},
        'no_array_access':True,'no_inference':True,'no_spatial_evaluation':True}
    write(OUT/'freeze_receipt.json',receipt)
    print(json.dumps({'protocol':str(OUT/'protocol.json'),'sha256':sha(OUT/'protocol.json'),'receipt_sha256':sha(OUT/'freeze_receipt.json')},ensure_ascii=False))

if __name__ == '__main__':
    main()
