"""Presentation-only summaries of independently accepted confirmation tables.

No models, solvers, selection, pixels or arrays are loaded. Existing outputs are
never overwritten. All consumed report artifacts are checked against its manifest.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PA = ROOT / 'precision_agriculture'
REPORT = ROOT / 'confirmation_v1/stages/report_tables'
REVIEW = ROOT / 'provenance/completed_confirmation_report_independent_20260916_142830_724677/review.json'
REVIEW_SHA = '3af89351cf4a813bae16c6c6ba82e7334e05be4aa048a49707ad4f521f57bed8'
BUDGETS = (.1, .2, .3, .5, 1.)
TOL_PP = 1e-9


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jread(path):
    return json.loads(path.read_text(encoding='utf-8'))


def n(x):
    return None if x is None or x == '' else float(x)


def ranges(rows, field, factor=1):
    values = [n(x[field]) for x in rows]
    defined = [x * factor for x in values if x is not None]
    return {'n': len(rows), 'n_defined': len(defined), 'n_undefined': len(rows)-len(defined),
            'min': min(defined) if defined else None, 'max': max(defined) if defined else None}


def directions(rows):
    d = ranges(rows, 'difference_capture_pct')
    z = [n(x['difference_capture_pct']) for x in rows]
    d.update(higher=sum(x is not None and x > TOL_PP for x in z),
             lower=sum(x is not None and x < -TOL_PP for x in z),
             tied=sum(x is not None and abs(x) <= TOL_PP for x in z))
    assert d['higher']+d['lower']+d['tied']+d['n_undefined'] == len(rows)
    return d


def fmt(x, places=2):
    if x is None:
        return 'NA'
    if abs(x) < .5*10**(-places):
        x = 0
    return f'{x:.{places}f}'


def span(x, places=2):
    if x['n_defined'] == 0:
        return 'NA'
    if x['max']-x['min'] <= TOL_PP:
        return fmt(x['min'], places)
    return f"{fmt(x['min'], places)}–{fmt(x['max'], places)}"


def mdtable(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def main():
    destinations = [PA/'PA_main_tables_confirmation_v5.md', PA/'PA_main_tables_confirmation_v5.json']
    assert all(not p.exists() for p in destinations), 'refuse overwrite'
    assert sha(REVIEW) == REVIEW_SHA
    review = jread(REVIEW)
    assert review['passed'] and review['scope'] == 'confirmation_report_tables_independent'
    assert review['stage'] == 'report_tables'
    assert sha(REPORT/'artifact_manifest.json') == review['audited_artifact_manifest_sha256']
    assert sha(REPORT/'summary.json') == review['summary_sha256']
    manifest = jread(REPORT/'artifact_manifest.json')['files']
    pins = {}

    def read(relative):
        p = REPORT/relative
        digest = sha(p)
        assert digest == manifest[relative], relative
        pins[relative] = {'path': str(p), 'sha256': digest}
        if relative.endswith('.csv'):
            with p.open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            pins[relative]['rows'] = len(rows)
            return rows
        result = jread(p)
        if isinstance(result, list):
            pins[relative]['rows'] = len(result)
        return result

    summary = read('summary.json')
    assert summary['local_spatial_models'] == 24 and summary['external_spatial_models'] == 6
    assert summary['R'] == [1,2,4,6]
    local = read('local/spatial_budget_all.csv')
    local_scalar = read('local/scalar_all.csv')
    weeds_scalar = read('weeds/scalar_all.csv')
    mc = read('local/spatial_budget_M_contrasts.csv')
    rc = read('local/spatial_budget_R_contrasts.csv')
    external = read('weeds/pooled_spatial_budget_all.csv')
    oracle = read('weeds/label_oracle_pooled_budget_metrics.csv')
    random_expected = read('weeds/random_expectation_pooled_budget_metrics.csv')
    dates = read('weeds/label_oracle_date_budget_metrics.csv')
    random_orders = read('local/shared_random_budgets.json')
    random_saved = read('local/shared_random_summary.json')
    label_prefix = read('local/shared_label_prefix_budgets.json')
    label_milp = read('local/shared_label_MILP_budgets.json')
    shared_summary = read('local/shared_summary.json')
    for file, rows in [('local/spatial_budget_all.csv',local),('local/spatial_budget_M_contrasts.csv',mc),
                       ('local/spatial_budget_R_contrasts.csv',rc),('weeds/pooled_spatial_budget_all.csv',external)]:
        assert len(rows) == summary['table_rows'][file]
    assert len({x['run_id'] for x in local}) == 24
    assert len({x['run_id'] for x in external}) == 6
    assert len({(x['run_id'],x['M'],x['R'],x['budget'],x['policy']) for x in local}) == 2880
    domain = float(local[0]['domain_area_m2'])
    target = float(local[0]['domain_target_m2'])
    assert all(abs(float(x['domain_area_m2'])-domain)<1e-8 and abs(float(x['domain_target_m2'])-target)<1e-8 for x in local)
    assert {(x['n'],x['n_zero'],x['n_positive']) for x in local_scalar} == {('338','284','54')}
    assert {(x['n'],x['n_zero'],x['n_positive']) for x in weeds_scalar} == {('26','0','26')}

    def ext_filter(rows, b):
        return [x for x in rows if x['grid_g']=='8' and x['budget_basis']=='nominal_primary'
                and float(x['parent_nominal_budget_fraction']) == b]

    date_rows = sorted(ext_filter(dates,.1), key=lambda x:x['scope_id'])
    assert len(date_rows)==4 and sum(int(x['frame_n']) for x in date_rows)==26
    e0 = ext_filter(oracle,.1)[0]
    external_weed = float(e0['total_weed_FAE'])
    external_crop = float(e0['total_crop_FAE'])
    assert abs(math.fsum(float(x['total_weed_FAE']) for x in date_rows)-external_weed)<1e-12
    assert abs(math.fsum(float(x['total_crop_FAE']) for x in date_rows)-external_crop)<1e-12
    t2 = {'local': {'windows':338,'domain_area_m2':domain,'target_area_m2':target,'zero_target_windows':284,
                    'positive_target_windows':54,'crop_area_m2':None,'date_groups':None,
                    'undefined_reason':'No crop annotation or verified acquisition-date grouping for this local analysis'},
          'external': {'frames':26,'support_FAE':26,'weed_FAE':external_weed,'crop_FAE':external_crop,
                       'zero_weed_frames':int(e0['zero_weed_frame_n']),'zero_crop_frames':int(e0['zero_crop_frame_n']),
                       'dates':[{'date':x['scope_id'],'frames':int(x['frame_n']),
                                 'weed_FAE':float(x['total_weed_FAE']),'crop_FAE':float(x['total_crop_FAE']),
                                 'positive_weed_frames':int(x['weed_eligible_frame_n']),
                                 'positive_crop_frames':int(x['crop_eligible_frame_n'])} for x in date_rows]}}
    t2_rows = [['Local confirmation',338,f'{domain:.4f} m²',f'{target:.4f} m²','Not annotated','54 / 338','Not annotated'],
               ['External: all dates',26,'26 FAE',f'{external_weed:.6f} FAE',f'{external_crop:.6f} FAE','26 / 26','24 / 26']]
    for x in t2['external']['dates']:
        t2_rows.append([x['date'],x['frames'],f"{x['frames']} FAE",f"{x['weed_FAE']:.6f} FAE",
                        f"{x['crop_FAE']:.6f} FAE",f"{x['positive_weed_frames']} / {x['frames']}",
                        f"{x['positive_crop_frames']} / {x['frames']}"])

    t3, t3_rows = [], []
    for b in BUDGETS:
        q=[x for x in local if x['M']=='2' and x['policy']=='prefix' and float(x['budget'])==b]
        assert len(q)==96 and len({(x['run_id'],x['R']) for x in q})==96
        ro=[x for x in random_orders if x['M']==2 and x['budget']==b]
        assert len(ro)==1000 and len({x['repetition'] for x in ro})==1000
        saved=next(x for x in random_saved if x['M']==2 and x['budget']==b)
        lp=next(x for x in label_prefix if x['M']==2 and x['budget']==b)
        lm=next(x for x in label_milp if x['M']==2 and x['budget']==b)
        random_stats={key:{'mean':statistics.mean(x[field]*factor for x in ro),
                           'sample_sd':statistics.stdev(x[field]*factor for x in ro)}
                      for key,field,factor in [('capture_pct','capture_fraction',100),('actual_area_m2','actual_area_m2',1)]}
        assert abs(random_stats['capture_pct']['mean']-100*saved['capture_fraction_mean'])<1e-10
        rec={'budget':b,'budget_area_m2':b*domain,'learned':{k:ranges(q,k) for k in
             ['capture_pct','actual_area_m2','actual_area_fraction','captured_target_m2','missed_target_m2']},
             'learned_by_R_population':{name:{k:ranges([x for x in q if int(x['R']) in rs],k) for k in
                 ['capture_pct','actual_area_m2','missed_target_m2']} for name,rs in [('original_R124',{1,2,4}),('additional_R6',{6})]},
             'random_order':{'n':1000,**random_stats},
             'label_prefix':{k:v for k,v in lp.items() if k!='selected_ids'},
             'label_MILP':{k:v for k,v in lm.items() if k!='selected_ids'}}
        rec['label_MILP']['upper_capture_pct']=100*lm['updated_upper_m2']/target
        t3.append(rec)
        btxt=str(round(100*b))
        t3_rows.append([btxt,'Learned prefix (96)',span(rec['learned']['actual_area_m2']),span(rec['learned']['capture_pct']),'—'])
        t3_rows.append([btxt,'Random orders (1000)',f"{fmt(random_stats['actual_area_m2']['mean'])} ± {fmt(random_stats['actual_area_m2']['sample_sd'])}",
                        f"{fmt(random_stats['capture_pct']['mean'])} ± {fmt(random_stats['capture_pct']['sample_sd'])}",'—'])
        t3_rows.append([btxt,'Label prefix (1)',fmt(lp['actual_area_m2']),fmt(100*lp['capture_fraction']),'—'])
        t3_rows.append([btxt,'Label MILP (1)',fmt(lm['actual_area_m2']),fmt(100*lm['capture_fraction']),
                        fmt(rec['label_MILP']['upper_capture_pct'])+('; met' if lm['updated_proven_flag'] else '; not met')])

    t4, t4_rows = [], []
    for name,rows,extra in [('M4_minus_M2',mc,lambda x:True),('R4_minus_R1',rc,lambda x:x['M']=='2')]:
        for pol in ['prefix','predicted_area']:
            for b in BUDGETS:
                q=[x for x in rows if x['contrast']==name and x['policy']==pol and float(x['budget'])==b and extra(x)]
                assert len(q)==(96 if name.startswith('M') else 24)
                # Verify the saved contrast against the two accepted budget rows.
                lookup={(x['run_id'],int(x['M']),int(x['R']),float(x['budget']),x['policy']):float(x['capture_pct']) for x in local}
                for x in q:
                    if name.startswith('M'):
                        a=lookup[(x['run_id'],4,int(x['R']),b,pol)]; z=lookup[(x['run_id'],2,int(x['R']),b,pol)]
                    else:
                        a=lookup[(x['run_id'],2,4,b,pol)]; z=lookup[(x['run_id'],2,1,b,pol)]
                    assert abs(a-z-float(x['difference_capture_pct']))<1e-12
                rec={'contrast':name,'policy':pol,'budget':b,**directions(q)}
                if name.startswith('M'):
                    rec['by_R_population']={name:directions([x for x in q if int(x['R']) in rs]) for name,rs in
                                            [('original_R124',{1,2,4}),('additional_R6',{6})]}
                t4.append(rec)
                t4_rows.append(['M4 − M2; all R' if name.startswith('M') else 'R4 − R1; M2',
                                'Prefix' if pol=='prefix' else 'P-policy',round(100*b),len(q),
                                f"{rec['higher']} / {rec['lower']} / {rec['tied']}",span(rec)])

    t5, t5_rows = [], []
    for b in BUDGETS:
        q=ext_filter(external,b);assert len(q)==24 and len({(x['run_id'],x['R']) for x in q})==24
        oo=ext_filter(oracle,b); rr=ext_filter(random_expected,b);assert len(oo)==len(rr)==1
        oo,rr=oo[0],rr[0]
        area=float(q[0]['selected_area_FAE']); fraction=float(q[0]['actual_fraction'])
        assert all(float(x['selected_area_FAE'])==area and float(x['actual_fraction'])==fraction for x in q+[oo,rr])
        assert abs(area-26*fraction)<1e-12
        rec={'budget':b,'k_per_frame':int(q[0]['k']),'actual_fraction':fraction,'selected_area_FAE':area,
             'learned':{k:ranges(q,k) for k in ['weed_capture_pct','crop_capture_pct','crop_fraction_pct']},
             'learned_by_R_population':{name:{k:ranges([x for x in q if int(x['R']) in rs],k) for k in
                ['weed_capture_pct','crop_capture_pct','crop_fraction_pct']} for name,rs in [('original_R124',{1,2,4}),('additional_R6',{6})]},
             'label_topk':oo,'random_expectation':rr}
        t5.append(rec)
        t5_rows.append([round(100*b),'Learned (24)',fmt(area,5),fmt(100*fraction,4),
                        span(rec['learned']['weed_capture_pct']),span(rec['learned']['crop_capture_pct']),span(rec['learned']['crop_fraction_pct'])])
        for name,x in [('Random expectation',rr),('Label top-k',oo)]:
            t5_rows.append([round(100*b),name,fmt(area,5),fmt(100*fraction,4),
                            fmt(100*float(x['weed_capture_ratio'])),fmt(100*float(x['crop_capture_ratio'])),
                            fmt(100*float(x['crop_fraction_of_selected_area']))])

    def contrast(name,pol,b):
        return next(x for x in t4 if x['contrast']==name and x['policy']==pol and x['budget']==b)
    p10,p20,p50=[contrast('M4_minus_M2','prefix',b) for b in [.1,.2,.5]]
    pp10,pp20,pp50=[contrast('M4_minus_M2','predicted_area',b) for b in [.1,.2,.5]]
    r10=contrast('R4_minus_R1','prefix',.1)
    statements=[
        {'id':'K1','table':2,'text':f'The local reserved partition contained 338 windows and {domain:.4f} m² of valid area, including {target:.4f} m² of annotated target. Target coverage was zero in 284 windows and positive in 54. The external confirmation set contained 26 frames across four dates (8, 8, 8 and 2 frames), with {external_weed:.6f} weed and {external_crop:.6f} crop frame-area equivalents (FAE). Two external frames had no crop annotation.'},
        {'id':'K2','table':3,'text':f'At the 10% local M2 budget, the complete set of 24 segmentation fits at R1/2/4/6 captured {span(t3[0]["learned"]["capture_pct"])}% of annotated target while selecting {span(t3[0]["learned"]["actual_area_m2"])} m². The saved random-order reference captured {fmt(t3[0]["random_order"]["capture_pct"]["mean"])} ± {fmt(t3[0]["random_order"]["capture_pct"]["sample_sd"])}% (mean ± sample SD across 1000 orders). Label-prefix and label-MILP captures were {fmt(100*t3[0]["label_prefix"]["capture_fraction"])}% and {fmt(100*t3[0]["label_MILP"]["capture_fraction"])}%, respectively, at their different reported actual areas.'},
        {'id':'K3','table':4,'text':f'M4−M2 capture was lower in {p10["lower"]}/96 prefix comparisons at 10% and {p20["lower"]}/96 at 20%, but higher in {p50["higher"]}/96 at 50% ({p50["lower"]} lower and {p50["tied"]} tied). The P-policy showed {pp10["lower"]}/96 and {pp20["lower"]}/96 decreases at 10% and 20%, followed by {pp50["higher"]}/96 increases at 50%. These counts include 72 comparisons at R1/2/4 and 24 additional comparisons at R6; the JSON retains both groups separately.'},
        {'id':'K4','table':4,'text':f'For the primary local M2 prefix policy at 10%, R4−R1 capture increased in {r10["higher"]} of 24 fits and decreased in {r10["lower"]}, with changes spanning {span(r10)} percentage points. R4−R1 therefore cannot be described as a universal loss from degradation. The table retains all five budgets and both policies.'},
        {'id':'K5','table':5,'text':f'For external g8 at the nominal 10% budget, all 24 fit×R settings selected 2.4375 FAE (9.375% of every frame). Their pooled weed capture was {span(t5[0]["learned"]["weed_capture_pct"])}%, crop capture {span(t5[0]["learned"]["crop_capture_pct"])}%, and crop composition of selected area {span(t5[0]["learned"]["crop_fraction_pct"])}%. The saved random expectation and exact equal-cost label top-k weed references were {fmt(100*float(t5[0]["random_expectation"]["weed_capture_ratio"]))}% and {fmt(100*float(t5[0]["label_topk"]["weed_capture_ratio"]))}%. Crop inclusion is a spatial proxy, not measured crop injury.'}
    ]
    notes = [
        'Only accepted confirmation outputs are used. No checkpoint was reselected and no model, solver, selection or annotation array was executed or read.',
        'Ranges retain every declared fit and R condition; they are not confidence intervals. Local learned ranges have 96 conditions per budget (24 fits × four R levels); external ranges have 24 (six fits × four R levels). These are dependent conditions within their respective scenes.',
        'R1/2/4 were the original development information factors. R6 is the additional frozen confirmation condition with a 42-pixel intermediate image. It is included explicitly, not substituted for an original R.',
        'All five nominal budgets are reported. A common nominal budget is an area cap, not equality of actual selected area. Local references and learned policies may use different actual areas.',
        'Undefined denominators remain NA. No pooled ratio in these four tables is undefined; two external frame-level crop capture denominators are zero and remain NA in the complete frame tables. Local crop metrics are unavailable, not zero.',
        'Direction counts use ±1e-9 percentage points on unrounded saved contrasts. The lower/upper endpoints and exact counts are retained in JSON; rounded 0.00 does not redefine a tie.',
        'Random-order sample SD is calculated from the 1000 previously saved orders. It is neither fitting-seed SD nor a field confidence interval. No new random order is generated.',
        'M2 label-MILP updated gap flags are met for all five budgets; this describes the accepted saved numerical bounds for an annotation problem. No minimum selected-area or biological-control optimum is claimed.',
        'External FAE counts image support separately per frame, without geographic mosaicking or cross-frame budget reallocation. Neither support nor absolute capture is pooled across datasets.',
        'External label top-k maximizes annotated weed at equal unit cost; its crop values describe those choices and were not optimized. Lower crop inclusion alone is not evidence of better agronomic performance.'
    ]
    md=['# Confirmation main tables 2–5 and source-backed result statements',
        '\nPrepared from the independently accepted confirmation report. Table numbers follow the fixed final-integration plan; Table 1 is maintained separately. Values are descriptive, and all declared conditions are retained.',
        '\n## Table 2',
        '\nTable 2. Confirmation support and annotation denominators.',
        '\n'+mdtable(['Dataset / date group','Windows / frames','Observed support','Target / weed area','Crop area','Target-positive / total','Crop-positive / total'],t2_rows),
        '\nLocal quantities use projected observed area in m². External quantities use frame-area equivalents (FAE), with one complete frame equal to one FAE. Date rows partition the external total; they are not additional samples. The local partition is from one scene, without a verified acquisition-date grouping for this analysis. Local crop annotation was unavailable. Two external crop-free frames retain undefined individual crop-capture ratios; pooled crop capture uses the positive pooled denominator.',
        '\n## Table 3',
        '\nTable 3. Local M2 prefix results and shared references across all nominal budgets.',
        '\n'+mdtable(['Budget (%)','Policy (number summarized)','Actual area (m²)','Target capture (%)','MILP upper (%) / gap flag'],t3_rows),
        '\nLearned rows show the complete minimum–maximum over 24 fits × R1/2/4/6. Random rows show mean ± sample SD across 1000 saved orders. Label rows are one shared reference per budget, not repeated model observations. “Met” means the saved updated gap flag met the requested 1e−8 tolerance; achieved values and bounds are displayed together. At budgets ≥20%, the saved label-MILP solution captured all annotated target using 223.8580 m²; no minimum-area claim follows. Label prefix orders by annotated target density, whereas label MILP maximizes captured annotation subject to the cap. The equal nominal cap does not imply equal actual area.',
        '\n## Table 4',
        '\nTable 4. Complete paired local capture contrasts across all five budgets.',
        '\n'+mdtable(['Contrast / population','Policy','Budget (%)','Pairs','Higher / lower / tied','Change range (pp)'],t4_rows),
        '\nPositive changes mean capture increased for the first condition in the contrast. M4−M2 includes all four R levels: 72 original-R1/2/4 pairs plus 24 additional-R6 pairs per budget and policy. R4−R1 is evaluated at M2 for all 24 fits and does not involve R6. Prefix is primary; P-policy is the secondary predicted-area knapsack policy with the accepted feasibility/bound handling. Counts are dependent within-scene model-condition pairs, not independent fields. Ties use ±1e−9 pp before rounding. All subgroup counts and unrounded ranges are provided in the companion JSON.',
        '\n## Table 5',
        '\nTable 5. External g8 nominal budgets: image area, weed capture and crop inclusion.',
        '\n'+mdtable(['Budget (%)','Policy','Selected area (FAE)','Actual area (%)','Weed capture (%)','Crop capture (%)','Crop composition (%)'],t5_rows),
        '\nLearned rows show complete ranges across six fits × R1/2/4/6; reference rows are shared and not replicated across fits. The same integer number of g8 cells is selected in every frame: 6, 12, 19, 32 and 64 at the five nominal budgets. Capture divides selected class area by total class area, whereas crop composition divides selected crop area by selected image area. Pooled ratios sum class areas before division. Random expectation is analytic; label top-k is exact for annotated weed under the equal-cost per-frame constraint and does not optimize crop inclusion. This table uses nominal budgets; complete matched-actual and alternate-grid results remain in the accepted supplementary tables.',
        '\n## Source-backed result statements',
        '\n'+'\n\n'.join(f"**{x['id']} (Table {x['table']}).** {x['text']}" for x in statements),
        '\n## Units, limits and completeness',
        '\n'+'\n'.join('- '+x for x in notes),
        '\n## Provenance',
        '\nThe companion JSON records exact source hashes, filters, row counts, unrounded values, undefined counts, reference qualifications and the accepted review. The main source is `confirmation_v1/stages/report_tables`; no development value substitutes for a confirmation result.',
        '\n- Table 2: local `spatial_budget_all.csv` and `scalar_all.csv`; external `scalar_all.csv`, `label_oracle_pooled_budget_metrics.csv` and `label_oracle_date_budget_metrics.csv`. Denominators are verified as invariant across their applicable model/condition rows.',
        '\n- Table 3: local `spatial_budget_all.csv`, `shared_random_budgets.json`, `shared_random_summary.json`, `shared_label_prefix_budgets.json` and `shared_label_MILP_budgets.json`; M=2, all five budgets, learned policy=prefix. Random sample SD uses the saved 1000 values with denominator 999.',
        '\n- Table 4: local `spatial_budget_M_contrasts.csv` and `spatial_budget_R_contrasts.csv`; exact filters are M4−M2 across all R, and R4−R1 at M2, both policies and all five budgets. Saved differences are cross-checked against `spatial_budget_all.csv`.',
        '\n- Table 5: external `pooled_spatial_budget_all.csv`, `random_expectation_pooled_budget_metrics.csv` and `label_oracle_pooled_budget_metrics.csv`; g=8, budget_basis=nominal_primary, all five budgets. Percentages are 100 × saved fractions.',
        '']
    markdown='\n'.join(md)
    assert sum(line.startswith('| ---') for line in markdown.splitlines())==4
    ledger={'version':'confirmation_main_tables_v5','created_at':datetime.now(timezone.utc).isoformat(),
            'status':'presentation summaries of independently accepted confirmation outputs',
            'report_acceptance':{'path':str(REVIEW),'sha256':REVIEW_SHA,'scope':review['scope'],
                                 'freeze_sha256':review['freeze_sha256'],
                                 'artifact_manifest_sha256':review['audited_artifact_manifest_sha256'],
                                 'summary_sha256':review['summary_sha256']},
            'implementation':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))},
            'integration_plan':{'path':str(PA/'FINAL_CONFIRMATION_INTEGRATION_PLAN.md'),'sha256':sha(PA/'FINAL_CONFIRMATION_INTEGRATION_PLAN.md')},
            'sources':pins,'direction_tolerance_pp':TOL_PP,'formatting_decimals':{'percent':2,'contrast_pp':2,'local_m2':2,'domain_m2':4,'external_FAE':5},
            'table_2':t2,'table_3':t3,'table_4':t4,'table_5':t5,'key_result_statements':statements,'limitations':notes,
            'filters':{'table_2':'Denominators, local scalar stratum counts; external g8/nominal b10 pooled+date reference rows (values invariant across budgets)',
                       'table_3':'M2/prefix/all24/R1,2,4,6/each of five budgets; saved M2 shared references',
                       'table_4':'M4_minus_M2/all24/all4R and R4_minus_R1/M2/all24; prefix and predicted_area; all five budgets',
                       'table_5':'g8/nominal_primary/pooled/all6/R1,2,4,6/all five budgets and unique reference rows'},
            'counts':{'markdown_tables':4,'table_2_rows':len(t2_rows),'table_3_rows':len(t3_rows),'table_4_rows':len(t4_rows),'table_5_rows':len(t5_rows),
                      'local_learned_conditions_per_budget':96,'external_learned_conditions_per_budget':24,
                      'local_M_contrast_pairs_per_budget_policy':96,'local_R_contrast_pairs_per_budget_policy':24},
            'operations':{'new_models':0,'new_solver_calls':0,'new_random_orders':0,'new_selections':0,'raw_array_or_pixel_reads':0,
                          'statistics':'Saved-table minimum, maximum, direction counts and sample SD of previously saved random orders only'},
            'markdown_sha256':hashlib.sha256(markdown.encode('utf-8')).hexdigest()}
    for pin in pins.values():
        assert sha(Path(pin['path']))==pin['sha256'], 'source changed while summarizing'
    assert sha(REVIEW)==REVIEW_SHA and sha(REPORT/'artifact_manifest.json')==review['audited_artifact_manifest_sha256']
    with destinations[0].open('x',encoding='utf-8',newline='\n') as f:
        f.write(markdown)
    with destinations[1].open('x',encoding='utf-8',newline='\n') as f:
        json.dump(ledger,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'outputs':[{str(p):sha(p)} for p in destinations], 'counts':ledger['counts']},ensure_ascii=False))


if __name__ == '__main__':
    main()
