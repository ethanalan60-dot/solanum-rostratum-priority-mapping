"""Read-only manuscript/source self-review; writes a new provenance directory only.

This is the drafting agent's own review, not fresh-context independent peer review.
No model, optimizer, solver, raw-image or label-cache modules are imported.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, math, re, shutil

ROOT=Path(__file__).resolve().parent
PA=ROOT/'precision_agriculture'
STEM='PA_manuscript_integrated_development_v2'
EXPECTED='9504ed3d23762365f29d249b2081de143e2b3d5487ef21445fb8f5798e2cace1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def nums(s):return [float(v.replace('−','-')) for v in re.findall(r'[−+\-]?\d+(?:\.\d+)?',s)]
def table(text,number):
    fragment=text.split(f'**Table {number}.',1)[1].split('\n### ',1)[0]
    return [[s.strip() for s in line.strip().strip('|').split('|')] for line in fragment.splitlines() if re.match(r'^\| \d+%',line)]

def main():
    manuscript=PA/(STEM+'.md');assert sha(manuscript)==EXPECTED
    evidence=read(PA/(STEM+'_evidence.json'));text=manuscript.read_text(encoding='utf-8')
    out=ROOT/'provenance'/('integrated_v2_author_self_review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    shutil.copy2(__file__,out/'review_source.py');shutil.copy2(manuscript,out/'reviewed_manuscript.md')
    hashes=[]
    for x in evidence['sources']:
        actual=sha(x['path']);hashes.append({**x,'actual_sha256':actual,'passed':actual==x['sha256']})
    assert len(hashes)==49 and all(x['passed'] for x in hashes)
    products=read(PA/(STEM+'_delivery_manifest.json'))['products'];product_checks=[]
    for name,x in products.items():product_checks.append({'name':name,'path':x['path'],'expected':x['sha256'],'actual':sha(x['path']),'passed':sha(x['path'])==x['sha256']})
    assert all(x['passed'] for x in product_checks)
    eq=read(PA/(STEM+'_equation_specs.json'))['equations'];assert [x['number'] for x in eq]==list(range(1,6))
    fig=read(PA/(STEM+'_figure_specs.json'))['figures'];figure_checks=[]
    for g in fig:
        for fmt,x in g['formats'].items():figure_checks.append({'number':g['number'],'format':fmt,'path':x['path'],'sha256':sha(x['path']),'passed':sha(x['path'])==x['sha256']})
    assert len(figure_checks)==13 and all(x['passed'] for x in figure_checks)
    assert 'pending' not in fig[3]['caption'].lower()

    scale=PA/'scale_contrasts_20260916_065523_428384'
    r_summary=csvread(scale/'R_fixed_summary.csv');t1=table(text,1);assert len(t1)==5
    numerical_checks=[]
    for row in t1:
        b=nums(row[0])[0]/100
        for pol,countcol,rangecol in [('prefix',1,2),('predicted_area',3,4)]:
            src=[x for x in r_summary if x['M']=='2' and x['from_level']=='1' and x['to_level']=='4' and x['policy']==pol and float(x['budget'])==b];assert len(src)==1
            x=src[0];expected_counts=[int(x[k]) for k in ['higher_count','lower_count','numerical_tie_count']]
            assert nums(row[countcol])==expected_counts
            got=nums(row[rangecol]);want=[float(x['minimum_difference_pp']),float(x['maximum_difference_pp'])]
            if len(got)==1:assert got==[0] and want==[0,0]
            else:assert all(abs(a-b)<=.00050000001 for a,b in zip(got,want))
            numerical_checks.append({'table':1,'budget':b,'policy':pol,'counts':expected_counts,'exact_range_pp':want,'passed':True})
    ext=ROOT/'external_weeds_v1/spatial_v1/results_20260916_065643_813785'
    data=[x for x in csvread(ext/'learned_pooled_budget_metrics.csv') if x['grid_g']=='8' and x['budget_basis']=='nominal_primary'];assert len(data)==90
    t2=table(text,2);assert len(t2)==5
    all30=[]
    for row in t2:
        b=nums(row[0])[0]/100;d=[x for x in data if float(x['parent_nominal_budget_fraction'])==b];assert len(d)==18
        actual=float(d[0]['actual_fraction'])*100;assert nums(row[1])==[actual]
        value=[float(x['weed_capture_ratio'])*100 for x in d];ranges=[[min(value),max(value)]]
        bymetric={k:[] for k in ['weed_capture_ratio','crop_capture_ratio','crop_fraction_of_selected_area']}
        for run in sorted({x['run_id'] for x in d}):
            sub={x['R']:x for x in d if x['run_id']==run};assert set(sub)=={'1','2','4'}
            assert float(sub['1']['selected_area_FAE'])==float(sub['4']['selected_area_FAE'])
            rec={'budget':b,'run_id':run,'same_selected_area':True}
            for k in bymetric:
                delta=(float(sub['4'][k])-float(sub['1'][k]))*100;bymetric[k].append(delta);rec[k+'_difference_pp']=delta
            all30.append(rec)
        ranges.extend([min(v),max(v)] for v in bymetric.values())
        for col,want in zip(range(2,6),ranges):
            got=nums(row[col])
            if len(got)==1:got=got*2
            assert all(abs(a-b)<=.00050000001 for a,b in zip(got,want)),(row,col,got,want)
        numerical_checks.append({'table':2,'budget':b,'actual_fraction_pct':actual,'exact_ranges':ranges,'difference_directions':{k:{'higher':sum(x>1e-9 for x in v),'lower':sum(x< -1e-9 for x in v),'tied':sum(abs(x)<=1e-9 for x in v)} for k,v in bymetric.items()},'passed':True})
    assert len(all30)==30

    # Formula 5 algebra on a known nonuniform complete-prefix staircase.
    s=[0,.2,.6,1];c=[0,.4,.8,1]
    auc=lambda u:math.fsum(c[k]*max(0,min(s[k+1],u)-s[k]) for k in range(3))/u
    assert abs(auc(1)-.48)<1e-15 and abs(auc(.5)-.24)<1e-15
    ext_protocol=read(ROOT/'external_weeds_v1/spatial_v1/protocol.json')
    random_curve=csvread(ext/'random_expectation_pooled_complete_prefix_curves.csv')
    nonempty=[x for x in random_curve if int(x['k'])>0]
    for x in nonempty:
        assert abs(float(x['weed_capture_ratio'])-int(x['k'])/int(x['N']))<1e-15
        assert abs(float(x['crop_capture_ratio'])-int(x['k'])/int(x['N']))<1e-15
        assert abs(float(x['crop_fraction_of_selected_area'])-float(x['total_crop_FAE'])/int(x['frame_n']))<1e-15
    findings=[
      {'id':'C1','severity':'required_method_clarification','location':'External retraining and frame-specific spatial evaluation, paragraph after Equation 4','current':'random expectation was k/g² without Monte Carlo sampling','problem':'This unrestricted phrase can be read as applying to crop composition as well as capture, whereas crop composition has a different random baseline.','required_text':'Expected weed and crop capture under uniform random selection were k/g²; for nonempty selections, expected crop composition equalled the annotated crop fraction of the relevant frame or pooled group. No Monte Carlo sampling was used.','protocol_field':'references.random_expectation.expected_crop_composition','numeric_outputs_correct':True},
      {'id':'C2','severity':'required_notation_clarification','location':'Equation 5 preceding definitions and equation sidecar variables','problem':'The formula is correct, but k is described as units without stating that external pooled/date steps advance by k units in each frame, not k globally ranked units.','required_text':'k indexes a feasible prefix step (k selected units per frame for the external pooled/date curves); s_k is the corresponding selected-area fraction and c_k the capture ratio.','numeric_outputs_correct':True},
      {'id':'N1','severity':'nonblocking_format_note','location':'Immediately after Equation anchors 1–5','problem':'A single leading space and no blank line before the next paragraph remain in Markdown. The Word builder should parse the anchor as its own line and strip the paragraph-leading space.','numeric_outputs_correct':True}
    ]
    equation_review=[
      {'equation':1,'verdict':'matches','reason':'P is integrated predicted area and A is valid observed unit area; zero-A units excluded.'},
      {'equation':2,'verdict':'matches','reason':'Original local budget applies to observed valid areas. Context confines the formula to local global-D allocation; external per-frame allocation is explicitly separate.'},
      {'equation':3,'verdict':'matches','reason':'Target capture has full fixed annotated-area denominator; no interpretation as plant recall/control efficacy.'},
      {'equation':4,'verdict':'matches','reason':'Crop capture and selected-area composition use different denominators; zero denominator NA explicitly stated.'},
      {'equation':5,'verdict':'formula_matches_definition_needs_C2','reason':'Exact step integral matches local protocol and external k/N formula. Whole-unit right-continuous integration gives 0.48 and normalized0.24 in analytic nonuniform example; no sparse linear interpolation.'}
    ]
    bib=[
      {'key':'He2016','verified':True,'source':'https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html','authors':['Kaiming He','Xiangyu Zhang','Shaoqing Ren','Jian Sun'],'title':'Deep Residual Learning for Image Recognition','venue':'Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition','year':2016,'pages':'770–778'},
      {'key':'Woo2023','verified':True,'source':'https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html','authors':['Sanghyun Woo','Shoubhik Debnath','Ronghang Hu','Xinlei Chen','Zhuang Liu','In So Kweon','Saining Xie'],'title':'ConvNeXt V2: Co-Designing and Scaling ConvNets With Masked Autoencoders','venue':'Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition','year':2023,'pages':'16133–16142'}
    ]
    receipt={'status':'self_review_pass_with_two_required_text_clarifications','reviewer':'same drafting agent /root/proposal_review','independence':'Author-agent self-review in the same context; NOT a fresh independent content review or human-author approval. Existing independently audited source results were reused.','created_utc':datetime.now(timezone.utc).isoformat(),'manuscript_path':manuscript.as_posix(),'manuscript_sha256':EXPECTED,'source_count':len(hashes),'source_hash_checks':hashes,'delivery_product_checks':product_checks,'figure_checks':figure_checks,'equation_review':equation_review,'analytic_auc_examples':{'full':auc(1),'half_normalized':auc(.5)},'table_checks':numerical_checks,'all30_external_pairs':all30,'findings':findings,'agricultural_scope':{'passed':True,'reasons':['Single local scene and external single field explicitly stated; dates/seeds not treated as fields.','No measured herbicide saving, crop injury, control, yield or operational optimum claim.','FAE, fixed per-frame budgets and global-D allocation separated; no absolute cross-dataset capture comparison.','Observed target only14.168571m² and coarse-region reference distinguished from canopy/plant count.','Actual selected area versus common budget disclosed; predictor objective gains not treated as target gains.']},'abstract_conclusion_balance':{'passed':True,'reasons':['Abstract reports R decline5/increase1/tie2 rather than universal degradation loss.','Full Table1 includes every declared budget for both policies; Table2 includes all5 and all6 models.','M4−M2 full budget direction counts, adverse soft examples, policy decreases and norm adverse outcomes retained.','Conclusion is qualified development evidence; no seed16/reserved test/R6 outcomes invented.']},'citation_checks':bib,'citation_retrieval_limit':'Original CVF web search-index entries exposed title/full author list/venue/year/pages; direct page fetch previously403. No full-paper performance claims were derived. Same-context verification, not independent citation audit.','writes':'This new provenance directory only; manuscript/sidecars/frozen sources/results unchanged.','not_performed':['model training','model inference','solver','label/cache/imagery loading','test evaluation','R6 evaluation','manuscript modification']}
    for x in hashes:assert sha(x['path'])==x['sha256']
    assert sha(manuscript)==EXPECTED
    write(out/'review.json',receipt)
    md='''# Integrated manuscript v2 — author-agent self-review

**Numerical/source checks passed; two Methods wording clarifications are required before the final Word text.** This is the drafting agent's own review in the same context. It is not a fresh independent content review, an external peer review or human-author approval.

## Scope and verified identity

The reviewed manuscript SHA256 is `'''+EXPECTED+'''`. All **49 evidence-source hashes**, all **11 delivery-product hashes**, and all **13 figure-format hashes** matched their ledger values. Original manuscript, sidecars, protocols, data and results were not changed. Only existing result CSVs and documents were read; no raw image/label cache, model, solver, test or R6 was accessed.

Both manuscript tables were checked against accepted complete result tables: all ten Table 1 policy–budget contrasts and all five Table 2 rows (30 paired external model–budget differences) agreed at displayed precision. Original selected area equality for each external R1/R4 pair was confirmed.

## Required clarifications

1. **Random expectation is outcome-specific.** The sentence “random expectation was k/g²” appears after discussing crop composition and could be read too broadly. The frozen protocol and saved random curves distinguish expected weed/crop capture (= k/g²) from expected crop composition (the full-frame or pooled annotated crop fraction for nonempty equal-area selections). The saved numerical outputs are correct. Replace that sentence with the explicit distinction supplied in `review.json`.
2. **Equation 5 requires a pooled-step definition.** Its algebra matches both protocols. For external pooled/date curves, k is the number of selected cells **per frame** at each common prefix step, not the number selected by a global cross-frame ranking. Clarify this in prose and the sidecar variables. No formula or numerical result needs to change.

The remaining four equations match their protocols and denominators. An independent analytic nonuniform step example gives AUC=0.48 over [0,1] and normalized AUC=0.24 over [0,0.5]. This verifies the displayed algebra only, without replacing the existing full curve audits.

## Agriculture and result balance

No unsupported field efficacy, herbicide saving, yield, crop injury or operational-optimum claim was found. Crop inclusion is explicitly an annotation-based proxy, and lower crop inclusion is not automatically labelled preferable. One scene/one field, development-set reuse, region-label limits and actual-versus-nominal area are stated. Abstract and conclusion do not suppress opposite R effects; full budget tables, policy decreases and adverse normalization outcomes are retained.

## Bibliography

The original CVF indexed pages verify He et al. (2016), four authors, CVPR, pp.770–778, and Woo et al. (2023), seven authors, CVPR, pp.16133–16142. The six-name-plus-et-al rendering of the latter is consistent with the chosen journal-specific style; full names remain in the evidence ledger. No claim is made to have read full architecture papers in this review. The original primary links and exact metadata are stored in `review.json`.

## Formatting note and remaining review

Each equation anchor is on a separate line, but a following paragraph begins with one space and no blank separator. The Word builder should handle these as separate elements and strip the space. This does not alter mathematics. Parent independent content review and rendered Word/PDF inspection remain necessary. Author contributions, conflicts and final approval are still missing facts, not inferred declarations.
'''
    (out/'review.md').write_text(md,encoding='utf-8')
    write(out/'artifact_manifest.json',{'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'manuscript_unchanged':sha(manuscript)==EXPECTED})
    print(json.dumps({'output':out.as_posix(),'review_json_sha256':sha(out/'review.json'),'review_md_sha256':sha(out/'review.md'),'sources_passed':49,'products_passed':len(product_checks),'figure_formats_passed':len(figure_checks),'required_clarifications':2},ensure_ascii=False))

if __name__=='__main__':main()
