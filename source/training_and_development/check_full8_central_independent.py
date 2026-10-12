"""Read-only central-table arithmetic and artifact audit; no production imports."""
from pathlib import Path
import csv,datetime,hashlib,itertools,json,math,re,shutil,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
PA=ROOT/'precision_agriculture'
REPORT=PA/'common_management_reports/full8_20260916_050927_560091'
BCA=PA/'bca_20260916_050803_135549'
PRIOR=PA/'bound_closure_independent_20260916_050846_426215/audit.json'
OUT=PA/('full8_central_independent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
OUT.mkdir();(OUT/'qa').mkdir()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-11),(a,b)
def boolean(v):
    assert v in (True,False,'True','False'),v
    return v is True or v=='True'
def key(r):return r['run_id'],int(r['M']),int(r['R']),float(r['budget'])
def pairkey(r):return r['encoder'],float(r['lr']),r['policy'],int(r['M']),int(r['R']),float(r['budget'])
def verify_relative(base,bindings):
    for rel,h in bindings.items():
        p=(base/rel.replace('\\','/')).resolve();assert p.is_relative_to(base.resolve()),rel
        assert sha(p)==h,(str(p),'hash mismatch')
before={n:sha(REPORT/n) for n in ['status.json','final_artifact_manifest.json']}
status=read(REPORT/'status.json');final=read(REPORT/'final_artifact_manifest.json')
provenance=read(REPORT/'provenance.json');figures=read(REPORT/'figure_manifest.json')
verify_relative(REPORT,status['file_sha256'])
verify_relative(REPORT,final['generated_file_sha256'])
assert final['immutable_table_snapshot_status_sha256']==before['status.json']
assert final['input_binding']==figures['table_snapshot_binding']
assert final['input_binding']['table_snapshot_hashes']==status['file_sha256']
assert final['input_binding']['status_sha256']==before['status.json']
assert final['table_snapshot_preserved'] and figures['no_inference_or_solver']
assert final['script_sha256']==figures['script_sha256']==sha(REPORT/'plotting_source/plot_full_common_management.py')==sha(ROOT/'plot_full_common_management.py')
assert provenance['script_sha256']==sha(REPORT/'source/build_full_common_management_report.py')==sha(ROOT/'build_full_common_management_report.py')
for path,h in provenance['source_sha256'].items():assert sha(path)==h,(path,'source mismatch')
for x in provenance['source_copies']:
    assert sha(x['original'])==sha(REPORT/x['copy'])==x['sha256']==provenance['source_sha256'][x['original']]
prior=read(PRIOR)
assert sha(PRIOR)=='cd88fe934e9c9d5dcee01a38a7571257f23d1ab8838ad9a1811e6e1f72913da2'
assert prior['passed'] and prior['passed_case_count']==360 and prior['final_ids_unchanged']
assert prior['all360_policy_outcomes_sha256']==sha(BCA/'all360_policy_outcomes.csv')
assert prior['source_binding_seal_sha256']==sha(BCA/'binding_seal.json')
assert str(PRIOR) in provenance['source_sha256']
bstatus=read(BCA/'status.json');assert bstatus['state']=='completed'
verify_relative(BCA,bstatus['files_sha256'])
data=rows(REPORT/'all_720_model_policy_budgets.csv');pairs=rows(REPORT/'all_360_soft_minus_hard_pairs.csv')
bs=rows(BCA/'all360_policy_outcomes.csv');bmap={key(x):x for x in bs}
assert len(data)==720 and len(pairs)==360 and len(bs)==len(bmap)==360
models=provenance['all8_models'];assert len(models)==8
runmap={x['run_id']:x for x in models}
assert {(x['encoder'],x['lr'],x['variant']) for x in models}==set(itertools.product(['resnet18','convnextv2_tiny'],[.0001,.0003],['hard_seg','soft_seg']))
assert {(*key(x),x['policy']) for x in data}==set(itertools.product(runmap,[1,2,4],[1,2,4],[.1,.2,.3,.5,1.],['prefix','predicted_area']))
numeric_checks=[]
for x in data:
    q=bmap[key(x)];m=runmap[x['run_id']];pre='prefix' if x['policy']=='prefix' else 'policy'
    for name in ['encoder','variant','checkpoint_sha256']:assert x[name]==q[name]==m[name]
    for name in ['lr','checkpoint_epoch']:close(x[name],q[name]);close(x[name],m[name])
    assert int(x['seed'])==17
    close(x['original_gpu_mae_pp'],m['original_gpu_mae_pp'])
    for name in ['budget_area_m2']:close(x[name],q[name])
    for dest,src in [('selected_count','selected_count'),('actual_area_m2','actual_area_m2'),('actual_area_fraction','actual_area_fraction'),('captured_target_m2','captured_target_m2'),('missed_target_m2','missed_target_m2'),('zero_annotation_area_m2','zero_annotation_area_m2'),('selected_zero_target_unit_area_m2','selected_zero_target_unit_area_m2'),('predicted_objective_m2','predicted_objective_m2')]:close(x[dest],q[pre+'_'+src])
    A=float(x['actual_area_m2']);T=float(x['captured_target_m2']);D=m['domain']['area_m2'];totalT=m['domain']['target_m2'];B=float(x['budget_area_m2'])
    assert A<=B;close(B,float(x['budget'])*D);close(x['capture_pct'],100*T/totalT)
    close(x['unused_budget_m2'],B-A);close(x['domain_area_m2'],D);close(x['domain_target_m2'],totalT)
    for path,h in [('source_csv','source_csv_sha256'),('fixed_budget_csv','fixed_budget_csv_sha256'),('binding_record','binding_record_sha256')]:assert sha(x[path])==x[h]
    assert Path(x['source_csv'])==BCA/'all360_policy_outcomes.csv'
    assert Path(x['binding_record'])==BCA/q['record_file']
    if pre=='policy':
        for dest,src in [('optimality_proven','updated_proven_flag'),('fallback_used','fallback_used'),('original_proven_flag','original_proven_flag'),('bound_anomaly','bound_anomaly'),('certification_downgraded','certification_downgraded')]:assert boolean(x[dest])==boolean(q[src])
        for dest,src in [('predicted_upper_m2','updated_upper_m2'),('predicted_relative_gap','updated_relative_gap'),('original_retained_upper_m2','retained_upper_m2'),('original_retained_gap','original_retained_gap')]:close(x[dest],q[src])
    else:
        assert x['optimality_proven']==x['predicted_upper_m2']==x['predicted_relative_gap']==''
        assert not any(boolean(x[n]) for n in ['fallback_used','bound_anomaly','certification_downgraded'])
    numeric_checks.append({'run_id':x['run_id'],'policy':x['policy'],'M':x['M'],'R':x['R'],'budget':x['budget'],'passed':True})
lookup={(x['encoder'],float(x['lr']),x['variant'],x['policy'],int(x['M']),int(x['R']),float(x['budget'])):x for x in data}
assert len(lookup)==720 and len({pairkey(x) for x in pairs})==360
for x in pairs:
    e,lr,pol,M,R,b=pairkey(x);h=lookup[e,lr,'hard_seg',pol,M,R,b];s=lookup[e,lr,'soft_seg',pol,M,R,b]
    for prefix,item in [('hard',h),('soft',s)]:
        close(x[prefix+'_capture_pct'],item['capture_pct']);close(x[prefix+'_actual_area_m2'],item['actual_area_m2'])
        assert x[prefix+'_fallback']==item['fallback_used'] and x[prefix+'_gap_qualified']==item['optimality_proven']
        assert x[prefix+'_source_sha256']==item['source_csv_sha256']
    close(x['soft_minus_hard_pp'],float(s['capture_pct'])-float(h['capture_pct']))
    close(x['soft_minus_hard_area_m2'],float(s['actual_area_m2'])-float(h['actual_area_m2']))
counts=rows(REPORT/'all_8_paired_direction_counts.csv')
for x in counts:
    values=[float(q['soft_minus_hard_pp']) for q in pairs if q['encoder']==x['encoder'] and float(q['lr'])==float(x['lr']) and q['policy']==x['policy']]
    assert len(values)==45
    assert [int(x[k]) for k in ['higher','lower','same']]==[sum(v>1e-8 for v in values),sum(v< -1e-8 for v in values),sum(abs(v)<=1e-8 for v in values)]
    close(x['min_pp'],min(values));close(x['max_pp'],max(values))
primary=rows(REPORT/'primary_M2_budget10_all_pairs.csv')
assert primary==[x for x in pairs if int(x['M'])==2 and float(x['budget'])==.1]
policies=[x for x in data if x['policy']=='predicted_area']
summary={name:sum(boolean(x[name]) for x in policies) for name in ['optimality_proven','certification_downgraded','bound_anomaly','fallback_used']}
assert summary=={'optimality_proven':257,'certification_downgraded':5,'bound_anomaly':5,'fallback_used':4}
for n,h in figures['input_sha256'].items():assert sha(REPORT/n)==h
assert figures['symmetric_difference_limits_pp']==[-20,20]
assert max(abs(float(x['soft_minus_hard_pp'])) for x in pairs)<=20
figure_checks=[];ns={'s':'http://www.w3.org/2000/svg'}
configs=list(itertools.product(['resnet18','convnextv2_tiny'],[.0001,.0003]))
for fig in figures['figures']:
    name=fig['name'];expected=pairs if name=='fig_all360_soft_minus_hard' else [x for x in data if int(x['M'])==2 and x['policy']==name.removeprefix('fig_primary_M2_')]
    assert fig['data']==expected
    for ext,item in fig['files'].items():assert Path(item['path'])==REPORT/(name+'.'+ext) and sha(item['path'])==item['sha256']
    svg=ET.parse(REPORT/(name+'.svg')).getroot()
    text_nodes=[''.join(x.itertext()) for x in svg.findall('.//s:text',ns)]
    assert any('One scene' in x for x in text_nodes)
    annotations=0
    if name=='fig_all360_soft_minus_hard':
        for panel,(enc,lr,pol) in enumerate((e,l,p) for e,l in configs for p in ['prefix','predicted_area']):
            g=svg.find(f'.//s:g[@id="axes_{panel+1}"]',ns)
            actual=[v for t in g.findall('.//s:text',ns) if re.fullmatch(r'[+-]\d+\.\d',v:=''.join(t.itertext()))]
            pmap={pairkey(x):x for x in pairs}
            wanted=[f"{float(pmap[enc,lr,pol,M,R,b]['soft_minus_hard_pp']):+.1f}" for M,R,b in itertools.product([1,2,4],[1,2,4],[.1,.2,.3,.5,1.])]
            assert actual==wanted,(panel,len(actual),len(wanted));annotations+=len(actual)
        assert annotations==360
    figure_checks.append({'name':name,'bound_data_rows':len(expected),'files':fig['files'],'svg_signed_annotations_verified':annotations})
import fitz
renderings=[]
for fig in figures['figures']:
    for ext in ['pdf','svg']:
        p=REPORT/(fig['name']+'.'+ext);d=fitz.open(str(p));assert d.page_count==1
        page=d[0];target=OUT/'qa'/(p.stem+'_'+ext+'.png')
        pix=page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False);pix.save(str(target))
        if ext=='pdf':
            txt=page.get_text();(OUT/'qa'/(p.stem+'_pdf_text.txt')).write_text(txt,encoding='utf8')
            assert 'One scene' in txt and '17' in txt
        renderings.append({'original':str(p),'sha256':sha(p),'qa_png':str(target),'qa_sha256':sha(target),'size_pixels':[pix.width,pix.height],'renderer':'PyMuPDF '+fitz.VersionBind,'render_scale':2,'page_count':1})
        d.close()
assert before=={n:sha(REPORT/n) for n in before}
result={'numerical_passed':True,'visual_review':'pending actual six-image inspection','audited_report_directory':str(REPORT),'audited_bca_directory':str(BCA),'prior_independent_audit':str(PRIOR),'prior_independent_audit_sha256':sha(PRIOR),'all360_policy_outcomes_sha256':sha(BCA/'all360_policy_outcomes.csv'),'source_binding_seal_sha256':sha(BCA/'binding_seal.json'),'table_rows_verified':720,'paired_rows_verified':360,'summary':summary,'snapshot_files_verified':len(status['file_sha256']),'current_source_and_copy_pairs_verified':len(provenance['source_copies']),'generated_files_verified':len(final['generated_file_sha256']),'bca_product_files_verified':len(bstatus['files_sha256']),'figure_checks':figure_checks,'renderings':renderings,'unchanged_original_locks':before,'new_solver_calls':0,'new_inference_calls':0,'test_R6_accessed':False,'source_modified':False,'auditor_source_sha256':sha(__file__),'limitations':'One-scene seed17 development. Arithmetic and source-binding review, not independent optimization proof, independent scenes, or measured agronomic efficacy.'}
save(OUT/'central_numeric_review.json',result)
shutil.copy2(__file__,OUT/Path(__file__).name)
print(json.dumps({'output':str(OUT),'numeric_passed':True,'rows':720,'pairs':360,'summary':summary,'renders':len(renderings)},ensure_ascii=False))
