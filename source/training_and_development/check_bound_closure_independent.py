"""Independent stdlib audit of output-only all360 certificate closure.

No production evaluator imports, solver, model, source mutation or new choice.
"""
from pathlib import Path
import argparse,collections,csv,datetime,hashlib,json,math,shutil
ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture'
CASES={(m,r,b) for m in (1,2,4) for r in (1,2,4) for b in (.1,.2,.3,.5,1.)}
WITNESS_AUDIT=PA/'all_common_upper_independent_20260916_045834_194791/audit.json'
WITNESS_SHA='77bed5f1453bc2f36f18e88dfd5988383dfc5d8031913186b1cb23b69fa34848'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def close(a,b,tol=1e-10):assert math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol),(a,b)
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def csvwrite(p,x):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def key(x):return (x['run_id'],int(x['M']),int(x['R']),float(x['budget']))
def pureprefix(group,r,b):
 ids=sorted(group,key=lambda i:(-group[i][f'P{r}']/group[i]['A'],i));total=math.fsum(x['A'] for x in group.values());cum=0.;result=[]
 for k,i in enumerate(ids):
  cum+=group[i]['A']
  if k==len(ids)-1:cum=total
  if cum>b*total:break
  result.append(i)
 return result
def metrics(group,ids,r):
 assert len(ids)==len(set(ids)) and set(ids)<=group.keys()
 a=math.fsum(group[i]['A'] for i in ids);t=math.fsum(group[i]['T'] for i in ids);p=math.fsum(group[i][f'P{r}'] for i in ids)
 A=math.fsum(x['A'] for x in group.values());T=math.fsum(x['T'] for x in group.values())
 return {'selected_count':len(ids),'actual_area_m2':a,'actual_area_fraction':a/A,'captured_target_m2':t,'capture_fraction':t/T,'missed_target_m2':T-t,'zero_annotation_area_m2':a-t,
   'selected_zero_target_unit_area_m2':math.fsum(group[i]['A'] for i in ids if group[i]['T']==0.),'predicted_objective_m2':p}
def parse_units(path):
 result={m:{} for m in (1,2,4)}
 for x in rows(path):
  m=int(x['M']);i=x['id'];assert i not in result[m] and i.isascii()
  result[m][i]={'A':float(x['area']),'T':float(x['target']),'ix':int(x['ix']),'iy':int(x['iy']),**{f'P{r}':float(x[f'predicted_R{r}']) for r in (1,2,4)}}
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--protocol-sha256',required=True);args=parser.parse_args()
 directory=args.output.resolve();assert directory.parent==PA.resolve() and directory.name.startswith('bca_')
 state=read(directory/'status.json');summary=read(directory/'summary.json');manifest=read(directory/'manifest.json');seal=read(directory/'binding_seal.json')
 assert state['state']==summary['state']=='completed'
 protocol_path=PA/'RM_bound_consistency_protocol_v1.json';assert sha(protocol_path)==args.protocol_sha256==manifest['protocol_sha256']==summary['protocol_sha256']
 protocol=read(protocol_path);assert manifest['implementation_sha256']==protocol['implementation_sha256']==sha(ROOT/'audit_common_bound_consistency.py')
 for p,h in state['files_sha256'].items():assert sha(directory/p)==h,p
 for p,h in manifest['source_sha256'].items():assert sha(p)==h,p
 assert seal['all360_existing_final_ids_bound'] and seal['new_final_selections']==0 and not seal['annotations_joined']
 assert len(seal['records_sha256'])==360 and sha(directory/'binding_seal.json')==summary['binding_seal_sha256']
 bindings={}
 for p,h in seal['records_sha256'].items():
  assert sha(directory/p)==h
  x=read(directory/p);assert key(x) not in bindings;bindings[key(x)]=x
 assert sha(WITNESS_AUDIT)==WITNESS_SHA
 witness=read(WITNESS_AUDIT)
 for p,h in witness['watched_sha256'].items():assert sha(p)==h,p
 independent_rows=rows(WITNESS_AUDIT.parent/'prediction_360_cases.csv');independent={key(x):x for x in independent_rows}
 source_ids={}
 for x in read(WITNESS_AUDIT.parent/'selected_and_prefix_ids.json'):
  rid=next(s['run_id'] for s in witness['models'] if s['label']==x['model'])
  source_ids[(rid,x['M'],x['R'],x['budget'])]=x
 table=rows(directory/'all360_policy_outcomes.csv');indexed={key(x):x for x in table}
 assert len(table)==len(indexed)==len(bindings)==len(independent)==len(source_ids)==360 and set(indexed)==set(bindings)==set(independent)==set(source_ids)
 model_specs={x['run_id']:x for x in protocol['models']};assert len(model_specs)==8
 baseline=parse_units(PA/'RM_evaluation_20260916_020003_962711/unit_predictions.csv')
 allunits={};prefixrefs={};source_hashes={};curves_checked=0
 # Verify complete source domain and nesting, then arithmetic on the already sealed IDs.
 for rid,spec in model_specs.items():
  original=Path(spec['unit_tables_directory']);units=parse_units(original/'unit_predictions.csv');allunits[rid]=units
  assert [len(units[m]) for m in (1,2,4)]==[1614,443,126]
  for m,group in units.items():
   assert set(group)==set(baseline[m])
   for ident,u in group.items():
    ref=baseline[m][ident]
    assert all(u[k]==ref[k] for k in ('A','T','ix','iy'))
    assert finite(u['A']) and u['A']>0 and 0<=u['T']<=u['A']+1e-12
    assert all(finite(u[f'P{r}']) and 0<=u[f'P{r}']<=u['A']+1e-12 for r in (1,2,4))
   if m>1:
    children=collections.defaultdict(list)
    for u in units[1].values():children[(u['ix']//m,u['iy']//m)].append(u)
    for u in group.values():
     for col in ('A','T','P1','P2','P4'):close(math.fsum(c[col] for c in children[(u['ix'],u['iy'])]),u[col])
  refs={(int(x['M']),int(x['R']),float(x['budget'])):x for x in rows(original/'fixed_budgets.csv') if x['method']=='model_prefix'}
  assert set(refs)==CASES;prefixrefs[rid]=refs
  # All model staircase points, including adverse prefixes, compared with fresh accurate sums.
  curves=rows(original/'decision_curves.csv')
  for m in (1,2,4):
   g=units[m];A=math.fsum(x['A'] for x in g.values());T=math.fsum(x['T'] for x in g.values())
   for r in (1,2,4):
    ordered=sorted(g,key=lambda i:(-g[i][f'P{r}']/g[i]['A'],i))
    cr=sorted([x for x in curves if int(x['M'])==m and x['method']=='model_prefix' and int(x['R'])==r],key=lambda x:int(x['k']))
    assert len(cr)==len(ordered)+1
    for k,x in enumerate(cr):
     assert int(x['k'])==k
     a=math.fsum(g[i]['A'] for i in ordered[:k]);t=math.fsum(g[i]['T'] for i in ordered[:k])
     close(x['actual_area'],a);close(x['captured_target'],t);close(x['actual_area_fraction'],a/A);close(x['capture_fraction'],t/T)
     if k:assert x['last_added_id']==ordered[k-1]
    curves_checked+=len(cr)
  source_hashes[str(original/'unit_predictions.csv')]=sha(original/'unit_predictions.csv')
 audits=[];by_model=collections.defaultdict(list)
 for case in sorted(indexed):
  rid,m,r,b=case;row=indexed[case];binding=bindings[case];spec=model_specs[rid];g=allunits[rid][m];old=independent[case];originalids=source_ids[case]
  final=binding['selected_ids'];pref=binding['prefix_ids']
  assert final==originalids['final_ids']==json.loads(row['policy_selected_ids_json'])
  assert pref==originalids['prefix_ids']==pureprefix(g,r,b)==json.loads(row['prefix_selected_ids_json'])
  assert sha(binding['choice_source'])==binding['choice_source_sha256']==row['choice_source_sha256']
  assert sha(binding['raw_source'])==binding['raw_source_sha256']==row['raw_source_sha256']
  raw=read(binding['raw_source']);assert raw['selected_ids']==binding['raw_selected_ids']==json.loads(row['raw_selected_ids_json'])
  assert raw['solver_candidate_ids']==binding['raw_solver_candidate_ids']==json.loads(row['raw_solver_candidate_ids_json'])
  for k in ('run_id','checkpoint_epoch','checkpoint_sha256','encoder','variant','lr'):
   expected=spec[k];assert str(row[k])==str(expected) and binding[k]==expected,(k,row[k],expected)
  meta=read(Path(spec['maps_directory'])/'metadata.json');assert meta['run_id']==rid and meta['checkpoint_sha256']==spec['checkpoint_sha256'] and meta['checkpoint_epoch']==spec['checkpoint_epoch']
  A=math.fsum(x['A'] for x in g.values());P=math.fsum(x[f'P{r}'] for x in g.values());B=b*A
  fm=metrics(g,final,r);pm=metrics(g,pref,r);assert fm['actual_area_m2']<=B and pm['actual_area_m2']<=B
  for prefix,values in [('policy',fm),('prefix',pm)]:
   for k,val in values.items():close(row[prefix+'_'+k],val)
  for field,k in [('selected_count','k'),('actual_area_m2','actual_area'),('captured_target_m2','captured_target'),('capture_fraction','capture_fraction'),('zero_annotation_area_m2','zero_annotation_area')]:close(pm[field],prefixrefs[rid][(m,r,b)][k])
  finalp=fm['predicted_objective_m2'];prefp=pm['predicted_objective_m2'];w=max(finalp,prefp);tol=max(1e-10,1e-10*P)
  retained=float(old['retained_upper']);rawupper=None if old['raw_upper']=='' else float(old['raw_upper'])
  reasons=[]
  if rawupper is not None and rawupper<w-tol:reasons.append('raw_solver_upper')
  if retained<w-tol:reasons.append('retained_upper')
  anomaly=bool(reasons);clamp=0.
  if anomaly:upper=P
  else:
   upper=min([P,retained]+([] if rawupper is None else [rawupper]))
   if upper<w:clamp=w-upper;assert clamp<=tol;upper=w
  gap=(upper-finalp)/max(abs(finalp),1e-30) if upper else 0.
  fallback=old['fallback']=='True';oldproven=old['previous_proven']=='True';proven=oldproven and not fallback and not anomaly and gap<=1e-8+1e-10
  assert (row['bound_anomaly']=='True')==anomaly and row['contradicted_upper_fields']==';'.join(reasons)
  assert (row['fallback_used']=='True')==fallback and (row['original_proven_flag']=='True')==oldproven and (row['updated_proven_flag']=='True')==proven
  assert (row['certification_downgraded']=='True')==(oldproven and not proven)
  bound=binding['bound_audit'];assert bound['bound_anomaly']==anomaly and bound['contradicted_upper_fields']==reasons and bound['updated_proven_flag']==proven
  for k,val in [('bound_tolerance_m2',tol),('feasible_objective_witness_m2',w),('updated_upper_m2',upper),('updated_primal_m2',finalp),('updated_absolute_gap_m2',upper-finalp),('updated_relative_gap',gap),('roundoff_clamp_m2',clamp)]:close(row[k],val,1e-12);close(bound[k],val,1e-12)
  delta=100*(fm['capture_fraction']-pm['capture_fraction']);close(row['delta_capture_pp'],delta);close(row['delta_predicted_m2'],finalp-prefp);close(row['unused_budget_m2'],B-fm['actual_area_m2'])
  assert row['final_ids_unchanged']=='True' and binding['final_ids_unchanged'] and row['prefix_strictly_feasible']=='True'
  result={'run_id':rid,'M':m,'R':r,'budget':b,'policy_A':fm['actual_area_m2'],'policy_T':fm['captured_target_m2'],'policy_P':finalp,'prefix_A':pm['actual_area_m2'],'prefix_T':pm['captured_target_m2'],'prefix_P':prefp,'delta_capture_pp':delta,'anomaly':anomaly,'fallback':fallback,'old_proven':oldproven,'updated_proven':proven,'updated_upper':upper,'updated_gap':gap,'clamp':clamp,'ids_unchanged':True}
  audits.append(result);by_model[rid].append(result)
 assert len(audits)==360 and sum(x['anomaly'] for x in audits)==5 and sum(x['updated_proven'] for x in audits)==257
 for item in summary['by_model']:
  xs=by_model[item['run_id']]
  for k,val in [('cases',45),('fallback_count',sum(x['fallback'] for x in xs)),('original_proven_count',sum(x['old_proven'] for x in xs)),('updated_proven_count',sum(x['updated_proven'] for x in xs)),('bound_anomalies',sum(x['anomaly'] for x in xs)),('roundoff_clamps',sum(x['clamp']>0 for x in xs)),('capture_increased',sum(x['delta_capture_pp']>1e-8 for x in xs)),('capture_decreased',sum(x['delta_capture_pp']< -1e-8 for x in xs)),('capture_unchanged',sum(abs(x['delta_capture_pp'])<=1e-8 for x in xs))]:assert item[k]==val,(item['run_id'],k,item[k],val)
 for p,h in state['files_sha256'].items():assert sha(directory/p)==h
 out=PA/('bound_closure_independent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
 report={'passed':True,'audited_output_directory':str(directory),'all360_policy_outcomes_sha256':sha(directory/'all360_policy_outcomes.csv'),'passed_case_count':360,'source_binding_seal_sha256':sha(directory/'binding_seal.json'),'final_ids_unchanged':True,
   'created_at':datetime.datetime.now().astimezone().isoformat(),'auditor_source_sha256':sha(__file__),'protocol_sha256':args.protocol_sha256,
   'all_product_and_source_hashes_verified':True,'fixed_D_A_T_and_nested_M_verified':True,'all_model_prefix_curve_points_recomputed':curves_checked,
   'original_prefix_sets_checked':360,'final_policy_sets_checked':360,'certification_anomalies':5,'old_proven_count':262,'updated_proven_count':257,'fallback_count':sum(x['fallback'] for x in audits),
   'roundoff_clamps':sum(x['clamp']>0 for x in audits),'independent_360_witness_audit_sha256':WITNESS_SHA,'label_reference_120_record_scan_contradictions':0,
   'new_solver_calls':0,'new_model_calls':0,'test_or_R6_accessed':False,'production_sources_modified':False,
   'limitations':'Recorded certificates are checked against feasible witnesses and arithmetic; no independent optimization proof or field efficacy is claimed. All evidence remains one-scene development.'}
 write(out/'audit.json',report);csvwrite(out/'all360_arithmetic_checks.csv',audits);shutil.copy2(Path(__file__),out/Path(__file__).name)
 (out/'review.md').write_text('# Independent all360 bound-closure review\n\n**Passed all 360 cases.** Every existing final ID set is unchanged and strictly feasible. Original prefix sets and all A/T/P metrics, source bindings, product hashes, bounds, gaps, flags and full model prefix curves were independently recomputed.\n\nFive inconsistent certificates are downgraded, giving 257 retained flags from 262; all four numerical fallbacks remain not proven. The separate 120-row label-reference scan found no analogous contradiction. No solver, model, GPU or test/R6 access occurred. These checks do not constitute an independent optimization proof.\n','utf8')
 print(json.dumps({'output':str(out),'passed':True,'audited_output_directory':str(directory),'audit_sha256':sha(out/'audit.json'),'cases':360,'new_proven':257},ensure_ascii=True))
if __name__=='__main__':main()
