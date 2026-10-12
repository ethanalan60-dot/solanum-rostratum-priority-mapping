"""Read-only solver-x/selector/choice cross-check for two sealed failed outputs."""
from pathlib import Path
import datetime,hashlib,json,math,shutil
import check_all_common_upper_independent as a
ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture'
def optional(x,y):
 if x is None or y is None:assert x is y,(x,y)
 else:a.close(x,y)
def main():
 reports=[];checks=[]
 for folder in ('cmv2_20260916_045131_211534','cmv2_20260916_045239_245304'):
  directory=PA/folder;dest=directory/'predicted_area_policy';seal=a.read(dest/'selection_seal.json');state=a.read(directory/'status.json')
  assert state['state']=='failed' and seal['all_45_final_choices_saved'] and not seal['target_join_started']
  for kind in ('solver_raw','raw','choices'):
   assert len(seal[kind+'_sha256'])==45
   for p,h in seal[kind+'_sha256'].items():assert a.sha(dest/p)==h
  rawrows=a.rows(directory/'unit_predictions.csv')
  groups={m:{x['id']:{'A':float(x['area']),**{f'P{r}':float(x[f'predicted_R{r}']) for r in (1,2,4)}} for x in rawrows if int(x['M'])==m} for m in (1,2,4)}
  for m,r,b in a.CASES:
   n=a.name(m,r,b);backend=a.read(dest/'solver_raw'/n);raw=a.read(dest/'raw'/n);c=a.read(dest/'choices'/n);g=groups[m];ids=sorted(g)
   totalA=math.fsum(x['A'] for x in g.values());totalP=math.fsum(x[f'P{r}'] for x in g.values());limit=b*totalA
   assert backend['case']==[m,r,b] and backend['actual_solver_calls_for_case']==1 and backend['solver_options_passed']==raw['options']
   result=backend['result'];x=result.get('x');err=None;binary=False;candidate=[];ca=cp=violation=None
   if x is not None and all(a.finite(v) for v in x):
    assert len(x)==len(ids);err=max(abs(v-round(v)) for v in x);binary=err<=1e-6 and all(-1e-6<=v<=1+1e-6 for v in x)
    if binary:
     candidate=[i for i,v in zip(ids,x) if round(v)];ca=a.sumfield(candidate,g,'A');cp=a.sumfield(candidate,g,f'P{r}');violation=max(0.,ca-limit)
   assert raw['solver_candidate_ids']==candidate
   for v,k in ((err,'candidate_integrality_error'),(ca,'candidate_area_m2'),(cp,'candidate_predicted_m2'),(violation,'candidate_budget_violation_m2')):optional(v,raw[k])
   feasible=binary and violation==0.
   assert raw['candidate_feasible_in_original_units']==feasible and raw['selected_ids']==(candidate if feasible else [])
   assert raw['solver_status']==result['status'] and raw['solver_message']==result['message']
   for bk,rk in (('fun','raw_minimization_fun'),('mip_dual_bound','raw_minimization_dual'),('mip_gap','raw_solver_gap'),('mip_node_count','mip_node_count')):
    optional(result[bk] if a.finite(result.get(bk)) else None,raw[rk])
   expectedupper=-result['mip_dual_bound']*(totalP if totalP>0 else 1.) if a.finite(result.get('mip_dual_bound')) else None
   optional(expectedupper,raw['raw_solver_dual_upper_m2'])
   assert c['raw_sha256']==a.sha(dest/'raw'/n) and c['fallback_used']==(not feasible)
   assert c['raw_candidate_audit']['candidate_ids']==candidate and c['raw_candidate_audit']['recorded_feasible_flag']==feasible
   expected=candidate if feasible else a.prefix(g,b,f'P{r}')
   if not feasible:
    before=expected.copy()
    while a.sumfield(expected,g,'A')>limit:expected.pop()
    assert c['prefix_before_tail_trim_ids']==before and c['prefix_tail_trim_count']==len(before)-len(expected)
   assert c['selected_ids']==expected and a.sumfield(expected,g,'A')<=limit
   a.close(a.sumfield(expected,g,'A'),c['selected_area_m2']);a.close(a.sumfield(expected,g,f'P{r}'),c['predicted_primal_m2'])
   checks.append({'output':folder,'M':m,'R':r,'budget':b,'solver_x_to_candidate_passed':True,'raw_to_final_choice_passed':True,'fallback':not feasible})
  reports.append({'directory':str(directory),'state_preserved':'failed','seal_sha256':a.sha(dest/'selection_seal.json'),'cases':45})
 out=PA/('failed_v2_layers_independent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
 report={'passed':True,'reports':reports,'case_count':len(checks),'source_sha256':a.sha(__file__),'solver_calls':0,'model_calls':0,'test_or_R6_accessed':False,'target_values_used':False,
   'scope':'Cross-check original backend x/status/scalars to selector audit and final choice, including strict area/fallback. Bound consistency and all T outcomes are separately reviewed in the all360 closure audit.'}
 a.write(out/'audit.json',report);a.csvwrite(out/'cases_90.csv',checks);shutil.copy2(Path(__file__),out/Path(__file__).name)
 print(json.dumps({'output':str(out),'passed':True,'cases':len(checks),'audit_sha256':a.sha(out/'audit.json')},ensure_ascii=True))
if __name__=='__main__':main()
