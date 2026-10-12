"""Independent 360-case bound witness audit; no optimizer/model imports.

Prediction-policy rules use only IDs, valid area and predicted area. A separate
final pass checks the label-reference bounds using their allowed annotations.
"""
from pathlib import Path
import csv,datetime,hashlib,json,math,shutil
ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture'
CASES=[(m,r,b) for m in (1,2,4) for r in (1,2,4) for b in (.1,.2,.3,.5,1.)]
SPECS=[
 ('r18_hard_lr1e4','common_RM_dev_common_v2_lr1e4_resnet18_hard_seg_L256_s17_20260916_031336_012442','raw',None),
 ('r18_soft_lr1e4','common_RM_dev_common_v2_lr1e4_resnet18_soft_seg_L256_s17_20260916_031402_620370','raw',None),
 ('conv_hard_lr1e4','common_policy_recovery_20260916_035027_773459','recovery','common_RM_dev_common_v2_lr1e4_convnextv2_tiny_hard_seg_L256_s17_20260916_033131_837962'),
 ('conv_soft_lr1e4','cmv2_20260916_045114_251507','v2',None),
 ('r18_hard_lr3e4','cmv2_20260916_045131_211534','v2',None),
 ('r18_soft_lr3e4','cmv2_20260916_045215_354647','v2',None),
 ('conv_hard_lr3e4','cmv2_20260916_045227_260901','v2',None),
 ('conv_soft_lr3e4','cmv2_20260916_045239_245304','v2',None)]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-10),(a,b)
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def csvwrite(p,x):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
def name(m,r,b):return f'M{m}_R{r}_B{round(b*100):03d}.json'
def prefix(units,b,key):
 order=sorted(units,key=lambda i:(-units[i][key]/units[i]['A'],i));total=math.fsum(x['A'] for x in units.values())
 cumulative=0.;chosen=[]
 for k,i in enumerate(order):
  cumulative+=units[i]['A']
  if k==len(order)-1:cumulative=total
  if cumulative>b*total:break
  chosen.append(i)
 return chosen
def sumfield(ids,units,k):
 assert len(ids)==len(set(ids)) and set(ids)<=units.keys()
 return math.fsum(units[i][k] for i in ids)
def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)

def main():
 outputs=[];watched={};sources=[];label_sources=[];ids_records=[]
 # Target columns are discarded as uninterpreted strings in this entire pass.
 for label,folder,kind,parent in SPECS:
  directory=PA/folder;unitdir=PA/parent if parent else directory;unitpath=unitdir/'unit_predictions.csv'
  watched[str(unitpath)]=sha(unitpath);rawrows=rows(unitpath)
  allowed=[{k:x[k] for k in ('M','id','area','predicted_R1','predicted_R2','predicted_R4')} for x in rawrows]
  units={m:{x['id']:{'A':float(x['area']),**{f'P{r}':float(x[f'predicted_R{r}']) for r in (1,2,4)}} for x in allowed if int(x['M'])==m} for m in (1,2,4)}
  assert [len(units[m]) for m in (1,2,4)]==[1614,443,126]
  for group in units.values():
   assert all(finite(v) for x in group.values() for v in x.values()) and all(x['A']>0 for x in group.values())
  source={'label':label,'directory':str(directory),'unit_directory':str(unitdir),'kind':kind,'run_id':None}
  if kind=='raw':source['run_id']=read(directory/'summary.json')['model_run']
  elif kind=='recovery':source['run_id']=read(directory/'summary.json')['run_id']
  else:source['run_id']=read(directory/'inputs.json')['run_id']
  if kind!='raw':
   base=directory if kind=='recovery' else directory/'predicted_area_policy';seal=read(base/'selection_seal.json')
   for field in ('choices_sha256','raw_sha256'):
    assert len(seal[field])==45
    for rel,h in seal[field].items():assert sha(base/rel)==h;watched[str(base/rel)]=h
   if kind=='v2':
    for rel,h in seal['solver_raw_sha256'].items():assert sha(base/rel)==h;watched[str(base/rel)]=h
  for m,r,b in CASES:
   group=units[m];A=math.fsum(x['A'] for x in group.values());P=math.fsum(x[f'P{r}'] for x in group.values());B=b*A
   if kind=='raw':
    cp=directory/'predicted_area_policy'/name(m,r,b);c=read(cp);selected=c['selected_ids'];upper=c['predicted_dual_upper_m2'];rawupper=c['raw_solver_dual_upper_m2'];proven=c['optimality_proven_within_tolerance'];fallback=False;gap=c['predicted_relative_gap']
   else:
    cp=base/'choices'/name(m,r,b);c=read(cp);selected=c['selected_ids'];upper=c['recomputed_bound']['upper_m2'];rawupper=c['recomputed_bound']['raw_upper_m2'];proven=c['optimality_proven'];fallback=c['fallback_used'];gap=c['recomputed_bound']['relative_gap']
   watched[str(cp)]=sha(cp)
   chosen_a=sumfield(selected,group,'A');chosen_p=sumfield(selected,group,f'P{r}');assert chosen_a<=B
   close(chosen_a,c['selected_area_m2']);close(chosen_p,c['predicted_primal_m2'])
   pref=prefix(group,b,f'P{r}');pref_a=sumfield(pref,group,'A');pref_p=sumfield(pref,group,f'P{r}');assert pref_a<=B
   witness=max(chosen_p,pref_p);tol=max(1e-10,1e-10*P)
   retained_deficit=witness-upper
   raw_deficit=witness-rawupper if finite(rawupper) else None
   contradictory=retained_deficit>tol or (raw_deficit is not None and raw_deficit>tol)
   # Reproduce the proposed rule as a diagnostic, never overwrite old certificates.
   suggested_upper=P if contradictory else max(upper,witness)
   clamp=0. if contradictory else max(0.,witness-upper)
   suggested_gap=(suggested_upper-chosen_p)/max(abs(chosen_p),1e-30) if suggested_upper else 0.
   suggested_proven=bool(proven and not fallback and not contradictory and suggested_gap<=1e-8+1e-10)
   record={'model':label,'run_id':source['run_id'],'M':m,'R':r,'budget':b,'budget_m2':B,'total_P':P,'final_A':chosen_a,'final_P':chosen_p,'prefix_A':pref_a,'prefix_P':pref_p,
      'final_count':len(selected),'prefix_count':len(pref),'prefix_strict_feasible':True,'retained_upper':upper,'raw_upper':rawupper,
      'witness_lower_bound':witness,'tolerance':tol,'retained_upper_deficit':retained_deficit,'raw_upper_deficit':raw_deficit,
      'contradiction':contradictory,'previous_proven':proven,'fallback':fallback,'previous_gap':gap,
      'suggested_upper':suggested_upper,'suggested_gap':suggested_gap,'suggested_clamp':clamp,'suggested_proven':suggested_proven,'selected_ids_unchanged':True,'choice_sha256':sha(cp)}
   outputs.append(record);ids_records.append({'model':label,'M':m,'R':r,'budget':b,'final_ids':selected,'prefix_ids':pref})
  sources.append(source);label_sources.append((source,unitpath,unitdir/'fixed_budgets.csv'))
 assert len(outputs)==360
 # Separate label-reference check, after all prediction-policy consistency rules.
 label_checks=[]
 for source,unitpath,fixedpath in label_sources:
  watched[str(fixedpath)]=sha(fixedpath)
  data=rows(unitpath);groups={m:{x['id']:{'A':float(x['area']),'T':float(x['target'])} for x in data if int(x['M'])==m} for m in (1,2,4)}
  refs={(int(x['M']),float(x['budget'])):x for x in rows(fixedpath) if x['method']=='label_milp'}
  assert len(refs)==15
  for (m,b),ref in sorted(refs.items()):
   g=groups[m];A=math.fsum(x['A'] for x in g.values());T=math.fsum(x['T'] for x in g.values());pref=prefix(g,b,'T')
   a=sumfield(pref,g,'A');t=sumfield(pref,g,'T');assert a<=b*A
   upper=float(ref['dual_target_bound']);primal=float(ref['primal_target']);tol=max(1e-10,1e-10*T)
   witness=max(primal,t);label_checks.append({'model':source['label'],'M':m,'budget':b,'prefix_A':a,'prefix_T':t,'prefix_strict_feasible':True,'primal_T':primal,'retained_upper_T':upper,'witness':witness,'tolerance':tol,'upper_deficit':witness-upper,'contradiction':witness-upper>tol,'reported_proven':ref['optimality_proven']})
 assert len(label_checks)==120
 for path,h in watched.items():assert sha(path)==h
 contradictions=[x for x in outputs if x['contradiction']]
 out=PA/('all_common_upper_independent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
 csvwrite(out/'prediction_360_cases.csv',outputs);csvwrite(out/'label_reference_120_cases.csv',label_checks);write(out/'selected_and_prefix_ids.json',ids_records)
 report={'state':'completed_readonly_audit','created_at':datetime.datetime.now().astimezone().isoformat(),'source_sha256':sha(__file__),'models':sources,'prediction_cases':360,'all_final_and_prefix_original_budgets_strictly_feasible':True,
   'contradiction_count':len(contradictions),'contradictions':contradictions,'within_tolerance_retained_inversions':sum(0<x['retained_upper_deficit']<=x['tolerance'] for x in outputs),
   'previous_proven_count':sum(x['previous_proven'] for x in outputs),'suggested_proven_count':sum(x['suggested_proven'] for x in outputs),
   'label_reference_rows_checked':120,'unique_label_cases':15,'label_reference_contradictions':[x for x in label_checks if x['contradiction']],
   'watched_sha256':watched,'prediction_rule_uses_target':False,'label_audit_is_separate':True,'selected_ids_changed':False,'solver_calls':0,'model_calls':0,'test_or_R6_accessed':False,
   'scope_limit':'The proposed adjusted values are diagnostics only. No source result, policy, selected ID, certificate or report was overwritten. Internal solver cause is unconfirmed.'}
 write(out/'audit.json',report);shutil.copy2(Path(__file__),out/Path(__file__).name)
 lines=['# Independent all-common bound witness audit','',f"Checked 360 prediction-policy cases and 120 label-reference records (15 repeated cases across eight models). All observed final and original prefix ID sets are strictly feasible under unchanged original area budgets. Prediction consistency used only ID/A/P; target data were used only in the separate label-reference check.",'',f"Prediction-bound contradictions: {len(contradictions)}. Label-reference contradictions: {len(report['label_reference_contradictions'])}.",'','| Model | M | R | Budget | Feasible-prefix P | Retained upper | Deficit | Tolerance |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for x in contradictions:lines.append(f"| {x['model']} | {x['M']} | {x['R']} | {x['budget']} | {x['prefix_P']:.15g} | {x['retained_upper']:.15g} | {x['retained_upper_deficit']:.15g} | {x['tolerance']:.15g} |")
 lines+=['','The existing selected IDs remain unchanged. A known feasible witness above a recorded upper beyond tolerance contradicts that recorded certificate; it does not justify another solve or a label-driven choice. The companion table computes the proposed sum-P safe-upper downgrade for diagnosis only. Internal solver cause is unconfirmed. No model, solver, GPU or test/R6 access occurred.']
 (out/'review.md').write_text('\n'.join(lines)+'\n','utf8')
 print(json.dumps({'output':str(out),'prediction_contradictions':len(contradictions),'label_contradictions':len(report['label_reference_contradictions']),'previous_proven':report['previous_proven_count'],'suggested_proven':report['suggested_proven_count'],'audit_sha256':sha(out/'audit.json'),'cases':[{k:x[k] for k in ('model','M','R','budget','retained_upper_deficit','tolerance')} for x in contradictions]},ensure_ascii=True))
if __name__=='__main__':main()
