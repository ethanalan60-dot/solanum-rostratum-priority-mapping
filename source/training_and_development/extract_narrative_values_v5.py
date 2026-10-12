"""Presentation-only extraction from the accepted report; no models or selectors."""
from pathlib import Path
import json,csv,hashlib,datetime
ROOT=Path(__file__).resolve().parent
B=ROOT/'confirmation_v1/stages/report_tables'
pins={}
def rows(p):
 f=B/p;pins[p]=hashlib.sha256(f.read_bytes()).hexdigest()
 with f.open(encoding='utf-8-sig',newline='') as s:return list(csv.DictReader(s))
def filt(rs,**kw):return [x for x in rs if all(str(x[k])==str(v) for k,v in kw.items())]
def stat(rs,k):
 z=[float(x[k]) for x in rs if x[k]!='']
 return dict(n=len(rs),defined=len(z),lo=min(z),hi=max(z),up=sum(x>1e-9 for x in z),down=sum(x< -1e-9 for x in z),tie=sum(abs(x)<=1e-9 for x in z)) if z else dict(n=len(rs),defined=0)
def main():
 review=ROOT/'provenance/completed_confirmation_report_independent_20260916_142830_724677/review.json'
 assert hashlib.sha256(review.read_bytes()).hexdigest()=='3af89351cf4a813bae16c6c6ba82e7334e05be4aa048a49707ad4f521f57bed8'
 out={};budgets=['0.1','0.2','0.3','0.5','1.0']
 for ds in ['local','weeds']:
  a=rows(ds+'/scalar_hard_soft_pairs.csv');out[ds+'_scalar_pairs']={k:stat(a,k) for k in a[0] if k.startswith('soft_minus_hard_')}
  a=rows(ds+'/scalar_seed_mean_sample_sd.csv');out[ds+'_scalar_group_values']=[{k:x[k] for k in ['encoder','variant','lr','R','n_models','mae_pp_mean','mae_pp_sample_sd','balanced_mae_pp_mean','balanced_mae_pp_sample_sd']} for x in a]
 a=rows('local/spatial_budget_hard_soft_pairs.csv');out['local_soft_hard_M2']={q:stat(filt(a,M=2,policy='prefix',budget=q),'soft_minus_hard_capture_pct') for q in budgets}
 a=rows('local/prefix_auc_hard_soft_pairs.csv');out['local_auc_soft_hard_M2']={k:stat(filt(a,M=2),k) for k in a[0] if k.startswith('soft_minus_hard_')}
 a=rows('weeds/pooled_spatial_budget_R_contrasts.csv');out['external_R4_R1_g8_nominal']={q:{k:stat(filt(a,grid_g=8,budget_basis='nominal_primary',parent_nominal_budget_code=q,contrast='R4_minus_R1'),k) for k in ['difference_weed_capture_pct','difference_crop_capture_pct','difference_crop_fraction_pct']} for q in ['b10','b20','b30','b50','b100']}
 a=rows('local/spatial_budget_M_contrasts.csv');out['local_M4_M2_R124']={q:stat([x for x in filt(a,contrast='M4_minus_M2',policy='prefix',budget=q) if x['R']!='6'],'difference_capture_pct') for q in budgets}
 a=rows('local/spatial_budget_all.csv');out['local_M2_R124_b10_capture']=stat([x for x in filt(a,M=2,policy='prefix',budget='0.1') if x['R']!='6'],'capture_pct')
 out['source_pins']=pins;out['report_review']={'path':str(review),'sha256':hashlib.sha256(review.read_bytes()).hexdigest()};out['note']='Exact saved contrasts and descriptive min/max/counts only. No fitting, inference, significance or reranking. Difference-count tolerance 1e-9 saved display units. Up/down for raw positive values is not interpreted.'
 out['presentation_extraction_recovery']={'prior_inline_attempt':'KeyError difference_capture_pct before writing any result file','cause':'Reporter hard/soft columns are named soft_minus_hard_*; R/M columns use difference_*.','correction':'Read actual CSV headers and use existing named columns; no outcome source changes.'}
 p=ROOT/'precision_agriculture/PA_results_narrative_extract_v5.json'
 with p.open('x',encoding='utf-8') as s:json.dump(out,s,ensure_ascii=False,indent=2)
 print(json.dumps({k:v for k,v in out.items() if 'group_values' not in k and k!='source_pins'},ensure_ascii=False))
if __name__=='__main__':main()
