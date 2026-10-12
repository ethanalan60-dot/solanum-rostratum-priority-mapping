"""Fixed complete descriptive findings from the accepted24-model development tables."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,math,statistics
ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary-directory',type=Path,required=True);p.add_argument('--independent-review',type=Path,required=True);a=p.parse_args();folder=a.summary_directory.resolve()
    watched={str(p.resolve()):sha(p) for p in (a.independent_review,folder/'status.json',folder/'summary.json',folder/'artifact_manifest.json',Path(__file__))}
    review=read(a.independent_review);assert review['passed'] and review['scope']=='full24_seed_summary_independent' and review['models']==24 and review['tables_audited']==25
    assert Path(review['output_directory']).resolve()==folder
    status=read(folder/'status.json');assert status['state']=='completed'
    assert review['summary_sha256']==sha(folder/'summary.json') and review['audited_artifact_manifest_sha256']==sha(folder/'artifact_manifest.json')==status['artifact_manifest_sha256']
    manifest=read(folder/'artifact_manifest.json')['file_sha256']
    names={'all2160_model_policy_budgets':2160,'all24_cpu_scalar_model_summary':24,'all1080_hard_soft_budget_pairs':1080,'all108_hard_soft_auc_pairs':108,'all_budget_R_contrasts':2160,'all_budget_M_contrasts':2160}
    tables={}
    for name,count in names.items():
        path=folder/(name+'.csv');assert sha(path)==manifest[path.name];watched[str(path)]=manifest[path.name];tables[name]=load(path)
        assert len(tables[name])==count==review['table_rows'][name]
    for path,digest in watched.items():assert sha(path)==digest
    def finite_values(values):
        values=[None if v in ('',None) else float(v) for v in values];assert all(v is None or math.isfinite(v) for v in values)
        return values,[v for v in values if v is not None]
    def directions(values,eps=1e-8):
        values,finite=finite_values(values)
        return dict(n=len(values),higher=sum(v>eps for v in finite),lower=sum(v< -eps for v in finite),tied=sum(abs(v)<=eps for v in finite),NA=sum(v is None for v in values),minimum=min(finite) if finite else None,maximum=max(finite) if finite else None,tolerance=eps)
    def range_summary(values):
        values,finite=finite_values(values)
        return dict(n=len(values),n_valid=len(finite),NA=len(values)-len(finite),minimum=min(finite) if finite else None,maximum=max(finite) if finite else None)
    def show_range(d):return 'NA' if d['minimum'] is None else f"{d['minimum']:.6f} to {d['maximum']:.6f}"
    report={'scope':'development, fixed24 models, all three seeds','field_replicates':1,'seeds':[17,29,43],'primary_M':2,'primary_budget':.1,'primary_policy':'prefix','direction_counts_are_not_independent_field_replicates':True,'units':{'capture_pct':'percent','capture_difference':'percentage points','CPU_error':'percentage points','AUC':'unitless fraction','actual_area':'m2'},'contrasts':[]}
    for policy in ('prefix','predicted_area'):
        for axis,criteria in [('R',{'M':'2','contrast':'R4_minus_R1'}),('M',{'contrast':'M4_minus_M2'})]:
            rows=[r for r in tables[f'all_budget_{axis}_contrasts'] if r['policy']==policy and float(r['budget'])==.1 and all(r[k]==v for k,v in criteria.items())]
            assert len(rows)==(24 if axis=='R' else 72)
            report['contrasts'].append(dict(policy=policy,axis=axis,nominal_budget=.1,criteria=criteria,capture_difference_pp=directions([r['difference_capture_pct'] for r in rows]),actual_area_difference_m2=directions([r['difference_actual_area_m2'] for r in rows],1e-10)))
    report['all_hard_soft_budget_differences']=directions([r['soft_minus_hard_capture_pct'] for r in tables['all1080_hard_soft_budget_pairs']]);assert report['all_hard_soft_budget_differences']['n']==1080
    report['primary_M2_hard_soft_AUC']={metric:directions([r['soft_minus_hard_'+metric] for r in tables['all108_hard_soft_auc_pairs'] if r['M']=='2'],1e-12) for metric in ('auc_0_1','auc_0_0p5_normalized')}
    assert all(d['n']==36 for d in report['primary_M2_hard_soft_AUC'].values())
    report['primary_M2_budget10_hard_soft_pairs']={policy:[r for r in tables['all1080_hard_soft_budget_pairs'] if r['M']=='2' and float(r['budget'])==.1 and r['policy']==policy] for policy in ('prefix','predicted_area')}
    assert all(len(group)==36 for group in report['primary_M2_budget10_hard_soft_pairs'].values())
    report['primary_M2_budget10_hard_soft_differences']={policy:directions([r['soft_minus_hard_capture_pct'] for r in group]) for policy,group in report['primary_M2_budget10_hard_soft_pairs'].items()}
    models=tables['all24_cpu_scalar_model_summary'];scalar=[];pairs=[]
    for enc in ('resnet18','convnextv2_tiny'):
        for lr in (.0001,.0003):
            for variant in ('hard_seg','soft_seg'):
                group=[r for r in models if (r['encoder'],float(r['lr']),r['variant'])==(enc,lr,variant)];assert {int(r['seed']) for r in group}=={17,29,43} and len(group)==3
                vals=[float(r['cpu_three_R_balanced_mae_pp']) for r in group];assert all(math.isfinite(v) for v in vals);scalar.append(dict(encoder=enc,lr=lr,variant=variant,n=3,CPU_three_R_balanced_MAE_pp_mean=statistics.mean(vals),CPU_three_R_balanced_MAE_pp_sample_SD=statistics.stdev(vals),original_epochs=';'.join(r['seed']+':'+r['checkpoint_epoch'] for r in sorted(group,key=lambda r:int(r['seed'])))))
            for seed in (17,29,43):
                hs={r['variant']:float(r['cpu_three_R_balanced_mae_pp']) for r in models if (r['encoder'],float(r['lr']),int(r['seed']))==(enc,lr,seed)};pairs.append(hs['soft_seg']-hs['hard_seg'])
    report['paired_model_CPU_error_soft_minus_hard_pp']=directions(pairs);assert len(pairs)==12
    report['primary_capture_range_by_policy']={policy:range_summary([r['capture_pct'] for r in tables['all2160_model_policy_budgets'] if r['M']=='2' and float(r['budget'])==.1 and r['policy']==policy]) for policy in ('prefix','predicted_area')}
    assert all(d['n']==72 for d in report['primary_capture_range_by_policy'].values())
    report['CPU_configuration_summary']=scalar
    lines=['# Complete three-seed development findings','', 'All24 fits retain their original checkpoint. Counts below describe repeated configurations/settings on one scene; they are not independent field replications. These are development results and do not replace confirmation results.','', '## Original-checkpoint CPU performance','', '| Encoder | Learning rate | Supervision | CPU balanced MAE, mean ± sample SD (pp) |','|---|---:|---|---:|']
    for row in scalar:lines.append(f"| {row['encoder']} | {row['lr']:g} | {row['variant']} | {row['CPU_three_R_balanced_MAE_pp_mean']:.6f} ± {row['CPU_three_R_balanced_MAE_pp_sample_SD']:.6f} |")
    lines.extend(['','## Fixed primary and scale comparisons',''])
    for policy,d in report['primary_capture_range_by_policy'].items():
        lines.append(f"- M2, 10% nominal budget, {policy}: {show_range(d)}% capture across all {d['n']} model–R settings ({d['NA']} NA).")
    for policy,d in report['primary_M2_budget10_hard_soft_differences'].items():
        lines.append(f"- M2, 10% nominal budget, {policy}, soft−hard: {d['higher']} higher / {d['lower']} lower / {d['tied']} tied / {d['NA']} NA across all {d['n']} matched settings; difference {show_range(d)} pp.")
    for row in report['contrasts']:
        d=row['capture_difference_pp'];scope='R4−R1, M2' if row['axis']=='R' else 'M4−M2, all R'
        lines.append(f"- {row['policy']}, 10% nominal budget, {scope}: {d['higher']} higher / {d['lower']} lower / {d['tied']} tied / {d['NA']} NA among {d['n']} paired settings; capture difference {show_range(d)} pp. Actual-area differences are retained in findings.json.")
    d=report['all_hard_soft_budget_differences'];lines.extend(['',f"All1,080 matched supervision/budget/settings pairs: soft−hard capture was higher in {d['higher']}, lower in {d['lower']}, tied in {d['tied']}, and undefined in {d['NA']}. Full grids and adverse cases remain in the source tables.", '', '## Interpretation','', 'The comparison measures annotated target-region capture and selection area under fixed budgets. Image-information degradation R and whole-unit size M are evaluated separately. Seed SD reflects fitting variation. These observations do not establish herbicide savings, realised control, crop injury, profit or independent-field generalisation.'])
    out=PA/('full24_key_findings_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();dump(out/'findings.json',report);(out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for path,digest in watched.items():assert sha(path)==digest
    dump(out/'provenance.json',{'source_bindings':watched,'summary_review':str(a.independent_review.resolve()),'new_model_calls':0,'new_solver_calls':0,'test_or_R6_accessed':False})
    dump(out/'artifact_manifest.json',{'files_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()}});print(json.dumps({'out':str(out),'models':24,'direction_counts':report['contrasts']}))
if __name__=='__main__':main()
