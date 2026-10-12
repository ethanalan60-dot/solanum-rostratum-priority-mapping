"""Descriptive complete-factor contrasts from accepted tables; no model/solver/data access."""
from pathlib import Path
import collections,csv,datetime,hashlib,itertools,json,math,shutil
ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture'
FIXED=PA/'common_management_reports/full8_20260916_050927_560091/all_720_model_policy_budgets.csv'
AUC=PA/'common_prefix_auc_full8_20260916_052212_839933/all72_model_prefix_auc.csv'
PAIRS=((1,2),(2,4),(1,4))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def readcsv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(p,rows):
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def compare_fixed(rows,axis):
    other='M' if axis=='R' else 'R';index={}
    for row in rows:
        key=(row['run_id'],row['policy'],int(row[other]),float(row['budget']))
        group=index.setdefault(key,{});assert int(row[axis]) not in group;group[int(row[axis])]=row
    out=[]
    for key,group in sorted(index.items()):
        assert set(group)=={1,2,4}
        for lo,hi in PAIRS:
            a,b=group[lo],group[hi]
            assert a['checkpoint_sha256']==b['checkpoint_sha256']
            assert float(a['budget_area_m2'])==float(b['budget_area_m2'])
            row={'axis':axis,'run_id':key[0],'policy':key[1],other:key[2],'budget':key[3],
                 'from_level':lo,'to_level':hi,'encoder':a['encoder'],'variant':a['variant'],'lr':a['lr'],'seed':a['seed'],
                 'checkpoint_sha256':a['checkpoint_sha256'],'common_budget_area_m2':float(a['budget_area_m2'])}
            for name in ('capture_pct','actual_area_m2','captured_target_m2','missed_target_m2','unused_budget_m2'):
                row['from_'+name]=float(a[name]);row['to_'+name]=float(b[name]);row['to_minus_from_'+name]=float(b[name])-float(a[name])
            row['capture_difference_unit']='percentage points'
            row['same_actual_selected_area']=float(a['actual_area_m2'])==float(b['actual_area_m2'])
            out.append(row)
    assert len(out)==720
    return out
def compare_auc(rows,axis):
    other='M' if axis=='R' else 'R';index={}
    for row in rows:index.setdefault((row['run_id'],int(row[other])),{})[int(row[axis])]=row
    out=[]
    for key,group in sorted(index.items()):
        assert set(group)=={1,2,4}
        for lo,hi in PAIRS:
            a,b=group[lo],group[hi]
            row={'axis':axis,'run_id':key[0],'policy':'prefix',other:key[1],'from_level':lo,'to_level':hi,
                 'checkpoint_sha256':a['checkpoint_sha256'],'auc_difference_unit':'dimensionless'}
            assert a['checkpoint_sha256']==b['checkpoint_sha256']
            for metric in ('auc_0_1','auc_0_0p5_normalized'):
                row['from_'+metric]=float(a[metric]);row['to_'+metric]=float(b[metric]);row['to_minus_from_'+metric]=float(b[metric])-float(a[metric])
            out.append(row)
    assert len(out)==72
    return out
def summaries(rows,axis):
    other='M' if axis=='R' else 'R';groups={}
    for row in rows:groups.setdefault((row['policy'],row[other],row['budget'],row['from_level'],row['to_level']),[]).append(row)
    out=[]
    for key,rr in sorted(groups.items()):
        assert len(rr)==8
        values=[x['to_minus_from_capture_pct'] for x in rr]
        out.append({'axis':axis,'policy':key[0],other:key[1],'budget':key[2],'from_level':key[3],'to_level':key[4],
            'configurations':8,'higher_count':sum(x>1e-9 for x in values),'lower_count':sum(x < -1e-9 for x in values),
            'numerical_tie_count':sum(abs(x)<=1e-9 for x in values),'minimum_difference_pp':min(values),'maximum_difference_pp':max(values),
            'mean_difference_pp':math.fsum(values)/8,'minimum_area_difference_m2':min(x['to_minus_from_actual_area_m2'] for x in rr),
            'maximum_area_difference_m2':max(x['to_minus_from_actual_area_m2'] for x in rr)})
    return out
def main():
    # Pin against the independently accepted central and AUC artifacts.
    status=json.loads((FIXED.parent/'status.json').read_text(encoding='utf-8-sig'))
    assert sha(FIXED)==status['file_sha256'][FIXED.name]
    auc_review=json.loads((PA/'prefix_auc_independent_20260916_052458_065751/review.json').read_text(encoding='utf-8-sig'))
    assert auc_review['passed'] and sha(AUC)==auc_review['all72_model_prefix_auc_sha256']
    out=PA/datetime.datetime.now().strftime('scale_contrasts_%Y%m%d_%H%M%S_%f');out.mkdir()
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Descriptive synthesis of the entire previously declared and already inspected development matrix; not a new preregistered experiment.',
        'source_sha256':{str(FIXED):sha(FIXED),str(AUC):sha(AUC)},'pairs':PAIRS,'axes':['R','M'],
        'expected_fixed_contrasts_per_axis':720,'expected_prefix_AUC_contrasts_per_axis':72,
        'difference':'higher level minus lower level; larger R means degraded image; larger M means coarser unit',
        'primary':'M2; R1-to-R4 contrasts with all five budgets and both policies; retain other pairs and M sensitivity',
        'numerical_tie_tolerance_pp':1e-9,'no_significance':True,'no_independent_field_inference':True,
        'M_caution':'Common nominal budget may differ in actual expenditure; include area differences; no isolated causal scale effect asserted.',
        'training':False,'new_spatial_selection':False,'test_accessed':False,'R6_accessed':False}
    write(out/'protocol.json',protocol);shutil.copy2(__file__,out/'source.py')
    fixed=readcsv(FIXED);aucs=readcsv(AUC);assert len(fixed)==720 and len(aucs)==72
    results={}
    for axis in ('R','M'):
        differences=compare_fixed(fixed,axis);a=compare_auc(aucs,axis);s=summaries(differences,axis)
        table(out/f'all720_{axis}_fixed_contrasts.csv',differences);table(out/f'all72_{axis}_prefix_auc_contrasts.csv',a);table(out/f'{axis}_fixed_summary.csv',s)
        results[axis]={'fixed_contrasts':len(differences),'prefix_auc_contrasts':len(a),'summary_rows':len(s)}
        if axis=='R':primary=[x for x in s if x['M']==2 and x['from_level']==1 and x['to_level']==4]
    text=['# Complete descriptive R and M contrasts','','All configurations retained. Differences are coarse minus fine. Counts refer to correlated model settings, not field replicates.','',
        '## Primary M = 2 m: R4 minus R1','','| Policy | Budget | Higher/lower/tied across 8 configurations | Capture difference range (pp) | Selected-area difference range (m²) |','|---|---:|---|---|---|']
    for x in primary:text.append(f"|{x['policy']}|{x['budget']:.0%}|{x['higher_count']}/{x['lower_count']}/{x['numerical_tie_count']}|{x['minimum_difference_pp']:.6f} to {x['maximum_difference_pp']:.6f}|{x['minimum_area_difference_m2']:.6f} to {x['maximum_area_difference_m2']:.6f}|")
    text+=['','Both adjacent contrasts and R1-to-R4 / M1-to-M4 are provided in full. AUC uses the already accepted complete staircase integrals, with no new curve fitting or integration.','',
        'This summary creates no new checkpoint, prediction, selected set or outcome. A numerical tie threshold of 1e-9 pp is a rounding convention, not a minimum agronomic effect or significance threshold.']
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    assert sha(FIXED)==protocol['source_sha256'][str(FIXED)] and sha(AUC)==protocol['source_sha256'][str(AUC)]
    write(out/'summary.json',{'completed':True,'counts':results,'primary_R1_to_R4_M2':primary,'independent_review':'pending','test_accessed':False,'R6_accessed':False})
    write(out/'artifact_manifest.json',{'files_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'output':str(out),'counts':results,'independent_review':'pending'},ensure_ascii=False))
if __name__=='__main__':main()
