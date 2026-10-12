"""Audit saved P-MILP decisions with stdlib only; no production code imports."""
from pathlib import Path
import json,csv,math,hashlib,datetime
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'precision_agriculture'/'RM_policy_diagnostic_20260916_021512_266975'

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)

def run():
    assert read(OUT/'status.json')['state']=='completed'
    inputs=read(OUT/'inputs.json')
    for p,h in inputs['original_input_hashes'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    hashes=read(OUT/'product_hashes.json')
    for p,h in hashes.items():assert hashlib.sha256((OUT/p).read_bytes()).hexdigest()==h
    units=read_csv(OUT/'source'/'unit_predictions.csv')
    rows=read_csv(OUT/'all_45_deltas.csv'); assert len(rows)==45
    combos={(int(r['M']),int(r['R']),float(r['budget'])) for r in rows}
    assert combos=={(m,r,b) for m in (1,2,4) for r in (1,2,4) for b in (.1,.2,.3,.5,1.)}
    decomposition=[]; proven=0; increased=decreased=unchanged=0
    for row in rows:
        m,r,b=int(row['M']),int(row['R']),float(row['budget'])
        group={x['id']:x for x in units if int(x['M'])==m}
        sol=read(OUT/row['solution_file'])
        ev=read((OUT/row['solution_file']).with_name(Path(row['solution_file']).stem+'_evaluation.json'))
        chosen=set(sol['selected_ids']); original=set(ev['prefix_selected_ids'])
        assert len(chosen)==len(sol['selected_ids']) and chosen<=group.keys()
        total_a=math.fsum(float(x['area']) for x in group.values())
        total_t=math.fsum(float(x['target']) for x in group.values())
        def agg(ids,col):return math.fsum(float(group[i][col]) for i in ids)
        assert agg(chosen,'area')<=b*total_a
        for prefix,ids in (('pmilp',chosen),('prefix',original)):
            close(agg(ids,'area'),float(row[prefix+'_actual_area_m2']))
            close(agg(ids,'target'),float(row[prefix+'_captured_target_m2']))
            close(agg(ids,f'predicted_R{r}'),float(row[prefix+'_predicted_objective_m2']))
            close(agg(ids,'target')/total_t,float(row[prefix+'_capture_fraction']))
        delta_t=agg(chosen,'target')-agg(original,'target')
        close(100*delta_t/total_t,float(row['delta_capture_percentage_points']))
        assert agg(chosen,f'predicted_R{r}')>=agg(original,f'predicted_R{r}')-1e-10
        close(delta_t,agg(chosen-original,'target')-agg(original-chosen,'target'))
        assert set(ev['added_ids'])==chosen-original and set(ev['removed_ids'])==original-chosen
        proven+=sol['optimality_proven_within_tolerance']
        d=100*delta_t/total_t
        increased+=d>1e-8; decreased+=d < -1e-8; unchanged+=abs(d)<=1e-8
        if m==4 and b==.1:
            decomposition.append({'M':m,'R':r,'budget':b,'added_units':len(chosen-original),'removed_units':len(original-chosen),
              'added_target_m2':agg(chosen-original,'target'),'removed_target_m2':agg(original-chosen,'target'),
              'delta_target_m2':delta_t,'delta_capture_pp':d,'proven':sol['optimality_proven_within_tolerance']})
    assert (proven,increased,decreased,unchanged)==(38,3,2,40)
    return {'passed':True,'reviewed_local':datetime.datetime.now().astimezone().isoformat(),'reviewer':'root',
      'all_input_and_product_hashes_match':True,'combinations_checked':45,'all_original_area_budgets_feasible':True,
      'all_objectives_and_target_outcomes_recomputed':True,'optimality_certificates_meeting_recorded_tolerance':proven,
      'true_capture_improved':increased,'true_capture_worsened':decreased,'true_capture_unchanged':unchanged,
      'M4_10pct_added_removed_decomposition':decomposition,
      'static_review':'Selector signature and body use only IDs, valid area and predictions; fixed protocol/hash and all choices precede annotation evaluation. No secondary target-based tie selection.',
      'limits':'Audit of supplied solver certificates, not a second optimization solver. Full objective improvements do not imply true target gains.',
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'test_accessed':False}
if __name__=='__main__':
    result=run(); p=ROOT/'precision_agriculture'/'RM_policy_independent_checks_20260916.json'
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
