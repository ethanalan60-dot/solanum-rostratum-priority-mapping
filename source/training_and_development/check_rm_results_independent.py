"""Read-only independent arithmetic audit; imports no production evaluator code."""
from pathlib import Path
import csv, json, math, hashlib, datetime, collections

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'precision_agriculture' / 'RM_evaluation_20260916_020003_962711'

def read_csv(name):
    with (OUT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def close(a, b, tol=1e-9):
    assert abs(a-b) <= tol * max(1., abs(a), abs(b)), (a, b)

def run():
    units = read_csv('unit_predictions.csv')
    fixed = read_csv('fixed_budgets.csv')
    curves = read_csv('decision_curves.csv')
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8-sig'))
    groups = {m: [r for r in units if int(r['M']) == m] for m in (1,2,4)}
    total_area = total_target = None
    cases = []
    for m, rows in groups.items():
        assert len({r['id'] for r in rows}) == len(rows)
        a = math.fsum(float(r['area']) for r in rows)
        t = math.fsum(float(r['target']) for r in rows)
        if total_area is None: total_area, total_target = a, t
        close(a,total_area); close(t,total_target)
        for r in rows:
            assert 0 <= float(r['target']) <= float(r['area']) + 1e-12
            for s in (1,2,4):
                p = float(r[f'predicted_R{s}'])
                assert 0 <= p <= float(r['area']) + 1e-12
                close(p/float(r['area']),float(r[f'q_R{s}']),1e-12)
        for s in (1,2,4,'label'):
            method = 'label_prefix' if s == 'label' else 'model_prefix'
            key = 'q_true' if s == 'label' else f'q_R{s}'
            ordered = sorted(rows, key=lambda r: (-float(r[key]),r['id']))
            selected_a, selected_t = [0.], [0.]
            for k in range(1,len(ordered)+1):
                selected_a.append(math.fsum(float(r['area']) for r in ordered[:k]))
                selected_t.append(math.fsum(float(r['target']) for r in ordered[:k]))
            cc = [r for r in curves if int(r['M']) == m and r['method'] == method and (s=='label' or int(r['R'])==s)]
            cc.sort(key=lambda r:int(r['k']))
            assert len(cc) == len(rows)+1
            for k, row in enumerate(cc):
                assert int(row['k']) == k
                close(float(row['actual_area']),selected_a[k])
                close(float(row['captured_target']),selected_t[k])
                if k: assert row['last_added_id'] == ordered[k-1]['id']
            ff = [r for r in fixed if int(r['M'])==m and r['method']==method and (s=='label' or int(r['R'])==s)]
            assert len(ff)==5
            for row in ff:
                b=float(row['budget']); budget_a=b*a
                k=len(rows) if b==1 else max(i for i,sa in enumerate(selected_a) if sa<=budget_a)
                assert k==int(row['k']), (m,s,b,k,row['k'])
                close(float(row['capture_fraction']),selected_t[k]/t)
                close(float(row['unused_budget_area']),budget_a-selected_a[k])
            cases.append({'M':m,'R':s,'prefix_points_checked':len(cc),'budgets_checked':len(ff)})
        by_id = {r['id']:r for r in rows}
        for opt in summary['results']['by_M'][str(m)]['label_milp']:
            ids=opt['selected_ids']; assert len(ids)==len(set(ids))
            selected=[by_id[i] for i in ids]
            sa=math.fsum(float(r['area']) for r in selected)
            st=math.fsum(float(r['target']) for r in selected)
            assert sa <= opt['budget']*a
            close(sa,opt['selected_area_feasibility_audit']); close(st,opt['primal_target'])
            assert opt['solver_status']==0 and opt['optimality_proven']
            close(st,opt['dual_target_bound'],1e-8)
        if m>1:
            parents=collections.defaultdict(list)
            for r in groups[1]: parents[(int(r['ix'])//m,int(r['iy'])//m)].append(r)
            assert set(parents)=={(int(r['ix']),int(r['iy'])) for r in rows}
            for r in rows:
                child=parents[(int(r['ix']),int(r['iy']))]
                for col in ('area','target','predicted_R1','predicted_R2','predicted_R4'):
                    close(math.fsum(float(c[col]) for c in child),float(r[col]))
    return {'passed':True,'reviewer':'root','created_local':datetime.datetime.now().astimezone().isoformat(),
            'independence':'Python standard library only; no production code imports and no model/raw-data access',
            'cases':cases,'label_milp_selected_set_audits':15,'nested_grid_checks':[2,4],
            'limits':'Recomputed arithmetic and recorded primal/dual certificates; not a separate MILP solver proof or independent field validation',
            'inputs_sha256':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ('unit_predictions.csv','fixed_budgets.csv','decision_curves.csv','summary.json')},
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'test_accessed':False}

if __name__=='__main__':
    result=run()
    target=ROOT/'precision_agriculture'/'RM_results_independent_checks_20260916.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'cases':len(result['cases']),'output':str(target)},ensure_ascii=False))
