"""Collect all eight declared spatial configurations; no inference or optimization."""
from pathlib import Path
import argparse, csv, datetime, hashlib, itertools, json, math, shutil
ROOT=Path(__file__).resolve().parent
PA=ROOT/'precision_agriculture'
MS=(1,2,4);RS=(1,2,4);BS=(.1,.2,.3,.5,1.)
CASES=set(itertools.product(MS,RS,BS))
HASHES={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):HASHES[str(p)]=sha(p);return p
def read(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with track(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvsave(p,data):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)
def boolean(x):
    assert x in (True,False,'True','False');return x is True or x=='True'
def case(x):return int(x['M']),int(x['R']),float(x['budget'])

def collect(bound_dir, independent_review):
    extension=read(PA/'RM_common_extension_protocol_v1.json')
    geometry=read(PA/'geometry_20260916_v1/metadata.json')
    domain={k:geometry[k] for k in ('native_valid_pixels','native_target_pixels','area_m2','target_m2','origin_utm_m','crs')}
    D=domain['area_m2'];T=domain['target_m2']
    status=read(bound_dir/'status.json');summary=read(bound_dir/'summary.json')
    manifest=read(bound_dir/'manifest.json');seal=read(bound_dir/'binding_seal.json')
    assert status['state']==summary['state']=='completed'
    assert summary['models']==8 and summary['cases']==360 and summary['all_existing_final_ids_unchanged']
    assert summary['all_final_selections_strictly_feasible']
    assert summary['solver_calls']==summary['new_inference_calls']==summary['new_final_selections']==0
    assert not summary['test_accessed'] and not summary['R6_accessed']
    assert {'summary.json','manifest.json','binding_seal.json','all360_policy_outcomes.csv'}.issubset(status['files_sha256'])
    for rel,h in status['files_sha256'].items():
        f=(bound_dir/rel.replace('\\','/')).resolve();assert f.is_relative_to(bound_dir.resolve())
        assert sha(f)==h;track(f)
    protocol=read(PA/'RM_bound_consistency_protocol_v1.json')
    assert sha(PA/'RM_bound_consistency_protocol_v1.json')==summary['protocol_sha256']==manifest['protocol_sha256']==seal['protocol_sha256']
    assert manifest['models']==protocol['models']
    assert sha(PA/'RM_common_extension_protocol_v1.json')==protocol['extension_protocol_sha256']
    assert sha(bound_dir/'binding_seal.json')==summary['binding_seal_sha256']
    review=read(independent_review)
    policy_path=bound_dir/'all360_policy_outcomes.csv'
    assert review['passed'] and review['passed_case_count']==360 and review['final_ids_unchanged']
    assert Path(review['audited_output_directory']).resolve()==bound_dir.resolve()
    assert review['all360_policy_outcomes_sha256']==sha(policy_path)
    assert review['source_binding_seal_sha256']==sha(bound_dir/'binding_seal.json')
    policy_rows=rows(policy_path)
    assert all(q['record_file'] in status['files_sha256'] and q['record_file'] in seal['records_sha256'] for q in policy_rows)
    assert all(status['files_sha256'][q['record_file']]==seal['records_sha256'][q['record_file']] for q in policy_rows)
    assert len(seal['records_sha256'])==360
    declared=[x['run_id'] for x in extension['declared_runs']]
    assert len(policy_rows)==360 and len(declared)==8
    assert {(q['run_id'],*case(q)) for q in policy_rows}==set(itertools.product(declared,MS,RS,BS))
    source_models={x['run_id']:x for x in manifest['models']}
    output=[];models=[]
    for run_id in declared:
        qs=[q for q in policy_rows if q['run_id']==run_id];assert len(qs)==45
        model=source_models[run_id]
        config=read(ROOT/'runs'/run_id/'config.json');training=read(ROOT/'runs'/run_id/'status.json')
        best=read(ROOT/'runs'/run_id/'best_validation.json')
        assert training['status']=='completed' and training['executed_epochs']==24 and not training['test_accessed']
        assert best['epoch']==model['checkpoint_epoch']==training['best_epoch']
        assert config['seed']==17 and config['variant'] in ('hard_seg','soft_seg')
        directory=Path(model['unit_tables_directory']);fixed_path=directory/'fixed_budgets.csv'
        assert sha(fixed_path)==model['table_sha256']['fixed_budgets.csv'];track(fixed_path)
        fixed={case(x):x for x in rows(fixed_path) if x['method']=='model_prefix'};assert set(fixed)==CASES
        meta={'run_id':run_id,'encoder':config['encoder'],'variant':config['variant'],'lr':config['lr'],
              'seed':17,'checkpoint_epoch':best['epoch'],'checkpoint_sha256':model['checkpoint_sha256'],
              'original_gpu_mae_pp':100*best['selection_balanced_mae'],'domain':domain,
              'source_directory':str(bound_dir),'original_output_directory':model['output'],
              'policy_csv':str(policy_path),'policy_csv_sha256':sha(policy_path),
              'recovery_applied':model['format']=='recovery_sealed','spatial_cases':45}
        current=[]
        for q in sorted(qs,key=case):
            m,r,b=case(q);f=fixed[(m,r,b)]
            assert q['encoder']==config['encoder'] and q['variant']==config['variant'] and float(q['lr'])==config['lr']
            assert int(q['checkpoint_epoch'])==best['epoch'] and q['checkpoint_sha256']==model['checkpoint_sha256']
            assert boolean(q['final_ids_unchanged']) and boolean(q['prefix_strictly_feasible'])
            record=read(bound_dir/q['record_file'])
            assert record['run_id']==run_id and (record['M'],record['R'],record['budget'])==(m,r,b)
            assert record['selected_ids']==json.loads(q['policy_selected_ids_json'])
            assert record['prefix_ids']==json.loads(q['prefix_selected_ids_json'])
            for a,z in [('actual_area','prefix_actual_area_m2'),('captured_target','prefix_captured_target_m2'),('capture_fraction','prefix_capture_fraction')]:close(f[a],q[z])
            fallback=boolean(q['fallback_used']);optimal=boolean(q['updated_proven_flag'])
            assert not(fallback and optimal) and not(boolean(q['bound_anomaly']) and optimal)
            for policy in ('prefix','predicted_area'):
                prefix='prefix' if policy=='prefix' else 'policy'
                area=float(q[prefix+'_actual_area_m2']);target=float(q[prefix+'_captured_target_m2'])
                budget=float(q['budget_area_m2']);close(budget,b*D);assert area<=budget
                close(q[prefix+'_capture_fraction'],target/T);close(q[prefix+'_actual_area_fraction'],area/D)
                close(q[prefix+'_missed_target_m2'],T-target);close(q[prefix+'_zero_annotation_area_m2'],area-target)
                row={k:meta[k] for k in ('run_id','encoder','variant','lr','seed','checkpoint_epoch','checkpoint_sha256','original_gpu_mae_pp')}
                row.update(policy=policy,M=m,R=r,budget=b,budget_area_m2=budget,
                    selected_count=int(q[prefix+'_selected_count']),actual_area_m2=area,actual_area_fraction=area/D,
                    captured_target_m2=target,capture_pct=100*target/T,missed_target_m2=T-target,
                    zero_annotation_area_m2=area-target,selected_zero_target_unit_area_m2=float(q[prefix+'_selected_zero_target_unit_area_m2']),
                    predicted_objective_m2=float(q[prefix+'_predicted_objective_m2']),unused_budget_m2=budget-area,
                    fallback_used=fallback if policy!='prefix' else False,
                    optimality_proven=optimal if policy!='prefix' else '',
                    predicted_relative_gap=float(q['updated_relative_gap']) if policy!='prefix' else '',
                    predicted_upper_m2=float(q['updated_upper_m2']) if policy!='prefix' else '',
                    original_proven_flag=boolean(q['original_proven_flag']) if policy!='prefix' else '',
                    original_retained_upper_m2=q['retained_upper_m2'] if policy!='prefix' else '',
                    original_retained_gap=q['original_retained_gap'] if policy!='prefix' else '',
                    bound_anomaly=boolean(q['bound_anomaly']) if policy!='prefix' else False,
                    certification_downgraded=boolean(q['certification_downgraded']) if policy!='prefix' else False,
                    domain_area_m2=D,domain_target_m2=T,source_csv=str(policy_path),source_csv_sha256=sha(policy_path),
                    fixed_budget_csv=str(fixed_path),fixed_budget_csv_sha256=sha(fixed_path),
                    binding_record=str(bound_dir/q['record_file']),binding_record_sha256=sha(bound_dir/q['record_file']),
                    historical_recovery=meta['recovery_applied'])
                current.append(row)
        pm=[x for x in current if x['policy']=='predicted_area'];pfx={(x['M'],x['R'],x['budget']):x for x in current if x['policy']=='prefix'}
        deltas=[x['capture_pct']-pfx[(x['M'],x['R'],x['budget'])]['capture_pct'] for x in pm]
        meta['policy_summary']={'gap_qualified':sum(x['optimality_proven'] for x in pm),'fallbacks':sum(x['fallback_used'] for x in pm),
            'bound_anomalies':sum(x['bound_anomaly'] for x in pm),'certification_downgrades':sum(x['certification_downgraded'] for x in pm),
            'capture_improved':sum(x>1e-8 for x in deltas),'capture_worsened':sum(x< -1e-8 for x in deltas),'capture_unchanged':sum(abs(x)<=1e-8 for x in deltas)}
        models.append(meta);output.extend(current)
    lookup={(x['encoder'],x['lr'],x['variant'],x['policy'],x['M'],x['R'],x['budget']):x for x in output}
    assert len(output)==len(lookup)==720
    pairs=[];counts=[]
    for encoder,lr,policy in itertools.product(('resnet18','convnextv2_tiny'),(1e-4,3e-4),('prefix','predicted_area')):
        current=[]
        for m,r,b in sorted(CASES):
            hard=lookup[(encoder,lr,'hard_seg',policy,m,r,b)];soft=lookup[(encoder,lr,'soft_seg',policy,m,r,b)]
            p={'encoder':encoder,'lr':lr,'policy':policy,'M':m,'R':r,'budget':b,
               'hard_capture_pct':hard['capture_pct'],'soft_capture_pct':soft['capture_pct'],
               'soft_minus_hard_pp':soft['capture_pct']-hard['capture_pct'],
               'hard_actual_area_m2':hard['actual_area_m2'],'soft_actual_area_m2':soft['actual_area_m2'],
               'soft_minus_hard_area_m2':soft['actual_area_m2']-hard['actual_area_m2'],
               'hard_fallback':hard['fallback_used'],'soft_fallback':soft['fallback_used'],
               'hard_gap_qualified':hard['optimality_proven'],'soft_gap_qualified':soft['optimality_proven'],
               'hard_source_sha256':hard['source_csv_sha256'],'soft_source_sha256':soft['source_csv_sha256']}
            pairs.append(p);current.append(p['soft_minus_hard_pp'])
        counts.append({'encoder':encoder,'lr':lr,'policy':policy,'higher':sum(x>1e-8 for x in current),
                       'lower':sum(x< -1e-8 for x in current),'same':sum(abs(x)<=1e-8 for x in current),'min_pp':min(current),'max_pp':max(current)})
    assert len(pairs)==360
    return output,pairs,models,counts

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bound-output',required=True);parser.add_argument('--independent-review',required=True)
    args=parser.parse_args()
    data,pairs,models,counts=collect(Path(args.bound_output).resolve(),Path(args.independent_review).resolve())
    out=PA/'common_management_reports'/('full8_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True);(out/'source').mkdir()
    shutil.copy2(__file__,out/'source'/Path(__file__).name)
    csvsave(out/'all_720_model_policy_budgets.csv',data)
    csvsave(out/'all_360_soft_minus_hard_pairs.csv',pairs)
    csvsave(out/'all_8_paired_direction_counts.csv',counts)
    primary=[x for x in pairs if x['M']==2 and x['budget']==.1]
    csvsave(out/'primary_M2_budget10_all_pairs.csv',primary)
    source_copies=[]
    for i,(path,h) in enumerate(HASHES.items()):
        p=Path(path);dest=out/'source'/(f'{i:03d}_'+p.name)
        assert sha(p)==h;shutil.copy2(p,dest);assert sha(dest)==h
        source_copies.append({'original':path,'copy':dest.relative_to(out).as_posix(),'sha256':h})
    save(out/'findings.json',{'models':models,'direction_counts':counts,'primary_M2_budget10':primary,
                            'scope':'all8 declared configurations; one scene, seed17, development only; no independent replication count'})
    lines=['# 完整共同骨干空间评价：8个配置','',
      '覆盖全部事先声明的8个分割配置：2骨干×2学习率×hard/soft。完整720条模型/政策结果与360组匹配差值均保留；这是同一场景、单种子开发比较，尚不是独立场景或封存确认结果。','',
      f"共同有效域{models[0]['domain']['area_m2']:.9f} m²，标注目标区域{models[0]['domain']['target_m2']:.9f} m²。M=2m为主分析，1/4m为敏感性；R1/2/4为模拟退化，预算10/20/30/50/100%。固定预算上限不表示各方法实际选入面积相等。",'',
      '## 全配置原检查点与政策状态','',
      '| 骨干 | LR | 标签 | epoch | 原MAE/pp | gap合格/45 | 后备/45 | 政策对前缀捕获升/降/平 |',
      '|---|---:|---|---:|---:|---:|---:|---|']
    for x in models:
        s=x['policy_summary']
        lines.append(f"|{x['encoder']}|{x['lr']}|{x['variant']}|{x['checkpoint_epoch']}|{x['original_gpu_mae_pp']:.6f}|{s['gap_qualified']}|{s['fallbacks']}|{s['capture_improved']}/{s['capture_worsened']}/{s['capture_unchanged']}|")
    lines+=['','## 全部45条件的soft−hard方向','', '| 骨干 | LR | 政策 | 较高 | 较低 | 相同 | 最小/pp | 最大/pp |','|---|---:|---|---:|---:|---:|---:|---:|']
    for x in counts:lines.append(f"|{x['encoder']}|{x['lr']}|{x['policy']}|{x['higher']}|{x['lower']}|{x['same']}|{x['min_pp']:+.6f}|{x['max_pp']:+.6f}|")
    lines+=['','这些相关条件的计数不是独立重复数，不作胜率或显著性检验。全部预算点和实际面积见CSV；不对五个离散预算点插值计算AUC。完整前缀曲线仍在各原始评价目录。','',
      '## 主分析：M2、10%预算的全部配对','',
      '| 骨干 | LR | 政策 | R | hard/% | soft/% | 差/pp | hard实际m² | soft实际m² |',
      '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for x in primary:lines.append(f"|{x['encoder']}|{x['lr']}|{x['policy']}|{x['R']}|{x['hard_capture_pct']:.6f}|{x['soft_capture_pct']:.6f}|{x['soft_minus_hard_pp']:+.6f}|{x['hard_actual_area_m2']:.6f}|{x['soft_actual_area_m2']:.6f}|")
    lines+=['','## 数值与证据边界','',
      '预测面积政策使用统一零超限规则。严格可行原解保留；否则采用原完整前缀后备（必要时只截尾）。后备永不标作已证明最优，原求解器status/gap与最终方案界限分开记录。原R18两组在新规则下选择完全等价；ConvNeXt hard LR1e-4保留原失败、单次诊断、一次后备及仅补齐缺项的恢复来源。其他五组通过新v2入口先保存原始solver返回与selector返回，再保存最终选择，45项封存后才关联标注。',
      f"随后统一对全部360个固定选择核对上界与已知可行前缀，{sum(x['policy_summary']['bound_anomalies'] for x in models)}项存在明显上界矛盾，{sum(x['policy_summary']['certification_downgrades'] for x in models)}项原认证降级。全部ID未变、0次重求；新表统一使用更新后的上界/gap/标记。最终{sum(x['policy_summary']['gap_qualified'] for x in models)}/360个标记满足已记录求解器标记、界限一致性与间隙条件。它仍不是独立重新求解证明，也不是实测管理收益。",
      '捕获为标注区域面积代理，不是实测防除率；非目标标注区域不能直接视为无任何杂草。未验证农药量、产量、作物损伤、可达性或经济成本。原标注为粗区域，不对应叶片覆盖或单株。',
      '本文不能把“预测误差较小未必决策较好”称为首次提出的理论。可能贡献是明确支持域、管理粒度与预算的具体联合证据；仍需外部来源复现、多种子稳定性、强回归基线诊断和最终封存确认。',
      '本汇总只读已有产物，未训练、推理、重求优化问题或使用test/R6。独立逐选择审核与图表人工可视检查在单独回执中衔接。']
    (out/'full_common_management_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(out/'provenance.json',{'source_sha256':HASHES,'source_copies':source_copies,'script_sha256':sha(__file__),
         'all8_models':models,'rows':720,'paired_rows':360,'test_accessed':False,'R6_accessed':False,
         'new_solver_calls':0,'inference_calls':0,'geometry_recomputed':False,'plots':'pending separate rendering and review'})
    save(out/'status.json',{'state':'tables_completed','rows':720,'pairs':360,'visual_review':'pending',
         'independent_production_audit':'required; see separate audit receipts before publication use',
         'file_sha256':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}})
    print(json.dumps({'directory':str(out),'direction_counts':counts},ensure_ascii=False))

if __name__=='__main__':main()
