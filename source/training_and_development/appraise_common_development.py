"""Audit all 12 completed original validation curves without inference or selection changes."""
from pathlib import Path
import csv, datetime, hashlib, json, math, shutil

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def main():
    plan_path = ROOT/'common_queue_plan_v2.json'
    plan = read(plan_path)
    assert len(plan['runs']) == 12 and not plan['test_accessed']
    scale_rows, run_rows, curves, inputs = [], [], [], {}
    zero_reference = None
    for entry in plan['runs']:
        run = ROOT/'runs'/entry['run_id']
        status = read(run/'status.json')
        assert status['status'] == 'completed' and status['executed_epochs'] == 24
        assert not status['test_accessed']
        config = read(run/'config.json')
        assert config['n_train'] == 985 and config['n_val'] == 357
        history = [json.loads(x) for x in (run/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [x['epoch'] for x in history] == list(range(1,25))
        best = min(history, key=lambda x:(x['selection_balanced_mae'], x['epoch']))
        assert best['epoch'] == status['best_epoch']
        assert math.isclose(best['selection_balanced_mae'], status['best_validation_balanced_mae'], rel_tol=1e-12)
        constants = read(run/'constant_baselines.json')
        if zero_reference is None:
            zero_reference = constants['zero']
        assert constants['zero'] == zero_reference
        for epoch in history:
            assert math.isclose(math.fsum(v['balanced_mae'] for v in epoch['scales'].values())/3,
                                epoch['selection_balanced_mae'], rel_tol=1e-12)
            for r, metrics in epoch['scales'].items():
                assert metrics['n'] == 357 and metrics['n_positive'] == 27
                assert math.isclose((metrics['mae_zero']+metrics['mae_positive'])/2,
                                    metrics['balanced_mae'], rel_tol=1e-12)
                assert math.isclose((330*metrics['mae_zero']+27*metrics['mae_positive'])/357,
                                    metrics['mae'], rel_tol=1e-10, abs_tol=1e-12)
            curves.append({'run_id':entry['run_id'], 'epoch':epoch['epoch'],
                           'balanced_mae_pp':100*epoch['selection_balanced_mae'],
                           'train_loss':epoch['train_loss'],
                           'actual_updates':epoch['optimizer_updates'],
                           'skipped_updates':epoch['optimizer_skipped_overflows']})
        meta = {k:entry[k] for k in ('run_id','encoder','variant','lr','seed')}
        row = dict(meta, best_epoch=best['epoch'], balanced_mae_pp=100*best['selection_balanced_mae'],
                   zero_baseline_mae_pp=100*constants['zero']['balanced_mae'],
                   improvement_over_zero_pp=100*(constants['zero']['balanced_mae']-best['selection_balanced_mae']),
                   train_mean_baseline_mae_pp=100*constants['train_mean']['balanced_mae'],
                   epoch24_mae_pp=100*history[-1]['selection_balanced_mae'],
                   actual_updates=status['optimizer_updates_total'], skipped_updates=status['optimizer_skips_total'])
        run_rows.append(row)
        for r, metrics in best['scales'].items():
            # bias = mean(prediction - truth), hence the same unweighted mean truth recovers mean prediction.
            mean_prediction = -constants['zero']['bias'] + metrics['bias']
            scale_rows.append(dict(meta, best_epoch=best['epoch'], R=int(r),
                                  **{k+'_pp':100*metrics[k] for k in ('mae','bias','mae_zero','mae_positive','balanced_mae')},
                                  mean_predicted_fraction=mean_prediction,
                                  zero_stratum_contribution_pp=50*metrics['mae_zero'],
                                  positive_stratum_contribution_pp=50*metrics['mae_positive']))
        for name in ('config.json','status.json','metrics.jsonl','constant_baselines.json','best_validation.json'):
            inputs[str(run/name)] = sha(run/name)
    stamp = datetime.datetime.now().astimezone()
    out = ROOT/'common_development_reports'/('appraisal_'+stamp.strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    write_csv(out/'all_12_original_checkpoints.csv', run_rows)
    write_csv(out/'all_36_original_scale_metrics.csv', scale_rows)
    write_csv(out/'all_288_training_epochs.csv', curves)
    pair_rows=[]
    for encoder in ('resnet18','convnextv2_tiny'):
        for lr in (1e-4,3e-4):
            hard=next(x for x in run_rows if x['encoder']==encoder and x['lr']==lr and x['variant']=='hard_seg')
            soft=next(x for x in run_rows if x['encoder']==encoder and x['lr']==lr and x['variant']=='soft_seg')
            pair_rows.append({'encoder':encoder,'lr':lr,'hard_mae_pp':hard['balanced_mae_pp'],
                              'soft_mae_pp':soft['balanced_mae_pp'],
                              'soft_minus_hard_pp':soft['balanced_mae_pp']-hard['balanced_mae_pp']})
    write_csv(out/'all_4_hard_soft_pairs.csv',pair_rows)
    lines=['# 完整单种子共同骨干开发比较：诊断与解释边界','',
           f'生成时间：{stamp.isoformat()}。全部12个原计划配置、24轮曲线、36个原最佳尺度指标已复算。',
           '这是原GPU AMP验证记录的只读分析；不重新推理、不更换检查点、不使用封存test或R6。', '',
           '| 骨干 | 方法 | LR | 原最佳epoch | 分层MAE/pp | 比全零改善/pp |',
           '|---|---|---:|---:|---:|---:|']
    for x in run_rows:
        lines.append(f"|{x['encoder']}|{x['variant']}|{x['lr']}|{x['best_epoch']}|{x['balanced_mae_pp']:.6f}|{x['improvement_over_zero_pp']:.6f}|")
    lines += ['', '## 完整配对', '', '| 骨干 | LR | soft−hard MAE/pp（负值为soft较低） |', '|---|---:|---:|']
    for p in pair_rows:
        lines.append(f"|{p['encoder']}|{p['lr']}|{p['soft_minus_hard_pp']:.6f}|")
    weak=next(x for x in run_rows if x['encoder']=='convnextv2_tiny' and x['variant']=='regression' and x['lr']==3e-4)
    weak_scales=[x for x in scale_rows if x['run_id']==weak['run_id']]
    means=[x['mean_predicted_fraction'] for x in weak_scales]
    lines += ['', '## 直接回归基线的局限','',
      f"ConvNeXt回归LR3e-4的原最佳为epoch{weak['best_epoch']}，相对全零基线仅改善{weak['improvement_over_zero_pp']:.8f}个百分点。由各尺度bias反算的非加权平均预测比例为{min(means):.10g}–{max(means):.10g}；对应全零基线的平均真实比例为{-zero_reference['bias']:.10g}。这些记录与接近全零的预测相符。",
      '这不是缺失或失败配置，原24轮记录继续保留；但它不能单独支持对直接比例回归的强结论。是否由损失、sigmoid饱和、抽样或学习率引起仍未经机制实验。其他LR和骨干的回归结果一并报告，未来诊断须给各方法匹配的开发预算。', '',
      '## 农业解释边界','',
      '分层MAE给330个零目标窗口与27个正目标窗口各一半权重；自然频率MAE约为92.44%零窗口和7.56%正窗口的加权平均。指标回答的问题不同，因此同时保留各分层误差、自然频率误差和bias。',
      '模型MAE排序不能替代同域、同预算、相同管理单元下的标注目标面积捕获比较。当前原始场景只有一个空间来源，seed17只有一次；学习率、尺度和预算条件不作为独立重复，不计算伪重复显著性。',
      '空间捕获可帮助评价有限面积下的复核或定点选择优先级；它不是实测防效、节药、产量或成本收益。完整空间比较、外部来源复现、多种子稳定性与封存确认仍需完成。', '',
      '此诊断在观察全部开发结果后形成，不是预注册假设；未据此增加、删除或重选任何当前配置。']
    (out/'critical_appraisal.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    shutil.copy2(__file__,out/Path(__file__).name)
    manifest={'created_local':stamp.isoformat(),'plan_sha256':sha(plan_path),'inputs':inputs,
              'rows':{'runs':len(run_rows),'scale_metrics':len(scale_rows),'epochs':len(curves),'paired_comparisons':len(pair_rows)},
              'test_accessed':False,'R6_accessed':False,'inference_calls':0,'checkpoint_selection_changed':False,
              'artifact_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'directory':str(out),'rows':manifest['rows']},ensure_ascii=False))

if __name__=='__main__':
    main()
