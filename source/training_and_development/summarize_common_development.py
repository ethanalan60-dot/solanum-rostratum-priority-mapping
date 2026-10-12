"""Read-only, timestamped summary of the full common_v2 development plan."""
from pathlib import Path
import csv,datetime,hashlib,json,math
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=read(ROOT/'common_queue_plan_v2.json')
    queue=read(ROOT/'common_queue_status.json')
    assert plan['version']=='common_queue_v2' and not plan['test_accessed']
    records=[];details=[];checks=[];inputs={}
    for entry in plan['runs']:
        run=ROOT/'runs'/entry['run_id']
        status=read(run/'status.json') if (run/'status.json').exists() else {'status':'not_started'}
        row={'run_id':entry['run_id'],'encoder':entry['encoder'],'variant':entry['variant'],'lr':entry['lr'],
             'seed':entry['seed'],'status':status['status'],'best_epoch':'','best_validation_mae_pp':'',
             'attempted_updates':'','actual_updates':'','skipped_overflows':''}
        if status['status']=='completed':
            config=read(run/'config.json')
            assert config['stage']=='development' and not config['smoke'] and not config['diagnostic']
            assert config['n_train']==985 and config['n_val']==357 and status['executed_epochs']==24
            assert not status['test_accessed']
            history=[json.loads(x) for x in (run/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            assert [x['epoch'] for x in history]==list(range(1,25))
            for x in history:
                assert set(x['scales'])=={'1','2','4'}
                expected=math.fsum(x['scales'][r]['balanced_mae'] for r in ('1','2','4'))/3
                assert math.isclose(expected,x['selection_balanced_mae'],rel_tol=1e-12)
                assert all(x['scales'][r]['n']==357 and x['scales'][r]['n_positive']==27 for r in ('1','2','4'))
            best=min(history,key=lambda x:(x['selection_balanced_mae'],x['epoch']))
            assert best['epoch']==status['best_epoch']
            assert math.isclose(best['selection_balanced_mae'],status['best_validation_balanced_mae'],rel_tol=1e-12)
            for status_key,history_key in [('optimizer_attempts_total','optimizer_update_attempts'),('optimizer_updates_total','optimizer_updates'),('optimizer_skips_total','optimizer_skipped_overflows')]:
                assert sum(x[history_key] for x in history)==status[status_key]
            assert status['optimizer_attempts_total']==status['optimizer_updates_total']+status['optimizer_skips_total']
            row.update(best_epoch=best['epoch'],best_validation_mae_pp=100*best['selection_balanced_mae'],
                       attempted_updates=status['optimizer_attempts_total'],actual_updates=status['optimizer_updates_total'],skipped_overflows=status['optimizer_skips_total'])
            details.append({'run_id':entry['run_id'],'best':best,'complete_curve':history,'status':status})
            for p in (run/'config.json',run/'metrics.jsonl',run/'status.json',run/'best.pt'):inputs[str(p)]=sha(p)
            checks.append({'run_id':entry['run_id'],'full_curve_selection_and_update_counts_passed':True})
        records.append(row)
    stamp=datetime.datetime.now().astimezone()
    out=ROOT/'common_development_reports'/stamp.strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    with (out/'plan_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    counts={k:sum(r['status']==k for r in records) for k in sorted({r['status'] for r in records})}
    summary={'created_local':stamp.isoformat(),'plan_sha256':sha(ROOT/'common_queue_plan_v2.json'),'state_counts':counts,
             'planned_runs':len(records),'rows':records,'completed_run_details':details,'input_sha256':inputs,
             'checks':checks,'queue_snapshot':queue,'script_sha256':sha(Path(__file__)),
             'test_accessed':False,'stage':'interim development snapshot' if counts.get('completed',0)<12 else 'completed single-seed development sweep',
             'interpretation':'Original GPU AMP stratified validation MAE and earliest best epoch; not sealed-test or decision-metric selection.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 公共骨干开发训练快照','',f'生成时间：{stamp.isoformat()}。状态计数：{counts}。','',
      '全部12个预定配置保留；表内分数是原GPU AMP验证分层MAE，不是封存测试成绩。未完成配置不参与最终比较。','',
      '| 骨干 | 方法 | 学习率 | 状态 | 最佳epoch | MAE（百分点） | 实际更新 / 溢出跳步 |','|---|---|---:|---|---:|---:|---:|']
    for r in records:
        metric=f"{r['best_validation_mae_pp']:.6f}" if r['status']=='completed' else '—'
        updates=f"{r['actual_updates']} / {r['skipped_overflows']}" if r['status']=='completed' else '—'
        lines.append(f"| {r['encoder']} | {r['variant']} | {r['lr']} | {r['status']} | {r['best_epoch']} | {metric} | {updates} |")
    lines+=['','已完成run的24轮曲线、原最佳epoch、357个验证窗口和更新计数均重新核对。',
      'hard_seg使用多数面积标签，与旧v2 nearest-hard定义不同；新旧解码器及训练预算也不同。多个种子与独立场景确认尚未完成。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    pointer={'created_local':stamp.isoformat(),'directory':str(out),'state_counts':counts,'summary_sha256':sha(out/'summary.json')}
    (ROOT/'latest_common_development_report.json').write_text(json.dumps(pointer,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(pointer,ensure_ascii=False))
if __name__=='__main__':main()
