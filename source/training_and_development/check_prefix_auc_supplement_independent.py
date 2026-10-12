"""Bounded AUC supplement review: formula/toys, outputs, 45 persisted references."""
from pathlib import Path
import ast,csv,datetime,hashlib,itertools,json,math,shutil
ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture'
TARGET=PA/'common_prefix_auc_full8_20260916_052212_839933'
OUT=PA/('prefix_auc_independent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));OUT.mkdir()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-13,abs_tol=1e-14),(a,b)
status=read(TARGET/'status.json');summary=read(TARGET/'summary.json');manifest=read(TARGET/'manifest.json')
assert status['state']==summary['state']=='completed' and status['summary']==summary
before=sha(TARGET/'status.json')
for rel,h in status['files_sha256'].items():assert sha(TARGET/rel)==h,(rel,'artifact hash')
for path,h in manifest['source_sha256'].items():assert sha(path)==h,(path,'source hash')
assert manifest['source_sha256']==summary['input_source_sha256']
assert manifest['implementation_sha256']==sha(ROOT/'summarize_common_prefix_auc.py')==sha(TARGET/'source/summarize_common_prefix_auc.py')
assert manifest['metric_protocol_sha256']=='527f716fe7db0125ceca70a76479d8f3337fc448ff197d5e16517f40bbe8ab8d'
assert manifest['independent_prior_point_audit_sha256']=='cd88fe934e9c9d5dcee01a38a7571257f23d1ab8838ad9a1811e6e1f72913da2'
prior=read(TARGET/'source/audit.json');assert prior['passed'] and prior['all_model_prefix_curve_points_recomputed']==52464
# Compile only the pure integral function, not any entry point.
tree=ast.parse((TARGET/'source/summarize_common_prefix_auc.py').read_text('utf8'))
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='step_auc')
namespace={'math':math};exec(compile(ast.Module(body=[function],type_ignores=[]),'reviewed_pure_step_auc','exec'),namespace)
f=namespace['step_auc']
toys=[([0.,.2,.5,1.],[0.,.3,.8,1.],1.,.49),([0.,.2,.5,1.],[0.,.3,.8,1.],.5,.18),([0.,.2,.8,1.],[0.,.3,.8,1.],.5,.18),([0.,1.],[0.,1.],1.,0.),([0.,.25,.25,.75,1.],[0.,.2,.6,.9,1.],.5,.3)]
for x,y,u,wanted in toys:close(f(x,y,u),wanted)
a=rows(TARGET/'all72_model_prefix_auc.csv');p=rows(TARGET/'all36_hard_soft_auc_pairs.csv')
assert len(a)==72 and len(p)==36
lookup={(x['encoder'],float(x['lr']),x['variant'],int(x['M']),int(x['R'])):x for x in a}
assert set(lookup)==set(itertools.product(['resnet18','convnextv2_tiny'],[.0001,.0003],['hard_seg','soft_seg'],[1,2,4],[1,2,4]))
assert {(x['encoder'],float(x['lr']),int(x['M']),int(x['R'])) for x in p}==set(itertools.product(['resnet18','convnextv2_tiny'],[.0001,.0003],[1,2,4],[1,2,4]))
models={m['run_id']:m for m in read(PA/'bca_20260916_050803_135549/manifest.json')['models']}
stored_cache={};matched=0;absent=0;maxerr=0.;points=0
for x in a:
    m=models[x['run_id']];parent=Path(x['parent_directory']);M=int(x['M']);R=int(x['R'])
    assert parent==Path(m['unit_tables_directory']) and x['checkpoint_sha256']==m['checkpoint_sha256']
    assert int(x['checkpoint_epoch'])==m['checkpoint_epoch'] and int(x['model_index'])==m['index'] and int(x['seed'])==17
    assert x['encoder']==m['encoder'] and x['variant']==m['variant'] and float(x['lr'])==m['lr']
    assert x['curve_csv_sha256']==sha(parent/'decision_curves.csv')==sha(TARGET/x['curve_csv_copy'])
    assert int(x['management_units'])=={1:1614,2:443,4:126}[M]
    assert int(x['curve_points'])==int(x['management_units'])+1;points+=int(x['curve_points'])
    assert x['input_capture_clipped']=='False' and float(x['max_input_capture_fraction_above_one'])<=2.220446049250313e-16
    available=(parent/'summary.json').exists();assert available==(x['stored_parent_auc_available']=='True')
    for metric in ['auc_0_1','auc_0_0p5_normalized']:assert 0<=float(x[metric])<=1
    if available:
        if parent not in stored_cache:stored_cache[parent]=read(parent/'summary.json')
        original=stored_cache[parent];assert original['state']=='completed' and original['model_run']==x['run_id'] and original['checkpoint_epoch']==int(x['checkpoint_epoch'])
        y=[q for q in original['results']['by_M'][str(M)]['prefix_results'] if q['method']=='model_prefix' and q['R']==R];assert len(y)==1
        for metric,errcol in [('auc_0_1','abs_error_vs_stored_auc_0_1'),('auc_0_0p5_normalized','abs_error_vs_stored_auc_0_0p5')]:
            close(x['stored_'+metric],y[0][metric]);err=abs(float(x[metric])-y[0][metric]);close(err,x[errcol]);assert err<1e-12;maxerr=max(maxerr,err)
        matched+=1
    else:
        assert x['stored_auc_0_1']==x['stored_auc_0_0p5_normalized']==''
        assert 'historical failure' in x['auc_source_note'];absent+=1
for x in p:
    enc,lr,M,R=x['encoder'],float(x['lr']),int(x['M']),int(x['R']);h=lookup[enc,lr,'hard_seg',M,R];s=lookup[enc,lr,'soft_seg',M,R]
    for prefix,y in [('hard',h),('soft',s)]:
        assert x[prefix+'_run']==y['run_id'] and x[prefix+'_checkpoint_epoch']==y['checkpoint_epoch'] and x[prefix+'_curve_csv_sha256']==y['curve_csv_sha256']
        for metric in ['auc_0_1','auc_0_0p5_normalized']:close(x[prefix+'_'+metric],y[metric])
    for metric in ['auc_0_1','auc_0_0p5_normalized']:close(x['soft_minus_hard_'+metric],float(s[metric])-float(h[metric]))
for M,metric in itertools.product([1,2,4],['auc_0_1','auc_0_0p5_normalized']):
    ds=[float(x['soft_minus_hard_'+metric]) for x in p if int(x['M'])==M];q=summary['by_M_direction_counts'][f'M{M}_{metric}']
    assert q['pairs']==len(ds)==12 and [q[k] for k in ['soft_higher','soft_lower','tied']]==[sum(v>1e-12 for v in ds),sum(v< -1e-12 for v in ds),sum(abs(v)<=1e-12 for v in ds)]
    close(q['min_difference'],min(ds));close(q['max_difference'],max(ds))
assert matched==summary['curves_with_saved_parent_auc_matched']==45 and absent==summary['curves_without_parent_summary']==27
assert points==summary['complete_model_curve_points']==52464;close(maxerr,summary['max_abs_difference_from_saved_parent_auc'])
assert sha(TARGET/'status.json')==before
central=PA/'common_management_reports/full8_20260916_050927_560091'
central_review=read(PA/'full8_central_independent_20260916_051750_163534/central_review.json')
assert sha(central/'status.json')==central_review['table_status_sha256'] and sha(central/'final_artifact_manifest.json')==central_review['final_artifact_manifest_sha256']
result={'passed':True,'audited_output_directory':str(TARGET),'all72_model_prefix_auc_sha256':sha(TARGET/'all72_model_prefix_auc.csv'),'all36_hard_soft_auc_pairs_sha256':sha(TARGET/'all36_hard_soft_auc_pairs.csv'),'source_implementation_sha256':manifest['implementation_sha256'],'formula_review':'Right-continuous constant prefix capture on [x_i,x_(i+1)); rectangle integral, clipping only interval width at u, normalized by u. Endpoint capture has zero measure; no trapezoid or five-budget interpolation.','pure_function_toys_passed':len(toys),'model_curves_checked':72,'hard_soft_pairs_checked':36,'persisted_parent_auc_pairs_crosschecked':45,'explicitly_absent_parent_summaries':27,'maximum_absolute_difference_from_stored':maxerr,'direction_counts_checked':6,'product_hashes_verified':len(status['files_sha256']),'current_source_hashes_verified':len(manifest['source_sha256']),'prior_full_curve_point_audit_sha256':manifest['independent_prior_point_audit_sha256'],'all_complete_curve_points_reaudited_here':False,'central_locks_unchanged':True,'new_model_calls':0,'new_solver_calls':0,'test_R6_accessed':False,'limitations':'Bounded formula/source/table QA. The existing independent 52,464-point audit remains the primary full-curve validation. The 27 curves without persisted summaries have formula and source validation here, not a newly claimed independent persisted-AUC crosscheck. Same-scene seed17 descriptive metrics, no model selection or field efficacy.','auditor_source_sha256':sha(__file__)}
(OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
(OUT/'review.md').write_text('# 完整前缀 AUC 补充表独立 QA\n\n结论：通过。\n\n- 右连续阶梯采用矩形积分，半区间在 0.5 截断后除以 0.5；端点无积分质量。5 个独立解析例通过，未使用梯形或五个稀疏预算点积分。\n- 全部 72 条 AUC 与 36 组 hard/soft 配对、来源/检查点、6 组方向计数、产物及来源哈希通过。\n- 45 条记录与原已持久化 AUC 对照，最大绝对差 '+str(maxerr)+'。另 27 条原父 summary 不存在，缺失标记准确，不伪称有旧 AUC 可对照。\n- 未重复 52,464 点原生面积审计；该部分明确承接此前独立回执。没有模型、求解器或 test/R6 访问，中央状态及图清单锁未改。\n- 同一场景、单种子描述统计；AUC 并不建立真实作业预算分布或农业收益。\n',encoding='utf8')
shutil.copy2(__file__,OUT/Path(__file__).name)
print(json.dumps({'review':str(OUT/'review.json'),'sha256':sha(OUT/'review.json'),'passed':True,'matched45_maxerror':maxerr},ensure_ascii=False))
