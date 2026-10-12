"""Descriptive summaries of ALL fixed grid-origin cases; no fitting or filtering."""
from pathlib import Path
import csv,json,hashlib,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def table(p,rs):
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

with (ROOT/'results_all.csv').open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
assert len(rows)==4608
for x in rows:
    for k in ('R','M'):x[k]=int(x[k])
    for k in ('budget','shift_x_fraction','shift_y_fraction','capture_pct','actual_area_fraction','domain_area_m2','domain_target_m2'):x[k]=float(x[k])
groups={}
for x in rows:groups.setdefault((x['run_id'],x['R'],x['M'],x['budget']),[]).append(x)
envelopes=[]
for (model,r,m,b),v in sorted(groups.items()):
    assert len(v)==4
    p=np.array([x['capture_pct'] for x in v]);baseline=next(x['capture_pct'] for x in v if x['shift_x_fraction']==0 and x['shift_y_fraction']==0)
    envelopes.append({'run_id':model,'R':r,'M':m,'budget':b,'baseline_capture_pct':baseline,'min_capture_pct':p.min(),'max_capture_pct':p.max(),'range_pp':np.ptp(p),'max_absolute_change_from_baseline_pp':np.max(np.abs(p-baseline)),
                      'min_change_from_baseline_pp':p.min()-baseline,'max_change_from_baseline_pp':p.max()-baseline})
table(ROOT/'origin_ranges_all.csv',envelopes)
summary=[]
for m in (1,2,4):
    for b in (.1,.2,.3,.5):
        group=[x for x in envelopes if x['M']==m and x['budget']==b]
        values=np.array([x['range_pp'] for x in group]);assert len(values)==96
        summary.append({'M':m,'budget':b,'model_resolution_cases':len(values),'range_pp_min':values.min(),'range_pp_q25':np.quantile(values,.25),'range_pp_median':np.median(values),'range_pp_q75':np.quantile(values,.75),'range_pp_max':values.max(),
                        'range_gt_5pp_cases':int(np.sum(values>5)),'range_gt_10pp_cases':int(np.sum(values>10))})
table(ROOT/'origin_range_summary.csv',summary)
lookup={(x['run_id'],x['R'],x['M'],x['budget'],x['shift_x_fraction'],x['shift_y_fraction']):x['capture_pct'] for x in rows}
md=sorted({x['run_id'] for x in rows});direction=[];percase=[]
for b in (.1,.2,.3,.5):
    all_diffs=[];all_lower=all_higher=mixed=0
    for model in md:
        for r in (1,2,4,6):
            diffs=[lookup[model,r,4,b,sx,sy]-lookup[model,r,2,b,sx,sy] for sx,sy in ((0,0),(.5,0),(0,.5),(.5,.5))]
            sign='lower_all_origins' if max(diffs)<-1e-9 else 'higher_all_origins' if min(diffs)>1e-9 else 'mixed_or_equal'
            all_lower+=sign=='lower_all_origins';all_higher+=sign=='higher_all_origins';mixed+=sign=='mixed_or_equal'
            percase.append({'run_id':model,'R':r,'budget':b,'M4_minus_M2_min_pp':min(diffs),'M4_minus_M2_max_pp':max(diffs),'direction_across_four_normalized_origins':sign})
            all_diffs+=diffs
    direction.append({'budget':b,'paired_model_R_origin_comparisons':384,'M4_lower_than_M2':sum(x<-1e-9 for x in all_diffs),'M4_higher_than_M2':sum(x>1e-9 for x in all_diffs),'equal_within_1e_9pp':sum(abs(x)<=1e-9 for x in all_diffs),
                      'model_R_cases_lower_at_all_origins':all_lower,'model_R_cases_higher_at_all_origins':all_higher,'model_R_cases_mixed_or_equal':mixed,'M4_minus_M2_median_pp':float(np.median(all_diffs))})
table(ROOT/'grid_size_comparison_by_case.csv',percase);table(ROOT/'grid_size_comparison_summary.csv',direction)
execution=json.loads((ROOT/'execution_complete.json').read_text(encoding='utf-8'))
result={'analysis':'Post-development grid-origin sensitivity; same scene and saved model predictions','results_rows':len(rows),'models':len(md),'R':[1,2,4,6],'M_m':[1,2,4],'budgets':[.1,.2,.3,.5],'origins_M_fraction':[[0,0],[.5,0],[0,.5],[.5,.5]],
        'independent_new_validation':False,'training_performed':False,'all_cases_reported':True,'statistical_unit_warning':'96 model-by-R settings share the same scene and labels; they are not independent fields or biological replicates.',
        'scope':'Whole-unit prefix policy only; no new knapsack solver evaluation. Mask denotes coarse annotated patches, not pure leaf cover.',
        'non_nested_shifted_grids':'A half-grid offset equals 0.5, 1 or 2 m at M=1,2,4 respectively. Equal normalized nonzero offsets need not form nested grids across M. The original nested-grid upper-bound argument is not extended to these shifted combinations; no new optimal upper bound is claimed.',
        'domain_area_m2':rows[0]['domain_area_m2'],'domain_target_m2':rows[0]['domain_target_m2'],'zero_origin_verification':execution,'range_summary':summary,'M4_vs_M2':direction,
        'output_generated_utc':datetime.now(timezone.utc).isoformat(),'source_results_sha256':sha(ROOT/'results_all.csv'),'summary_script_sha256':sha(__file__)}
dump(ROOT/'experiment_summary.json',result)

# All 96 paired model-resolution cases appear at each budget; deterministic jitter
# separates symbols only. No inferential confidence bars or significance tests.
colors=['#0072B2','#D55E00','#009E73']
cnfont=font_manager.FontProperties(family=['Microsoft YaHei', 'Noto Sans CJK SC', 'DejaVu Sans'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none'})
for lang in ('EN','ZH'):
    fig,axes=plt.subplots(1,3,figsize=(8.2,3.7),sharey=True)
    maxval=max(x['range_pp_max'] for x in summary)
    for ai,(ax,m,c) in enumerate(zip(axes,(1,2,4),colors)):
        for j,b in enumerate((.1,.2,.3,.5)):
            vals=np.array([x['range_pp'] for x in envelopes if x['M']==m and x['budget']==b])
            jitter=((np.arange(96)*37)%97/96-.5)*.4
            ax.scatter(j+jitter,vals,s=11,c=c,alpha=.3,edgecolors='none',zorder=2)
            q=np.quantile(vals,[.25,.5,.75])
            ax.plot([j,j],[q[0],q[2]],c='#222222',lw=2,zorder=3)
            ax.scatter([j],[q[1]],s=27,c='white',marker='D',edgecolors='#222222',zorder=4)
        title=f'{chr(97+ai)}  {m} m grid' if lang=='EN' else f'{chr(97+ai)}  {m} m网格'
        ax.set_title(title,loc='left',fontproperties=cnfont if lang=='ZH' else None)
        ax.set_xticks(range(4),['10','20','30','50']);ax.set_ylim(-.4,maxval*1.1);ax.set_xlim(-.55,3.55)
        ax.set_xlabel('Area limit (%)' if lang=='EN' else '可选面积上限（%）',fontproperties=cnfont if lang=='ZH' else None)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',lw=.4,alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Capture range across origins\n(percentage points)' if lang=='EN' else '四种原点下选中斑块比例的极差\n（百分点）',fontproperties=cnfont if lang=='ZH' else None)
    note='Dots: all 96 model–resolution settings; diamond: median; line: interquartile range.' if lang=='EN' else '圆点：全部96组模型与分辨率设置；菱形：中位数；竖线：四分位范围。'
    fig.text(.08,.025,note,fontsize=9,fontproperties=cnfont if lang=='ZH' else None)
    fig.subplots_adjust(left=.115,right=.985,bottom=.22,top=.86,wspace=.17)
    base=ROOT/f'Figure_S18_Grid_origin_{lang}'
    fig.savefig(str(base)+'.png',dpi=400,facecolor='white');fig.savefig(str(base)+'.pdf');fig.savefig(str(base)+'.svg')
    plt.close(fig)

by={(x['M'],x['budget']):x for x in summary};cmp={x['budget']:x for x in direction}
zh=(f'网格原点敏感性分析使用同一评价区内24个已训练分割模型在4种图像细节条件下保存的预测。对1、2、4 m网格分别设定原点偏移(0,0)、(M/2,0)、(0,M/2)、(M/2,M/2)，并保留10%、20%、30%、50%的面积上限，共计算4608种组合。选格仍按预测斑块比例从高到低排序，采用不超出面积上限的整格前缀规则。未重新训练模型，未根据结果选择原点。\n\n'
    f'原点为0时，1152种组合的选区编号与原结果完全一致；保存预测的独立积分也通过逐格核对。不同原点下有效评价面积和标注斑块总面积保持不变，分别约为{result["domain_area_m2"]:.6f} m²和{result["domain_target_m2"]:.6f} m²。\n\n'
    f'当可选面积上限为10%时，四个原点间选中标注斑块比例的极差中位数，在1、2、4 m网格下分别为{by[1,.1]["range_pp_median"]:.2f}、{by[2,.1]["range_pp_median"]:.2f}和{by[4,.1]["range_pp_median"]:.2f}个百分点；最大极差分别为{by[1,.1]["range_pp_max"]:.2f}、{by[2,.1]["range_pp_max"]:.2f}和{by[4,.1]["range_pp_max"]:.2f}个百分点。\n\n'
    f'在各自网格尺寸的相同相对原点下进行比较，10%面积上限时，4 m网格在{cmp[.1]["M4_lower_than_M2"]}/384组比较中低于2 m网格；50%面积上限时，4 m网格在{cmp[.5]["M4_higher_than_M2"]}/384组比较中高于2 m网格。这些为共享同一场景的设置比较，不构成独立田块重复。\n\n'
    '半格偏移在1、2、4 m网格下分别对应0.5、1、2 m。因此，相同相对偏移下不同网格大小不一定构成嵌套划分，原补充材料中针对嵌套网格的上界结论不适用于全部平移组合。本次只比较实际整格前缀选区，不计算或声明新的最优上界。\n\n'
    '该分析在原研究开发与评价之后开展，用于检查已有结论对网格摆放的敏感程度，不属于新的独立验证。标注对应粗略斑块范围，不等同于纯叶片覆盖或实际防除效果。\n\n'
    '图S18. 网格原点平移引起的选区结果变化。纵轴为同一模型、图像细节条件、网格大小和面积上限下，四种原点的选中标注斑块比例最大值减最小值。每个面积上限展示全部96组模型与分辨率设置；圆点为各组极差，菱形为中位数，竖线为四分位范围。点的水平偏移仅用于避免重叠。该分布描述设置之间的变化，不是跨田块置信区间。\n')
en=(f'Grid-origin sensitivity was assessed using saved predictions from all 24 fitted segmentation models at four image-detail settings within the same evaluation domain. For each grid size (1, 2 and 4 m), four origin offsets, (0,0), (M/2,0), (0,M/2) and (M/2,M/2), were evaluated at area limits of 10%, 20%, 30% and 50%, yielding 4,608 combinations. Units were ranked by predicted annotated-patch proportion and selected using the original whole-unit prefix rule. No models were refitted and no origin was selected based on its outcome.\n\n'
    f'At the original origin, selected unit identifiers matched the archived results in all 1,152 combinations. Independent integration of the saved predictions also reproduced the archived unit values. Valid domain area and total annotated-patch area were conserved across origins, at approximately {result["domain_area_m2"]:.6f} m² and {result["domain_target_m2"]:.6f} m², respectively.\n\n'
    f'At a 10% area limit, the median range in captured annotated-patch percentage across the four origins was {by[1,.1]["range_pp_median"]:.2f}, {by[2,.1]["range_pp_median"]:.2f} and {by[4,.1]["range_pp_median"]:.2f} percentage points for the 1, 2 and 4 m grids, respectively. The corresponding maximum ranges were {by[1,.1]["range_pp_max"]:.2f}, {by[2,.1]["range_pp_max"]:.2f} and {by[4,.1]["range_pp_max"]:.2f} percentage points.\n\n'
    f'Comparing the same relative origin within each grid size, the 4 m grid captured less annotated-patch area than the 2 m grid in {cmp[.1]["M4_lower_than_M2"]} of 384 comparisons at the 10% limit, whereas it captured more in {cmp[.5]["M4_higher_than_M2"]} of 384 comparisons at the 50% limit. These are comparisons of settings within one scene, not independent field replicates.\n\n'
    'A half-grid offset corresponds to 0.5, 1 and 2 m for the 1, 2 and 4 m grids, respectively. Equal normalized offsets therefore need not produce nested grids across sizes. The original nested-grid upper-bound argument does not apply to all shifted combinations. This analysis compares realized whole-unit prefix selections and claims no new optimal upper bound.\n\n'
    'This post-development analysis tests sensitivity to grid placement and provides no new independent validation. The reference masks delineate coarse patches rather than pure leaf cover, and capture does not measure control efficacy.\n\n'
    'Figure S18. Sensitivity of selected annotated-patch capture to grid origin. For a fixed model, image-detail setting, grid size and area limit, the range equals the highest minus the lowest captured annotated-patch percentage across the four origins. Each area limit includes all 96 model-by-resolution settings. Dots show individual ranges, diamonds show medians, and vertical lines show interquartile ranges. Horizontal jitter only separates points. The distribution describes variation among settings and is not a confidence interval across fields.\n')
(ROOT/'results_description_ZH.txt').write_text(zh,encoding='utf-8');(ROOT/'results_description_EN.txt').write_text(en,encoding='utf-8')
dump(ROOT/'plot_provenance.json',{'utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),'data_sha256':sha(ROOT/'origin_ranges_all.csv'),'matplotlib':matplotlib.__version__,'generative_image_used':False,'figure_type':'computed statistical plot','all_96_cases_each_cell':True,'jitter':'deterministic function of index for visualization only','figure_files':{p.name:sha(p) for p in ROOT.glob('Figure_S18*')}})
print(json.dumps(result,ensure_ascii=False,indent=2))
