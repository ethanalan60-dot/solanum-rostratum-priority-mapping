"""Render all full common-management comparisons from verified CSVs only."""
from pathlib import Path, PureWindowsPath
import argparse,csv,hashlib,itertools,json,math,os,shutil

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save_json(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def load_json(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def strict_bool(value):
    assert value in ('True','False'),value
    return value=='True'

def finite(value):
    number=float(value);assert math.isfinite(number),value
    return number

def equal_number(a,b):
    assert math.isclose(finite(a),finite(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)

def snapshot_files(out,status):
    """Verify immutable table snapshot, including its archived source copies."""
    assert status['state']=='tables_completed' and status['rows']==720 and status['pairs']==360
    hashes=status['file_sha256']
    required={'all_720_model_policy_budgets.csv','all_360_soft_minus_hard_pairs.csv',
              'provenance.json','findings.json','full_common_management_report.md'}
    assert required.issubset(hashes)
    for relative,digest in hashes.items():
        win=PureWindowsPath(relative)
        assert not win.drive and not win.is_absolute() and '..' not in win.parts,relative
        path=(out/relative).resolve();assert path.is_relative_to(out),relative
        assert path.is_file() and sha(path)==digest,(relative,'snapshot hash mismatch')
    provenance=load_json(out/'provenance.json')
    assert provenance['rows']==720 and provenance['paired_rows']==360
    assert not provenance['test_accessed'] and not provenance['R6_accessed']
    assert provenance['new_solver_calls']==0 and provenance['inference_calls']==0
    for item in provenance['source_copies']:
        relative=item['copy'];assert relative in hashes and hashes[relative]==item['sha256']
        assert provenance['source_sha256'][item['original']]==item['sha256']
    return provenance

def preflight(out):
    """No plotting side effects: bind 720 observations and 360 differences before rendering."""
    out=Path(out).resolve();status=load_json(out/'status.json')
    provenance=snapshot_files(out,status)
    data=read(out/'all_720_model_policy_budgets.csv');pairs=read(out/'all_360_soft_minus_hard_pairs.csv')
    assert len(data)==720 and len(pairs)==360
    lookup={(x['encoder'],finite(x['lr']),x['variant'],x['policy'],int(x['M']),int(x['R']),finite(x['budget'])):x for x in data}
    plookup={(x['encoder'],finite(x['lr']),x['policy'],int(x['M']),int(x['R']),finite(x['budget'])):x for x in pairs}
    encoders=('resnet18','convnextv2_tiny');lrs=(1e-4,3e-4);variants=('hard_seg','soft_seg')
    policies=('prefix','predicted_area');ms=(1,2,4);rs=(1,2,4);bs=(.1,.2,.3,.5,1.)
    assert set(lookup)==set(itertools.product(encoders,lrs,variants,policies,ms,rs,bs))
    assert set(plookup)==set(itertools.product(encoders,lrs,policies,ms,rs,bs))
    models=provenance['all8_models'];assert len(models)==8
    model_lookup={(x['encoder'],float(x['lr']),x['variant']):x for x in models}
    assert set(model_lookup)==set(itertools.product(encoders,lrs,variants))
    domain=None
    for key,row in lookup.items():
        model=model_lookup[key[:3]]
        assert row['run_id']==model['run_id'] and int(row['seed'])==int(model['seed'])==17
        assert int(row['checkpoint_epoch'])==int(model['checkpoint_epoch'])
        assert row['checkpoint_sha256']==model['checkpoint_sha256']
        equal_number(row['original_gpu_mae_pp'],model['original_gpu_mae_pp'])
        assert row['source_csv']==model['policy_csv'] and row['source_csv_sha256']==model['policy_csv_sha256']
        D=finite(row['domain_area_m2']);T=finite(row['domain_target_m2']);assert D>0 and 0<T<=D
        assert D==model['domain']['area_m2'] and T==model['domain']['target_m2']
        if domain is None:domain=(D,T)
        assert domain==(D,T)
        a=finite(row['actual_area_m2']);t=finite(row['captured_target_m2']);b=finite(row['budget_area_m2'])
        assert 0<=a<=b and 0<=t<=min(a,T)+1e-10
        equal_number(b,key[-1]*D);equal_number(row['actual_area_fraction'],a/D)
        equal_number(row['capture_pct'],100*t/T);equal_number(row['missed_target_m2'],T-t)
        equal_number(row['zero_annotation_area_m2'],a-t);equal_number(row['unused_budget_m2'],b-a)
        assert int(row['selected_count'])>=0
        p=finite(row['predicted_objective_m2']);assert 0<=p<=a+1e-10
        fallback=strict_bool(row['fallback_used'])
        if row['policy']=='prefix':
            assert not fallback and row['optimality_proven']==''
        else:
            optimal=strict_bool(row['optimality_proven']);assert not(fallback and optimal)
            upper=finite(row['predicted_upper_m2']);gap=finite(row['predicted_relative_gap'])
            assert upper>=p and gap>=0
    for key,row in plookup.items():
        enc,lr,policy,m,r,b=key
        hard=lookup[(enc,lr,'hard_seg',policy,m,r,b)];soft=lookup[(enc,lr,'soft_seg',policy,m,r,b)]
        for prefix,source in [('hard',hard),('soft',soft)]:
            equal_number(row[prefix+'_capture_pct'],source['capture_pct'])
            equal_number(row[prefix+'_actual_area_m2'],source['actual_area_m2'])
            assert row[prefix+'_fallback']==source['fallback_used']
            assert row[prefix+'_gap_qualified']==source['optimality_proven']
            assert row[prefix+'_source_sha256']==source['source_csv_sha256']
        equal_number(row['soft_minus_hard_pp'],finite(soft['capture_pct'])-finite(hard['capture_pct']))
        equal_number(row['soft_minus_hard_area_m2'],finite(soft['actual_area_m2'])-finite(hard['actual_area_m2']))
    return data,pairs,lookup,plookup,{'status_sha256':sha(out/'status.json'),
        'table_snapshot_hashes':dict(status['file_sha256']),
        'rows':720,'paired_rows':360,'exact_keys_and_pair_values_checked':True}

parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);args=parser.parse_args()
OUT=Path(args.report).resolve();SCRIPT_BYTES=Path(__file__).read_bytes()
data,pairs,lookup,plookup,INPUT_BINDING=preflight(OUT)
new_paths=['figure_manifest.json','final_artifact_manifest.json','full_common_management_report_with_figures.md','plotting_source']
new_paths += [name+'.'+ext for name in ('fig_primary_M2_prefix','fig_primary_M2_predicted_area','fig_all360_soft_minus_hard') for ext in ('png','svg','pdf')]
assert all(not (OUT/name).exists() for name in new_paths),'Prior plotting artifacts must be preserved; use a new report snapshot.'
os.environ['MPLCONFIGDIR']=str(OUT/'mpl_config');os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

CONFIGS=(('resnet18',1e-4),('resnet18',3e-4),('convnextv2_tiny',1e-4),('convnextv2_tiny',3e-4))
NAMES={'resnet18':'ResNet18','convnextv2_tiny':'ConvNeXt V2-T'}
RS=(1,2,4);MS=(1,2,4);BS=(.1,.2,.3,.5,1.)
policies=('prefix','predicted_area');labels={'prefix':'Fraction-ranked prefix','predicted_area':'Predicted-area policy'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
figures=[]
def emit(fig,name,caption,binding):
    for ext in ('png','svg','pdf'):
        target=OUT/(name+'.'+ext);assert not target.exists();fig.savefig(target,dpi=240,facecolor='white',bbox_inches='tight')
    figures.append({'name':name,'caption':caption,'data':binding,
                    'files':{ext:{'path':str(OUT/(name+'.'+ext)),'sha256':sha(OUT/(name+'.'+ext))} for ext in ('png','svg','pdf')}})
    plt.close(fig)

for policy in policies:
    fig,axes=plt.subplots(4,3,figsize=(12,11),sharex=True,sharey=True)
    for iy,(enc,lr) in enumerate(CONFIGS):
        for ix,r in enumerate(RS):
            ax=axes[iy,ix]
            for variant,color,marker,offset in [('hard_seg','#2166ac','o',-1.1),('soft_seg','#b35806','s',1.1)]:
                ys=[float(lookup[(enc,lr,variant,policy,2,r,b)]['capture_pct']) for b in BS]
                ax.plot([100*b+offset for b in BS],ys,linestyle='none',marker=marker,markersize=5.8,color=color,
                        markerfacecolor='white' if variant=='hard_seg' else color,label='Hard' if variant=='hard_seg' else 'Soft')
            ax.set(ylim=(0,104),xlim=(5,105),xticks=[10,20,30,50,100],yticks=[0,25,50,75,100],
                   title=f'{NAMES[enc]}, LR={lr:g} | R={r}')
            ax.grid(axis='y',alpha=.22)
            if ix==0:ax.set_ylabel('Annotated target captured (%)')
            if iy==3:ax.set_xlabel('Valid-area budget (%)')
    handles,ll=axes[0,0].get_legend_handles_labels();fig.legend(handles,ll,loc='upper center',bbox_to_anchor=(.5,.96),ncol=2,frameon=False)
    fig.suptitle(f'M = 2 m primary analysis: {labels[policy]}',fontsize=16,y=.995)
    fig.subplots_adjust(left=.075,right=.98,top=.9,bottom=.11,hspace=.37,wspace=.16)
    fig.text(.5,.051,'All 8 declared models; all 5 budgets. Markers offset by ±1.1 budget pp for readability; no interpolation.',ha='center',fontsize=8.5)
    fig.text(.5,.025,'One scene, seed 17, development results | Same budget cap; actual selected areas differ | Capture is not field efficacy',ha='center',fontsize=8.4)
    emit(fig,'fig_primary_M2_'+policy,
         f'M=2m primary analysis, {labels[policy]}. All eight models and R1/2/4 are shown at all five declared area-budget caps, on a shared0–100% capture scale. No line interpolation or sparse-point AUC. Horizontal offsets disclose ties; original actual areas and solver/fallback status are retained in the table. One-scene, single-seed development evidence.',
         [x for x in data if x['policy']==policy and int(x['M'])==2])

limit=max(1.,math.ceil(max(abs(float(x['soft_minus_hard_pp'])) for x in pairs)/5)*5)
norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
fig,axes=plt.subplots(4,2,figsize=(17,11))
for iy,(enc,lr) in enumerate(CONFIGS):
    for ix,policy in enumerate(policies):
        ax=axes[iy,ix]
        matrix=np.array([[float(plookup[(enc,lr,policy,m,r,b)]['soft_minus_hard_pp']) for r,b in itertools.product(RS,BS)] for m in MS])
        im=ax.imshow(matrix,cmap='RdBu',norm=norm,aspect='auto')
        for rr,cc in itertools.product(range(3),range(15)):
            v=matrix[rr,cc];ax.text(cc,rr,f'{v:+.1f}',ha='center',va='center',fontsize=7.6,color='white' if abs(v)>.62*limit else '#111111')
        for x in (4.5,9.5):ax.axvline(x,color='#333333',lw=1)
        ax.add_patch(Rectangle((-.5,.5),15,1,fill=False,edgecolor='#333333',lw=1.5))
        ax.set(xticks=range(15),xticklabels=[int(b*100) for r,b in itertools.product(RS,BS)],
               yticks=range(3),yticklabels=['1 m','2 m*','4 m'])
        ax.set_title(f'{NAMES[enc]}, LR={lr:g} | {labels[policy]}',pad=28)
        for x,r in [(2,1),(7,2),(12,4)]:ax.text(x,-.75,f'R={r}',ha='center',va='bottom',fontsize=8.5)
        ax.set_xlabel('Valid-area budget (%) within each R group')
        if ix==0:ax.set_ylabel('Management unit M')
fig.suptitle('Soft − hard: all 360 paired target-capture differences (pp)',fontsize=17,y=.996)
fig.subplots_adjust(left=.055,right=.92,top=.92,bottom=.105,hspace=.76,wspace=.12)
cax=fig.add_axes([.94,.15,.013,.71]);fig.colorbar(im,cax=cax,label='Soft − hard capture (pp)')
fig.text(.49,.049,'Shared symmetric color scale; positive = greater capture with soft labels. * 2 m primary row; 1/4 m sensitivity.',ha='center',fontsize=9)
fig.text(.49,.024,'All configurations and adverse results retained | One scene, seed 17 | Correlated conditions; no significance test',ha='center',fontsize=9)
emit(fig,'fig_all360_soft_minus_hard',
     'Every paired capture difference across4 encoder/LR pairs×2 policies×3 management sizes×3 image scales×5 budgets. Shared symmetric color limits preserve negative, positive and unchanged values. M2 primary rows outlined. Budgets are common upper bounds; different actual selected areas are reported separately. Descriptive correlated conditions, not independent replicates or field efficacy.',pairs)
assert Path(__file__).read_bytes()==SCRIPT_BYTES,'Plotting source changed during rendering'
(OUT/'plotting_source').mkdir()
(OUT/'plotting_source'/Path(__file__).name).write_bytes(SCRIPT_BYTES)
save_json(OUT/'figure_manifest.json',{'figures':figures,'matplotlib':matplotlib.__version__,'numpy':np.__version__,
           'input_sha256':{n:sha(OUT/n) for n in ('all_720_model_policy_budgets.csv','all_360_soft_minus_hard_pairs.csv')},
           'script_sha256':sha(__file__),'symmetric_difference_limits_pp':[-limit,limit],
           'table_snapshot_binding':INPUT_BINDING,'no_inference_or_solver':True,'visual_review':'pending'})
with (OUT/'full_common_management_report_with_figures.md').open('x',encoding='utf-8') as f:
    f.write((OUT/'full_common_management_report.md').read_text(encoding='utf-8-sig'))
    f.write('\n## 完整图表\n\n')
    for item in figures:f.write(f"### {item['name']}\n\n![{item['name']}]({(OUT/(item['name']+'.png')).as_posix()})\n\n{item['caption']}\n\n")
assert sha(OUT/'status.json')==INPUT_BINDING['status_sha256']
snapshot_files(OUT,load_json(OUT/'status.json'))
generated=[name for name in new_paths if name not in ('plotting_source','final_artifact_manifest.json')]
generated.append('plotting_source/'+Path(__file__).name)
save_json(OUT/'final_artifact_manifest.json',{'state':'figures_rendered_visual_review_pending',
    'immutable_table_snapshot_status_sha256':INPUT_BINDING['status_sha256'],
    'table_snapshot_preserved':True,'input_binding':INPUT_BINDING,
    'generated_file_sha256':{name:sha(OUT/name) for name in generated},
    'script_sha256':sha(__file__),'visual_review':'pending; independent receipt required',
    'new_solver_calls':0,'inference_calls':0,'test_accessed':False,'R6_accessed':False})
print(json.dumps({'report':str(OUT),'figures':[x['name'] for x in figures],'color_limit_pp':limit},ensure_ascii=False))
