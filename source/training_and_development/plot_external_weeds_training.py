"""Render the sealed nine-run external development comparison; no model imports."""
from pathlib import Path
import csv, datetime, hashlib, json, math, os, shutil, sys, traceback

ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / 'external_weeds_training_reports/20260916_062524_429008'
AUDIT_SHA = '172bb72f0b31c761ccb584cba29b8817b4f58b71e9bcff33a6665d88cd2d75fc'
ARTIFACT_SHA = 'b4cac236aa58a1d6201720b0100e142258a7e494ba775bb166367e6aa8714439'
SEEDS = (17, 29, 43)
VARIANTS = ('regression', 'hard_seg', 'soft_seg')
COLORS = {17: '#0072B2', 29: '#D55E00', 43: '#009E73'}
MARKERS = {17: 'o', 29: 's', 43: '^'}
INPUTS = {}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def csvread(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def near(a, b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-13), (a,b)

def main():
    out = ROOT / 'external_weeds_training_reports' / ('figure_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key]='2'
    temp = out / 'temp'; temp.mkdir()
    for key in ('TEMP','TMP','MPLCONFIGDIR'):
        os.environ[key]=str(temp)
    try:
        assert sha(AUDIT_DIR/'audit.json') == AUDIT_SHA
        assert sha(AUDIT_DIR/'artifact_manifest.json') == ARTIFACT_SHA
        manifest=json.loads((AUDIT_DIR/'artifact_manifest.json').read_text(encoding='utf-8-sig'))
        for name, digest in manifest['files_sha256'].items():
            path=(AUDIT_DIR/name).resolve()
            assert path.is_relative_to(AUDIT_DIR.resolve())
            assert sha(path)==digest, name
        for name in ('audit.json','artifact_manifest.json','all9_configs.csv',
                     'all3_hard_soft_seed_pairs.csv','constant_baselines.csv'):
            path=AUDIT_DIR/name; INPUTS[str(path)]=sha(path)
            shutil.copy2(path,out/name)
        audit=json.loads((AUDIT_DIR/'audit.json').read_text(encoding='utf-8-sig'))
        assert audit['passed'] and audit['run_count']==9 and len(audit['val_ids'])==26
        assert audit['independent_field_count']==1 and audit['dates']==4
        assert audit['validation_is_development'] and not audit['seed_count_is_field_replication']
        runs=csvread(AUDIT_DIR/'all9_configs.csv')
        pairs=csvread(AUDIT_DIR/'all3_hard_soft_seed_pairs.csv')
        baselines=csvread(AUDIT_DIR/'constant_baselines.csv')
        assert len(runs)==9 and {(x['variant'],int(x['seed'])) for x in runs}=={(v,s) for v in VARIANTS for s in SEEDS}
        assert len(pairs)==3 and {int(x['seed']) for x in pairs}==set(SEEDS)
        assert len(baselines)==2 and {x['baseline'] for x in baselines}=={'zero','train_mean'}
        lookup={(x['variant'],int(x['seed'])):x for x in runs}
        audited={x['run_id']:x for x in audit['runs']}
        for row in runs:
            source=audited[row['run_id']]
            assert row['checkpoint_sha256']==source['checkpoint_sha256']
            assert int(row['best_epoch'])==source['best_epoch']
            near(row['mae_pp'],source['mae_pp'])
        for row in pairs:
            seed=int(row['seed'])
            hard=float(lookup['hard_seg',seed]['mae_pp']); soft=float(lookup['soft_seg',seed]['mae_pp'])
            near(row['hard_mae_pp'],hard);near(row['soft_mae_pp'],soft)
            near(row['soft_minus_hard_mae_pp'],soft-hard)
        for row in baselines:
            near(row['mae_pp'],100*audit['constant_baselines'][row['baseline']]['mae'])
            assert int(row['n'])==26
        protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'source_code_sha256':sha(__file__),'input_sha256':INPUTS,
                  'all9_runs':True,'all3_paired_seeds':True,'show_constant_baselines':True,
                  'use_best_checkpoint_already_selected_on_validation':True,
                  'no_confidence_interval_or_significance_tests':True,
                  'axis_units':'percentage points','lower_is_better':True,
                  'no_training_or_new_inference':True,'test_accessed':False,'R6_accessed':False,
                  'independent_field_count':1,'seed_count_is_field_replication':False}
        dump(out/'figure_protocol.json',protocol)
        data={'left_panel':[{'run_id':x['run_id'],'variant':x['variant'],'seed':int(x['seed']),
                            'best_epoch':int(x['best_epoch']),'mae_pp':float(x['mae_pp']),
                            'checkpoint_sha256':x['checkpoint_sha256']} for x in runs],
              'right_panel':[{'seed':int(x['seed']),'soft_minus_hard_mae_pp':float(x['soft_minus_hard_mae_pp'])} for x in pairs],
              'baselines':[{**x,'mae_pp':float(x['mae_pp'])} for x in baselines],
              'n_train':104,'n_val':26,'R':[1,2,4],'source_audit_sha256':AUDIT_SHA}
        dump(out/'plotted_data.json',data)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.text import Text
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                             'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':9,
                             'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path',
                             'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8})
        fig,(left,right)=plt.subplots(1,2,figsize=(11.8,6.15),gridspec_kw={'width_ratios':[1.45,1]})
        fig.subplots_adjust(left=.075,right=.975,bottom=.255,top=.78,wspace=.30)
        fig.text(.075,.955,'WeedsGalore: all nine development runs',fontsize=17,weight='bold',ha='left')
        fig.text(.075,.905,'ResNet18  |  104 training frames / 26 validation frames  |  24 epochs per run',fontsize=10.5,color='#46515C')
        left.set_title('a   Coverage error at the selected epoch',loc='left',pad=14)
        right.set_title('b   Paired soft − hard difference',loc='left',pad=14)
        for ax in (left,right):
            ax.set_axisbelow(True);ax.grid(axis='y',color='#E3E7EB',linewidth=.65)
            ax.tick_params(length=3,color='#747C83')
            ax.spines['left'].set_color('#747C83');ax.spines['bottom'].set_color('#747C83')
        left.set_xlim(-.36,2.36);left.set_ylim(0,5.55)
        left.set_xticks([0,1,2],['Regression','Hard segmentation','Soft segmentation'])
        left.set_yticks([0,1,2,3,4,5]);left.set_ylabel('Validation coverage MAE (pp) ↓')
        for i,seed in enumerate(SEEDS):
            offset=(i-1)*.12
            x=[j+offset for j in range(3)]
            y=[float(lookup[v,seed]['mae_pp']) for v in VARIANTS]
            left.plot(x,y,color=COLORS[seed],marker=MARKERS[seed],markersize=7.5,
                      markeredgecolor='white',markeredgewidth=.65,linewidth=1.15,alpha=.95,zorder=3,label=f'Seed {seed}')
        baseline_map={x['baseline']:float(x['mae_pp']) for x in baselines}
        for name,label,style,color in [('zero','Zero predictor',(0,(5,3)),'#6C747C'),
                                       ('train_mean','Training-mean predictor',(0,(1.5,2.0)),'#313B45')]:
            y=baseline_map[name];left.axhline(y,color=color,linewidth=1.1,linestyle=style,zorder=2)
            left.annotate(f'{label}: {y:.3f} pp',xy=(2.31,y),xytext=(0,4),textcoords='offset points',
                          ha='right',va='bottom',fontsize=9,color=color,
                          bbox={'facecolor':'white','edgecolor':'none','pad':1.2})
        handles=[Line2D([0],[0],color=COLORS[s],marker=MARKERS[s],markersize=6.5,linewidth=1,label=f'Seed {s}') for s in SEEDS]
        left.legend(handles=handles,loc='upper left',bbox_to_anchor=(.005,.765),ncol=3,
                    frameon=False,fontsize=9,handlelength=1.5,columnspacing=1.3)
        left.text(.02,.55,'Lines link the same seed; horizontal offsets separate points.',
                  transform=left.transAxes,fontsize=8.2,color='#5B646D')
        right.set_xlim(-.5,2.5);right.set_ylim(-.004,.058)
        right.set_xticks(range(3),[f'Seed {s}' for s in SEEDS]);right.set_yticks([0,.01,.02,.03,.04,.05])
        right.set_ylabel('Soft − hard validation MAE (pp) ↓')
        right.axhline(0,color='#6C747C',linewidth=.9)
        deltas={int(x['seed']):float(x['soft_minus_hard_mae_pp']) for x in pairs}
        for i,seed in enumerate(SEEDS):
            value=deltas[seed]
            right.plot([i,i],[0,value],color=COLORS[seed],linewidth=1.1,alpha=.55,zorder=2)
            right.scatter([i],[value],color=COLORS[seed],marker=MARKERS[seed],s=65,
                          edgecolor='white',linewidth=.6,zorder=3)
            right.annotate(f'{value:+.4f}',xy=(i,value),xytext=(0,9),textcoords='offset points',
                           ha='center',va='bottom',color=COLORS[seed],fontsize=10,weight='bold')
        right.text(.02,.95,'Positive values favor hard supervision.',transform=right.transAxes,
                   va='top',fontsize=9,color='#46515C')
        right.text(.02,.04,'Zero = equal MAE',transform=right.transAxes,fontsize=8.4,color='#5B646D')
        fig.text(.075,.158,'Lower is better. MAE is averaged over the same 26 validation frames at R = 1, 2, 4.',fontsize=10)
        fig.text(.075,.117,'The best epoch was selected using this validation set; these are development results.',fontsize=10)
        fig.text(.075,.076,'One field, four dates. Seeds are optimization repeats, not independent fields. No confidence intervals or significance tests.',fontsize=9.3,color='#46515C')
        fig.canvas.draw();renderer=fig.canvas.get_renderer();frame=fig.bbox
        clipped=[]
        for item in fig.findobj(Text):
            if item.get_visible() and item.get_text():
                box=item.get_window_extent(renderer)
                if box.x0<frame.x0-.5 or box.x1>frame.x1+.5 or box.y0<frame.y0-.5 or box.y1>frame.y1+.5:
                    clipped.append(item.get_text())
        assert not clipped,clipped
        stem=out/'weeds_all9_development_comparison'
        fig.savefig(stem.with_suffix('.png'),dpi=300,facecolor='white')
        fig.savefig(stem.with_suffix('.svg'),facecolor='white')
        fig.savefig(stem.with_suffix('.pdf'),facecolor='white')
        plt.close(fig)
        # Independent MuPDF rasterization of both vector outputs; no source images/models.
        import fitz
        from PIL import Image
        qa={'passed':True,'matplotlib_text_inside_canvas':True,'python':sys.executable,
            'matplotlib_version':matplotlib.__version__,'fitz_version':fitz.VersionBind,
            'seed_shapes_distinct_in_grayscale':True,'rendered_files':[]}
        for suffix in ('pdf','svg'):
            document=fitz.open(stem.with_suffix('.'+suffix))
            assert len(document)==1
            pix=document[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
            destination=out/('qa_'+suffix+'_render.png');pix.save(destination)
            qa['rendered_files'].append({'format':suffix,'path':destination.name,'width':pix.width,'height':pix.height})
            document.close()
        with Image.open(stem.with_suffix('.png')) as im:
            assert im.width==3540 and im.height==1845
            qa['png_dimensions']=[im.width,im.height]
            im.convert('L').save(out/'qa_grayscale.png')
        for path,digest in INPUTS.items():assert sha(path)==digest,('Input changed',path)
        dump(out/'render_qa.json',qa)
        (out/'caption.md').write_text(
            '# External development comparison\n\n'
            'All nine predeclared ResNet18 runs on WeedsGalore are shown, including all three paired random seeds. '
            'Panel a gives coverage MAE at the earliest best epoch selected on the official 26-frame validation set, '
            'averaged over input factors R=1,2,4; each connected series represents one seed. '
            'The zero and training-mean constants use the same validation frames. '
            'Panel b shows each paired soft-minus-hard MAE difference; positive values favor hard supervision. '
            'The field of view and 256-pixel output support are fixed. '
            'The dataset represents one field over four dates. Seeds quantify optimization repeat variability and are not independent fields. '
            'No confidence interval or significance test is shown. These are development scores, not independent test estimates.\n\n'
            'PNG, SVG and PDF derive from the SHA-verified audit and original result CSV files copied into this directory. '
            'Both vector formats were independently rasterized with MuPDF; root review of the PNG is recorded separately. '
            'No training, inference, test access or changes to source run/audit files occurred.\n',encoding='utf-8')
        dump(out/'figure_manifest.json',{'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_code_sha256':sha(__file__),'input_sha256':INPUTS,'source_audit_sha256':AUDIT_SHA,
            'figure_protocol_sha256':sha(out/'figure_protocol.json'),'plotted_data_sha256':sha(out/'plotted_data.json'),
            'files_sha256':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
            'run_count':9,'paired_seed_count':3,'constant_baseline_count':2,
            'no_new_inference_or_training':True,'test_accessed':False,'R6_accessed':False,
            'human_or_root_visual_review':'pending; separate receipt required'})
        print(json.dumps({'passed':True,'directory':str(out),'png':str(stem.with_suffix('.png')),
                          'source_sha256':sha(__file__),'figure_manifest_sha256':sha(out/'figure_manifest.json')},ensure_ascii=False))
    except BaseException as error:
        dump(out/'failure.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc(),
                                'source_code_sha256':sha(__file__),'input_sha256':INPUTS})
        raise

if __name__=='__main__':main()
