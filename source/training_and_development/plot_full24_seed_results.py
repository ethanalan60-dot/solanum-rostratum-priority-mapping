"""Publication figures from accepted full24 tables; no model, solver or new science metrics."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,math,os,shutil
ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture'
PAIRS=[('resnet18',.0001,'ResNet18 · LR 10⁻⁴'),('resnet18',.0003,'ResNet18 · LR 3×10⁻⁴'),('convnextv2_tiny',.0001,'ConvNeXt V2-T · LR 10⁻⁴'),('convnextv2_tiny',.0003,'ConvNeXt V2-T · LR 3×10⁻⁴')]
COLORS={'hard_seg':'#0072B2','soft_seg':'#D55E00'};MARKERS={17:'o',29:'s',43:'^'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save_csv(p,data):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def numbers(row):
    result={}
    for k,v in row.items():
        try:result[k]=float(v)
        except ValueError:result[k]=None if v=='' else v
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--summary-directory',type=Path,required=True);parser.add_argument('--independent-review',type=Path,required=True);args=parser.parse_args()
    folder=args.summary_directory.resolve()
    watched={str(p.resolve()):sha(p) for p in (args.independent_review,folder/'status.json',folder/'summary.json',folder/'artifact_manifest.json',Path(__file__))}
    review=read(args.independent_review);status=read(folder/'status.json');manifest=read(folder/'artifact_manifest.json')
    assert review['passed'] and review['scope']=='full24_seed_summary_independent' and review['models']==24 and review['tables_audited']==25
    assert Path(review['output_directory']).resolve()==folder
    assert review['summary_sha256']==sha(folder/'summary.json') and review['audited_artifact_manifest_sha256']==sha(folder/'artifact_manifest.json')==status['artifact_manifest_sha256']
    assert status['state']=='completed'
    for name,digest in manifest['file_sha256'].items():assert sha(folder/name)==digest
    names={'all2160_model_policy_budgets':2160,'budget_three_seed_mean_sample_sd':720,'all72_cpu_scalar_by_R':72,'cpu_scalar_three_seed_mean_sample_sd':24}
    data={}
    for name,count in names.items():
        path=folder/(name+'.csv');assert sha(path)==manifest['file_sha256'][path.name];watched[str(path)]=manifest['file_sha256'][path.name];data[name]=[numbers(x) for x in rows(path)]
        assert len(data[name])==count==review['table_rows'][name]
    for name,metric in (('budget_three_seed_mean_sample_sd','capture_pct'),('cpu_scalar_three_seed_mean_sample_sd','balanced_mae_pp')):
        for row in data[name]:
            assert row['n_expected']==3 and row['seeds']=='17;29;43' and row['independent_fields']==1 and row[metric+'_n_valid']==3
            assert all(isinstance(row[metric+suffix],float) and math.isfinite(row[metric+suffix]) for suffix in ('_mean','_sample_sd')) and row[metric+'_sample_sd']>=0
    for path,digest in watched.items():assert sha(path)==digest
    out=PA/('full24_publication_figures_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();(out/'temp').mkdir();(out/'mpl_config').mkdir()
    os.environ.update(MPLCONFIGDIR=str(out/'mpl_config'),TEMP=str(out/'temp'),TMP=str(out/'temp'),CUDA_VISIBLE_DEVICES='')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from PIL import Image,ImageOps
    import fitz
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','axes.linewidth':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,'lines.linewidth':.75})
    figures=[];captions=[]
    def export(fig,name,points,means,caption):
        fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height();sizes=[]
        for item in fig.findobj(Text):
            if not item.get_visible() or not item.get_text():continue
            sizes.append(item.get_fontsize());b=item.get_window_extent(renderer);assert b.x0>=-.5 and b.y0>=-.5 and b.x1<=w+.5 and b.y1<=h+.5,(name,item.get_text(),b.bounds)
        assert min(sizes)>=8;files={}
        for extension in ('png','pdf','svg'):
            p=out/(name+'.'+extension);fig.savefig(p,dpi=600 if extension=='png' else None);files[extension]={'path':str(p),'sha256':sha(p)}
        pdf=fitz.open(files['pdf']['path']);page=pdf[0];assert abs(page.rect.width*25.4/72-174)<.001 and page.rect.height*25.4/72<=234
        page.get_pixmap(matrix=fitz.Matrix(180/72,180/72)).save(str(out/(name+'_pdf_qa.png')));pdf.close()
        with Image.open(files['png']['path']) as im:ImageOps.grayscale(im).resize((1740,round(im.height*1740/im.width))).save(out/(name+'_grayscale_qa.png'))
        save_csv(out/(name+'_individual_points.csv'),points);save_csv(out/(name+'_mean_sd.csv'),means)
        figures.append(dict(name=name,files=files,individual_points=len(points),mean_SD_points=len(means),minimum_font_pt=min(sizes),size_mm=[x*25.4 for x in fig.get_size_inches()],data_binding_verified=True,visual_review_pending=True));captions.append('### '+name+'\n\n'+caption)
        plt.close(fig)
    def axes_style(ax):
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#dedede',lw=.4,zorder=0)
    seed_handles=[Line2D([],[],marker=MARKERS[s],ls='none',color='#555555',markerfacecolor='none',markersize=3,label=f'Seed {s}') for s in (17,29,43)]
    supervision=[Line2D([],[],marker='o' if v=='hard_seg' else 'x',ls='none',color=COLORS[v],markerfacecolor='none',markersize=4,label='Hard' if v=='hard_seg' else 'Soft') for v in COLORS]
    # Six figures retain every2160 row. M2 prefix is primary; the rest are prespecified sensitivity/supplement figures.
    for m in (2,1,4):
        for policy in ('prefix','predicted_area'):
            fig,axes=plt.subplots(4,3,figsize=(174/25.4,196/25.4),sharex=True,sharey=True);fig.subplots_adjust(left=.10,right=.97,bottom=.09,top=.875,hspace=.62,wspace=.22)
            displayed_stats=[x for x in data['budget_three_seed_mean_sample_sd'] if (x['M'],x['policy'])==(m,policy)]
            ylo=min(0.,min(x['capture_pct_mean']-x['capture_pct_sample_sd'] for x in displayed_stats))-3
            yhi=max(100.,max(x['capture_pct_mean']+x['capture_pct_sample_sd'] for x in displayed_stats))+3
            fig.legend(handles=supervision+seed_handles,loc='upper center',bbox_to_anchor=(.53,.996),ncol=5,frameon=False,columnspacing=.75,handletextpad=.3)
            fig.text(.53,.941,'Individual seeds and mean ± sample SD',ha='center',fontsize=8)
            fig.supxlabel('Available area budget (%)',x=.53,y=.026,fontsize=8);fig.supylabel('Annotated target area captured (%)',x=.014,y=.49,fontsize=8)
            points=[];means=[]
            for ri,(enc,lr,label) in enumerate(PAIRS):
                for ci,r in enumerate((1,2,4)):
                    ax=axes[ri,ci];axes_style(ax);ax.set_title(label+'\nR = '+str(r),pad=6);ax.set_xlim(0,103);ax.set_ylim(ylo,yhi);ax.set_xticks([10,20,30,50,100]);ax.set_yticks([0,50,100])
                    for v in COLORS:
                        raw=[x for x in data['all2160_model_policy_budgets'] if (x['encoder'],x['lr'],x['variant'],x['M'],x['R'],x['policy'])==(enc,lr,v,m,r,policy)]
                        stats=[x for x in data['budget_three_seed_mean_sample_sd'] if (x['encoder'],x['lr'],x['variant'],x['M'],x['R'],x['policy'])==(enc,lr,v,m,r,policy)]
                        assert len(raw)==15 and len(stats)==5
                        for row in raw:
                            x=row['budget']*100;y=row['capture_pct'];assert y is not None
                            line,=ax.plot([x],[y],ls='none',marker=MARKERS[int(row['seed'])],ms=2.5,mfc='none',mec=COLORS[v],mew=.55,alpha=.6,zorder=2)
                            assert float(line.get_xdata()[0])==x and float(line.get_ydata()[0])==y;points.append({'panel_row':ri,'panel_col':ci,'x':x,'y':y,**row})
                        for row in stats:
                            assert row['capture_pct_n_valid']==3;x=row['budget']*100;y=row['capture_pct_mean'];sd=row['capture_pct_sample_sd']
                            artist=ax.errorbar([x],[y],yerr=[sd],fmt='o' if v=='hard_seg' else 'x',ms=4,mfc='none',mec=COLORS[v],color=COLORS[v],capsize=1.5,elinewidth=.6,mew=.8,zorder=3)
                            assert float(artist[0].get_ydata()[0])==y;means.append({'panel_row':ri,'panel_col':ci,'x':x,'y':y,'sd':sd,**row})
            assert len(points)==360 and len(means)==120
            export(fig,f'local_M{m}_{policy}_three_seeds',points,means,f'Local development targeting at M = {m} m using the {policy.replace("_"," ")} policy. Each point is one of all24 fixed fits at its original error-selected epoch. Budget observations are unconnected because they are samples from discrete allocation staircases. Large symbols and bars give the three-seed mean and sample SD; smaller shapes identify each seed. Bars describe training variability on one scene, not uncertainty across fields. Nominal budgets retain their exact x positions; actual used areas are in the bound tables. All five budgets and R1/R2/R4 are shown; no outcome-based selection of seeds or configurations was made.')
    fig,axes=plt.subplots(2,2,figsize=(174/25.4,145/25.4),sharex=True);fig.subplots_adjust(left=.105,right=.97,bottom=.13,top=.86,hspace=.45,wspace=.31)
    fig.legend(handles=supervision+seed_handles,loc='upper center',bbox_to_anchor=(.53,.99),ncol=5,frameon=False,columnspacing=.7,handletextpad=.3);fig.text(.53,.922,'Individual seeds and mean ± sample SD',ha='center')
    fig.supxlabel('Image-information degradation factor R',x=.53,y=.035,fontsize=8);fig.supylabel('Balanced MAE (percentage points)',x=.015,y=.48,fontsize=8);points=[];means=[]
    for ax,(enc,lr,label) in zip(axes.ravel(),PAIRS):
        axes_style(ax);ax.set_title(label,pad=6);ax.set_xticks([1,2,4]);ax.set_xlim(.6,4.4)
        for v in COLORS:
            raw=[x for x in data['all72_cpu_scalar_by_R'] if (x['encoder'],x['lr'],x['variant'])==(enc,lr,v)]
            stats=[x for x in data['cpu_scalar_three_seed_mean_sample_sd'] if (x['encoder'],x['lr'],x['variant'])==(enc,lr,v)]
            assert len(raw)==9 and len(stats)==3
            for row in raw:
                x=row['R'];y=row['balanced_mae_pp'];line,=ax.plot([x],[y],ls='none',marker=MARKERS[int(row['seed'])],ms=3,mfc='none',mec=COLORS[v],alpha=.6)
                assert float(line.get_ydata()[0])==y;points.append({'x':x,'y':y,**row})
            for row in stats:
                assert row['balanced_mae_pp_n_valid']==3;x=row['R'];y=row['balanced_mae_pp_mean'];sd=row['balanced_mae_pp_sample_sd']
                artist=ax.errorbar([x],[y],yerr=[sd],fmt='o' if v=='hard_seg' else 'x',ms=4,mfc='none',color=COLORS[v],capsize=2)
                assert float(artist[0].get_ydata()[0])==y;means.append({'x':x,'y':y,'sd':sd,**row})
    export(fig,'local_cpu_balanced_MAE_three_seeds',points,means,'Local CPU FP32 balanced window MAE for all24 fixed segmentation fits. Zero-target and positive-target strata are weighted equally. Original AMP-validation-selected epochs are retained; CPU evaluation does not reselect checkpoints. Mean ± sample SD reflects three training seeds within the same spatial split; there are no independent-field confidence intervals.')
    for path,digest in watched.items():assert sha(path)==digest
    shutil.copy2(__file__,out/'plot_source.py');dump(out/'figure_specs.json',dict(source_directory=str(folder),independent_review={'path':str(args.independent_review.resolve()),'sha256':sha(args.independent_review)},figures=figures,source_bindings=watched,all_budget_rows_plotted=2160,all_cpu_rows_plotted=72,primary_figure='local_M2_prefix_three_seeds',primary_budget_fraction=.1,primary_budget_raw_points=72,visual_review_pending=True,caption_interpretation='seed SD only; not field CI; mean plus/minus SD is untruncated and can exceed the physical capture range'))
    (out/'captions.md').write_text('# Complete local three-seed figures\n\n'+'\n\n'.join(captions)+'\n',encoding='utf-8');dump(out/'artifact_manifest.json',{'files_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'out':str(out),'figures':len(figures),'all_budget_rows':2160,'all_CPU_rows':72,'visual_review_pending':True}))
if __name__=='__main__':main()
