"""Publication figures from independently accepted confirmation report CSVs only."""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, math, os, shutil

ROOT = Path(__file__).resolve().parent
PA = ROOT / 'precision_agriculture'
RS = (1, 2, 4, 6)
CONFIGS = [('resnet18', .0001, 'ResNet18 | LR 10⁻⁴'), ('resnet18', .0003, 'ResNet18 | LR 3×10⁻⁴'),
           ('convnextv2_tiny', .0001, 'ConvNeXt V2-T | LR 10⁻⁴'), ('convnextv2_tiny', .0003, 'ConvNeXt V2-T | LR 3×10⁻⁴')]
COLORS = {'hard_seg': '#0072B2', 'soft_seg': '#D55E00', 'regression': '#009E73'}
MARKERS = {17: 'o', 29: 's', 43: '^'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def numeric(row):
    result = {}
    for k, value in row.items():
        if value == '': result[k] = None; continue
        try:
            number = float(value)
            assert math.isfinite(number), (k, value)
            result[k] = number
        except ValueError: result[k] = value
    return result


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return [numeric(r) for r in csv.DictReader(f)]


def csv_write(path, data):
    assert data
    fields = list(dict.fromkeys(k for r in data for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-directory', type=Path, required=True)
    parser.add_argument('--independent-review', type=Path, required=True)
    args = parser.parse_args()
    folder = args.report_directory.resolve()
    watched = {str(p.resolve()): sha(p) for p in [Path(__file__), args.independent_review, folder/'artifact_manifest.json', folder/'summary.json', folder/'status.json']}
    review, manifest, summary = read(args.independent_review), read(folder/'artifact_manifest.json'), read(folder/'summary.json')
    assert review['passed'] is True and review['scope'] == 'confirmation_report_tables_independent'
    assert Path(review['output_directory']).resolve() == folder
    assert review['audited_artifact_manifest_sha256'] == sha(folder/'artifact_manifest.json')
    assert review['summary_sha256'] == sha(folder/'summary.json') and read(folder/'status.json')['state'] == 'completed'
    assert summary['R'] == list(RS) and summary['models'] == {'local': 28, 'weeds': 9}
    assert summary['local_spatial_models'] == 24 and summary['external_spatial_models'] == 6 and summary['datasets_pooled'] is False
    freeze = ROOT/'confirmation_v1/FREEZE.json'
    assert sha(freeze) == review['freeze_sha256']; watched[str(freeze)] = sha(freeze)
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'}
    assert actual == set(manifest['files'])
    for name, digest in manifest['files'].items():
        p = (folder/name).resolve(); assert p.is_relative_to(folder) and sha(p) == digest
    def table(name, expected=None):
        p = folder/name; assert sha(p) == manifest['files'][name]; watched[str(p)] = sha(p)
        result = csv_rows(p)
        assert len(result) == summary['table_rows'][name]
        if expected is not None: assert len(result) == expected
        return result
    local = table('local/spatial_budget_all.csv', 2880)
    local_mean = table('local/spatial_budget_seed_mean_sample_sd.csv', 960)
    cpu_local = table('local/scalar_all.csv', 112)
    cpu_local_mean = table('local/scalar_seed_mean_sample_sd.csv', 48)
    cpu_weeds = table('weeds/scalar_all.csv', 36)
    cpu_weeds_mean = table('weeds/scalar_seed_mean_sample_sd.csv', 12)
    external = table('weeds/pooled_spatial_budget_all.csv', 720)
    external_mean = table('weeds/pooled_spatial_budget_seed_mean_sample_sd.csv', 240)
    for r in local_mean + external_mean:
        assert r['n_models'] == 3 and r['seeds'] == '17;29;43' and r['independent_field_replicates'] == 1
    out = PA/('confirmation_publication_figures_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(); (out/'temp').mkdir(); (out/'mpl_config').mkdir()
    os.environ.update(MPLCONFIGDIR=str(out/'mpl_config'), TEMP=str(out/'temp'), TMP=str(out/'temp'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from PIL import Image, ImageOps
    import fitz
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8, 'axes.labelsize':8, 'axes.titlesize':8,
        'xtick.labelsize':8, 'ytick.labelsize':8, 'legend.fontsize':8, 'pdf.fonttype':42, 'ps.fonttype':42,
        'svg.fonttype':'path', 'axes.linewidth':.6, 'xtick.major.size':2.5, 'ytick.major.size':2.5})
    figures, captions = [], []
    def handles(variants):
        return [Line2D([], [], marker='D' if v=='regression' else 'o' if v=='hard_seg' else 'x', ls='none',
                color=COLORS[v], mfc='none', ms=4, label=v.replace('_seg','').title()) for v in variants] + [
                Line2D([], [], marker=MARKERS[s], ls='none', color='#555555', mfc='none', ms=3, label=f'Seed {s}') for s in MARKERS]
    def style(ax):
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', color='#dddddd', lw=.4)
    def draw(ax, raw, means, metric, xkey, xscale, variants):
        points, averages = [], []
        for row in raw:
            x = row[xkey]*xscale; y = row[metric]; v = row['variant']
            points.append({'x':x, 'y':y, 'plotted':y is not None, **row})
            if y is not None:
                assert ax.get_ylim()[0] <= y <= ax.get_ylim()[1], ('individual clipped', metric, y)
                artist, = ax.plot([x], [y], ls='none', marker=MARKERS[int(row['seed'])], ms=2.7, mfc='none', mec=COLORS[v], mew=.55, alpha=.65, zorder=2)
                assert float(artist.get_xdata()[0]) == x and float(artist.get_ydata()[0]) == y
        for row in means:
            v = row['variant']; x = row[xkey]*xscale; y = row[metric+'_mean']; sd = row[metric+'_sample_sd']
            assert row['n_models'] in (1,3) and 0 <= row[metric+'_defined_models'] <= row['n_models']
            averages.append({'x':x, 'y':y, 'sd':sd, 'plotted':y is not None, **row})
            if y is None:
                assert sd is None and row[metric+'_defined_models'] < row['n_models']; continue
            assert row[metric+'_defined_models'] == row['n_models']
            assert sd is None if row['n_models']==1 else sd is not None and sd >= 0
            assert ax.get_ylim()[0] <= y-(sd or 0) and y+(sd or 0) <= ax.get_ylim()[1], ('mean/SD clipped', metric)
            artist = ax.errorbar([x], [y], yerr=None if sd is None else [sd], fmt='D' if v=='regression' else 'o' if v=='hard_seg' else 'x',
                    ms=4, mfc='none', mec=COLORS[v], color=COLORS[v], capsize=1.5, elinewidth=.6, mew=.8, zorder=3)
            assert float(artist[0].get_ydata()[0]) == y
        if not any(r['plotted'] for r in points):
            ax.text(.5,.5,'Undefined ratio\n(zero denominator)', transform=ax.transAxes, ha='center', va='center', fontsize=8)
        return points, averages
    def ylim(means, metric, percent=False, raw=()):
        vals = [r[metric+'_mean']+sign*(r[metric+'_sample_sd'] or 0) for r in means if r[metric+'_mean'] is not None for sign in (-1,1)]
        vals += [r[metric] for r in raw if r[metric] is not None]
        lo = min([0]+vals); hi = max(([100] if percent else [1])+vals)
        pad = max(.05, (hi-lo)*.04); return lo-pad, hi+pad
    def export(fig, name, points, means, caption):
        fig.canvas.draw(); renderer=fig.canvas.get_renderer(); width,height=fig.canvas.get_width_height(); sizes=[]
        for text in fig.findobj(Text):
            if not text.get_visible() or not text.get_text(): continue
            sizes.append(text.get_fontsize()); b=text.get_window_extent(renderer)
            assert b.x0 >= -.5 and b.y0 >= -.5 and b.x1 <= width+.5 and b.y1 <= height+.5, (name,text.get_text(),b.bounds)
        assert min(sizes)>=8
        files={}
        for ext in ('png','pdf','svg'):
            p=out/(name+'.'+ext); fig.savefig(p,dpi=600 if ext=='png' else None); files[ext]={'path':str(p),'sha256':sha(p)}
        pdf=fitz.open(files['pdf']['path']); assert len(pdf)==1
        assert abs(pdf[0].rect.width*25.4/72-174)<.001 and pdf[0].rect.height*25.4/72<=234
        pdf[0].get_pixmap(matrix=fitz.Matrix(180/72,180/72)).save(str(out/(name+'_pdf_qa.png'))); pdf.close()
        with Image.open(files['png']['path']) as image: ImageOps.grayscale(image).resize((1740,round(image.height*1740/image.width))).save(out/(name+'_grayscale_qa.png'))
        csv_write(out/(name+'_individual_points.csv'),points); csv_write(out/(name+'_mean_sd.csv'),means)
        figures.append({'name':name,'files':files,'individual_rows':len(points),'plotted_individual_rows':sum(r['plotted'] for r in points),
            'mean_rows':len(means),'plotted_mean_rows':sum(r['plotted'] for r in means),'minimum_font_pt':min(sizes),'visual_review_pending':True})
        captions.append('### '+name+'\n\n'+caption); plt.close(fig)
    seed_note = ('Small shapes identify all fixed training seeds. Larger symbols and bars give the mean and sample SD, not field confidence intervals. '
                 'Undefined ratios remain absent with their rows retained in the figure data. Error bars are not clipped to the physical ratio range. '
                 'Budget points are unconnected observations of discrete allocation curves. R=6 is the additional frozen degradation condition; its intermediate size is 42 pixels.')
    for m in (2,1,4):
        for policy in ('prefix','predicted_area'):
            fig,axes=plt.subplots(4,4,figsize=(174/25.4,205/25.4),sharex=True,sharey=True)
            fig.subplots_adjust(left=.09,right=.98,bottom=.09,top=.865,hspace=.78,wspace=.20)
            fig.legend(handles=handles(('hard_seg','soft_seg')),loc='upper center',bbox_to_anchor=(.53,.998),ncol=5,frameon=False,columnspacing=.7,handletextpad=.3)
            fig.text(.53,.953,f'Confirmation | M = {m} m | '+policy.replace('_',' '),ha='center',fontsize=8)
            fig.supxlabel('Available area budget (%)',x=.53,y=.025,fontsize=8); fig.supylabel('Annotated target area captured (%)',x=.008,y=.48,fontsize=8)
            limits=ylim([r for r in local_mean if r['M']==m and r['policy']==policy],'capture_pct',True,[r for r in local if r['M']==m and r['policy']==policy])
            points,means=[],[]
            for ri,(enc,lr,label) in enumerate(CONFIGS):
                top=axes[ri,0].get_position().y1
                fig.text(.535,top+.035,label,ha='center',va='bottom',fontsize=8)
                for ci,r in enumerate(RS):
                    ax=axes[ri,ci]; style(ax); ax.set_title(f'R = {r}',pad=4); ax.set_xlim(0,106); ax.set_xticks([10,20,30,50,100]); ax.tick_params(axis='x', labelrotation=45); ax.set_ylim(*limits); ax.set_yticks([0,50,100])
                    raw=[x for x in local if (x['encoder'],x['lr'],x['M'],x['R'],x['policy'])==(enc,lr,m,r,policy)]
                    avg=[x for x in local_mean if (x['encoder'],x['lr'],x['M'],x['R'],x['policy'])==(enc,lr,m,r,policy)]
                    assert len(raw)==30 and len(avg)==10
                    p,s=draw(ax,raw,avg,'capture_pct','budget',100,('hard_seg','soft_seg')); points+=p; means+=s
            assert len(points)==480 and len(means)==160
            export(fig,f'confirmation_local_M{m}_{policy}',points,means,f'Frozen local confirmation at M={m} m with the {policy.replace("_"," ")} policy. '+seed_note+' Actual selected areas, unused allowances and missed targets are retained in the accepted report tables.')
    for dataset,raw_all,mean_all,metric in [('local',cpu_local,cpu_local_mean,'mae_pp'),('local',cpu_local,cpu_local_mean,'balanced_mae_pp'),('weeds',cpu_weeds,cpu_weeds_mean,'mae_pp')]:
        local_dataset=dataset=='local'; configs=CONFIGS if local_dataset else [CONFIGS[0]]
        fig,axs=plt.subplots(2 if local_dataset else 1,2 if local_dataset else 1,figsize=(174/25.4,(150 if local_dataset else 100)/25.4),squeeze=False)
        fig.subplots_adjust(left=.12,right=.96,bottom=.14,top=.81,hspace=.55,wspace=.32)
        fig.legend(handles=handles(('hard_seg','soft_seg','regression')),loc='upper center',bbox_to_anchor=(.53,.99),ncol=3,frameon=False)
        fig.supxlabel('Image-information degradation factor R',x=.53,y=.03,fontsize=8)
        fig.supylabel(('Balanced MAE' if metric=='balanced_mae_pp' else 'MAE')+' (percentage points)',x=.018,y=.48,fontsize=8)
        points,means=[],[]
        for ax,(enc,lr,label) in zip(axs.ravel(),configs):
            style(ax); ax.set_title(label,pad=6); ax.set_xticks(RS); ax.set_xlim(.6,6.4)
            raw=[r for r in raw_all if (r['encoder'],r['lr'])==(enc,lr)]
            avg=[r for r in mean_all if (r['encoder'],r['lr'])==(enc,lr)]
            ax.set_ylim(*ylim(avg,metric,raw=raw)); p,s=draw(ax,raw,avg,metric,'R',1,tuple(COLORS)); points+=p; means+=s
        assert len(points)==(112 if local_dataset else 36)
        export(fig,f'confirmation_{dataset}_{metric}',points,means,'CPU float32 confirmation error at every original selected checkpoint. '+('Local regression retains its original single seed and has no sample SD; local segmentation uses three seeds.' if local_dataset else 'Each external method uses three seeds. All methods were retrained on the external dataset.')+' Balanced MAE, when shown, gives equal weight to zero-target and positive-target strata and is undefined if either stratum is empty. These are fitting repetitions within one scene, not independent fields.')
    for grid in (8,4,16):
        for basis in ('nominal_primary','matched_actual_secondary'):
            fig,axes=plt.subplots(3,4,figsize=(174/25.4,174/25.4),sharex=True)
            fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.86,hspace=.88,wspace=.23)
            fig.legend(handles=handles(('hard_seg','soft_seg')),loc='upper center',bbox_to_anchor=(.53,.998),ncol=5,frameon=False,columnspacing=.7,handletextpad=.3)
            fig.text(.53,.952,f'External confirmation | grid g={grid} | '+basis.replace('_',' '),ha='center',fontsize=8)
            fig.supxlabel('Actual selected frame area (%)',x=.53,y=.03,fontsize=8)
            raw=[x for x in external if x['grid_g']==grid and x['budget_basis']==basis]
            avg=[x for x in external_mean if x['grid_g']==grid and x['budget_basis']==basis]
            points,means=[],[]
            for ri,(metric,label) in enumerate([('weed_capture_pct','Weed area captured (%)'),('crop_capture_pct','Crop area included (%)'),('crop_fraction_pct','Crop fraction of selected area (%)')]):
                top=axes[ri,0].get_position().y1;fig.text(.535,top+.043,label,ha='center',va='bottom',fontsize=8)
                for ci,r in enumerate(RS):
                    ax=axes[ri,ci];style(ax);ax.set_title(f'R = {r}',pad=4);ax.set_xlim(0,106);ax.set_xticks([0,20,50,100]);ax.set_ylim(*ylim(avg,metric,True,raw));ax.set_yticks([0,50,100])
                    rr=[dict(x,plot_x=x['actual_fraction']) for x in raw if x['R']==r]
                    ss=[]
                    for x in avg:
                        if x['R']!=r:continue
                        same=[z for z in rr if z['variant']==x['variant'] and z['parent_nominal_budget_code']==x['parent_nominal_budget_code']]
                        xs={z['plot_x'] for z in same};assert len(xs)==1
                        ss.append(dict(x,plot_x=next(iter(xs))))
                    assert len(rr)==30 and len(ss)==10
                    p,s=draw(ax,rr,ss,metric,'plot_x',100,('hard_seg','soft_seg'))
                    points += [dict(z,plotted_metric=metric) for z in p];means += [dict(z,plotted_metric=metric) for z in s]
            assert len(points)==360 and len(means)==120
            export(fig,f'confirmation_weeds_g{grid}_{basis}',points,means,'External pooled-frame confirmation weed capture, crop capture and crop composition under '+basis.replace('_',' ')+' budgets. All 26 frame areas are counted separately, not as unique geographic ground area. Axes show actual selected area; nominal and matched-budget families remain separate. '+seed_note+' Crop inclusion is a spatial proxy, not measured contact, dose or crop injury.')
    for path,digest in watched.items(): assert sha(path)==digest
    shutil.copy2(__file__,out/'plot_source.py')
    dump(out/'figure_specs.json',{'scope':'independently_accepted_confirmation_tables','report_directory':str(folder),'independent_review':{'path':str(args.independent_review.resolve()),'sha256':sha(args.independent_review)},
        'figures':figures,'source_bindings':watched,'local_budget_rows':2880,'unique_scalar_rows':148,'external_pooled_budget_rows':720,
        'external_rows_repeated_across_three_metrics':True,'local_scalar_rows_repeated_for_ordinary_and_balanced_error':True,
        'fitting_seed_semantics':{'local_segmentation':{'n':3,'seeds':[17,29,43]},'local_regression':{'n':1,'seeds':[17],'sample_sd':None},'external_all_methods':{'n':3,'seeds':[17,29,43]},'independent_fields_per_dataset':1},
        'visual_review_pending':True,'new_model_calls':0,'new_solver_calls':0,'raw_arrays_loaded':0})
    (out/'captions.md').write_text('# Frozen confirmation figures\n\n'+'\n\n'.join(captions)+'\n',encoding='utf-8')
    dump(out/'artifact_manifest.json',{'files_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'out':str(out),'figures':len(figures),'visual_review_pending':True}))


if __name__ == '__main__': main()
