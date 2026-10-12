"""Plot only completed, source-bound R x M validation tables; no model access."""
from pathlib import Path
import argparse,csv,hashlib,json,math,os
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prefix(rows,score):
    sorted_rows=sorted(rows,key=lambda r:(-score(r),r['id']))
    A=math.fsum(r['area'] for r in rows);T=math.fsum(r['target'] for r in rows)
    area=np.r_[0.,np.cumsum([r['area'] for r in sorted_rows])]
    target=np.r_[0.,np.cumsum([r['target'] for r in sorted_rows])]
    area[-1]=A;target[-1]=T
    assert T>0 and A>0
    return area/A,target/T

def main(directory):
    directory=Path(directory).resolve()
    summary_path=directory/'summary.json'
    summary=json.loads(summary_path.read_text(encoding='utf-8-sig'))
    assert summary['state']=='completed' and summary['model_run']=='dev_v2_03_softseg_aug_s17'
    assert summary['checkpoint_epoch']==12 and not summary['test_accessed'] and not summary['R6_accessed']
    status=json.loads((directory/'status.json').read_text(encoding='utf-8-sig'))
    assert status['state']=='completed'
    source=directory/'unit_predictions.csv'
    with source.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert sha(source)==status['files_sha256'][source.name]
    with (directory/'fixed_budgets.csv').open(encoding='utf-8-sig',newline='') as stream:
        fixed=list(csv.DictReader(stream))
    for r in rows:
        r['M']=int(r['M'])
        for key in ('area','target','predicted_R1','predicted_R2','predicted_R4'):
            r[key]=float(r[key])
    assert {r['M'] for r in rows}=={1,2,4}
    out=directory/'figures'
    out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors={1:'#0072B2',2:'#D55E00',4:'#009E73'}
    capture=np.empty((3,3));records=[]
    fig,axes=plt.subplots(1,3,figsize=(13.1,4.8),sharex=True,sharey=True)
    for mi,M in enumerate((1,2,4)):
        block=[r for r in rows if r['M']==M]
        for ri,R in enumerate((1,2,4)):
            x,y=prefix(block,lambda r:r['predicted_R'+str(R)]/r['area'])
            axes[mi].step(100*x,100*y,where='post',color=colors[R],linewidth=1.8,
                          linestyle={1:'-',2:'--',4:':'}[R],label=f'R = {R}')
            k=np.searchsorted(x,.1,side='right')-1
            capture[mi,ri]=100*y[k]
            reference=next(r for r in fixed if int(r['M'])==M and r['method']=='model_prefix'
                           and r['R']==str(R) and float(r['budget'])==.1)
            assert abs(y[k]-float(reference['capture_fraction']))<1e-10
            assert abs(x[k]-float(reference['actual_area_fraction']))<1e-10
            records.append({'M':M,'R':R,'budget':.1,'actual_area_fraction':float(x[k]),
                            'capture_fraction':float(y[k])})
        x,y=prefix(block,lambda r:r['target']/r['area'])
        axes[mi].step(100*x,100*y,where='post',color='#555555',linewidth=1.2,
                      linestyle='-.',label='Label-based ranking')
        axes[mi].axvline(10,color='#BBBBBB',linewidth=1)
        axes[mi].set(title=f'{chr(65+mi)}  Management unit: {M} m',xlabel='Valid-area budget (%)',
                     xlim=(0,100),ylim=(0,103),xticks=[0,10,25,50,75,100],yticks=[0,25,50,75,100])
        axes[mi].grid(alpha=.18)
    axes[0].set_ylabel('Annotated target area captured (%)')
    axes[-1].legend(loc='lower right',frameon=False,fontsize=9)
    fig.suptitle('Spatial targeting across image resolution and management unit size',fontsize=14,y=.98)
    fig.text(.5,.05,'Development validation | one scene, 357 source windows | original soft-seg checkpoint, seed 17\n'
             'R = synthetic degradation factor. Whole-unit prefix policy; label-based ranking uses reference labels.',
             ha='center',fontsize=9)
    fig.subplots_adjust(left=.065,right=.99,bottom=.24,top=.83,wspace=.13)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'rm_budget_curves.{ext}',dpi=190 if ext=='png' else 300)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,5.8))
    im=ax.imshow(capture,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    for mi in range(3):
        for ri in range(3):ax.text(ri,mi,f'{capture[mi,ri]:.2f}%',ha='center',va='center',
                                  fontsize=16,fontweight='bold',color='white' if capture[mi,ri]>65 else '#111111')
    ax.set(xticks=range(3),xticklabels=['R = 1','R = 2','R = 4'],yticks=range(3),yticklabels=['1 m','2 m','4 m'],
           xlabel='Synthetic image degradation factor',ylabel='Management unit size')
    ax.set_title('Annotated target capture at a 10% valid-area budget',pad=16,fontsize=13)
    fig.colorbar(im,ax=ax,label='Target area captured (%)',shrink=.85,pad=.035)
    fig.text(.5,.045,'Same observed domain and budget limit; actual selected area can differ.\n'
             'Single-scene development result; area capture is not weed-control efficacy.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.12,right=.92,bottom=.22,top=.85)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'rm_capture_10pct.{ext}',dpi=190 if ext=='png' else 300)
    plt.close(fig)
    (out/'figure_source.json').write_text(json.dumps({'unit_table':str(source),'unit_table_sha256':sha(source),
        'summary_sha256':sha(summary_path),'script_sha256':sha(__file__),'plotted_10pct_values':records,
        'curve_interpolation':'post step; no partial units','summary_loaded':bool(summary),
        'test_accessed':False,'interpretation':'development validation only'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'figures':str(out),'capture_10pct_percent':capture.tolist()},ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory');args=ap.parse_args();main(args.directory)
