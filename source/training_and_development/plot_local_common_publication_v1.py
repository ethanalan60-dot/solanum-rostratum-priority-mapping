"""Publication-size copies of three accepted local figures. Existing CSV values only."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,os,shutil,sys

ROOT=Path(__file__).resolve().parent
PA=ROOT/'precision_agriculture'
SOURCE=PA/'common_management_reports/full8_20260916_050927_560091'
OUT=PA/('local_publication_figures_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
OUT.mkdir(exist_ok=False)
os.environ['MPLCONFIGDIR']=str(OUT/'mpl_config')
os.environ['TMP']=str(OUT/'temp');os.environ['TEMP']=str(OUT/'temp');(OUT/'temp').mkdir()
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.patches import Rectangle
import fitz
from PIL import Image,ImageOps

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def js(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvout(p,x):
    with Path(p).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)

PAIRS=[('resnet18','0.0001','ResNet18 · LR 10⁻⁴'),('resnet18','0.0003','ResNet18 · LR 3×10⁻⁴'),('convnextv2_tiny','0.0001','ConvNeXt V2-T · LR 10⁻⁴'),('convnextv2_tiny','0.0003','ConvNeXt V2-T · LR 3×10⁻⁴')]
BUDGETS=[.1,.2,.3,.5,1.]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','axes.linewidth':.6,'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2.5,'ytick.major.size':2.5,'lines.linewidth':.7,'savefig.facecolor':'white'})

def export(fig,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
    outside=[];sizes=[]
    for a in fig.findobj(Text):
        if not a.get_visible() or not a.get_text():continue
        sizes.append(a.get_fontsize());bb=a.get_window_extent(renderer)
        if bb.x0<-.5 or bb.y0<-.5 or bb.x1>w+.5 or bb.y1>h+.5:outside.append({'text':a.get_text(),'bounds':list(bb.bounds)})
    assert min(sizes)>=8,(name,min(sizes));assert not outside,(name,outside)
    sizes_mm=[float(v)*25.4 for v in fig.get_size_inches()];assert abs(sizes_mm[0]-174)<1e-10 and sizes_mm[1]<=200
    paths={}
    for ext in ['png','pdf','svg']:
        p=OUT/(name+'.'+ext);fig.savefig(p,dpi=600 if ext=='png' else None);paths[ext]={'path':p.as_posix(),'sha256':sha(p)}
    doc=fitz.open(paths['pdf']['path']);page=doc[0];page.get_pixmap(matrix=fitz.Matrix(200/72,200/72)).save(str(OUT/(name+'_pdf_qa.png')))
    fonts=page.get_fonts(full=True);pdf_sizes=[]
    for b in page.get_text('dict')['blocks']:
        for l in b.get('lines',[]):
            for s in l.get('spans',[]):pdf_sizes.append(s['size'])
    pdf_pts=[page.rect.width,page.rect.height];assert abs(pdf_pts[0]*25.4/72-174)<.001 and pdf_pts[1]*25.4/72<=200.001
    with Image.open(paths['png']['path']) as im:
        pixel_size=list(im.size);ImageOps.grayscale(im).resize((1740,round(im.height*1740/im.width))).save(OUT/(name+'_grayscale_qa.png'))
    doc.close();plt.close(fig)
    return {'name':name,'formats':paths,'size_mm':sizes_mm,'raster_pixels':pixel_size,'png_dpi':600,'minimum_artist_font_pt':min(sizes),'minimum_extracted_pdf_font_pt':min(pdf_sizes),'pdf_fonts':fonts,'text_outside_figure':outside,'svg_font_encoding':'paths from same DejaVu Sans glyphs; no remote font substitution','vector_pdf_embedded_true_type':all(x[2]=='Type0' for x in fonts)}

def primary(data,policy,name):
    fig,axes=plt.subplots(4,3,figsize=(174/25.4,180/25.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.105,right=.975,bottom=.09,top=.913,hspace=.69,wspace=.20)
    legend=[Line2D([],[],linestyle='none',marker='o',markersize=5,markerfacecolor='none',markeredgecolor='#0072B2',markeredgewidth=.8,label='Hard supervision'),Line2D([],[],linestyle='none',marker='x',markersize=4,markeredgewidth=.8,color='#D55E00',label='Soft supervision')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.54,.992),ncol=2,frameon=False,handletextpad=.5,columnspacing=1.7)
    fig.supxlabel('Available area budget (%)',x=.54,y=.025,fontsize=8)
    fig.supylabel('Annotated target area captured (%)',x=.017,y=.50,fontsize=8)
    bindings=[]
    for row,(enc,lr,label) in enumerate(PAIRS):
        for col,R in enumerate([1,2,4]):
            ax=axes[row,col];ax.set_title(label+'\nR = '+str(R),pad=7,fontsize=8)
            ax.set_xlim(0,100);ax.set_ylim(0,100);ax.set_xticks([0,10,20,30,50,100]);ax.set_yticks([0,50,100])
            ax.tick_params(axis='x',pad=3);ax.tick_params(axis='y',pad=3)
            ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
            ax.grid(axis='y',color='#dddddd',linewidth=.4,zorder=0)
            for variant in ['hard_seg','soft_seg']:
                records=sorted([x for x in data if x['encoder']==enc and x['lr']==lr and x['policy']==policy and x['M']=='2' and x['R']==str(R) and x['variant']==variant],key=lambda x:float(x['budget']))
                assert len(records)==5 and [float(x['budget']) for x in records]==BUDGETS
                X=[float(x['budget'])*100 for x in records];Y=[float(x['capture_pct']) for x in records]
                kw=({'marker':'o','markersize':5,'markerfacecolor':'none','markeredgecolor':'#0072B2','markeredgewidth':.8} if variant=='hard_seg' else {'marker':'x','markersize':4,'color':'#D55E00','markeredgewidth':.8})
                line,=ax.plot(X,Y,linestyle='none',clip_on=False,zorder=3 if variant=='hard_seg' else 4,**kw)
                assert np.array_equal(line.get_xdata(),X) and np.array_equal(line.get_ydata(),Y)
                for i,record in enumerate(records):bindings.append({'figure':name,'panel_row':row,'panel_col':col,'artist_variant':variant,'artist_point_index':i,'artist_x_budget_pct':float(line.get_xdata()[i]),'artist_y_capture_pct':float(line.get_ydata()[i]),**record})
    assert len(bindings)==120
    csvout(OUT/(name+'_all120_bound_points.csv'),bindings)
    result=export(fig,name);result.update({'original_rows':120,'bound_artist_points':120,'max_coordinate_difference':0.0,'all_points_original_budget_x':True,'no_line_interpolation':True,'coincident_mark_handling':'Unfilled hard circle with soft cross; coordinates never jittered or offset.'})
    return result

def differences(data):
    name='fig3_all360_soft_minus_hard';fig,axes=plt.subplots(4,2,figsize=(174/25.4,196/25.4))
    fig.subplots_adjust(left=.13,right=.975,bottom=.155,top=.925,hspace=.40,wspace=.24)
    fig.text(.318,.973,'Fraction-ranked prefix',ha='center',va='center',fontsize=8)
    fig.text(.79,.973,'Predicted-area policy',ha='center',va='center',fontsize=8)
    bindings=[];MR=[(m,r) for m in [1,2,4] for r in [1,2,4]]
    for row,(enc,lr,label) in enumerate(PAIRS):
        for col,policy in enumerate(['prefix','predicted_area']):
            ax=axes[row,col];array=np.empty((9,5));records={}
            for yi,(M,R) in enumerate(MR):
                for xi,b in enumerate(BUDGETS):
                    found=[x for x in data if x['encoder']==enc and x['lr']==lr and x['policy']==policy and int(x['M'])==M and int(x['R'])==R and float(x['budget'])==b];assert len(found)==1
                    records[yi,xi]=found[0];array[yi,xi]=float(found[0]['soft_minus_hard_pp'])
            im=ax.imshow(array,origin='upper',aspect='auto',cmap='RdBu_r',vmin=-20,vmax=20,interpolation='nearest')
            assert np.array_equal(np.asarray(im.get_array()),array)
            ax.set_title(label,fontsize=8,pad=5)
            ax.set_yticks(range(9),[f'{M}/{R}' for M,R in MR]);ax.set_xticks(range(5),['10','20','30','50','100'])
            ax.tick_params(axis='both',length=0,pad=3)
            ax.set_ylabel('M (m) / R',fontsize=8,labelpad=5)
            if row==3:ax.set_xlabel('Available area budget (%)',fontsize=8,labelpad=5)
            ax.add_patch(Rectangle((-.5,2.5),5,3,fill=False,edgecolor='#555555',linewidth=.8))
            for yi in range(9):
                for xi in range(5):
                    value=float(array[yi,xi]);glyph='0' if abs(value)<=1e-9 else ('+' if value>0 else '−')
                    color='white' if abs(value)>12 else '#202020'
                    ax.text(xi,yi,glyph,ha='center',va='center',fontsize=8,color=color)
                    bindings.append({'figure':name,'panel_row':row,'panel_col':col,'cell_row':yi,'cell_col':xi,'artist_color_value_pp':value,'glyph':glyph,'glyph_numerical_tie_tolerance_pp':1e-9,**records[yi,xi]})
            for spine in ax.spines.values():spine.set_linewidth(.5)
    assert len(bindings)==360 and max(abs(float(x['artist_color_value_pp'])) for x in bindings)<20
    cax=fig.add_axes([.24,.065,.63,.013]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-20,-10,0,10,20]);cb.set_label('Soft − hard annotated-area capture (pp)',fontsize=8,labelpad=4);cb.ax.tick_params(labelsize=8)
    csvout(OUT/(name+'_all360_bound_cells.csv'),bindings)
    result=export(fig,name);result.update({'original_rows':360,'bound_heatmap_cells':360,'max_color_value_difference':0.0,'color_limits_pp':[-20,20],'cell_labels':'Signs only: + / − / 0, each8pt; all exact values preserved in bound CSV.','primary_M2_rows_outlined':True})
    return result

def main():
    sourcepaths=[SOURCE/'all_720_model_policy_budgets.csv',SOURCE/'all_360_soft_minus_hard_pairs.csv',PA/'full8_central_independent_20260916_051750_163534/central_review.json']
    expected=['b7baa56fb153e924ca8f7ebf8f2a3a0f793806e993010fc7d9da76150086d71b','0216df67184de321048371523675b3f68ca31f537695e13794fed68637cbf07d','6225b0b64dbd15df24dc69952af78a5919e3a98b6a434d30682f04e5a29ff9af']
    (OUT/'source').mkdir();inputs=[]
    for p,h in zip(sourcepaths,expected):assert sha(p)==h;shutil.copy2(p,OUT/'source'/p.name);inputs.append({'path':p.as_posix(),'sha256':h})
    shutil.copy2(__file__,OUT/'source/plot_source.py')
    data=load(sourcepaths[0]);paired=load(sourcepaths[1]);assert len(data)==720 and len(paired)==360
    figs=[primary(data,'prefix','fig1_primary_M2_prefix'),primary(data,'predicted_area','fig2_primary_M2_predicted_area'),differences(paired)]
    for x in inputs:assert sha(x['path'])==x['sha256']
    receipt={'state':'rendered_pending_actual_visual_review','created_utc':datetime.now(timezone.utc).isoformat(),'source_inputs':inputs,'source_sha256':sha(__file__),'python':sys.executable,'matplotlib':matplotlib.__version__,'figures':figs,'zero_new_statistics_inference_selection_solver':True,'scope':'Read audited full8 CSV only; re-encode layout/marks, preserve complete original values. No manuscript/old figure/source modification.'}
    js(OUT/'build_receipt.json',receipt)
    captions={1:'Primary 2 m grid, fraction-ranked prefix. All eight configurations, three R levels and five budgets are shown as 120 original points. Hard supervision uses open circles and soft supervision crosses at exact budget coordinates; coincident symbols are overlaid without jitter. Axes span0–100%. Actual area can differ under a common cap. One-scene seed17 development evidence.',2:'Primary 2 m grid, predicted-area selection. All120 original points use the same axes and symbols as Figure1. Strict-budget final choices include numerical fallbacks and unresolved gaps; no universal optimality claim is implied.',3:'All360 soft-minus-hard capture differences, with a shared symmetric colour scale. Panels include every encoder/LR pair, policy, management size, degradation and budget. Symbols indicate positive, negative or numerical-zero differences; exact values remain in the companion CSV. M2 rows are outlined. These are correlated development conditions, not independent field replicates.'}
    js(OUT/'figure_specs.json',{'figures':[{'number':i+1,'anchor':f'[Figure {i+1}]','formats':figs[i]['formats'],'caption':captions[i+1]} for i in range(3)],'width_mm':174,'old_figures_unchanged':True})
    print(json.dumps({'output':OUT.as_posix(),'receipt_sha256':sha(OUT/'build_receipt.json'),'figures':[{'name':x['name'],'size_mm':x['size_mm'],'min_font':x['minimum_artist_font_pt'],'min_pdf_font':x['minimum_extracted_pdf_font_pt']} for x in figs]},ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        js(OUT/'failure.json',{'type':type(exc).__name__,'message':str(exc),'source_sha256':sha(__file__)})
        raise
