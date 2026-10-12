"""Vector study schematic; scientific definitions only, no uncompleted empirical results."""
from pathlib import Path
import datetime,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=PA/('allocation_framework_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    plt.rcParams.update({'font.family':'Arial','font.size':8.5,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(174/25.4,154/25.4));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ink='#202d35';blue='#315f78';gray='#657078'
    def box(x,y,w,h,title,body,fill='#f1f5f7',titlecolor=ink):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.008',linewidth=.7,edgecolor=gray,facecolor=fill))
        ax.text(x+w/2,y+h-.027,title,ha='center',va='top',fontsize=9,fontweight='bold',color=titlecolor)
        ax.text(x+w/2,y+h-.070,body,ha='center',va='top',fontsize=8.5,linespacing=1.5,color=ink)
    def arrow(p,q):ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=9,color=gray,linewidth=.8))
    ax.text(.018,.975,'a',va='top',fontweight='bold',fontsize=10)
    ax.text(.072,.975,'Vary information and selectable-unit size separately',va='top',fontweight='bold',fontsize=9)
    box(.05,.782,.42,.135,'Image information, R','Controlled RGB degradation\nFixed field of view and output support')
    box(.53,.782,.42,.135,'Selectable units, M or g','Nested metre grids locally\nRelative equal-area grids externally')
    arrow((.26,.774),(.37,.722));arrow((.74,.774),(.63,.722))
    box(.16,.59,.68,.13,'Integrate predictions onto each selectable unit','P: predicted target area     A: observed valid area\nLocal: projected area; external: frame-area equivalents',fill='#eaf1f5',titlecolor=blue)
    arrow((.5,.584),(.5,.546))
    ax.text(.018,.544,'b',va='top',fontweight='bold',fontsize=10)
    box(.05,.36,.90,.175,'Allocate complete units under an area allowance','Primary: fraction-ranked prefix, with fixed ties\nLocal diagnostic: maximize predicted area under the same allowance\nSave all declared rankings and selections before reference scoring',fill='#f4f4f4')
    arrow((.29,.354),(.29,.31));arrow((.71,.354),(.71,.31))
    ax.text(.018,.312,'c',va='top',fontweight='bold',fontsize=10)
    box(.05,.135,.43,.165,'Targeting outcomes','Annotated target captured and missed\nActual area selected and unused\nComplete budget curves and exact AUC')
    box(.52,.135,.43,.165,'External crop outcome','Annotated crop inside selected units\nCrop capture and crop composition\nSeparate from crop injury')
    ax.text(.5,.079,'Metre-based global budgets and frame-specific budgets are evaluated separately.',ha='center',va='center',fontsize=8.5,color=ink)
    ax.text(.5,.039,'Reference rankings and random allocations describe the same observed domain.',ha='center',va='center',fontsize=8.5,color=ink)
    for suffix in ('png','pdf','svg'):fig.savefig(out/('allocation_framework.'+suffix),dpi=600,facecolor='white')
    plt.close(fig)
    caption=('Study design for assessing map value under area constraints. (a) Image-information degradation changes prediction inputs, whereas management-unit size changes the complete units available for selection. '
      '(b) The primary policy ranks predicted fractions and retains the largest feasible complete prefix. The local predicted-area policy is a separately declared diagnostic. '
      '(c) Saved selections are evaluated for annotated target capture and actual area; the external dataset additionally supports crop-annotation inclusion. Local metre-based global allocation and external frame-specific allocation are separate problems. '
      'The diagram states the analysis structure and contains no confirmation results. Image degradation does not represent an additional flight, and selected area does not measure herbicide use, field efficacy or profit.')
    (out/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    sources=[Path(__file__),PA/'PA_manuscript_integrated_development_v2_1.md',ROOT/'confirmation_v1/DESIGN_PREPARATION.md']
    manifest={'status':'candidate schematic; visual review pending','dimensions_mm':[174,154],'minimum_font_pt':8.5,'data_used':'protocol definitions only; no results/images/masks/predictions',
      'sources':{str(p):sha(p) for p in sources},'artifacts':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'empirical_claims_added':False}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(out)
if __name__=='__main__':main()
