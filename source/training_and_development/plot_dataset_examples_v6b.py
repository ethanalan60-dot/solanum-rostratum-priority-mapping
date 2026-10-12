"""Render supplied training RGB and annotations, without any model prediction."""
from pathlib import Path
import json, hashlib, datetime as dt, string
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
REV = ROOT / 'precision_agriculture/final_v6_dataset_examples'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pin(p):
    return {'path':str(p), 'sha256':sha(p)}

def dump(p,d):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')

def main():
    source = REV/'data_examples.json'
    spec = json.loads(source.read_text(encoding='utf-8'))
    out = REV/'figures_v2'; out.mkdir(exist_ok=False)
    (out/'panels').mkdir()
    colors = {'target':np.array([230,159,0],dtype=np.uint8),
              'crop':np.array([0,114,178],dtype=np.uint8),
              'background':np.array([37,37,37],dtype=np.uint8)}
    plt.rcParams.update({'font.family':'Arial','font.size':8.5,'pdf.fonttype':42,'svg.fonttype':'none',
                         'savefig.facecolor':'white','figure.facecolor':'white'})
    rows, records = {}, []
    for item in spec['samples']:
        for key in ('rgb','mask'):
            assert sha(item[key]['path']) == item[key]['sha256']
        rgb = np.asarray(Image.open(item['rgb']['path']).convert('RGB'))
        mask = np.asarray(Image.open(item['mask']['path']))
        assert mask.ndim == 2 and rgb.shape[:2] == mask.shape
        assert list(mask.shape) == item['shape']
        if item['dataset']=='local':
            target = mask > 0; crop = np.zeros_like(target)
        else:
            assert set(np.unique(mask)).issubset({0,1,2,3,4,5})
            target = mask > 1; crop = mask == 1
        assert int(target.sum()) == item['target_pixels']
        assert int(crop.sum()) == item['crop_pixels']
        label = np.empty_like(rgb); label[:] = colors['background']
        label[target] = colors['target']; label[crop] = colors['crop']
        overlay = rgb.copy(); foreground = target | crop
        # Fixed 40% annotation opacity in every sample; no per-image contrast changes.
        overlay[foreground] = np.rint(.6*rgb[foreground]+.4*label[foreground]).astype(np.uint8)
        target_only = np.zeros_like(rgb); target_only[target] = rgb[target]
        views = [rgb,label,overlay,target_only]
        row = {'id':item['id'],'dataset':item['dataset'],'split':'train',
               'target_pixels':int(target.sum()),'crop_pixels':int(crop.sum()),
               'native_pixels':int(mask.size),'target_percent':100*float(target.mean()),
               'rgb':item['rgb'],'mask':item['mask'],'panels':{}}
        for name,array in zip(['rgb','annotation_mask','annotation_overlay','target_only_rgb'],views):
            p=out/'panels'/(item['id']+'_'+name+'.png'); Image.fromarray(array).save(p)
            row['panels'][name]=pin(p)
        row['checks']={'RGB_unchanged':bool(np.array_equal(views[0],rgb)),
                       'target_only_inside_equals_RGB':bool(np.array_equal(target_only[target],rgb[target])),
                       'target_only_outside_zero':bool(np.all(target_only[~target]==0)),
                       'overlay_background_unchanged':bool(np.array_equal(overlay[~foreground],rgb[~foreground])),
                       'mask_geometry_not_resized':True}
        assert all(row['checks'].values())
        records.append(row); rows[item['id']] = (item, views)
    figures = []
    for name,ids in spec['figures'].items():
        assert len(ids)==3
        fig=plt.figure(figsize=(174/25.4,174/25.4))
        left=.014; gap=.012; width=(.972-3*gap)/4; height=width*174/174
        letters=iter(string.ascii_lowercase)
        for ri,(y,identifier) in enumerate(zip([.665,.362,.059],ids)):
            item,views=rows[identifier]
            coverage=100*item['target_pixels']/(item['shape'][0]*item['shape'][1])
            label=('Local S. rostratum' if item['dataset']=='local' else 'WeedsGalore maize')
            label += ' | '+identifier+' | annotated '+('target' if item['dataset']=='local' else 'weed')+' '+f'{coverage:.2f}%'
            fig.text(left,y+height+.033,label,fontsize=8.5,fontweight='bold',va='bottom')
            for ci,(title,view) in enumerate(zip(['RGB','Annotation mask','Annotation overlay','Target-only RGB'],views)):
                ax=fig.add_axes([left+ci*(width+gap),y,width,height])
                ax.imshow(view,interpolation='nearest',aspect='equal')
                ax.set_title('('+next(letters)+') '+title,loc='left',fontsize=8.2,pad=4)
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_linewidth(.4);spine.set_edgecolor('#666666')
        legend=[Patch(facecolor=colors['target']/255,label='Local target / external pooled weeds'),
                Patch(facecolor=colors['crop']/255,label='Crop (external only)'),
                Patch(facecolor=colors['background']/255,label='Mask background')]
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.006),ncol=3,
                   frameon=False,fontsize=8,handlelength=1.0,columnspacing=1.2)
        formats={}
        for ext in ['png','pdf','svg']:
            p=out/(name+'.'+ext)
            fig.savefig(p,dpi=600)
            formats[ext]=pin(p)
        plt.close(fig)
        figures.append({'name':name,'sample_ids':ids,'width_mm':174,'height_mm':174,'formats':formats})
    result={'created_at':dt.datetime.now().astimezone().isoformat(),'source':pin(source),
            'script':pin(Path(__file__)),'figures':figures,'samples':records,
            'display':{'overlay_alpha':.4,'local_RGB':'Native uint8 without colour adjustment',
                       'external_RGB':'Fixed round(uint16/65535*255), no per-image stretching',
                       'mask_classes':{'local':'positive supplied region mask','external':'0 background; 1 crop; >1 pooled weeds'},
                       'target_only':'Unchanged RGB inside supplied target/weed annotation; all other RGB pixels set to zero',
                       'interpolation':'nearest','native_orientation_preserved':True},
            'new_model_calls':0,'new_solver_calls':0,'performance_based_selection':False,
            'scientific_results_modified':False,'visual_review_complete':False}
    dump(out/'render_manifest.json',result)
    print(json.dumps({'directory':str(out),'samples':len(records),'figures':len(figures)},ensure_ascii=False))

if __name__=='__main__':main()
