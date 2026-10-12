"""Add training-image illustrations to a new manuscript revision; retain all results."""
from pathlib import Path
import json, hashlib, re, datetime as dt

ROOT=Path(__file__).resolve().parent
PA=ROOT/'precision_agriculture'
OLD=PA/'final_v5'
NEW=PA/'final_v6_dataset_examples'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def text(p,s):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:f.write(s)
def dump(p,d):text(p,json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    assert sha(OLD/'PA_manuscript.md')=='07bca0bebd178e5071bf9b8e2e329cc55532c708724ab35f7b97fc08835ecff6'
    assert sha(OLD/'PA_supplementary.md')=='1343fede7c05aa70e781c7da8cea95f2d98a0eaac147aee4464eb604e8af18c6'
    figures={x['name']:x for x in read(NEW/'figures_v2/render_manifest.json')['figures']}
    caption_main=('Training imagery and supplied annotations. The first two rows show local windows nearest the median and 90th percentile of positive training annotation coverage; the last shows the first official WeedsGalore training frame containing both crop and weed labels. Columns show RGB, the annotation mask, a 40% opacity overlay, and unchanged RGB retained only inside the target or weed mask, with other pixels set to zero. Gold denotes local target regions or pooled external weeds; blue denotes external crop. Local masks delineate coarse regions, not exact plant crowns; white field frames belong to the original imagery. Native 512 × 512 local and 600 × 600 external footprints are shown without a common ground scale. These are supplied annotations, not model predictions. External RGB uses a fixed uint16/65535 conversion without contrast enhancement. WeedsGalore imagery and annotations: Celikkan et al. (2025), CC BY 4.0, shown as RGB composites and transformed annotation views. Further examples and selection details appear in Figure S14 and Supplement S11.')
    caption_supp=('Additional training imagery and supplied annotation views. Local rows show the window nearest the 10th percentile of positive training annotation coverage and the zero-annotation window with greatest valid fraction, with ties broken by ascending identifier. Zero annotation does not establish biological absence; the all-black target-only panel follows directly from the supplied mask. The external row is the first crop-and-weed-positive training frame from a different date than the main example, following official list order. Columns and colours follow Figure 2: RGB, supplied mask, 40% opacity overlay and target-only RGB. Local RGB is unchanged uint8; external RGB uses fixed uint16/65535 conversion. Local masks represent coarse regions. Native pixel alignment and orientation are retained; panels do not share a ground scale. No model prediction or performance-based selection is shown. WeedsGalore imagery and annotations: Celikkan et al. (2025), CC BY 4.0; RGB composite and transformed label views.')
    oldmain=(OLD/'PA_manuscript.md').read_text(encoding='utf-8')
    main=re.sub(r'Figure ([234])(?!\d)',lambda m:'Figure '+str(int(m.group(1))+1),oldmain)
    main_figure=figures['dataset_examples_main']
    insertion=('Figure 2 shows training imagery with the supplied masks, annotation overlays and masked RGB. These illustrations describe the annotation support and are not predictions or evidence of segmentation accuracy. Sparse and zero-annotation examples are retained in Figure S14; Supplement S11 gives the deterministic selection and display rules.\n\n'
               +'![Figure 2]('+main_figure['formats']['png']['path'].replace('\\','/')+')\n\n**Figure 2.** '+caption_main+'\n\n')
    anchor='**Table 1.** Retained model populations for confirmation.'
    assert main.count(anchor)==1;main=main.replace(anchor,insertion+anchor)
    oldright='Permission for public redistribution has not been established, and these materials are not included in the internal reproducibility package.'
    newright='Permission for public redistribution of the local dataset has not been established. The internal review materials include the selected training-image excerpts and annotation visualizations in Figure 2 and Supplementary Figure S14, but exclude the full raw dataset.'
    assert oldright in main;main=main.replace(oldright,newright)
    main=main.replace('It excludes raw imagery, array caches and weight binaries.','It excludes complete raw-image collections, array caches and weight binaries.')
    text(NEW/'PA_manuscript.md',main)
    supp=(OLD/'PA_supplementary.md').read_text(encoding='utf-8')
    addition='''## S11 Dataset illustration selection and display

Figure 2 and Figure S14 illustrate training data and provided annotations only. They were added after the completed confirmation analysis for dataset description, without fitting a model, changing a score or selecting examples from prediction performance. Four local examples were chosen from the accepted training manifest: the nearest observed positive annotation fractions to the 0.10, 0.50 and 0.90 linear-interpolated quantiles, with ascending ASCII identifiers breaking ties, plus the zero-annotation window with the greatest valid fraction and then the smallest identifier. The chosen identifiers are 4_6_17_17, 4_6_28_15, 4_5_29_28 and 4_5_32_26, respectively. All four have complete valid support. Parent RGB windows and masks retain their native 512 × 512 orientation; mask values greater than zero define target regions.

External examples follow the official training-list order: the first frame containing both crop and weed annotations, 2023-06-15_0472, and the first such frame from another date, 2023-06-06_0197. Native 600 × 600 labels retain class 0 as background, class 1 as crop and classes greater than 1 as pooled weeds. RGB display values are round(255 × float32(uint16)/65535); no sample-specific contrast stretch or gamma adjustment is applied. Local uint8 RGB is unchanged.

The same four views are shown for every example. The annotation view uses gold for local target regions or external pooled weeds, blue for external crop and dark grey for background. The overlay mixes annotation colour at 40% opacity on labelled foreground pixels and leaves background RGB unchanged. The target-only view retains original RGB within the local target mask or external weed mask and sets all other RGB pixels to zero. Native geometry and orientation remain unchanged; no new labels are inferred. A zero-annotation window may still contain vegetation, and the local coarse regions should not be read as complete plant-level masks. The native windows and external frames have different ground supports, so their equally sized panels do not imply a common ground scale. The internal revision materials retain the exact example identifiers, source identities, class counts and rendering code. Dataset rights and author-review requirements continue to apply.

'''
    assert supp.count('## Supplementary figure and table guide')==1
    supp=supp.replace('## Supplementary figure and table guide',addition+'## Supplementary figure and table guide')
    supp=supp.replace('Raw imagery, annotation/cache arrays and weight binaries are excluded','Complete raw-image collections, annotation/cache arrays and weight binaries are excluded')
    supp += '\n- Figure S14: additional training RGB, annotation masks, overlays and target-only views.\n'
    text(NEW/'PA_supplementary.md',supp)
    fs=read(OLD/'main_figures.json');fs['manuscript']=pin(NEW/'PA_manuscript.md')
    for f in fs['figures']:
        if f['number']>=2:f['number']+=1
    fs['figures'].insert(1,{'number':2,'name':'dataset_training_examples','caption':caption_main,'formats':main_figure['formats']})
    dump(NEW/'main_figures.json',fs)
    ss=read(OLD/'supplement_figures.json');ss['manuscript']=pin(NEW/'PA_supplementary.md')
    ss['figures'].append({'number':'S14','name':'additional_training_examples','caption':caption_supp,'formats':figures['dataset_examples_supplement']['formats']})
    dump(NEW/'supplement_figures.json',ss)
    es=read(OLD/'equations.json');es['manuscript']=pin(NEW/'PA_manuscript.md');dump(NEW/'equations.json',es)
    # The scientific result sections are byte-identical after reversing only figure numbering.
    start='## Results';end='## Funding'
    oldblock=oldmain.split(start,1)[1].split(end,1)[0]
    newblock=main.split(start,1)[1].split(end,1)[0]
    inverse=re.sub(r'Figure ([345])(?!\d)',lambda m:'Figure '+str(int(m.group(1))-1),newblock)
    assert inverse==oldblock
    dump(NEW/'integration_record.json',{'created_at':dt.datetime.now().astimezone().isoformat(),
        'scope':'dataset illustration addition only','source_manuscript':pin(OLD/'PA_manuscript.md'),
        'source_supplement':pin(OLD/'PA_supplementary.md'),'main':pin(NEW/'PA_manuscript.md'),
        'supplement':pin(NEW/'PA_supplementary.md'),'figure_rendering':pin(NEW/'figures_v2/render_manifest.json'),
        'added_main_figure':2,'added_supplement_figure':'S14','main_figures':5,'supplement_figures':14,
        'results_discussion_conclusions_unchanged_after_figure_renumbering':True,
        'new_training':False,'new_inference':False,'changed_scientific_results':False,
        'historical_v5_preserved':True,'layout_revision':'First 162-mm-high illustration retained in figures; 174-mm v2 gives row labels more space.',
        'publication_authorized':False,'author_facts_still_pending':True})
    print(json.dumps({'main':pin(NEW/'PA_manuscript.md'),'supplement':pin(NEW/'PA_supplementary.md')},ensure_ascii=False))

if __name__=='__main__':main()
