"""Assemble manuscript presentation from the accepted comparison and reviewed prose."""
from pathlib import Path
import re,json,hashlib,datetime
ROOT=Path(__file__).resolve().parent;PA=ROOT/'precision_agriculture';OUT=PA/'final_v5'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return Path(p).read_text(encoding='utf-8-sig')
def dump(p,j):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2)
def main():
 report=ROOT/'provenance/completed_confirmation_report_independent_20260916_142830_724677/review.json'
 assert sha(report)=='3af89351cf4a813bae16c6c6ba82e7334e05be4aa048a49707ad4f521f57bed8'
 inputs=[PA/x for x in ['PA_final_integration_base_v5.md','PA_abstract_v5_working.md','PA_results_and_discussion_v5_working.md','PA_main_tables_confirmation_v5.md','PA_main_tables_confirmation_v5.json','PA_supplementary_v5_working.md','FINAL_REFERENCES_VERIFIED.md','FINAL_DECLARATIONS_AND_AVAILABILITY_DRAFT.md']]
 original={str(p):sha(p) for p in inputs}
 abstract=read(inputs[1]).replace('the complete model–information population','24 local segmentation models across four information conditions').replace('Changing R1 to R4 increased','Changing from undegraded inputs (R1) to fourfold degradation (R4) increased')
 assert 150<=len(abstract.split())<=250
 text=read(inputs[0]).replace('[[ABSTRACT_FROM_ACCEPTED_RESULTS]]',abstract.strip())
 text=text.replace('| Model population | Retained fits | Spatial maps |','**Table 1.** Retained model populations for confirmation.\n\n| Model population | Retained fits | Spatial maps |')
 text=text.replace('The ranking score is','The ranking score (Eq. 1) is').replace('a selected set S satisfies','a selected set S satisfies the area cap in Eq. 2:').replace('Annotated-target capture is','Annotated-target capture (Eq. 3) is').replace('Crop capture and crop composition are','Crop capture and crop composition (Eq. 4) are').replace('normalized AUC is','normalized AUC (Eq. 5) is')
 text=text.replace('is\n\n$$','is\n\n$$').replace('2:\n\n$$','2:\n\n$$')
 results=read(inputs[2]).replace('two contained no crop annotation','two had zero annotated crop area').replace('Local 10% allocations concentrated','Local 10% allocations at 2 m units concentrated')
 tables=read(inputs[3]);tableblocks={}
 for n in range(2,6):
  block=re.search(r'## Table '+str(n)+r'\n(.*?)(?=\n## |\Z)',tables,re.S).group(1).strip()
  block=block.replace('Table '+str(n)+'.','**Table '+str(n)+'.**',1).replace('Two external crop-free frames','Two external frames with zero annotated crop area').replace('P-policy','Predicted area')
  block=block.replace('the companion JSON','the supplementary main-table source file')
  tableblocks[n]=block;results=results.replace('[[TABLE_'+str(n)+']]',block)
 figdir=PA/'confirmation_publication_figures_20260916_142928_858120';spec=json.loads(read(figdir/'figure_specs.json'))
 byname={f['name']:f for f in spec['figures']};caps={m.group(1):m.group(2).strip() for m in re.finditer(r'### ([^\n]+)\n\n(.*?)(?=\n### |\Z)',read(figdir/'CAPTIONS.md'),re.S)}
 concept=PA/'allocation_framework_20260916_103409_851025';spatial=PA/'confirmation_spatial_example_20260916_142955_540018'
 conceptcap=read(PA/'framework_confirmation_qa_20260916_123648_022081/caption_for_final_manuscript.txt').strip();conceptcap=re.sub(r'^\*?\*?Figure 1\.\*?\*?\s*','',conceptcap)
 mainfigs=[{'number':1,'name':'allocation_framework','caption':conceptcap,'formats':{e:pin(concept/('allocation_framework.'+e)) for e in ['png','pdf','svg']}},
  {'number':2,'name':'confirmation_local_M2_prefix','caption':'Local capture changes with the available area and information condition. '+caps['confirmation_local_M2_prefix'],'formats':byname['confirmation_local_M2_prefix']['files']},
  {'number':3,'name':'fixed_spatial_example','caption':read(spatial/'caption.md').strip(),'formats':{e:pin(spatial/('spatial_example.'+e)) for e in ['png','pdf','svg']}},
  {'number':4,'name':'confirmation_weeds_g8_nominal_primary','caption':'Weed capture and crop inclusion have distinct responses to image information. '+caps['confirmation_weeds_g8_nominal_primary'],'formats':byname['confirmation_weeds_g8_nominal_primary']['files']}]
 def markup(f):return '![Figure '+str(f['number'])+']('+Path(f['formats']['png']['path']).as_posix()+')\n\n**Figure '+str(f['number'])+'.** '+f['caption']
 text=text.replace('## Materials and methods','Figure 1 summarizes the prediction, selection and reference-scoring sequence.\n\n'+markup(mainfigs[0])+'\n\n## Materials and methods',1)
 for f in mainfigs[1:]:results=results.replace('[[FIGURE_'+str(f['number'])+']]',markup(f))
 text+='\n'+results.strip()+'\n\n'
 declarations=read(inputs[7])
 funding=declarations.split('### Funding\n',1)[1].strip().split('\n\n',1)[0]
 text+='## Funding\n\n'+funding.strip()+'\n\n'
 data=declarations.split('## 1. Data availability\n',1)[1].split('\n\n**Editorial boundary',1)[0].strip()
 text+='## Data and code availability\n\n'+data+'\n\nThe analysis code, configurations, frozen protocols, comparison tables and computational review records accompany the internal review package. It excludes raw imagery, array caches and weight binaries. No public study repository or licence for newly written code has been assigned. Third-party data, code and model weights retain their original terms; Supplement S9 and the package access guide describe these distinctions.\n\n'
 ai=declarations.split('## 4. Generative AI and reproducibility statement\n',1)[1].split('\n\n**For final integration',1)[0].strip()
 text+='## Generative AI disclosure\n\n'+ai+'\n\n## Author declarations for review\n\nIndividual CRediT roles, competing-interest declarations, the role of funders, final author approval and authority for local-data access have not yet been supplied. These factual declarations require confirmation before submission. The supplied author order, equal contribution, correspondence and three funding sources are preserved.\n\n'
 refs=read(inputs[6]).split('## Reference list\n',1)[1].split('\n## ',1)[0].strip().replace('effect?.','effect?')
 text+='## References\n\n'+refs+'\n'
 assert '[[' not in text and len(re.findall(r'\\tag\{\d\}',text))==5
 assert len(re.findall(r'^\|\s*:?-{3}',text,re.M))==5
 main=OUT/'PA_manuscript.md';main.write_text(text,encoding='utf-8')
 supplement=read(inputs[5]);suppfig=[]
 for f in spec['figures']:
  if f['name'] in {mainfigs[1]['name'],mainfigs[3]['name']}:continue
  suppfig.append({'number':'S'+str(len(suppfig)+1),'name':f['name'],'caption':caps[f['name']],'formats':f['files']})
 assert len(suppfig)==13
 supplement+='\n## Supplementary figure and table guide\n\nFigures S1–S13 retain the alternate local policies and unit sizes, all scalar model conditions, the external matched-actual comparison and alternate grids. The complete report tables retain all frame, date, pooled, seed, paired-contrast, AUC and reference-policy values. The 76 comparison-family tables and separate primary/reference files are indexed in the report summary and package manifest. Local and external units remain separate, and undefined ratios retain their recorded reasons.\n\n'
 for f in suppfig:supplement+='- Figure '+f['number']+': '+f['name'].replace('confirmation_','').replace('_',' ')+'.\n'
 supp=OUT/'PA_supplementary.md';supp.write_text(supplement,encoding='utf-8')
 dump(OUT/'main_figures.json',{'manuscript':pin(main),'figures':mainfigs})
 dump(OUT/'supplement_figures.json',{'manuscript':pin(supp),'figures':suppfig})
 eq=json.loads(read(PA/'PA_manuscript_integrated_development_v3_1_equation_specs.json'));eq['manuscript']=pin(main);dump(OUT/'equations.json',eq)
 for p,h in original.items():assert sha(p)==h
 dump(OUT/'integration_record.json',{'created_at':datetime.datetime.now().astimezone().isoformat(),'source_pins':original,'accepted_report':pin(report),'main':pin(main),'supplement':pin(supp),'abstract_words':len(abstract.split()),'main_tables':5,'main_figures':4,'supplement_figures':13,'equations':5,'references':16,'minor_clarifications':['Abstract identifies local24 segmentation population and R1/R4 meaning','Residual-area discussion explicitly uses M2','Zero annotated crop area does not mean missing labels or verified biological absence'],'scientific_outputs_changed':False,'final_content_and_word_QA':'pending'})
 print(json.dumps({'main':pin(main),'supplement':pin(supp),'abstract_words':len(abstract.split()),'main_words':len(text.split())},ensure_ascii=False))
if __name__=='__main__':main()
