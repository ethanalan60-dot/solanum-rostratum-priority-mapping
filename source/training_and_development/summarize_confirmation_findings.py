"""Traceable writing aid from an actually accepted confirmation report; stdlib only.

No outcome-stage, image, tensor, model, solver, or development-result access.
The complete accepted CSVs remain authoritative. This tool does not select results.
"""
from pathlib import Path
import argparse, collections, csv, datetime, hashlib, itertools, json, math, shutil, traceback

ROOT=Path(__file__).resolve().parent
REVIEWER_SHA='106cde94fca04c2ac612bb7ec55bf8dad1b2eb615a87149972658b3025894381'
RS=(1,2,4,6);BS=(.1,.2,.3,.5,1.)
SM=('mae_pp','bias_pp','mae_zero_pp','mae_positive_pp','balanced_mae_pp')
LM=('capture_pct','captured_target_m2','actual_area_m2','unused_budget_m2','missed_target_m2','selected_non_target_area_m2')
EM=('weed_capture_pct','crop_capture_pct','crop_fraction_pct','selected_area_FAE','selected_weed_FAE','selected_crop_FAE','unselected_weed_FAE','selected_background_FAE','unused_effective_budget_area_FAE')
AM=('auc_0_1','auc_0_0p5_normalized')
FLAGS=('fallback_used','bound_anomaly','updated_proven_flag','certification_downgraded')
IDENTIFIERS=('dataset','run_id','encoder','variant','lr','seed','seeds','M','R','budget','policy','scope','scope_id','frame_id','acquisition_date','grid_g','budget_basis','parent_nominal_budget_code','contrast')
TIE_PP=1e-9

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def number(value):
    if value is None or value=='':return None
    x=float(value);require(math.isfinite(x),'Nonfinite saved number');return x
def families():
    result=[('local','scalar',None,SM),('weeds','scalar',None,SM),('local','spatial_budget','M',LM),('local','prefix_auc','M',AM)]
    result += [('weeds',scope+'_'+kind,'grid_g',EM if kind=='spatial_budget' else AM) for scope in ('frame','date','pooled') for kind in ('spatial_budget','prefix_auc')]
    return result
def suffixes(axis):
    return ('all','hard_soft_pairs','seed_mean_sample_sd','hard_soft_seed_mean_sample_sd','R_contrasts','R_contrast_seed_mean_sample_sd')+((axis+'_contrasts',axis+'_contrast_seed_mean_sample_sd') if axis else ())
def report_names():
    names={'status.json','input_seal.json','begin_binding.json','end_binding.json','summary.json','model_metadata.json','saved_curve_reference_index.json'}
    for ds,stem,axis,_ in families():names.update(f'{ds}/{stem}_{s}.csv' for s in suffixes(axis))
    names.add('local/primary_M2_budget10_prefix.csv')
    names.update('weeds/primary_g8_pooled_'+b+'.csv' for b in ('nominal_primary','matched_actual_secondary'))
    names.update(ds+'/constant_scalar_baselines.csv' for ds in ('local','weeds'))
    names.add('local/complete_prefix_curves.csv')
    names.update('local/shared_'+n+'.json' for n in ('label_prefix_budgets','label_MILP_budgets','label_prefix_curves','label_prefix_auc','random_budgets','random_auc','random_summary','summary','reference_seal'))
    names.update(f'weeds/{a}_{b}_{c}.csv' for a,b,c in itertools.product(('learned','label_oracle','random_expectation'),('frame','date','pooled'),('complete_prefix_curves','budget_metrics','auc')) if a!='learned' or c=='complete_prefix_curves')
    require(len(names)==119,'Internal report schema differs');return names

class Reader:
    def __init__(self):self.watched={};self.folder=None;self.files={};self.inventory=None
    def pin(self,path,expected=None):
        p=Path(path).resolve();require(p.suffix.lower() in ('.py','.json','.csv'),'Only source/metadata/CSV permitted')
        h=sha(p);require(expected is None or h==expected,'SHA mismatch: '+str(p))
        require(str(p) not in self.watched or self.watched[str(p)]==h,'Beginning source drift')
        self.watched[str(p)]=h;return h
    def json(self,path,expected=None):
        self.pin(path,expected);return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    def path(self,name):
        require(name in self.files,'Not in exact accepted report inventory');p=(self.folder/name).resolve()
        require(p.is_relative_to(self.folder),'Escaped report');return p
    def csv(self,name):
        p=self.path(name);require(p.suffix=='.csv','CSV only');self.pin(p,self.files[name])
        with p.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);require(reader.fieldnames and len(reader.fieldnames)==len(set(reader.fieldnames)),'Malformed columns')
            rows=list(reader);require(all(None not in row for row in rows),'Malformed rows');return rows
    def finish(self):
        for path,h in self.watched.items():require(sha(path)==h,'Ending source drift: '+path)
        if self.folder:
            require({p.relative_to(self.folder).as_posix() for p in self.folder.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}==self.inventory,'Ending inventory drift')

def gate(folder,review_path,review_sha,reader):
    reader.pin(__file__)
    folder=Path(folder).resolve();require(folder==(ROOT/'confirmation_v1/stages/report_tables').resolve(),'Only canonical report stage allowed')
    reader.folder=folder
    # Pin every identity before parsing the corresponding objects.
    reviewer=ROOT/'audit_completed_confirmation_report.py';reader.pin(reviewer,REVIEWER_SHA)
    review_path=Path(review_path).resolve();require(review_path.is_relative_to(ROOT/'provenance'),'Actual independent receipt must be under campaign provenance')
    reader.pin(review_path,review_sha);reader.pin(folder/'artifact_manifest.json');reader.pin(folder/'summary.json');reader.pin(folder/'status.json')
    freeze=ROOT/'confirmation_v1/FREEZE.json';reader.pin(freeze)
    review=reader.json(review_path,review_sha)
    require(review.get('passed') is True and review.get('scope')=='confirmation_report_tables_independent' and review.get('stage')=='report_tables','Actual independent report acceptance missing')
    require(Path(review['output_directory']).resolve()==folder,'Acceptance output differs')
    require(review['reviewer_sha256']==REVIEWER_SHA and review['numeric_comparisons']>0,'Wrong or synthetic-only auditor receipt')
    require(review['audited_artifact_manifest_sha256']==reader.watched[str(folder/'artifact_manifest.json')] and review['summary_sha256']==reader.watched[str(folder/'summary.json')],'Accepted report bytes differ')
    require(review['freeze_sha256']==reader.watched[str(freeze)],'Freeze differs')
    f=reader.json(freeze);require(f.get('passed') is True and f.get('frozen') is True and f.get('scope')=='confirmation_v1','Formal freeze missing')
    manifest=reader.json(folder/'artifact_manifest.json');reader.files=manifest['files'];reader.inventory=report_names()
    require(set(reader.files)==reader.inventory,'Exact 119-file report required')
    require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}==reader.inventory,'Report file inventory differs')
    for name,h in reader.files.items():reader.pin(reader.path(name),h)
    require(reader.json(folder/'status.json')['state']=='completed','Report incomplete')
    summary=reader.json(folder/'summary.json')
    require(summary['R']==list(RS) and summary['models']=={'local':28,'weeds':9} and summary['local_spatial_models']==24 and summary['external_spatial_models']==6,'Fixed population differs')
    require(summary['datasets_pooled'] is False and summary['table_rows']==review['table_rows'],'Different table population')
    require(summary['primary_local']=={'M':2,'budget':.1,'policy':'prefix'},'Local primary differs')
    for name,h in reader.watched.items():
        if name.startswith(str(folder)):require(review['source_and_output_hashes'].get(name)==h,'Actual reviewer did not bind consumed report identity')
    reader.finish();return summary,review

def base_metric(column):
    c=column
    for suffix in ('_defined_models','_sample_sd','_mean'):
        if c.endswith(suffix):c=c[:-len(suffix)];break
    for prefix in ('soft_minus_hard_','difference_','comparison_','reference_','hard_','soft_'):
        if c.startswith(prefix):return c[len(prefix):]
    return c
def semantics(column):
    base=base_metric(column)
    if column.endswith('_defined_models'):return {'unit':'fitting-model count','denominator':'complete declared fitting-seed group; not fields'}
    den={'capture_pct':'domain_target_m2','weed_capture_pct':'total_weed_FAE within this scope','crop_capture_pct':'total_crop_FAE within this scope','crop_fraction_pct':'selected_area_FAE within this scope',
         'mae_pp':'n saved samples in this dataset','bias_pp':'n saved samples in this dataset','mae_zero_pp':'n_zero saved samples','mae_positive_pp':'n_positive saved samples','balanced_mae_pp':'equal average of zero and positive strata; NA if either is absent',
         'auc_0_1':'total target in this domain and complete area-fraction interval [0,1]','auc_0_0p5_normalized':'total target; complete [0,0.5] area-fraction interval normalized by 0.5'}
    aliases={'capture_fraction':'capture_pct','weed_capture_ratio':'weed_capture_pct','crop_capture_ratio':'crop_capture_pct','crop_fraction_of_selected_area':'crop_fraction_pct',
             'mae':'mae_pp','bias':'bias_pp','mae_zero':'mae_zero_pp','mae_positive':'mae_positive_pp','balanced_mae':'balanced_mae_pp','auc_0_half_normalized':'auc_0_0p5_normalized'}
    if base in aliases:den[base]=den[aliases[base]]
    unit='percentage points' if base.endswith(('_pct','_pp')) else 'm2' if base.endswith('_m2') else 'frame-area equivalents (FAE)' if base.endswith('_FAE') else 'fraction' if base.startswith('auc_') or base in ('budget','actual_area_fraction','effective_budget_fraction','parent_nominal_budget_fraction','actual_fraction') else 'count' if base in ('n','n_zero','n_positive','n_models','frame_n','k','N','independent_field_replicates') else 'dimensionless optimizer bound gap' if base=='updated_relative_gap' else 'flag' if base in FLAGS else 'metadata or exact saved source unit'
    if base in aliases:unit='fraction'
    if base in ('domain_area_m2','domain_target_m2'):unit='m2'
    result={'unit':unit,'denominator':den.get(base,'not a ratio' if unit in ('m2','frame-area equivalents (FAE)','count','flag') else 'see exact saved field; not interpreted by this summary')}
    if column.startswith(('difference_','soft_minus_hard_')):result['direction']='comparison minus reference' if column.startswith('difference_') else 'soft minus hard'
    if column.endswith('_sample_sd'):result['statistic']='saved sample SD across fitting seeds; n-1; not a confidence interval'
    if column.endswith('_mean'):result['statistic']='saved complete fitting-seed mean; any missing observation retains NA'
    return result
def matches(row,filters):
    for k,want in filters.items():
        require(k in row,'Unknown filter column')
        if isinstance(want,(float,int)):got=number(row[k])
        else:got=row[k]
        if got!=want:return False
    return True
def view(reader,name,rows,filters=None,embed=False):
    filters=filters or {};selected=[(i+2,r) for i,r in enumerate(rows) if matches(r,filters)]
    columns=list(rows[0]) if rows else []
    result={'csv':str(reader.path(name)),'sha256':reader.files[name],'filters':filters,'row_count':len(selected),'csv_line_numbers_include_header':True,
            'field_semantics':{k:semantics(k) for k in columns},'full_source_retained':True}
    if embed:
        # Exact text prevents accidental float conversion of IDs, aliases, flags or NA.
        result['rows']=[{'csv_line':i,'identity_filter':{k:r[k] for k in IDENTIFIERS if k in r},'saved_fields':r} for i,r in selected]
    return result
def common_fact(reader,name,rows,column,unit):
    vals=[number(r[column]) for r in rows];require(vals and all(v is not None for v in vals),'Missing domain/count field')
    require(all(math.isclose(v,vals[0],rel_tol=1e-10,abs_tol=1e-9) for v in vals),'Domain/count differs across fixed report rows')
    return {'value':vals[0],'unit':unit,'denominator':'not a ratio','source':{'csv':str(reader.path(name)),'sha256':reader.files[name],'filters':{},'column':column,'representative_csv_line':2,'all_rows_checked':len(rows)}}
def direction_counts(reader,name,rows,column,filters=None):
    filters=filters or {};chosen=[(i+2,r) for i,r in enumerate(rows) if matches(r,filters)]
    c={'positive':0,'negative':0,'numeric_tie':0,'NA':0}
    for _,r in chosen:
        x=number(r[column]);c['NA' if x is None else 'numeric_tie' if abs(x)<=TIE_PP else 'positive' if x>0 else 'negative']+=1
    return {'counts':c,'count_unit':'saved paired comparisons, not independent samples','numeric_tie_tolerance_pp':TIE_PP,'source':{'csv':str(reader.path(name)),'sha256':reader.files[name],'filters':filters,'column':column,'csv_lines':[i for i,_ in chosen]},'metric_semantics':semantics(column),'interpretation':'sign only; a positive error difference is worse, while a positive capture difference is higher capture'}
def flag_counts(reader,name,rows):
    result={}
    for field in FLAGS:
        if not rows or field not in rows[0]:result[field]={'status':'not present in this accepted table'};continue
        counts={'true':0,'false':0,'NA':0}
        for r in rows:
            v=r[field];require(v in ('','True','False','true','false'),'Unknown saved flag')
            counts['NA' if v=='' else v.lower()]+=1
        result[field]={'counts':counts,'unit':'saved model-budget policy cases, not independent samples','source':{'csv':str(reader.path(name)),'sha256':reader.files[name],'filters':{},'column':field}}
    return result

def build(reader,summary):
    cached={};catalog=[];all_views=[]
    def table(name):
        if name not in cached:
            cached[name]=reader.csv(name)
            if name in summary['table_rows']:require(len(cached[name])==summary['table_rows'][name],'Saved row count differs')
        return cached[name]
    # Every family has a source view, including frame/date/pooled, adverse pairs and all contrasts.
    for ds,stem,axis,_ in families():
        for suffix in suffixes(axis):
            name=f'{ds}/{stem}_{suffix}.csv';all_views.append(view(reader,name,table(name)))
    for name,h in sorted(reader.files.items()):catalog.append({'relative_path':name,'sha256':h,'source_path':str(reader.path(name)),'handling':'parsed aggregate CSV' if name in cached else 'preserved by exact hash; not interpreted'})
    local_name='local/spatial_budget_all.csv';local=table(local_name)
    domain={c:common_fact(reader,local_name,local,c,'m2') for c in ('domain_area_m2','domain_target_m2')}
    strata={ds:{c:common_fact(reader,f'{ds}/scalar_all.csv',table(f'{ds}/scalar_all.csv'),c,'saved samples') for c in ('n','n_zero','n_positive')} for ds in ('local','weeds')}
    primary=[]
    # Include all five budgets, both policies, all four R and all 24 segmentation models.
    for suffix in ('all','hard_soft_pairs','seed_mean_sample_sd','hard_soft_seed_mean_sample_sd','R_contrasts','R_contrast_seed_mean_sample_sd'):
        name='local/spatial_budget_'+suffix+'.csv';primary.append(view(reader,name,table(name),{'M':2},True))
    for ds in ('local','weeds'):
        for suffix in ('all','hard_soft_pairs','seed_mean_sample_sd','hard_soft_seed_mean_sample_sd','R_contrasts','R_contrast_seed_mean_sample_sd'):
            name=f'{ds}/scalar_{suffix}.csv';primary.append(view(reader,name,table(name),embed=True))
        name=f'{ds}/constant_scalar_baselines.csv';primary.append(view(reader,name,table(name),embed=True))
    for scope in ('date','pooled'):
        for suffix in ('all','hard_soft_pairs','seed_mean_sample_sd','hard_soft_seed_mean_sample_sd','R_contrasts','R_contrast_seed_mean_sample_sd'):
            name=f'weeds/{scope}_spatial_budget_{suffix}.csv';primary.append(view(reader,name,table(name),{'grid_g':8},True))
    # AUC values and paired/seed summaries are copied, never integrated from sparse budgets.
    auc=[]
    for ds,stem in [('local','prefix_auc')]+[('weeds',s+'_prefix_auc') for s in ('date','pooled')]:
        for suffix in ('all','hard_soft_pairs','seed_mean_sample_sd','hard_soft_seed_mean_sample_sd'):
            name=f'{ds}/{stem}_{suffix}.csv';auc.append(view(reader,name,table(name),embed=True))
    signs=[]
    for suffix,column in [('R_contrasts','difference_capture_pct'),('M_contrasts','difference_capture_pct'),('hard_soft_pairs','soft_minus_hard_capture_pct')]:
        name=f'local/spatial_budget_{suffix}.csv';rr=table(name)
        # All budgets, plus each declared 10/20/... budget by policy/contrast/R-or-M.
        signs.append(direction_counts(reader,name,rr,column))
        keys=tuple(k for k in ('contrast','policy','budget','M','R') if k in rr[0])
        groups=sorted({tuple(r[k] for k in keys) for r in rr})
        for key in groups:signs.append(direction_counts(reader,name,rr,column,dict(zip(keys,key))))
    external_signs=[]
    for scope in ('date','pooled'):
        for suffix,prefix in [('hard_soft_pairs','soft_minus_hard_'),('R_contrasts','difference_'),('grid_g_contrasts','difference_')]:
            name=f'weeds/{scope}_spatial_budget_{suffix}.csv';rr=table(name)
            keys=tuple(k for k in ('scope_id','budget_basis','parent_nominal_budget_code','contrast','grid_g','R') if k in rr[0])
            for key in sorted({tuple(r[k] for k in keys) for r in rr}):
                for metric in ('weed_capture_pct','crop_capture_pct','crop_fraction_pct'):
                    external_signs.append(direction_counts(reader,name,rr,prefix+metric,dict(zip(keys,key))))
    for entry in catalog:
        entry['handling']='parsed aggregate CSV' if entry['relative_path'] in cached else 'preserved by exact hash; not interpreted'
    return {'scope':'writing_aid_from_accepted_confirmation_report','primary_local_definition':summary['primary_local'],'primary_external_definition':summary['primary_external'],
        'local_domain':domain,'scalar_strata':strata,'primary_and_scalar_values':primary,'complete_prefix_auc_values':auc,'complete_family_views':all_views,
        'local_direction_counts':signs,'external_direction_counts':external_signs,'local_policy_flags':flag_counts(reader,local_name,local),'accepted_report_catalog':catalog,
        'NA_representation':'Exact CSV blanks retained in saved_fields; derived count bins use NA. No missing ratios or sample SD filled with zero.',
        'reporting_rules':['No pooling between local and weeds, or between external date and pooled scopes.','All means and n-1 sample SD are taken from accepted CSVs; no new confidence interval or significance calculation.','The original local regressors have seed17 only and NA sample SD.','Fitting seeds, frames, dates, budgets and cells are not independent field samples.','R contrasts are comparison minus reference: 4-1, 2-1, 4-2, 6-1, 6-4; M contrasts are 4-1, 2-1, 4-2.','External grid counts g increase toward finer cells; grid_g16-minus-grid_g4 is not a coarser-minus-finer metric.','Nominal and matched allocation rows/aliases remain separate; actual frame-area budgets are preserved.','Whole complete curves and reference files are catalogued unchanged, without sparse interpolation.'],
        'not_summarized':['Historical execution attempts and failed runs are not reconstructed from development outputs; the accepted report is not a full execution-attempt ledger.','Shared label/random reference JSON and raw solver return details are only catalogued, not numerically re-audited by this writing aid.','Frame-level full tables are source-indexed but not duplicated in the short narrative; all date and pooled primary values remain separate.','A field sampling independence claim is not derived from model/seed/frame counts.']}

def markdown(findings):
    lines=['# Confirmation findings: source-backed writing aid','','This is a report-table index and exact-value extract, not a new experiment or independent reanalysis.','',
           '## Local domain and scalar denominators','']
    for key,item in findings['local_domain'].items():
        src=item['source'];lines.append(f"- {key}: {item['value']:.12g} {item['unit']} — `{Path(src['csv']).name}`, column `{src['column']}`, all {src['all_rows_checked']} rows consistent.")
    for ds,items in findings['scalar_strata'].items():
        lines.append(f"- {ds}: "+', '.join(f"{k}={v['value']:.0f}" for k,v in items.items())+' saved samples. Exact source fields are in JSON.')
    lines += ['', '## Fixed reporting coverage','',
        '- Local primary: M=2 m, 10% prefix; its full context includes all five budgets, both policies, all four R values and all 24 segmentation models. Values and fitting-seed summaries are copied, with adverse signs and NA retained.',
        '- Ordinary and balanced CPU MAE remain distinct. Zero and positive strata, the single-seed local regressors, and all three-seed segmentation/external summaries remain explicit.',
        '- All R and M contrasts are indexed; direction counts preserve every budget and highlight none by performance. Exact source lines and filters are attached to every count.',
        '- External g=8 primary values are separated by date and pooled scope, nominal and matched budgets. Full g=4/16 sensitivity tables and frame-level tables remain linked. Weed capture, crop capture and crop composition have different denominators.',
        '- AUC comes from accepted complete-curve AUC tables, never sparse budget interpolation. Actual allocations, unused areas, missed targets, flags and duplicate-selection aliases stay in source rows.',
        '', '## Limits','']
    lines += ['- '+s for s in findings['not_summarized']]
    lines += ['', 'Use findings.json for each CSV/hash/filter/line/field/unit/denominator. Sample SD describes fitting seeds; no confidence intervals or independent-field sample sizes are added.','']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-directory',type=Path,required=True);p.add_argument('--independent-review',type=Path,required=True);p.add_argument('--independent-review-sha256',required=True);p.add_argument('--execute',action='store_true',required=True);a=p.parse_args()
    out=ROOT/'precision_agriculture'/('confirmation_findings_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));reader=Reader();reader.pin(__file__)
    primary_error=None
    try:
        out.mkdir();shutil.copy2(__file__,out/'source.py');require(sha(out/'source.py')==reader.watched[str(Path(__file__).resolve())],'Source changed before snapshot')
        summary,review=gate(a.report_directory,a.independent_review,a.independent_review_sha256,reader)
        findings=build(reader,summary);reader.finish()
        findings['acceptance_binding']={'report_directory':str(reader.folder),'review_path':str(a.independent_review.resolve()),'review_sha256':a.independent_review_sha256,'freeze_sha256':review['freeze_sha256'],'report_artifact_manifest_sha256':review['audited_artifact_manifest_sha256']}
        findings['source_bindings']=reader.watched
        dump(out/'findings.json',findings);(out/'findings.md').write_text(markdown(findings),encoding='utf-8');reader.finish()
        dump(out/'status.json',{'state':'completed','scope':'accepted_report_writing_aid','implementation_sha256':reader.watched[str(Path(__file__).resolve())],'model_solver_calls':0,'raw_arrays_read':0,'source_stages_read':False})
    except BaseException as e:
        primary_error=e
        try:
            out.mkdir(exist_ok=True);dump(out/'failure.json',{'state':'failed','error':repr(e),'traceback':traceback.format_exc(),'watched_sha256':reader.watched,'no_automatic_retry':True})
        except BaseException:pass
        raise
    finally:
        try:
            if out.exists():dump(out/'artifact_manifest.json',{'files_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='artifact_manifest.json'}})
        except BaseException:
            if primary_error is None:raise
    print(str(out))
if __name__=='__main__':main()
