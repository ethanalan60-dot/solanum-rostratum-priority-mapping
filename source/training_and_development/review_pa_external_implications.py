"""Read-only arithmetic and evidence check for two development manuscript drafts."""
from pathlib import Path
import collections,csv,datetime,hashlib,json,re,shutil,traceback

ROOT=Path(__file__).resolve().parent
PA=ROOT/'precision_agriculture'
WATCH={}
def sha(p):
    p=Path(p).resolve();s=hashlib.sha256(p.read_bytes()).hexdigest();WATCH[str(p)]=s;return s
def read(p):sha(p);return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def text(p):sha(p);return Path(p).read_text(encoding='utf-8-sig')
def table(p):
    sha(p)
    with Path(p).open(encoding='utf-8-sig',newline='') as s:return list(csv.DictReader(s))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    out=ROOT/'provenance'/datetime.datetime.now().strftime('PA_external_implications_review_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True);shutil.copy2(__file__,out/Path(__file__).name)
    try:
        external=PA/'PA_external_training_draft_v1.md';agricultural=PA/'PA_agricultural_implications_draft_v1.md'
        drafts={str(p):sha(p) for p in (external,agricultural)}
        exttext=text(external);agrtext=text(agricultural)
        assert sha(external)=='8bac905fab11f602ba0d9b10318ef368bb3f1bedd33ea123ee56227f0ece6780'
        assert sha(agricultural)=='6589729d1b88894de153dee024984dafd115d25f3f487ea5d3ca2b04ee18f0e8'
        assert 'fractions were ≥ 0.5 and zero otherwise' in exttext
        assert 'including its declared feasibility checks and prefix fallback' in agrtext
        assert 'not all certified as globally optimal' in agrtext
        for p in (external,agricultural):
            evidence=read(p.with_name(p.stem+'_evidence.json'))
            assert evidence['draft_sha256']==sha(p)
            for rel,digest in evidence['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
        training=read(ROOT/'external_weeds_training_reports/20260916_062524_429008/audit.json')
        assert training['passed'] and training['run_count']==9
        stats=training['variant_seed_summaries'];runs=training['runs'];pairs=training['paired_seed_results']
        numeric=[]
        for row in stats:
            for k in ('mean_mae_pp','sample_sd_mae_pp'):
                formatted=f'{row[k]:.4f}';assert formatted in exttext
                numeric.append({'claim':row['variant']+' '+k,'actual':row[k],'display':formatted})
        lookup={(x['variant'],x['seed']):x for x in runs}
        for pair in pairs:
            s=pair['seed'];line='| '+str(s)+' | '+' | '.join(f"{lookup[v,s]['mae_pp']:.4f}" for v in ('regression','hard_seg','soft_seg'))+f" | {pair['soft_minus_hard_mae_pp']:+.4f} |"
            assert line in exttext
            numeric.append({'claim':'external full table seed '+str(s),'actual':[lookup[v,s]['mae_pp'] for v in ('regression','hard_seg','soft_seg')]+[pair['soft_minus_hard_mae_pp']],'display':line})
        assert all(x['actual_updates']==312 and x['attempted_updates']==312 and x['skipped_updates']==0 for x in runs)
        for name in ('zero','train_mean'):
            value=100*training['constant_baselines'][name]['mae'];assert f'{value:.4f}' in exttext
            assert all(x['mae_pp']<value for x in runs)
            numeric.append({'claim':name+' baseline pp','actual':value,'display':f'{value:.4f}'})
        train=read(ROOT/'weedsgalore_data_v1/train.json');val=read(ROOT/'weedsgalore_data_v1/val.json')
        assert len(train)==104 and len(val)==26 and all(x['coverage']>0 for x in train+val)
        dates=dict(collections.Counter(x['acquisition_date'] for x in val))
        assert dates=={'2023-05-25':8,'2023-05-30':8,'2023-06-06':8,'2023-06-15':2}
        cfg=read(ROOT/'external_weeds_v1/runs/weeds_v1_resnet18_hard_seg_L256_s17/config.json')
        assert cfg['lr']==1e-4 and cfg['epochs']==24 and cfg['batch_size']==4 and cfg['accumulation_steps']==2
        local_evidence=read(PA/'PA_results_full_common_draft_v1_evidence.json')
        for item in local_evidence['source_files'].values():assert sha(item['path'])==item['sha256']
        rows=table(local_evidence['source_files']['central720']['path']);assert len(rows)==720
        assert len({x['run_id'] for x in rows})==8
        for policy,expectedcapture,expectedarea in [('prefix','88.415–93.589','142.815–145.701'),('predicted_area','86.063–94.076','145.703–145.749')]:
            group=[x for x in rows if float(x['M'])==2 and float(x['budget'])==.1 and x['policy']==policy]
            assert len(group)==24 and {int(x['R']) for x in group}=={1,2,4}
            capture=[float(x['capture_pct']) for x in group];area=[float(x['actual_area_m2']) for x in group]
            c=f'{min(capture):.3f}–{max(capture):.3f}';a=f'{min(area):.3f}–{max(area):.3f}'
            assert c==expectedcapture and a==expectedarea and c in agrtext and a in agrtext
            numeric.extend([{'claim':policy+' M2 B.1 capture percent range','actual':[min(capture),max(capture)],'display':c},
                            {'claim':policy+' M2 B.1 selected area m2 range','actual':[min(area),max(area)],'display':a}])
        for enc,r,expected in [('resnet18',4,['64.295','82.533']),('convnextv2_tiny',1,['83.993','69.015'])]:
            group=[x for x in rows if x['encoder']==enc and float(x['lr'])==.0003 and int(x['M'])==4 and int(x['R'])==r and float(x['budget'])==.1 and x['policy']=='prefix']
            assert len(group)==2
            mapped={x['variant']:float(x['capture_pct']) for x in group}
            actual=[mapped[v] for v in ('hard_seg','soft_seg')]
            assert [f'{x:.3f}' for x in actual]==expected and all(x in agrtext for x in expected)
            numeric.append({'claim':enc+f' M4 R{r} B.1 prefix hard/soft capture','actual':actual,'display':expected})
        geom=read(local_evidence['source_files']['geometry']['path']);ha=geom['area_m2']/10000
        assert f'{ha:.3f}'=='0.146' and '0.146 ha' in agrtext
        numeric.append({'claim':'local observed domain ha','actual':ha,'display':'0.146 ha'})
        local_pairs=table(local_evidence['source_files']['training_pairs4']['path'])
        assert len(local_pairs)==4 and sum(float(x['soft_minus_hard_pp'])<0 for x in local_pairs)==1
        assert len(pairs)==3 and all(x['soft_minus_hard_mae_pp']>0 for x in pairs)
        spatial=read(ROOT/'external_weeds_v1/spatial_v1/protocol.json')
        assert spatial['status']=='frozen_development_protocol_before_spatial_evaluation'
        assert {x['grid_cells_per_axis'] for x in spatial['matched_actual_budget_plan']}=={4,8,16}
        anchors={x['budget_fraction'] for x in spatial['matched_actual_budget_plan']}
        assert anchors=={1/16,3/16,4/16,8/16,1}
        assert len(spatial['eligible_models'])==6 and {x['variant'] for x in spatial['eligible_models']}=={'hard_seg','soft_seg'}
        # Source citation is extracted exactly, not invented from a secondary database.
        readme=ROOT/'external_data/weedsgalore/official_source_20260916_043356/static/README.md'
        official=text(readme);match=re.search(r'(@InProceedings\{Celikkan_2025_WACV,[\s\S]+?\n\})',official)
        assert match
        (out/'WeedsGalore_official.bib').write_text(match.group(1)+'\n',encoding='utf-8')
        citation=('Celikkan, E., Kunzmann, T., Yeskaliyev, Y., Itzerott, S., Klein, N., & Herold, M. (2025). '
                  'WeedsGalore: A Multispectral and Multitemporal UAV-Based Dataset for Crop and Weed Segmentation in Agricultural Maize Fields. '
                  'In Proceedings of the Winter Conference on Applications of Computer Vision (WACV), pp. 4767–4777. February 2025.')
        (out/'citation_supplement.md').write_text('# Verified WeedsGalore citation supplement\n\n'+citation+'\n\n'
            '[Official fixed-commit README](https://github.com/GFZ/weedsgalore/blob/0ddf3a77bce282daf9d795de6f6c3d98e7fc9dcc/README.md) · '
            '[Official WACV paper link given by the authors](https://openaccess.thecvf.com/content/WACV2025/html/Celikkan_WeedsGalore_A_Multispectral_and_Multitemporal_UAV-Based_Dataset_for_Crop_and_WACV_2025_paper.html)\n\n'
            'Six authors, title, venue, February 2025 and pages are copied from the locally archived official README citation. '
            'DOI is not supplied in that entry and is not guessed. The literal BibTeX is preserved separately. '
            'Dataset CC BY 4.0 and repository Apache 2.0 are separate licenses, as stated in the same official README.\n',encoding='utf-8')
        notes=[{'severity':'resolved','file':str(external),'text':'Hard targets were defined by thresholding weed fractions at 0.5',
                'resolution':'Root explicitly changed to fractions ≥ 0.5 and zero otherwise; independently checked final text.'},
               {'severity':'resolved','file':str(agricultural),'text':'prediction-area optimization',
                'resolution':'Root added feasibility checks and prefix fallback, and stated outputs are not all certified globally optimal; independently checked final text.'},
               {'severity':'integration','file':str(external),'text':'Dataset citation must be integrated',
                'recommendation':'Insert Celikkan et al. (2025) using the official citation supplement before submission.'}]
        for path,digest in WATCH.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
        result={'passed':True,'status':'passed_after_two_clarity_edits_citation_integration_pending','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'draft_sha256':drafts,'blocking_issues':0,'numerical_discrepancies':0,'arithmetic_checks':numeric,
            'minor_notes':notes,'val_date_counts':dates,
            'completed_planned_boundary':'Nine training runs and scalar metrics completed; external spatial outcomes and normalization intervention outcomes described as pending, not evidence.',
            'agricultural_scope':'Decision-variable evaluation in the observed domain; no efficacy, chemical saving, crop injury, yield, cost or physical operating-resolution claim.',
            'interpretive_passes':['Original GPU selected-best table matches audit; CPU export does not reselect.',
                'Both directions and all eight local models/all three external seeds are retained.',
                'Nominal budget versus actual expenditure and whole-unit quantization are explicit.',
                'Repeated dates/seeds and unmatched field support are not treated as independent field replication.',
                'Matched-area controls and crop denominators match the frozen protocol.',
                'Prior established prediction-versus-decision principle is not claimed as a new theory.'],
            'official_readme_sha256':sha(readme),'source_code_sha256':sha(__file__),'source_sha256':WATCH,
            'modified_drafts':False,'new_experiment_or_inference':False,'test_accessed':False}
        write(out/'review.json',result)
        md=['# External training and agricultural implications: independent content review','',
            '**PASS after the two agreed clarity edits; zero numerical discrepancies or blocking overclaims.**','',
            'The exact external means/sample SDs, all nine errors, all three differences, both constants and update counts agree with the accepted audit. '
            'The local 2 m/10% ranges, actual areas, two opposite 4 m examples and 0.146 ha conversion were recomputed from the complete 720-row table and geometry. '
            'Only one of four local hard/soft pairs favors soft for stratified MAE; all three external seed pairs favor hard for ordinary MAE.','',
            'Completed training/scalar evaluation is distinguished from pending external spatial and intervention evidence. '
            'The agricultural discussion appropriately concerns what area is selected, missed and charged under whole-unit budgets. '
            'It does not infer herbicide savings, control efficacy, crop injury, yield, field-generalization or an operational flight/grid recommendation.','',
            '## Resolved edits and remaining integration','',
            '1. Root now states the hard-label condition explicitly as fraction ≥ 0.5; final text verified.',
            '2. Root now identifies feasibility checks and prefix fallback, and explicitly disclaims universal certification of global optimality; final text verified.',
            '3. Integrate the official Celikkan et al. (2025) reference from the new citation supplement.','',
            '## Reviewed draft hashes','']+[f'- `{p}`: `{h}`' for p,h in drafts.items()]
        (out/'review.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
        write(out/'artifact_manifest.json',{'passed':True,'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}})
        print(json.dumps({'passed':True,'directory':str(out),'drafts':drafts,'review_sha256':sha(out/'review.json')},ensure_ascii=False))
    except BaseException as e:
        write(out/'failure.json',{'passed':False,'error':repr(e),'traceback':traceback.format_exc(),'source_sha256':WATCH});raise

if __name__=='__main__':main()
