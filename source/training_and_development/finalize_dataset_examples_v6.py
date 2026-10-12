"""Deliver the presentation-only v6 addendum without changing frozen science."""
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PA = ROOT/'precision_agriculture'
NEW = PA/'final_v6_dataset_examples'
NOW = datetime.now().astimezone()
STAMP = NOW.strftime('%Y%m%d_%H%M%S_%f')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def pin(path):
    return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(path,data):
    assert not path.exists(), str(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

assert not (NEW/'delivery.json').exists(), 'Already finalized; do not duplicate.'
review_names = ['main_pages1_12_visual_QA.json','main_pages13_24_visual_QA.json',
                'supplement_pages1_24_visual_QA.json','delta_content_and_figure_review.json',
                'docx_asset_verification.json']
for name in review_names:
    assert read(NEW/name)['passed'],name

# Bind the exact already-reviewed assets and actual page PNGs.
assets=[]
for stem,prefix in [('PA_manuscript','main'),('PA_manuscript_supplement','supplement')]:
    docx=NEW/'word'/f'{stem}.docx'
    pdf=NEW/'word'/f'{stem}.pdf'
    receipt=read(NEW/'word'/f'{stem}.pdf.v6c_receipt.json')
    assert receipt['passed'] and receipt['source_unchanged']
    assert sha(docx)==receipt['source_sha256'] and sha(pdf)==receipt['pdf_sha256']
    idx=read(NEW/'word'/f'{prefix}_render_pages.json')
    assert idx['pdf_sha256']==sha(pdf) and idx['page_count']==24
    assert [p['page'] for p in idx['pages']]==list(range(1,25))
    for p in idx['pages']:
        assert sha(Path(p['path']))==p['sha256']
    assets.extend([pin(docx),pin(pdf)])

locked={
    'FREEZE.json':'24f8d1605c58df6bd6169b8434a451975abca0c3f8176e9c55c252abf8f3e732',
    'SOURCE_LOCK.json':'37edeb2586568cb46837f4259dc28a4477a64ed569661480cf07d35900fdf11f',
    'contract.json':'ce710241422ccbb54a000adceb4d400cfdbc4b04bd9d105b6d40f8298ffd77d9',
    'checkpoints.json':'5cc9c54b2c8ee4630404da0eebc0f5d0857145380d3ccde4d7c99fbe3f0ff4d5'}
for name,expected in locked.items():
    assert sha(ROOT/'confirmation_v1'/name)==expected,name
assert sha(PA/'author_metadata/authors_funding_current.json')=='3ce8863d8e7de61c3b69cffc2f717ac8f0ff74416b83c79e2cae778d73e2edd7'

acceptance={
    'passed':True,'created_at':NOW.isoformat(),'scope':'Presentation-only dataset illustration revision',
    'documents':assets,'main_pages_actually_viewed':24,'supplement_pages_actually_viewed':24,
    'reviews':[pin(NEW/n) for n in review_names],
    'main_figures':5,'supplement_figures':14,'new_figures':['2','S14'],
    'training_examples':6,'new_panels':24,'source_images_and_annotations':'Actual supplied training data',
    'model_prediction_images_added':False,'new_training':False,'new_inference':False,
    'scientific_results_unchanged':True,'frozen_contract_files_unchanged':locked,
    'author_metadata_unchanged':True,'submitted':False,
    'render_history':{
        'accepted_exporter':pin(PA/'render_with_word_v6c.ps1'),
        'failed_attempts_preserved':True,
        'zero_byte_main_render_success_json':'Failed v6b output placeholder; not an acceptance receipt.',
        'bundled_renderer':'Unavailable LibreOffice; failure log retained.',
        'first_word_helper':'Open argument conversion error; failure receipt retained.',
        'second_word_helper':'Timed out at an unconfirmed COM phase; cleanup receipts retained.',
        'final_word_helper':'Probe-document ownership, exact source export and cleanup passed for both documents.'}}
write(NEW/'visual_and_content_acceptance.json',acceptance)

base=ROOT/'internal_submission_review_bundles/20260916_151139_042531/internal_review_only.zip'
assert sha(base)=='66d7c5bd47c74c797bde3cf942f07c363857a7a0d83ef8150f516abfa11c1b4b'
base_reference={'role':'Immutable complete v5 scientific base; use together with this v6 presentation addendum',
                'zip':pin(base),'manifest':pin(base.parent/'package_manifest.json'),
                'v5_delivery':pin(ROOT/'provenance/final_delivery_v5_20260916_151243_890148/delivery.json'),
                'v6_addendum_is_complete_scientific_archive':False}
write(NEW/'base_package_reference.json',base_reference)
readme='''# 数据集图片修订版 v6

按用户要求加入真实训练样例：4 个本地刺萼龙葵样例、2 个 WeedsGalore 样例。
每例包括原始 RGB、数据集标注 mask、40% 透明度叠加图，以及 mask 后仅保留目标区域的 RGB，共 24 个面板。

- 主文 Figure 2 位于 PDF 第 6 页；补充 Figure S14 位于第 24 页。
- 新版主文与补充材料均为 24 页。主图共 5 张，补图共 14 张。
- 图像取自训练集，以已记录的覆盖率分位数和官方列表次序选取。未按模型表现挑选。
- 这些是原始标注示意，不是模型预测。局部粗多边形不表示精确植株轮廓；零标注不等于生物学上的无杂草。
- 本地 RGB 原样展示；外部 uint16 RGB 固定除以 65535 转为展示值。无逐图增强、合成或新推理。
- 新版保留已完成研究的数值、结论、作者及三项基金，旧版保留。

## 文件

`word/`：新版 Word/PDF；`figures_v2/`：600 dpi PNG、PDF、SVG 和 24 张独立面板；
`data_source_audit/`、`external_source_audit/`：所选训练样例及来源校验；
`visual_and_content_acceptance.json`：48 页实际视觉检查及内容/资产审核的汇总。

## 内部审阅增量包

`delivery_bundle/dataset_examples_v6_addendum.zip` 仅是图片修订增量包。
完整科学记录使用 `base_package_reference.json` 指向的已封存 v5 基础包，两者合用。
增量包包含所选训练图片及衍生展示图，不包含完整原始数据集或模型权重。
WeedsGalore 的 CC BY 4.0 来源署名已在图注及来源审核中保留；本地数据未获得新的公开再许可。
尚未投稿；既有作者声明及最终批准待确认状态不变。

旧导出失败日志保留；`word/main_render_success.json` 是失败尝试留下的空文件，
最终成功凭据为两份 `*.pdf.v6c_receipt.json`，不得混用。
'''
(NEW/'README.md').write_text(readme,encoding='utf-8')

# Assemble only this revision. The base scientific archive stays byte-identical.
bundle=NEW/'delivery_bundle'
bundle.mkdir(exist_ok=False)
selected={}
for p in NEW.rglob('*'):
    if not p.is_file(): continue
    rel=p.relative_to(NEW)
    if rel.parts[0] in ('delivery_bundle','figures'): continue
    if any(x in rel.parts for x in ('main_pages','supplement_pages','bundled_render','__pycache__')): continue
    selected['revision/'+rel.as_posix()]=p
for p in [ROOT/'plot_dataset_examples_v6.py',ROOT/'plot_dataset_examples_v6b.py',
          ROOT/'integrate_dataset_examples_v6.py',Path(__file__),
          PA/'build_final_manuscript_v5b.py',PA/'render_with_word_v6.ps1',
          PA/'render_with_word_v6b.ps1',PA/'render_with_word_v6c.ps1']:
    selected['code/'+p.name]=p
rows=[{'member':k,**pin(p)} for k,p in sorted(selected.items())]
manifest={'created_at':NOW.isoformat(),'scope':'v6 illustration addendum; not a complete scientific archive',
          'base_reference':base_reference,'files':rows,'file_count':len(rows),
          'omitted':'Full datasets, weights, already archived v5 science; page-review PNG intermediates are pinned in retained QA indexes.'}
write(bundle/'package_manifest.json',manifest)
archive=bundle/'dataset_examples_v6_addendum.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for row in rows: z.write(row['path'],row['member'])
    z.write(bundle/'package_manifest.json','package_manifest.json')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(rows)+1
    for row in rows:
        assert hashlib.sha256(z.read(row['member'])).hexdigest()==row['sha256']
        assert sha(Path(row['path']))==row['sha256']
verification={'passed':True,'zip':pin(archive),'manifest':pin(bundle/'package_manifest.json'),
              'verified_file_count':len(rows),'member_count':len(rows)+1,
              'crc_passed':True,'all_members_sha256_and_source_bytes_passed':True}
write(bundle/'verification.json',verification)

# Preserve all four entry points before promotion; append, never remove, historical file rows.
snapshot=ROOT/'provenance'/f'dataset_examples_v6_delivery_{STAMP}'
snapshot.mkdir()
entrypoints=[ROOT/'CAMPAIGN_STATUS.md',ROOT/'progress.md',PA/'README.md',PA/'latest_results.json']
snapshots=[]
for p in entrypoints:
    target=snapshot/'snapshots'/p.relative_to(ROOT)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,target)
    assert sha(p)==sha(target)
    snapshots.append({'original':str(p),'snapshot':pin(target)})
index=read(PA/'latest_results.json')
old_rows=list(index['files'])
prefix='final_v6_dataset_examples/'
replacement={
    'manuscript_docx':'word/PA_manuscript.docx','methods_evidence_supplement':'PA_supplementary.md',
    'current_common_results_draft':'PA_manuscript.md','integrated_manuscript_markdown':'PA_manuscript.md',
    'integrated_manuscript_word':'word/PA_manuscript.docx','integrated_manuscript_pdf':'word/PA_manuscript.pdf',
    'integrated_supplement_word':'word/PA_manuscript_supplement.docx',
    'integrated_supplement_pdf':'word/PA_manuscript_supplement.pdf','integrated_supplement_markdown':'PA_supplementary.md',
    'integrated_manuscript_figure_specs':'main_figures.json','integrated_manuscript_equation_specs':'equations.json',
    'integrated_supplement_figure_specs':'supplement_figures.json',
    'integrated_manuscript_integration_report':'integration_record.json',
    'integrated_manuscript_integration_supplement':'integration_record.json',
    'integrated_manuscript_visual_review':'visual_and_content_acceptance.json',
    'integrated_supplement_visual_review':'visual_and_content_acceptance.json',
    'integrated_manuscript_content_review':'delta_content_and_figure_review.json',
    'integrated_manuscript_delivery':'delivery.json','terminal_index_verification':'delivery.json'}
index['presentation_revision_prior_pointers']={k:index.get(k) for k in replacement}
index.update({k:prefix+v for k,v in replacement.items()})
index.update({'updated_local':NOW.isoformat(),'integrated_manuscript_pages':24,
              'integrated_supplement_pages':24,'dataset_examples_v6_complete':True,
              'presentation_revision_role':'6 actual training samples, 24 image panels; no new science',
              'internal_review_bundle_role':'Immutable complete v5 scientific base; apply v6 addendum for current prose and figures',
              'presentation_revision_bundle':prefix+'delivery_bundle/dataset_examples_v6_addendum.zip',
              'presentation_revision_bundle_sha256':verification['zip']['sha256'],
              'presentation_revision_bundle_verification':prefix+'delivery_bundle/verification.json',
              'presentation_revision_base_reference':prefix+'base_package_reference.json',
              'presentation_revision_prior_entrypoints':snapshots})
known={r['path'].replace('\\','/') for r in old_rows}
for p in sorted(NEW.rglob('*')):
    if not p.is_file(): continue
    if any(x in p.relative_to(NEW).parts for x in ('main_pages','supplement_pages','figures','bundled_render','__pycache__')):continue
    rel=p.relative_to(PA).as_posix()
    if rel not in known: index['files'].append({'path':rel,'sha256':sha(p)})
assert index['files'][:len(old_rows)]==old_rows
for key in ('final_confirmation_complete','computational_work_complete','confirmation_outcome_report_independently_accepted'):
    assert index[key] is True
assert index['submitted'] is False and index['submission_ready'] is False and index['automation_status']=='PAUSED'
(PA/'latest_results.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
header='''# 刺萼龙葵研究：数据集图片修订版 v6 已完成

新增 6 个真实训练样例，每例含原始 RGB、标注 mask、叠加图和目标区域 RGB，共 24 个面板。
主文图 2 在 PDF 第 6 页，补图 S14 在第 24 页；主文与补充材料各 24 页，均已逐页检查。
科学结果、冻结确认记录、作者及三项基金不变。未投稿，既有作者声明待确认；自动推进保持暂停。

'''
for p in [ROOT/'CAMPAIGN_STATUS.md',ROOT/'progress.md',PA/'README.md']:
    rel='' if p.parent==PA else 'precision_agriculture/'
    v=rel+prefix
    base_rel='../' if p.parent==PA else ''
    body=header+f'更新：{NOW.isoformat()}。目标期刊 Precision Agriculture。\n\n'
    body+=f'[主文 Word]({v}word/PA_manuscript.docx) · [主文 PDF]({v}word/PA_manuscript.pdf)\n\n'
    body+=f'[补充 Word]({v}word/PA_manuscript_supplement.docx) · [补充 PDF]({v}word/PA_manuscript_supplement.pdf)\n\n'
    body+=f'[新图及说明]({v}README.md) · [v6 图片修订增量包]({v}delivery_bundle/dataset_examples_v6_addendum.zip) · [交付回执]({v}delivery.json)\n\n'
    body+=f'[完整 v5 科学基础包]({base_rel}internal_submission_review_bundles/20260916_151139_042531/internal_review_only.zip) 继续保留；与 v6 增量包合用。\n\n'
    body+='新增图像为原始训练标注示意，不是模型预测。内部审阅未赋予本地数据新的公开再许可。旧版本与所有历史索引已保留，旧状态刷新脚本不得覆盖本终态。\n'
    p.write_text(body,encoding='utf-8')
delivery={'passed':True,'created_at':NOW.isoformat(),'scope':'Completed v6 dataset illustration addition',
          'assets':assets,'acceptance':pin(NEW/'visual_and_content_acceptance.json'),
          'revision_bundle':verification,'base_package_reference':pin(NEW/'base_package_reference.json'),
          'prior_entrypoints':snapshots,'updated_entrypoints':[pin(p) for p in entrypoints],
          'old_index_rows_retained':len(old_rows),'current_index_rows':len(index['files']),
          'frozen_science_unchanged':locked,'submitted':False,'automation_status':'PAUSED',
          'author_facts_pending':index['author_facts_pending']}
write(NEW/'delivery.json',delivery)
print(json.dumps({'passed':True,'delivery':pin(NEW/'delivery.json'),'bundle':verification['zip'],
                  'assets':assets,'old_rows':len(old_rows),'new_rows':len(index['files'])},ensure_ascii=False,indent=2))
