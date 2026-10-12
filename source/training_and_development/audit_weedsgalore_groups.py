"""Independent metadata-only WeedsGalore split/source audit; no test pixels."""
from pathlib import Path
from html.parser import HTMLParser
import collections, datetime, hashlib, json, shutil, zipfile

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'weedsgalore_data_v1'
EXT=ROOT/'external_data'/'weedsgalore'
SOURCE=EXT/'official_source_20260916_043356'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Paragraphs(HTMLParser):
    def __init__(self):super().__init__();self.current=None;self.paragraphs=[]
    def handle_starttag(self,tag,attrs):
        if tag=='p':self.current=[]
    def handle_data(self,data):
        if self.current is not None:self.current.append(data)
    def handle_endtag(self,tag):
        if tag=='p' and self.current is not None:
            self.paragraphs.append(' '.join(''.join(self.current).split()));self.current=None

def main():
    lock=read(DATA/'DATASET_LOCK.json');audit=read(SOURCE/'source_and_preprocessing_audit.json')
    static=SOURCE/'static';splits=static/'src/datasets/weedsgalore/splits'
    protocol=read(DATA/'source/data_protocol_v1.json')
    assert sha(DATA/'source/data_protocol_v1.json')==lock['protocol_sha256']
    checked={}
    for name,h in audit['static_file_sha256'].items():
        assert sha(static/name)==h;checked[str(static/name)]=h
    for name,h in lock['manifest_sha256'].items():
        assert sha(DATA/name)==h;checked[str(DATA/name)]=h
    archive=Path(protocol['archive']);assert sha(archive)==lock['archive_sha256']
    checked[str(archive)]=lock['archive_sha256']
    sets={};rows={};dates={}
    with zipfile.ZipFile(archive) as z:
        for split,n in [('train',104),('val',26),('test',26)]:
            official=(splits/(split+'.txt')).read_text().splitlines()
            zipped=z.read('weedsgalore-dataset/splits/'+split+'.txt').decode().splitlines()
            records=read(DATA/('test_sealed_metadata.json' if split=='test' else split+'.json'))
            assert official==zipped==[r['id'] for r in records]
            assert len(official)==len(set(official))==n
            assert all(r['split']==split and r['scene_group']=='WeedsGalore_Marquardt_2023_single_field' for r in records)
            sets[split]=set(official);rows[split]=records
            dates[split]=dict(sorted(collections.Counter(x[:10] for x in official).items()))
    assert len(set.union(*sets.values()))==156 and sum(map(len,sets.values()))==156
    assert set((splits/'scenes_list.txt').read_text().splitlines())==set.union(*sets.values())
    assert all(x['cache'] is None and x['metadata_only'] and 'coverage' not in x for x in rows['test'])
    assert {p.stem for p in (DATA/'cache').glob('*.npz')}==sets['train']|sets['val']
    reads=read(DATA/'archive_member_read_audit.json')
    assert set(reads['decoded_image_or_label_ids'])==sets['train']|sets['val']
    assert not reads['test_pixels_decoded']
    assert all(x.get('id') not in sets['test'] for x in reads['reads'])
    parser=Paragraphs()
    paper=ROOT/'external_data/candidate_screening/weedsgalore_availability_20260916_034329/paper_original.html'
    parser.feed(paper.read_text(encoding='utf-8'))
    evidence=[p for p in parser.paragraphs if ('split' in p.lower() and ('spatial' in p.lower() or 'patch' in p.lower()))]
    assert evidence
    out=EXT/('group_review_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    shutil.copy2(__file__,out/Path(__file__).name)
    result={'passed':True,'scope':'metadata only; official code read statically, never executed',
            'source_commit':audit['commit'],'counts':{s:len(v) for s,v in sets.items()},'date_counts':dates,
            'official_zip_manifest_membership_and_order_equal':True,'all_ids_disjoint_and_cover_156':True,
            'test_no_cache_or_coverage':True,'test_pixels_read_this_audit':False,
            'trainval_exact_cache_population_130':True,'actual_per_image_physical_patch_id_available':False,
            'source_field_count':1,'date_counts_are_not_independent_fields':True,
            'author_spatial_split_claim_independently_georeferenced':False,
            'paper_source':'https://arxiv.org/html/2502.13103v1','paper_sha256':sha(paper),
            'paper_paragraphs_for_internal_audit':evidence,'checked_sha256':checked,
            'dataset_lock_sha256':sha(DATA/'DATASET_LOCK.json'),'script_sha256':sha(__file__)}
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    report='''# WeedsGalore 分组与来源独立核查

固定官方提交、ZIP清单和新cache清单的104/26/26成员及顺序全部一致，156 ID互斥且无缺项。全部静态源码与数据清单哈希通过。130个开发cache严格对应train+val；test只有成员元数据，无cache或coverage。本次未读取任何test像素。

原论文报告按12个空间区域作8/2/2分割，并维持跨日期位置的split归属；包内没有可供独立复算的逐帧位置组ID或已核实地理变换。本次确认的是官方split完整复现，不能独立证明跨split无物理足迹重叠。四日期来自同一田块，不能视为四个独立田间重复。

可作为额外数据来源的开发复现，不能称跨田块独立测试，不能把26幅val或多个日期当独立田块做显著性检验。未来物理组验证需要真实位置映射，现有原始分割保持不变。该数据没有可用于本地米制M主分析的已验证逐帧地理变换；名义GSD不替代正射几何。

本次仅审核分组、来源及隔离；像素适配器与空间重采样数值检查另外记录，未启动外部训练。
'''
    (out/'review.md').write_text(report,encoding='utf-8')
    print(json.dumps({'directory':str(out),'date_counts':dates,'paper_evidence':evidence},ensure_ascii=False))

if __name__=='__main__':main()
