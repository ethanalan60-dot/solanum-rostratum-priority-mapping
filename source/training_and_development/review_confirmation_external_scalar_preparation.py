"""Bounded independent source/artificial review; no actual outcome entry."""
from pathlib import Path
from unittest.mock import patch
from fractions import Fraction
import argparse,ast,copy,datetime,importlib.util,json,math,sys
ROOT=Path(__file__).resolve().parent;HOME=ROOT/'confirmation_v1'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-sha',required=True);parser.add_argument('--readiness',required=True,type=Path);args=parser.parse_args()
    sys.path.insert(0,str(HOME));import core
    source=ROOT/'review_confirmation_external_scalar.py';core.bound_file(source,args.source_sha)
    readiness=core.read(args.readiness);assert readiness['implementation_sha256']==args.source_sha and readiness['preparation_passed']
    for p,rec in readiness['bindings'].items():core.bound_file(p,rec['sha256']);core.bound_file(rec['archive_path'],rec['sha256'])
    spec=importlib.util.spec_from_file_location('independent_external_scalar_subject',source);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    out=ROOT/'provenance'/('confirmation_ext_scalar_independent_preparation_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    checks=[]
    def yes(name,ok=True):assert ok,name;checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,FileNotFoundError):checks.append(name);return
        raise AssertionError('Missing refusal '+name)
    try:
        yes('Reviewed module imports no array model or solver','numpy' not in sys.modules and 'torch' not in sys.modules and 'scipy' not in sys.modules)
        core.bound_file(readiness['old_source'],readiness['old_source_sha256'])
        def function_ast(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
        old,new=function_ast(readiness['old_source']),function_ast(source)
        yes('All seven scientific functions unchanged by provenance repair',all(old[n]==new[n] for n in readiness['scientific_function_AST_unchanged']))
        home=out/'mock_gate';home.mkdir();fixed=('FREEZE.json','SOURCE_LOCK.json','contract.json','checkpoints.json','scalar_baselines.json')
        for name in fixed:core.write(home/name,{'artificial':True})
        grant=home/'grant.json';choices=home/'choices.json';core.write(grant,{});core.write(choices,{})
        calls=[]
        def stop_after_capture(**kwargs):
            calls.append(1)
            assert all(str((home/name).resolve()) in a.WATCH for name in fixed)
            assert str(grant.resolve()) in a.WATCH and str(choices.resolve()) in a.WATCH
            raise ValueError('ARTIFICIAL stop at freeze validator; no dataset access')
        with patch.object(a,'HOME',home),patch.object(core,'validate_freeze',side_effect=stop_after_capture):
            reject('Initial source snapshot exists before freeze validator',lambda:a.gate(grant,choices))
        yes('Only mocked freeze validator entered',calls==[1] and 'numpy' not in sys.modules)
        old=(home/'contract.json').read_bytes();core.write(home/'contract.json',{'artificial':'changed'})
        reject('Begin/end rejects contract replacement',a.verify_watched);(home/'contract.json').write_bytes(old)
        a.verify_watched();checks.append('Restored artificial source identity verifies')
        metrics=a.independent_metrics([.2,.4,.7],[0.,.2,1.])
        expected=dict(n=3,n_zero=1,n_positive=2,mae=7/30,bias=1/30,mae_zero=.2,mae_positive=.25,balanced_mae=.225)
        yes('Nondegenerate scalar errors recomputed by explicit fractions',all(abs(metrics[k]-v)<1e-15 for k,v in expected.items()))
        yes('Zero-only stratum has NA positive/balanced',a.independent_metrics([.2,.4],[0.,0.])['balanced_mae'] is None and a.independent_metrics([.2,.4],[0.,0.])['mae_positive'] is None)
        yes('Positive-only stratum has NA zero/balanced',a.independent_metrics([.2,.4],[.1,.5])['balanced_mae'] is None and a.independent_metrics([.2,.4],[.1,.5])['mae_zero'] is None)
        yes('Seed n-1 SD has independent nonconstant result',abs(a.sd([.1,.2,.9])-math.sqrt(.19))<1e-15)
        yes('Single seed and partial NA never invent SD',a.sd([.2]) is None and a.sd([.2,None,.3]) is None and a.mean([.2,None]) is None)
        reject('NaN scalar rejected',lambda:a.independent_metrics([float('nan')],[.1]))
        reject('Prediction above one rejected',lambda:a.independent_metrics([1.1],[.1]))
        reject('NA is not silently zero',lambda:a.check(0.,None))
        yes('Crop zero denominator and k0 composition remain NA',a.ratio(0.,0.) is None and a.ratio(.1,.2)==.5)
        for g in (4,8,16):
            budget=a.budgets(g);yes(f'g{g} complete two bases',len(budget)==10)
            seen={}
            for row in budget:
                f=Fraction(row['parent_nominal_budget_fraction']).limit_denominator(100)
                n=g*g
                k=n*f.numerator//f.denominator if row['budget_basis']=='nominal_primary' else (16*f.numerator//f.denominator)*(n//16)
                yes(f'g{g}/{row["budget_basis"]}/{row["parent_nominal_budget_code"]} independent floor and alias',row['k']==k and row['actual_fraction']==k/n and row['prior_alias']==seen.get(k))
                seen.setdefault(k,(row['budget_basis'],row['parent_nominal_budget_code']))
        nominal=[a.budgets(g)[0]['actual_fraction'] for g in (4,8,16)]
        yes('10percent actual fractions differ by grid as declared',nominal==[1/16,6/64,25/256])
        yes('Matched 10percent anchor equals1/16 for all grids',all(a.budgets(g)[5]['actual_fraction']==1/16 for g in (4,8,16)))
        # Deliberately incomplete scalar files must fail from identity/sample checks
        # before any downstream table is opened.
        folder=out/'artificial_scalar';folder.mkdir()
        model=dict(run_id='SYNTHETIC',dataset='local',encoder='resnet18',variant='regression',lr=.0001,seed=17,best_epoch=3)
        base=dict(model,R=1,id='f',prediction=.2,truth=.1,signed_error=.1,absolute_error=.1)
        exported={('SYNTHETIC',R,'f'):.2 for R in a.RS};ids={'local':['f']};truth={'local':{'f':.1}}
        def scalarcall():return a.audit_scalar(folder,{'SYNTHETIC':model},ids,truth,exported,{},False)
        core.table(folder/'all_scalar_samples.csv',[base,base])
        reject('Duplicate sample identity rejected',scalarcall)
        a.WATCH.pop(str((folder/'all_scalar_samples.csv').resolve()),None)
        bad=dict(base,id='FOREIGN');core.table(folder/'all_scalar_samples.csv',[bad])
        reject('Foreign sample identity rejected',scalarcall)
        a.WATCH.pop(str((folder/'all_scalar_samples.csv').resolve()),None)
        bad=dict(base,best_epoch=4);core.table(folder/'all_scalar_samples.csv',[bad])
        reject('Original epoch identity enforced',scalarcall)
        # Test tiny binary32 sums separately from the full saved table fixture.
        import numpy as np
        manifest=out/'mock_source.json'
        def sourcecase(records):
            core.write(manifest,records);a.WATCH.pop(str(manifest.resolve()),None)
            return {'local':{'source_manifest':core.binding(manifest),'records':[{'id':'a'},{'id':'b'}]}}
        inputs=sourcecase([{'id':'a','coverage':1/3},{'id':'b','coverage':.123456789}])
        values=a.local_source_coverage(inputs,np)
        yes('Original coverage is cast to float32 not soft-grid mean',values=={'a':float(np.float32(1/3)),'b':float(np.float32(.123456789))})
        inputs=sourcecase([{'id':'b','coverage':.1},{'id':'a','coverage':.2}])
        reject('Source coverage order mismatch rejected',lambda:a.local_source_coverage(inputs,np))
        inputs=sourcecase([{'id':'a','coverage':.1},{'id':'b','coverage':2.}])
        reject('Source coverage domain violation rejected',lambda:a.local_source_coverage(inputs,np))
        plane=np.zeros((256,256),np.float32);plane[:64,:64]=np.float32(2.**-60);plane[0,0]=.5
        plane[:64,64:128]=np.float32(2.**-60);plane[63,127]=.5
        vals=a.aggregate(plane,4,np)
        exact=np.array([math.fsum(map(float,plane[r*64:(r+1)*64,c*64:(c+1)*64].ravel()))/65536 for r in range(4) for c in range(4)])
        analytic=float((Fraction(1,2)+4095*Fraction(1,2**60))/65536)
        yes('Tiny binary32 cell sums agree with rational oracle',exact[0]==exact[1]==analytic)
        yes('Exact-equal tiny cells receive ASCII tie break',sorted((0,1),key=lambda i:(-exact[i],a.ids_for('f',4)[i]))==[0,1])
        yes('Fast reduction is diagnostic only within tolerance',np.max(abs(vals-exact))<=2e-13)
        plane=np.arange(256*256,dtype=np.float32).reshape(256,256)/np.float32(65536)
        g4=a.aggregate(plane,4,np).reshape(4,4);g8=a.aggregate(plane,8,np).reshape(4,2,4,2).sum((1,3));g16=a.aggregate(plane,16,np).reshape(4,4,4,4).sum((1,3))
        yes('Independent direct nested grid area integrals agree',np.max(abs(g4-g8))<1e-15 and np.max(abs(g4-g16))<1e-15)
        for n in (16,64,256):
            for upper in (Fraction(1),Fraction(1,2)):
                total=sum(Fraction(k,n)*max(Fraction(0),min(Fraction(k+1,n),upper)-Fraction(k,n)) for k in range(n))/upper
                formula=Fraction(n-1,2*n) if upper==1 else Fraction(1,4)-Fraction(1,2*n)
                yes(f'N{n} analytic random step AUC u{upper}',total==formula)
        # Source-reviewed equations and consumer fields; no execute(), real gate,
        # NPZ loader or actual reserved stage is invoked.
        tree=ast.parse(source.read_text(encoding='utf-8'));functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        body=ast.unparse(functions['execute'])
        yes('Prediction cell ranking uses exact fsum result', 'math.fsum' in body and 'P[rid, R, item[\'id\'], g] = exact' in body)
        yes('Actual gate precedes numpy array import',body.index('gate(')<body.index('import numpy'))
        gatebody=ast.unparse(functions['gate'])
        yes('Global prediction and input manifests captured before validators',gatebody.index("pin(HOME / 'stages/predictions_seal/all_predictions.json'")<gatebody.index('predictions = validate_global_predictions()') and gatebody.index("pin(HOME / 'stages' / ('inputs_' + dataset) / 'input_manifest.json')")<gatebody.index('inputs[dataset] = validate_input_manifest(dataset)'))
        yes('Final report follows every watched hash check',body.index('for path, digest in WATCH.items()')<body.index("common = {'passed': True"))
        for key in ('stage','passed','audited_artifact_manifest_sha256','freeze_sha256'):
            yes('Report consumer required field '+key,key in source.read_text(encoding='utf-8') and key in (HOME/'report_tables.py').read_text(encoding='utf-8'))
        core.bound_file(source,args.source_sha)
        review=dict(passed=True,scope='independent_preparation_source_and_small_synthetic',implementation_sha256=args.source_sha,reviewer_sha256=core.sha(__file__),author_readiness=core.binding(args.readiness),checks=checks,check_count=len(checks),
          original_full_fixture_repeated=False,actual_reserved_arrays_read=0,actual_outcome_entry_calls=0,model_calls=0,solver_calls=0,
          limits=['Author full38792/complete external cross-implementation fixture accepted by source/hash receipt; not redundantly re-executed.','Raw production NPZ decoding and completed real stages were not executed.','No claim of real confirmation acceptance; a formal source-bound execution grant remains necessary.'])
        core.write(out/'review.json',review)
        (out/'review.md').write_text('# Independent preparation review\n\nPASS: '+str(len(checks))+' focused checks and full source review. Scalar stratum NA, n-1 seed SD, identity rejection, grid/budget aliases, exact tiny-value ranking sums and analytic step AUC were independently checked. Author full-matrix receipts remain separately bound. No real reserved inputs or actual entry was used.\n',encoding='utf-8')
        print(json.dumps({'passed':True,'out':str(out),'checks':len(checks),'review_sha256':core.sha(out/'review.json')}))
    except BaseException as error:
        import traceback
        core.write(out/'failure.json',dict(passed=False,error=repr(error),traceback=traceback.format_exc(),checks_completed=checks));raise
if __name__=='__main__':main()
