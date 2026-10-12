"""CPU prediction-only export, global seals, and explicit post-selection truth gate."""
from pathlib import Path
import argparse,csv,gc,json,math,sys
from core import HERE,R_VALUES,attempt,binding,bound_file,completed_stage,configure_cpu,read,require,sha,table,validate_freeze,validate_source_lock,write
from data_io import validate_input_manifest

def prediction_arrays(model,x,valid,variant,factor):
    """Only image/valid arguments; no reference tensor is accepted."""
    import torch
    import frozen_nn as nn
    logits=model(nn.degrade_rgb(x,factor,valid),valid);nn.require_finite_forward(logits,model)
    probability=logits.float().sigmoid()
    if variant=='regression':
        require(probability.shape==(len(x),),'Regression must be scalar only');prediction=probability
    else:
        require(probability.shape==(len(x),1,256,256),'Spatial output support changed');prediction=nn.coverage(probability,valid)
    require(bool(torch.isfinite(prediction).all() and ((prediction>=0)&(prediction<=1)).all()),'Nonfinite/out-of-range prediction')
    return probability,prediction

def export_run(run_id):
    contract,models,freeze=validate_freeze();entry=next((x for x in models if x['run_id']==run_id),None)
    require(entry is not None,'Undeclared run');dataset=entry['dataset'];manifest=validate_input_manifest(dataset)
    model=None
    with attempt('predict_'+run_id) as out:
        configure_cpu(out)
        import numpy as np
        import torch
        import frozen_nn as nn
        torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(entry['seed'])
        require(not torch.cuda.is_available(),'GPU must be hidden')
        write(out/'prediction_input_seal.json',{'checkpoint':binding(entry['checkpoint_path']),'config':binding(entry['config_path']),'input_manifest':binding(HERE/'stages'/('inputs_'+dataset)/'input_manifest.json'),'input_artifact_manifest':binding(HERE/'stages'/('inputs_'+dataset)/'artifact_manifest.json'),'original_best_epoch':entry['best_epoch'],'checkpoint_reselected':False,'truth_read_by_prediction':False})
        try:
            bound_file(entry['checkpoint_path'],entry['checkpoint_sha256'])
            if dataset=='weeds':
                from safe_checkpoint import load_checkpoint
                payload=load_checkpoint(entry['checkpoint_path'],entry['checkpoint_sha256'])
            else:payload=torch.load(entry['checkpoint_path'],map_location='cpu',weights_only=True)
            require(payload['config']==read(entry['config_path']) and payload['epoch']==entry['best_epoch'] and payload['validation']==read(entry['best_validation_path']),'Selected payload differs')
            model=nn.make_model(entry['encoder'],entry['variant'],pretrained=False).cpu().eval();model.load_state_dict(payload['model'],strict=True);del payload
            records=manifest['records'];seg=entry['variant']!='regression';maps={row['id']:[] for row in records} if seg else None
            valid_arrays={};rows=[]
            for factor in R_VALUES:
                for start in range(0,len(records),4):
                    batch=records[start:start+4];images=[];valids=[]
                    for record in batch:
                        bound_file(record['input_cache'],record['input_cache_sha256'])
                        with np.load(record['input_cache'],allow_pickle=False) as z:
                            require(set(z.files)=={'image','valid'},'Non-input cache keys');x=z['image'];v=z['valid']
                        require(x.shape==(3,256,256) and x.dtype==np.float32 and np.isfinite(x).all() and ((x>=0)&(x<=1)).all(),'RGB input schema')
                        require(v.shape==(256,256) and v.dtype==np.float32 and np.isfinite(v).all() and ((v>=0)&(v<=1)).all() and v.sum(dtype=np.float64)>0,'Valid input schema')
                        images.append(x);valids.append(v)
                    x=torch.from_numpy(np.stack(images));v=torch.from_numpy(np.stack(valids))[:,None]
                    with torch.inference_mode():prob,pred=prediction_arrays(model,x,v,entry['variant'],factor)
                    for j,record in enumerate(batch):
                        identifier=record['id'];estimate=float(pred[j]);row={k:entry[k] for k in ('run_id','dataset','encoder','variant','seed','best_epoch')}
                        row.update(id=identifier,R=factor,prediction_cpu=estimate)
                        rows.append(row)
                        if seg:
                            array=prob[j,0].numpy().copy();va=valids[j]
                            q=float((array.astype(np.float64)*va).sum(dtype=np.float64)/va.sum(dtype=np.float64))
                            require(abs(q-estimate)<=(1e-7 if dataset=='weeds' else 1e-6),'Map/scalar integral mismatch')
                            maps[identifier].append(array);valid_arrays[identifier]=va
            require(len(rows)==len(records)*4,'Incomplete predictions');table(out/'predicted_scalars.csv',rows);index=[]
            if seg:
                (out/'maps').mkdir()
                for record in records:
                    identifier=record['id'];a=np.stack(maps[identifier]);v=valid_arrays[identifier]
                    require(a.dtype==np.float32 and a.shape==(4,256,256),'Map stack mismatch')
                    path=out/'maps'/(identifier+'.npz');np.savez_compressed(path,probabilities=a,valid=v,R=np.asarray(R_VALUES,dtype=np.int64),id=np.asarray(identifier))
                    with np.load(path,allow_pickle=False) as z:require(set(z.files)=={'probabilities','valid','R','id'} and np.array_equal(z['probabilities'],a) and np.array_equal(z['valid'],v),'Map roundtrip')
                    index.append({'run_id':run_id,'dataset':dataset,'id':identifier,'path':str(path),'sha256':sha(path),'R':list(R_VALUES),'shape':[4,256,256],'dtype':'float32','input_cache_sha256':record['input_cache_sha256'],'native_window_xywh':record['native_window_xywh'],'source_id':record['source_id'],'valid_semantics':record['valid_semantics']})
            write(out/'map_index.json',index)
            # Guard changing source or checkpoint during computation; no second forward or reselection.
            validate_source_lock();bound_file(entry['checkpoint_path'],entry['checkpoint_sha256'])
            write(out/'metadata.json',{**entry,'version':'confirmation_prediction_only_v1','R':list(R_VALUES),'CPU_dtype':'float32','batch_size':4,'CPU_threads':2,'torch':str(torch.__version__),'member_ids':[x['id'] for x in records],'scalar_rows':len(rows),'map_count':len(index),'input_manifest':binding(HERE/'stages'/('inputs_'+dataset)/'input_manifest.json'),'freeze_sha256':sha(HERE/'FREEZE.json'),'checkpoint_reselected':False,'truth_read_by_prediction':False,'test_vs_development_samples_compared':False})
        finally:model=None;gc.collect()
    return out

def inspect_prediction(entry):
    out=HERE/'stages'/('predict_'+entry['run_id']);completed_stage(out);meta=read(out/'metadata.json');maps=read(out/'map_index.json')
    manifest=validate_input_manifest(entry['dataset']);ids=[x['id'] for x in manifest['records']]
    for key,value in entry.items():require(meta[key]==value,'Prediction model binding changed: '+key)
    require(meta['member_ids']==ids and meta['R']==list(R_VALUES) and meta['freeze_sha256']==sha(HERE/'FREEZE.json'),'Prediction population/freeze differs')
    require(meta['truth_read_by_prediction'] is False and meta['checkpoint_reselected'] is False,'Wrong prediction phase')
    with (out/'predicted_scalars.csv').open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    expected=[(identifier,r) for r in R_VALUES for identifier in ids]
    require([(x['id'],int(x['R'])) for x in rows]==expected and len(rows)==meta['scalar_rows'],'Incomplete scalar identity/order')
    for row in rows:require(row['run_id']==entry['run_id'] and math.isfinite(float(row['prediction_cpu'])) and 0<=float(row['prediction_cpu'])<=1,'Invalid scalar')
    expected_maps=ids if entry['variant']!='regression' else []
    require([x['id'] for x in maps]==expected_maps and len(maps)==meta['map_count'],'Incomplete maps/regression map')
    for row in maps:
        require(row['R']==list(R_VALUES) and row['run_id']==entry['run_id'],'Wrong map R/run');bound_file(row['path'],row['sha256'])
    return {'run_id':entry['run_id'],'dataset':entry['dataset'],'variant':entry['variant'],'stage':str(out),'artifact_manifest':binding(out/'artifact_manifest.json'),'metadata':binding(out/'metadata.json'),'map_index':binding(out/'map_index.json'),'predicted_scalars':binding(out/'predicted_scalars.csv'),'scalar_rows':len(rows),'map_count':len(maps)}

def seal_predictions():
    contract,models,freeze=validate_freeze()
    with attempt('predictions_seal') as out:
        entries=[inspect_prediction(x) for x in models]
        require(sum(x['scalar_rows'] for x in entries)==38792 and sum(x['map_count'] for x in entries)==8268,'Global prediction count mismatch')
        write(out/'all_predictions.json',{'version':'confirmation_all_predictions_v1','freeze_sha256':sha(HERE/'FREEZE.json'),'run_ids':[x['run_id'] for x in models],'models':entries,'scalar_rows':38792,'map_count':8268,'truth_scoring_completed':False,'all_declared_models_completed':True})
    return out

def validate_global_predictions():
    out=HERE/'stages'/'predictions_seal';completed_stage(out);seal=read(out/'all_predictions.json')
    contract=read(HERE/'contract.json');require(seal['freeze_sha256']==sha(HERE/'FREEZE.json') and seal['run_ids']==[x['run_id'] for x in contract['roster']],'Global seal scope mismatch')
    require(seal['all_declared_models_completed'] is True and seal['scalar_rows']==38792 and seal['map_count']==8268,'Global seal counts')
    require([x['run_id'] for x in seal['models']]==[x['run_id'] for x in contract['roster']],'Global model entries incomplete/duplicated/reordered')
    for entry,model in zip(seal['models'],contract['roster']):
        count=contract['datasets'][model['dataset']]['count']
        require(entry['dataset']==model['dataset'] and entry['variant']==model['variant'],'Global model identity mismatch')
        require(entry['scalar_rows']==count*4 and entry['map_count']==(0 if model['variant']=='regression' else count),'Global per-model count mismatch')
        expected=HERE/'stages'/('predict_'+model['run_id'])
        require(Path(entry['stage']).resolve()==expected.resolve(),'Global stage identity mismatch')
        completed_stage(entry['stage'])
        for key in ('artifact_manifest','metadata','map_index','predicted_scalars'):
            filename=key+('.csv' if key=='predicted_scalars' else '.json')
            require(Path(entry[key]['path']).resolve()==(expected/filename).resolve(),'Global artifact path mismatch')
            bound_file(entry[key]['path'],entry[key]['sha256'])
    require(sum(x['scalar_rows'] for x in seal['models'])==seal['scalar_rows'] and sum(x['map_count'] for x in seal['models'])==seal['map_count'],'Global counts do not sum')
    return seal

def validate_choices(review_path):
    """Consumer contract. Geometry/selection have their own later frozen implementation and audit."""
    review=read(review_path);contract=read(HERE/'contract.json')
    require(review.get('passed') is True and review.get('scope')=='confirmation_all_label_free_selections','Independent root selection acceptance required')
    require(review['freeze_sha256']==sha(HERE/'FREEZE.json') and review['all_predictions_sha256']==sha(HERE/'stages'/'predictions_seal'/'all_predictions.json'),'Choice seal precedes/differs from predictions')
    expected=[x for x in contract['roster'] if x['variant']!='regression']
    require([x['run_id'] for x in review['models']]==[x['run_id'] for x in expected],'All 30 segmentation choices must be sealed in roster order')
    require(review.get('truth_used_for_choices') is False and review.get('complete_case_inventory_verified') is True,'Selection information/case boundary violated')
    for entry,model in zip(review['models'],expected):
        for key in ('selection_seal','independent_review'):bound_file(entry[key]['path'],entry[key]['sha256'])
        audit=read(entry['independent_review']['path'])
        audited=audit.get('models',[audit]);matched=[x for x in audited if x.get('run_id')==model['run_id']]
        require(audit.get('passed') is True and len(matched)==1 and matched[0]['selection_seal_sha256']==entry['selection_seal']['sha256'],'Selection audit binding missing')
        require(entry['case_count']==(120 if model['dataset']=='local' else 3120),'Incomplete grid/R/budget/policy or frame cases')
    return review

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    sub.add_parser('inspect');sub.add_parser('validate-freeze')
    p=sub.add_parser('prepare-inputs');p.add_argument('--dataset',choices=['local','weeds'],required=True)
    p=sub.add_parser('predict');p.add_argument('--run-id',required=True)
    sub.add_parser('seal-predictions')
    p=sub.add_parser('materialize-truth');p.add_argument('--dataset',choices=['local','weeds'],required=True);p.add_argument('--choice-review',required=True)
    args=parser.parse_args()
    if args.stage=='inspect':
        c=read(HERE/'contract.json');validate_contract_local(c);print(json.dumps({'phase':'preparation_only','freeze_exists':(HERE/'FREEZE.json').exists(),'checkpoint_manifest_frozen':read(HERE/'checkpoints.json')['frozen'],'expected':c['expected']},ensure_ascii=False));return
    if args.stage=='validate-freeze':validate_freeze();print('Freeze metadata accepted; no test file opened');return
    if args.stage=='prepare-inputs':
        from data_io import prepare_inputs
        result=prepare_inputs(args.dataset)
    elif args.stage=='predict':result=export_run(args.run_id)
    elif args.stage=='seal-predictions':result=seal_predictions()
    else:
        from data_io import materialize_truth
        result=materialize_truth(args.dataset,args.choice_review)
    print(str(result))

def validate_contract_local(c):
    from core import validate_contract
    validate_contract(c)

if __name__=='__main__':main()
