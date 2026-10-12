"""Reproducible development-only training. No final-test manifest is read here."""
from pathlib import Path
import argparse, csv, hashlib, json, math, os, random, sys, time, traceback
ROOT=Path(__file__).resolve().parent
for key,sub in [('TORCH_HOME','cache/torch'),('HF_HOME','cache/huggingface'),('TEMP','tmp'),('TMP','tmp')]:
    folder=ROOT/sub;folder.mkdir(parents=True,exist_ok=True);os.environ[key]=str(folder)
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Tiles(Dataset):
    def __init__(self,records,augment=False):self.records=records;self.augment=augment;self.verified=set()
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        if 'cache_sha256' in r and r['id'] not in self.verified:
            assert sha(r['cache'])==r['cache_sha256'], f"Cache changed: {r['id']}"
            self.verified.add(r['id'])
        with np.load(r['cache']) as d:
            x=d['image'].copy();soft=d['soft'].astype(np.float32);valid=d['valid'].astype(np.float32);hard=d['hard'].astype(np.float32)
        if self.augment:
            if random.random()<.5:x=x[:,::-1];soft=soft[:,::-1];valid=valid[:,::-1];hard=hard[:,::-1]
            if random.random()<.5:x=x[::-1];soft=soft[::-1];valid=valid[::-1];hard=hard[::-1]
            x=np.clip(x.astype(np.float32)*random.uniform(.9,1.1)+random.uniform(-8,8),0,255)
            x[valid==0]=0
        x=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).float()/255
        return x,torch.from_numpy(soft.copy())[None],torch.from_numpy(valid.copy())[None],torch.from_numpy(hard.copy())[None],torch.tensor(r['coverage'],dtype=torch.float32)

class ConvBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__();self.layers=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.ReLU(inplace=True),nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.ReLU(inplace=True))
    def forward(self,x):return self.layers(x)

class Model(nn.Module):
    def __init__(self,regression=False,pretrained=True):
        super().__init__();self.regression=regression
        self.encoder=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None,progress=False)
        if regression:
            self.head=nn.Linear(512,1);nn.init.constant_(self.head.bias,-2.5)
        else:
            self.up3=ConvBlock(512+256,128);self.up2=ConvBlock(128+128,64)
            self.up1=ConvBlock(64+64,32);self.up0=ConvBlock(32+64,32)
            self.head=nn.Conv2d(32,1,1);nn.init.constant_(self.head.bias,-2.5)
        self.encoder.fc=nn.Identity()
    def forward(self,x,valid=None):
        e=self.encoder;s0=e.relu(e.bn1(e.conv1(x)));s1=e.layer1(e.maxpool(s0));s2=e.layer2(s1);s3=e.layer3(s2);s4=e.layer4(s3)
        if self.regression:
            if valid is None:features=s4.mean((2,3))
            else:
                weights=F.interpolate(valid,size=s4.shape[-2:],mode='area')
                features=(s4*weights).sum((2,3))/weights.sum((2,3)).clamp_min(1e-6)
            return self.head(features).squeeze(1)
        def up(a,b):return torch.cat([F.interpolate(a,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1)
        z=self.up3(up(s4,s3));z=self.up2(up(z,s2));z=self.up1(up(z,s1));z=self.up0(up(z,s0))
        return F.interpolate(self.head(z),size=x.shape[-2:],mode='bilinear',align_corners=False)

def degrade(x,factor,valid):
    if factor!=1:
        size=(max(8,x.shape[-2]//factor),max(8,x.shape[-1]//factor))
        vsmall=F.interpolate(valid,size=size,mode='area')
        numerator=F.interpolate(x*valid,size=size,mode='area')
        small=numerator/vsmall.clamp_min(1e-6)
        numerator=F.interpolate(small*vsmall,size=x.shape[-2:],mode='bilinear',align_corners=False)
        denominator=F.interpolate(vsmall,size=x.shape[-2:],mode='bilinear',align_corners=False)
        x=numerator/denominator.clamp_min(1e-6)
    x=torch.where(valid>0,x,torch.zeros_like(x))
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return (x-mean)/std

def coverage(p,valid):return (p*valid).sum((1,2,3))/valid.sum((1,2,3)).clamp_min(1)

def summarize(pred,true):
    p=np.asarray(pred);t=np.asarray(true);e=p-t;pos=t>0;zero=~pos
    z=float(np.abs(e[zero]).mean()) if zero.any() else None
    a=float(np.abs(e[pos]).mean()) if pos.any() else None
    strata=[v for v in [z,a] if v is not None]
    return {'n':len(t),'n_positive':int(pos.sum()),'mae':float(np.abs(e).mean()),'bias':float(e.mean()),'mae_zero':z,'mae_positive':a,'balanced_mae':float(np.mean(strata))}

@torch.inference_mode()
def evaluate(model,loader,device,regression):
    model.eval();results={}
    for factor in [1,2,4]:
        pred=[];true=[];tp=fp=fn=0.
        for x,soft,valid,hard,q in loader:
            x=x.to(device);valid=valid.to(device);soft=soft.to(device)
            with torch.autocast(device_type='cuda',enabled=device.type=='cuda'):
                logits=model(degrade(x,factor,valid),valid);p=logits.float().sigmoid();est=p if regression else coverage(p,valid)
            pred.extend(est.cpu().tolist());true.extend(q.tolist())
            if not regression:
                binary=(p>.5).float();w=valid
                # A coarse binary prediction covers a fine-grid fraction soft; do not threshold away ground-truth area.
                tp+=float((binary*soft*w).sum());fp+=float((binary*(1-soft)*w).sum());fn+=float(((1-binary)*soft*w).sum())
        results[str(factor)]=summarize(pred,true)
        if not regression:results[str(factor)].update({'target_iou':tp/max(tp+fp+fn,1),'dice':2*tp/max(2*tp+fp+fn,1)})
    return {'selection_balanced_mae':float(np.mean([v['balanced_mae'] for v in results.values()])), 'scales':results}

def run(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed);torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir=ROOT/'runs'/args.run_id
    if (run_dir/'config.json').exists():raise FileExistsError(f'Run id already exists: {run_dir}')
    run_dir.mkdir(parents=True,exist_ok=True)
    data_dir=Path(args.data_dir)
    if not data_dir.is_absolute():data_dir=ROOT/data_dir
    train_path=data_dir/'train.json';val_path=data_dir/'val.json'
    lock_path=data_dir/'DATASET_LOCK.json'
    if not lock_path.exists():raise RuntimeError('Seal the prepared dataset before training.')
    lock=json.loads(lock_path.read_text(encoding='utf-8'))
    assert sha(train_path)==lock['manifest_sha256']['train.json'] and sha(val_path)==lock['manifest_sha256']['val.json'], 'Manifest changed after sealing'
    train=json.loads(train_path.read_text(encoding='utf-8'));val=json.loads(val_path.read_text(encoding='utf-8'))
    train_ids=[r['id'] for r in train];val_ids=[r['id'] for r in val]
    assert len(train_ids)==len(set(train_ids)) and len(val_ids)==len(set(val_ids)), 'Duplicate locations in manifest'
    assert not (set(train_ids)&set(val_ids)), 'Train/validation overlap'
    # Optional bounded smoke run retains positives and zeros deterministically.
    def subset(rows,n):
        if n<=0:return rows
        pos=[r for r in rows if r['coverage']>0];neg=[r for r in rows if r['coverage']==0]
        return pos[:n//2]+neg[:n-len(pos[:n//2])]
    train=subset(train,args.max_train);val=subset(val,args.max_val)
    cfg=vars(args)|{'python':sys.version,'torch':torch.__version__,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                    'train_manifest_sha256':sha(train_path),'val_manifest_sha256':sha(val_path),'script_sha256':sha(__file__),
                    'dataset_lock_sha256':sha(lock_path),'validation_input_sizes':{'1':256,'2':128,'4':64},
                    'n_train':len(train),'n_val':len(val),'stage':'smoke' if args.max_train else 'development',
                    'test_accessed':False,'pretrained_source':ResNet18_Weights.DEFAULT.url}
    (run_dir/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    truth=[r['coverage'] for r in val];mean=float(np.mean([r['coverage'] for r in train]))
    constants={'zero':summarize([0]*len(val),truth),'train_mean':summarize([mean]*len(val),truth)}
    (run_dir/'constant_baselines.json').write_text(json.dumps(constants,indent=2),encoding='utf-8')
    npos=sum(r['coverage']>0 for r in train);nneg=len(train)-npos
    weights=[.5/max(npos,1) if r['coverage']>0 else .5/max(nneg,1) for r in train]
    gen=torch.Generator().manual_seed(args.seed)
    sampler=WeightedRandomSampler(weights,len(train),replacement=True,generator=gen)
    train_loader=DataLoader(Tiles(train,True),batch_size=args.batch_size,sampler=sampler,num_workers=0,pin_memory=device.type=='cuda')
    val_loader=DataLoader(Tiles(val),batch_size=args.batch_size,num_workers=0,pin_memory=device.type=='cuda')
    regression=args.variant=='regression';model=Model(regression).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda')
    best=float('inf');best_epoch=None;start=time.time()
    for epoch in range(1,args.epochs+1):
        model.train();total=0.;count=0
        for x,soft,valid,hard,q in train_loader:
            x=x.to(device);soft=soft.to(device);valid=valid.to(device);hard=hard.to(device);q=q.to(device)
            factor=random.choice([1,2,4]) if args.scale_augmentation else 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',enabled=device.type=='cuda'):
                logits=model(degrade(x,factor,valid),valid);p=logits.float().sigmoid()
                if regression:
                    loss=F.smooth_l1_loss(p,q,beta=.1)
                else:
                    target=hard if args.variant=='hard_seg' else soft
                    bce=F.binary_cross_entropy_with_logits(logits.float(),target,reduction='none',pos_weight=logits.new_tensor(args.pos_weight))
                    bce=(bce*valid).sum()/valid.sum().clamp_min(1)
                    dice=1-(2*(p*target*valid).sum((1,2,3))+1)/((p*valid).sum((1,2,3))+(target*valid).sum((1,2,3))+1)
                    qpred=coverage(p,valid)
                    loss=bce+args.dice_weight*dice.mean()+args.coverage_weight*F.l1_loss(qpred,q)
                    if args.variant=='consistent_seg':
                        second_logits=model(degrade(x,random.choice([1,2,4]),valid),valid).float()
                        second=second_logits.sigmoid()
                        consistency=F.l1_loss(coverage(second,valid),qpred)
                        # Supervise both views to prevent a mutually consistent constant solution.
                        extra=(F.binary_cross_entropy_with_logits(second_logits,soft,reduction='none')*valid).sum()/valid.sum().clamp_min(1)
                        loss=loss+args.consistency_weight*consistency+.25*extra
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            scaler.step(optimizer);scaler.update();total+=float(loss.detach())*len(x);count+=len(x)
            if args.step_pause>0:time.sleep(args.step_pause)
        metrics=evaluate(model,val_loader,device,regression)
        metrics.update({'epoch':epoch,'train_loss':total/max(count,1),'elapsed_seconds':time.time()-start,'lr':optimizer.param_groups[0]['lr']})
        if metrics['selection_balanced_mae']<best:
            best=metrics['selection_balanced_mae'];best_epoch=epoch
            torch.save({'model':model.state_dict(),'config':cfg,'epoch':epoch,'validation':metrics},run_dir/'best.pt')
            (run_dir/'best_validation.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        with (run_dir/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(metrics)+'\n')
        print(json.dumps({'run':args.run_id,'epoch':epoch,'train_loss':round(metrics['train_loss'],6),'validation_balanced_mae':round(metrics['selection_balanced_mae'],6),'best':round(best,6),'seconds':round(metrics['elapsed_seconds'])}),flush=True)
        scheduler.step()
    final={'status':'completed','best_epoch':best_epoch,'best_validation_balanced_mae':best,'elapsed_seconds':time.time()-start,'test_accessed':False}
    (run_dir/'status.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    print(json.dumps(final),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True)
    ap.add_argument('--data-dir',default='local_data_v2')
    ap.add_argument('--variant',choices=['regression','hard_seg','soft_seg','consistent_seg'],required=True)
    ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--seed',type=int,default=17);ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--weight-decay',type=float,default=1e-4);ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--gpu-memory-fraction',type=float,default=.16);ap.add_argument('--step-pause',type=float,default=.05)
    ap.add_argument('--scale-augmentation',action='store_true');ap.add_argument('--coverage-weight',type=float,default=2.)
    ap.add_argument('--consistency-weight',type=float,default=2.);ap.add_argument('--pos-weight',type=float,default=1.)
    ap.add_argument('--dice-weight',type=float,default=0.)
    ap.add_argument('--max-train',type=int,default=0);ap.add_argument('--max-val',type=int,default=0)
    args=ap.parse_args()
    try:run(args)
    except Exception as e:
        run_dir=ROOT/'runs'/args.run_id;run_dir.mkdir(parents=True,exist_ok=True)
        (run_dir/'failure.json').write_text(json.dumps({'status':'failed','error':str(e),'traceback':traceback.format_exc()},indent=2),encoding='utf-8')
        raise

if __name__=='__main__':main()
