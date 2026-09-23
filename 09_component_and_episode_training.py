"""Train matched component ablations and episode aggregation controls (new results)."""
import argparse
import json
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from project_paths import PROJECT_ROOT
from revision_core import (base,VARIANTS,DEFAULT_VARIANTS,ComponentModel,EpisodeDataset,
    load_data,fixed_splits,digest,stable_seed,json_write,seed_everything,metrics,selection_score,predict,collate_bags)

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=PROJECT_ROOT/'09_editor_revision_results')
    p.add_argument('--frame-csv',type=Path,default=base.FRAME_CSV)
    p.add_argument('--qc-frame-csv',type=Path,default=base.QC_FRAME_CSV)
    p.add_argument('--qc-bag-csv',type=Path,default=base.QC_BAG_CSV)
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=DEFAULT_VARIANTS)
    p.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
    p.add_argument('--folds',nargs='+',type=int,choices=[0,1,2],default=[0,1,2])
    p.add_argument('--split-seed',type=int,default=2026)
    p.add_argument('--epochs',type=int,default=70)
    p.add_argument('--img-size',type=int,default=160)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--depth-crop-ratio',type=float,default=.70)
    p.add_argument('--min-good-frames',type=int,default=3)
    p.add_argument('--val-fraction',type=float,default=.22)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--no-amp',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--audit-only',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.img_size,a.batch_size,a.min_good_frames)<1 or a.num_workers<0: p.error('Invalid numeric settings')
    if not 0<a.val_fraction<1:p.error('Validation fraction must be between 0 and 1')
    return a

def main():
    a=parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    raw,qc,bags=load_data(a.frame_csv,a.qc_frame_csv,a.qc_bag_csv,a.min_good_frames)
    splits=fixed_splits(bags,a.split_seed,a.val_fraction)
    input_paths={k:str(getattr(a,k).resolve()) for k in ['frame_csv','qc_frame_csv','qc_bag_csv']}
    conf={k:v for k,v in vars(a).items() if k not in ['output','resume','audit_only','variants','seeds','folds','frame_csv','qc_frame_csv','qc_bag_csv']}
    conf.update(input_paths=input_paths,input_sha256={k:digest(v) for k,v in input_paths.items()},
                code_sha256={f:digest(PROJECT_ROOT/f) for f in ['revision_core.py','09_component_and_episode_training.py','04_train_full_ablation.py']},
                metric_policy='fixed five and three label sets; undefined QWK contributes zero only to checkpoint score',
                frame_order='natural sort of frame_key, not independently verified timestamps')
    config_path=a.output/'protocol.json'
    if config_path.exists() and json.loads(config_path.read_text())!=conf:raise RuntimeError('Protocol/data/code changed: use a new output directory')
    json_write(config_path,conf)
    json_write(a.output/'environment.json',dict(python=sys.version,platform=platform.platform(),torch=str(torch.__version__),
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    manifest=[];class_rows=[]
    for fold,parts in splits.items():
        for name,df in zip(['train','validation','test'],parts):
            for _,r in df.iterrows():manifest.append(dict(test_fold=fold,role=name,bag_id=r.bag_id,animal_id=r.animal_id,outer_fold=r.fold_id,bcs_raw=r.bcs_raw))
            for b in range(2,7):class_rows.append(dict(test_fold=fold,role=name,bcs_raw=b,n_episodes=int((df.bcs_raw==b).sum())))
    pd.DataFrame(manifest).to_csv(a.output/'split_manifest.csv',index=False)
    pd.DataFrame(class_rows).to_csv(a.output/'split_class_counts.csv',index=False)
    bags.drop(columns=['rgb_path','depth_npy_path','frame_keys']).to_csv(a.output/'eligible_episodes.csv',index=False)
    raw.groupby('bag_id')[['animal_id','fold_id','bcs_raw']].first().reset_index().to_csv(a.output/'all_episodes.csv',index=False)
    print(f'Audit: {len(bags)}/{raw.bag_id.nunique()} eligible episodes; {bags.animal_id.nunique()} eligible animals',flush=True)
    if a.audit_only:return
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Install a CUDA build; --device cpu is for small smoke tests')
    device=torch.device(a.device);amp=not a.no_amp and device.type=='cuda'
    for variant in a.variants:
        cfg=VARIANTS[variant]
        for seed in a.seeds:
            for fold in a.folds:
                folder=a.output/variant/f'seed{seed}'/f'fold{fold}';folder.mkdir(parents=True,exist_ok=True)
                done=folder/'completed.json'
                if done.exists():
                    if not a.resume:raise RuntimeError(f'{folder} already complete; use --resume or a new output')
                    check=json.loads(done.read_text())
                    for name,h in check.items():
                        if not (folder/name).exists() or digest(folder/name)!=h:raise RuntimeError(f'Changed/incomplete output: {folder/name}')
                    print('Resume skip:',folder,flush=True);continue
                train,val,test=splits[fold]
                model_seed=stable_seed(seed,fold,'model');seed_everything(model_seed)
                model=ComponentModel(variant).to(device)
                mean,std=base.fit_angle_scaler(train)
                val_ds=EpisodeDataset(val,mean,std,a,cfg['frames'])
                test_ds=EpisodeDataset(test,mean,std,a,cfg['frames'])
                val_loader=DataLoader(val_ds,batch_size=a.batch_size,num_workers=a.num_workers,collate_fn=collate_bags)
                test_loader=DataLoader(test_ds,batch_size=a.batch_size,num_workers=a.num_workers,collate_fn=collate_bags)
                optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
                scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs)
                scaler=torch.amp.GradScaler('cuda',enabled=amp)
                ce_weight=base.class_weights_from_bags(train,device)
                counts=train.label5.value_counts();weights=[1/counts[int(x)] for x in train.label5]
                best=-float('inf');best_state=None;best_epoch=0;history=[]
                for epoch in range(1,a.epochs+1):
                    epoch_seed=stable_seed(seed,fold,epoch)
                    seed_everything(epoch_seed)
                    train_ds=EpisodeDataset(train,mean,std,a,cfg['frames'],True,epoch_seed)
                    generator=torch.Generator().manual_seed(epoch_seed)
                    sampler=WeightedRandomSampler(weights,len(weights),replacement=True,generator=generator)
                    loader=DataLoader(train_ds,batch_size=a.batch_size,sampler=sampler,num_workers=a.num_workers,
                                      collate_fn=collate_bags,pin_memory=device.type=='cuda')
                    loss=base.train_one_epoch(model,loader,optimizer,scaler,device,ce_weight,cfg['aux'],amp)
                    scheduler.step()
                    vp=predict(model,val_loader,device,amp);vm=metrics(vp);score=selection_score(vp)
                    history.append(dict(epoch=epoch,loss=loss,score=score,**vm))
                    if score>best:
                        best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    if epoch==1 or epoch%5==0 or epoch==a.epochs:print(f'{variant} seed={seed} fold={fold} epoch={epoch}/{a.epochs} loss={loss:.4f} score={score:.4f} best={best_epoch}',flush=True)
                if best_state is None:raise RuntimeError('No finite validation checkpoint')
                model.load_state_dict(best_state)
                pred=predict(model,test_loader,device,amp)
                pred=pred.merge(test[['bag_id','animal_id','fold_id']],on='bag_id',validate='one_to_one')
                pred['variant']=variant;pred['seed']=seed
                pred.to_csv(folder/'predictions.csv',index=False)
                fm=metrics(pred);fm.update(variant=variant,seed=seed,fold_id=fold,best_epoch=best_epoch,
                    raw_test_episodes=int(raw[raw.fold_id==fold].bag_id.nunique()),eligible_test_episodes=len(test))
                fm['coverage']=len(test)/fm['raw_test_episodes']
                pd.DataFrame([fm]).to_csv(folder/'metrics.csv',index=False)
                pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
                checkpoint=dict(state_dict=best_state,variant=variant,seed=seed,fold=fold,best_epoch=best_epoch,
                    angle_mean=mean.tolist(),angle_std=std.tolist(),config=conf,
                    train_ids=train.bag_id.tolist(),val_ids=val.bag_id.tolist(),test_ids=test.bag_id.tolist())
                torch.save(checkpoint,folder/'best.pt')
                json_write(done,{n:digest(folder/n) for n in ['predictions.csv','metrics.csv','history.csv','best.pt']})
                del model
                if device.type=='cuda':torch.cuda.empty_cache()
    print('Training complete:',a.output)

if __name__=='__main__':main()
