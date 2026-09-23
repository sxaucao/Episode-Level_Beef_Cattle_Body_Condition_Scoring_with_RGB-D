"""Held-out prediction sensitivity: frame resampling and disjoint episode halves."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import torch
from project_paths import PROJECT_ROOT
from revision_core import (VARIANTS,PROBS,ComponentModel,base,load_data,digest,stable_seed,
                           select_indices,load_checkpoint,json_write)

@torch.no_grad()
def encode_episode(model,row,args,device,batch=32):
    chunks=[]
    for start in range(0,len(row.rgb_path),batch):
        rr=row.rgb_path[start:start+batch];dd=row.depth_npy_path[start:start+batch]
        rgb=torch.from_numpy(np.stack([base.load_rgb(p,args.img_size) for p in rr])).unsqueeze(0).to(device)
        dep=torch.from_numpy(np.stack([base.load_depth(p,args.img_size,args.depth_crop_ratio) for p in dd])).unsqueeze(0).to(device)
        chunks.append(model.encode_frames(rgb,dep).squeeze(0).cpu())
    return torch.cat(chunks,dim=0)

@torch.no_grad()
def probabilities(model,features,indices,raw_features,mean,std,device):
    # Encoding is cached only in eval mode; no dropout or batch-stat updates.
    f=features[indices].unsqueeze(0).to(device)
    g=torch.tensor((np.asarray(raw_features,dtype=np.float32)-mean)/std).unsqueeze(0).to(device)
    return torch.softmax(model.from_features(f,g)['logits5'],1).squeeze(0).cpu().numpy()

def profile_map(profile_csv,qc):
    prof=pd.read_csv(profile_csv,dtype={'bag_id':str,'frame_key':str})
    required={'bag_id','frame_key','is_good_angle_frame','angle'}
    if not required<=set(prof):raise ValueError('Profile QC missing required columns')
    prof=prof[prof.is_good_angle_frame==1].copy()
    result={}
    for key,g in prof.groupby(['bag_id','frame_key']):
        v=g.angle.to_numpy(float)
        if not np.isfinite(v).all():raise ValueError(f'Nonfinite profile angles: {key}')
        result[key]=v
    for r in qc[qc.is_good_angle_frame==1].itertuples():
        if (r.bag_id,r.frame_key) not in result or len(result[(r.bag_id,r.frame_key)])!=int(r.n_angles):
            raise ValueError(f'Incomplete profile export: {r.bag_id}/{r.frame_key}; regenerate stage 03')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=PROJECT_ROOT/'09_editor_revision_results')
    p.add_argument('--profile-csv',type=Path,default=PROJECT_ROOT/'03_quality_controlled_dorsal_angle/03_profile_fit_quality.csv')
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=['VAQ','V','V_single','V_mean'])
    p.add_argument('--repeats',type=int,default=50)
    p.add_argument('--frame-counts',nargs='+',type=int,default=[1,3,6,12])
    p.add_argument('--resampling-seed',type=int,default=1709)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--encode-batch',type=int,default=32)
    p.add_argument('--allow-partial',action='store_true',help='For smoke tests only; not a manuscript analysis')
    a=p.parse_args()
    if a.repeats<2 or min(a.frame_counts)<1 or a.encode_batch<1:p.error('Invalid repeats/frame count/batch')
    config=json.loads((a.results/'protocol.json').read_text())
    for k,path in config['input_paths'].items():
        if digest(path)!=config['input_sha256'][k]:raise ValueError('Training input changed: '+path)
    for name,h in config['code_sha256'].items():
        if digest(PROJECT_ROOT/name)!=h:raise ValueError('Training code changed: '+name)
    args=SimpleNamespace(**config)
    raw,qc,bags=load_data(**config['input_paths'],min_good_frames=args.min_good_frames)
    amap=profile_map(a.profile_csv,qc)
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    destination=a.results/'reliability';destination.mkdir(exist_ok=True)
    json_write(destination/'protocol.json',dict(repeats=a.repeats,frame_counts=a.frame_counts,
        resampling_seed=a.resampling_seed,profile_sha256=digest(a.profile_csv),
        stage10_sha256=digest(__file__),variants=a.variants,allow_partial=a.allow_partial,
        geometry_for_resampling='fixed from complete episode',
        geometry_for_halves='recomputed within each disjoint half; original frame QC labels retained',
        inference='float32 eval-mode cached frame embeddings',order='natural frame_key; chronology must be checked by author'))
    instability=[];halves=[];baseline_rows=[];resampled=[]
    for variant in a.variants:
        files=sorted((a.results/variant).glob('seed*/fold*/best.pt'))
        if not files:raise FileNotFoundError(f'No stage 09 checkpoints for {variant}')
        if not a.allow_partial:
            groups={}
            for f in files:groups.setdefault(f.parent.parent.name,set()).add(f.parent.name)
            if any(x!={'fold0','fold1','fold2'} for x in groups.values()):raise ValueError('Incomplete three-fold checkpoints')
        for path in files:
            done=path.parent/'completed.json'
            if not done.exists() or json.loads(done.read_text()).get('best.pt')!=digest(path):
                raise ValueError(f'Incomplete or changed checkpoint: {path}')
            ck=load_checkpoint(path)
            if ck['config']!=config:raise ValueError('Checkpoint protocol mismatch')
            model=ComponentModel(variant).to(device);model.load_state_dict(ck['state_dict']);model.eval()
            mean=np.asarray(ck['angle_mean'],np.float32);std=np.asarray(ck['angle_std'],np.float32)
            test=bags[bags.bag_id.isin(ck['test_ids'])]
            if set(test.bag_id)!=set(ck['test_ids']):raise ValueError('Checkpoint test cohort mismatch')
            if set(test.animal_id)&set(bags[bags.bag_id.isin(ck['train_ids']+ck['val_ids'])].animal_id):raise ValueError('Test animal leakage')
            saved=pd.read_csv(path.parent/'predictions.csv').set_index('bag_id')
            cfg=VARIANTS[variant]
            for row in test.itertuples():
                feats=encode_episode(model,row,args,device,a.encode_batch)
                full_angles=np.concatenate([amap[(row.bag_id,k)] for k in row.frame_keys])
                if not np.isclose(np.median(full_angles),row.bag_good_angle_median,atol=1e-3):
                    raise ValueError(f'Profile median inconsistent with training QC: {row.bag_id}')
                geo=[row.bag_good_angle_median,row.good_frame_ratio,np.log1p(row.good_angle_frame_count)]
                full_idx=select_indices(len(feats),cfg['frames'])
                prob=probabilities(model,feats,full_idx,geo,mean,std,device)
                discrepancy=float(np.max(np.abs(prob-saved.loc[row.bag_id,PROBS].to_numpy(float))))
                if discrepancy>.02:raise ValueError(f'Cached baseline differs materially from stage 09: {row.bag_id}: {discrepancy}')
                meta=dict(bag_id=row.bag_id,animal_id=row.animal_id,fold_id=ck['fold'],seed=ck['seed'],variant=variant,true_label5=row.label5)
                baseline_rows.append(dict(**meta,pred_label5=int(prob.argmax()),max_probability_difference_from_stage09=discrepancy,
                                          **dict(zip(PROBS,prob.astype(float)))))
                for k in a.frame_counts:
                    pp=[];unique=[]
                    for rep in range(a.repeats):
                        rng=np.random.default_rng(stable_seed(a.resampling_seed,row.bag_id,k,rep))
                        ids=select_indices(len(feats),k,True,rng)
                        pr=probabilities(model,feats,ids,geo,mean,std,device);pp.append(pr);unique.append(len(set(ids)))
                        resampled.append(dict(**meta,frame_count=k,repeat=rep,pred_label5=int(pr.argmax()),n_unique_frames=len(set(ids)),**dict(zip(PROBS,pr.astype(float)))))
                    pp=np.asarray(pp);pred=pp.argmax(1);expected=pp@np.arange(2,7)
                    instability.append(dict(**meta,frame_count=k,available_good_frames=len(feats),mean_unique_frames=float(np.mean(unique)),
                        repeat_agreement_with_reference=float(np.mean(pred==prob.argmax())),
                        modal_consistency=float(np.bincount(pred,minlength=5).max()/len(pred)),
                        mean_abs_bcs_change=float(np.abs(pred-int(prob.argmax())).mean()),
                        expected_bcs_sd=float(expected.std(ddof=1)),
                        mean_total_variation=float((.5*np.abs(pp-prob).sum(axis=1)).mean()),
                        repeated_accuracy=float(np.mean(pred==row.label5))))
                # Split ALL frames, including QC failures; recompute Q and pooled angle per half.
                fr=qc[qc.bag_id==row.bag_id].copy()
                fr=fr.iloc[sorted(range(len(fr)),key=lambda i:__import__('revision_core').natural_key(fr.iloc[i].frame_key))]
                mid=len(fr)//2
                for half,part in [('first',fr.iloc[:mid]),('second',fr.iloc[mid:])]:
                    good=part[part.is_good_angle_frame==1]
                    valid=len(good)>=args.min_good_frames
                    h=dict(**meta,half=half,n_all_frames=len(part),n_good_frames=len(good),eligible=int(valid),reference_pred_label5=int(prob.argmax()))
                    if valid:
                        keys=good.frame_key.tolist();ids=[row.frame_keys.index(key) for key in keys]
                        angle=float(np.median(np.concatenate([amap[(row.bag_id,key)] for key in keys])))
                        hgeo=[angle,len(good)/len(part),np.log1p(len(good))]
                        sampled=select_indices(len(ids),cfg['frames'])
                        hp=probabilities(model,feats,[ids[i] for i in sampled],hgeo,mean,std,device)
                        h.update(pred_label5=int(hp.argmax()),angle=angle,good_frame_ratio=len(good)/len(part),**dict(zip(PROBS,hp.astype(float))))
                    halves.append(h)
            print('Reliability completed:',variant,'seed',ck['seed'],'fold',ck['fold'],flush=True)
    for name,rows in [('baseline_predictions',baseline_rows),('resampling_predictions',resampled),('episode_stability',instability),('disjoint_halves',halves)]:
        pd.DataFrame(rows).to_csv(destination/(name+'.csv'),index=False)
    print('Saved:',destination)

if __name__=='__main__':main()
