"""Summarize new out-of-fold experiments without treating frames/seeds as animals."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from project_paths import PROJECT_ROOT
from revision_core import PROBS,DEFAULT_VARIANTS,metrics,json_write,digest

CORE_METRICS=['acc5','macro_f1_5','qwk5','mae_bcs']
CONTRASTS={
 'angle_given_quality':{'VAQ':1,'VQ':-1},
 'angle_without_quality':{'VA':1,'V':-1},
 'quality_given_angle':{'VAQ':1,'VA':-1},
 'quality_without_angle':{'VQ':1,'V':-1},
 'angle_quality_interaction':{'VAQ':1,'VQ':-1,'VA':-1,'V':1},
 'multi_frame_vs_single_frame':{'V':1,'V_single':-1},
 'attention_vs_mean_pooling':{'V':1,'V_mean':-1},
 'auxiliary_supervision':{'VAQ':1,'VAQ_noaux':-1},
}

def fast_scores(y,p):
    cm=np.bincount(5*y+p,minlength=25).reshape(5,5).astype(float)
    n=cm.sum();tr=cm.sum(1);pr=cm.sum(0)
    denom=tr+pr;f=np.divide(2*np.diag(cm),denom,out=np.zeros(5),where=denom>0).mean()
    distance=np.subtract.outer(np.arange(5),np.arange(5))**2
    expected=np.outer(tr,pr)/n;de=(expected*distance).sum()
    qwk=1-(cm*distance).sum()/de if de>0 else np.nan
    return np.array([np.trace(cm)/n,f,qwk,float(np.abs(y-p).mean())])

def cluster_indices(meta,rng):
    indices=[]
    for _,part in meta.groupby('fold_id',sort=True):
        animals=sorted(part.animal_id.unique())
        for animal in rng.choice(animals,size=len(animals),replace=True):
            indices.extend(part.index[part.animal_id==animal].tolist())
    return np.asarray(indices,int)

def interval(x):
    finite=np.asarray(x)[np.isfinite(x)]
    if not len(finite):return np.nan,np.nan,0
    lo,hi=np.percentile(finite,[2.5,97.5]);return float(lo),float(hi),len(finite)

def calibration(df,bins=10):
    p=df[PROBS].to_numpy(float);y=df.true_label5.to_numpy(int)
    p=p/p.sum(1,keepdims=True)
    if not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid probabilities')
    confidence=p.max(1);correct=(p.argmax(1)==y)
    bucket=np.minimum((confidence*bins).astype(int),bins-1)
    rows=[];ece=0
    for b in range(bins):
        m=bucket==b
        if not m.any():continue
        gap=abs(float(correct[m].mean())-float(confidence[m].mean()))
        ece+=m.mean()*gap
        rows.append(dict(bin=b,lower=b/bins,upper=(b+1)/bins,n=int(m.sum()),accuracy=float(correct[m].mean()),mean_confidence=float(confidence[m].mean())))
    return dict(brier_multiclass=float(((p-np.eye(5)[y])**2).sum(1).mean()),
                nll=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),ece_10=float(ece)),rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=PROJECT_ROOT/'09_editor_revision_results')
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--bootstrap-seed',type=int,default=2209)
    p.add_argument('--allow-partial',action='store_true',help='Diagnostic only: do not use partial results in manuscript')
    a=p.parse_args()
    if a.bootstrap<20:p.error('Use at least 20 bootstrap repetitions; default 2000')
    out=a.results/'11_revision_tables';out.mkdir(exist_ok=True)
    files=sorted(a.results.glob('*/seed*/fold*/predictions.csv'))
    if not files:raise FileNotFoundError('No predictions. Run stage 09.')
    for f in files:
        done=f.parent/'completed.json'
        if not done.exists() or json.loads(done.read_text()).get('predictions.csv')!=digest(f):
            raise ValueError(f'Incomplete or changed stage 09 predictions: {f}')
    pred=pd.concat([pd.read_csv(f,dtype={'bag_id':str,'animal_id':str}) for f in files],ignore_index=True)
    expected=pd.read_csv(a.results/'eligible_episodes.csv',dtype={'bag_id':str,'animal_id':str})
    if pred.duplicated(['variant','seed','bag_id']).any():raise ValueError('Duplicate out-of-fold predictions')
    available=set(pred.variant)
    if not a.allow_partial and not set(DEFAULT_VARIANTS)<=available:raise ValueError('Complete the four component models and both aggregation controls first')
    groups={}
    for (variant,seed),df in pred.groupby(['variant','seed']):
        if not a.allow_partial and set(df.bag_id)!=set(expected.bag_id):raise ValueError(f'Incomplete OOF cohort: {variant}, {seed}')
        joined=df.merge(expected[['bag_id','animal_id','fold_id','bcs_raw']],on='bag_id',suffixes=('','_ref'),validate='one_to_one')
        if len(joined)!=len(df) or not (joined.animal_id==joined.animal_id_ref).all() or not (joined.fold_id==joined.fold_id_ref).all() or not (joined.true_bcs==joined.bcs_raw).all():raise ValueError('Metadata disagreement')
        groups[(variant,int(seed))]=df.sort_values('bag_id').reset_index(drop=True)
    seed_sets={v:set(pred[pred.variant==v].seed) for v in available}
    if len({tuple(sorted(s)) for s in seed_sets.values()})>1:raise ValueError('Models must have the same training seeds')
    seeds=sorted(next(iter(seed_sets.values())))
    first=next(iter(groups.values()));cohort=first.bag_id.tolist()
    if any(g.bag_id.tolist()!=cohort for g in groups.values()):raise ValueError('Models/seeds do not share exactly the same OOF episodes')
    meta=first[['animal_id','fold_id']].copy();y=first.true_label5.to_numpy(int)
    pool_rows=[];fold_rows=[];cal_rows=[];bins=[];class_rows=[]
    for (v,s),g in groups.items():
        m=metrics(g);pool_rows.append(dict(variant=v,seed=s,**m))
        for fold,sub in g.groupby('fold_id'):fold_rows.append(dict(variant=v,seed=s,fold_id=fold,**metrics(sub)))
        cm,br=calibration(g);cal_rows.append(dict(variant=v,seed=s,**cm));bins.extend(dict(variant=v,seed=s,**r) for r in br)
        for label in range(5):
            true=g.true_label5.to_numpy(int)==label;pr=g.pred_label5.to_numpy(int)==label
            class_rows.append(dict(variant=v,seed=s,bcs=label+2,support=int(true.sum()),predicted_count=int(pr.sum()),
                recall=float((true&pr).sum()/true.sum()) if true.any() else np.nan))
    pooled=pd.DataFrame(pool_rows);folds=pd.DataFrame(fold_rows)
    pooled.to_csv(out/'component_pooled_by_seed.csv',index=False)
    folds.to_csv(out/'component_by_fold.csv',index=False)
    pd.DataFrame(class_rows).to_csv(out/'per_class_support_recall.csv',index=False)
    pd.DataFrame(cal_rows).to_csv(out/'calibration_by_seed.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'calibration_bins.csv',index=False)
    fold_summary=folds.groupby(['variant','seed'])[CORE_METRICS+['within_one_acc','macro_f1_3']].agg(['mean','std'])
    fold_summary.columns=['_'.join(c) for c in fold_summary.columns];fold_summary.reset_index().to_csv(out/'component_fold_mean_sd_by_seed.csv',index=False)
    seed_summary=pooled.groupby('variant')[CORE_METRICS+['within_one_acc','macro_f1_3']].agg(['mean','std'])
    seed_summary.columns=['_'.join(c) for c in seed_summary.columns];seed_summary.reset_index().to_csv(out/'component_pooled_seed_mean_sd.csv',index=False)
    arrays={v:np.stack([groups[(v,int(s))].pred_label5.to_numpy(int) for s in seeds]) for v in available}
    contrasts={name:weights for name,weights in CONTRASTS.items() if set(weights)<=available}
    draws={name:[] for name in contrasts};base_scores={v:np.nanmean([fast_scores(y,p) for p in pp],axis=0) for v,pp in arrays.items()}
    rng=np.random.default_rng(a.bootstrap_seed)
    for rep in range(a.bootstrap):
        ix=cluster_indices(meta,rng)
        scores={v:np.nanmean([fast_scores(y[ix],p[ix]) for p in pp],axis=0) for v,pp in arrays.items()}
        for name,w in contrasts.items():draws[name].append(sum(weight*scores[v] for v,weight in w.items()))
    contrast_rows=[]
    for name,w in contrasts.items():
        point=sum(weight*base_scores[v] for v,weight in w.items());dd=np.asarray(draws[name])
        for j,metric in enumerate(CORE_METRICS):
            lo,hi,n=interval(dd[:,j]);contrast_rows.append(dict(contrast=name,metric=metric,estimate=point[j],ci_low=lo,ci_high=hi,valid_bootstrap=n,
                n_animals=meta.animal_id.nunique(),n_episodes=len(meta),n_seeds=len(seeds),positive_is_better=metric!='mae_bcs'))
    pd.DataFrame(contrast_rows).to_csv(out/'paired_animal_cluster_contrasts.csv',index=False)
    # Conditional uncertainty of fixed OOF models, not new-training or population-generalization uncertainty.
    absolute=[]
    rng=np.random.default_rng(a.bootstrap_seed)
    absolute_draws={v:[] for v in available}
    for rep in range(a.bootstrap):
        ix=cluster_indices(meta,rng)
        for v,pp in arrays.items():absolute_draws[v].append(np.nanmean([fast_scores(y[ix],p[ix]) for p in pp],axis=0))
    for v in available:
        for j,metric in enumerate(CORE_METRICS):
            lo,hi,n=interval(np.asarray(absolute_draws[v])[:,j]);absolute.append(dict(variant=v,metric=metric,estimate=base_scores[v][j],ci_low=lo,ci_high=hi,valid_bootstrap=n))
    pd.DataFrame(absolute).to_csv(out/'absolute_animal_cluster_intervals.csv',index=False)
    rel=a.results/'reliability';rel_status='NOT RUN'
    if (rel/'episode_stability.csv').exists():
        stability=pd.read_csv(rel/'episode_stability.csv',dtype={'bag_id':str,'animal_id':str})
        if stability.duplicated(['variant','seed','bag_id','frame_count']).any():raise ValueError('Duplicate stability summaries')
        if not a.allow_partial:
            for (v,s,k),g in stability.groupby(['variant','seed','frame_count']):
                if set(g.bag_id)!=set(cohort):raise ValueError('Incomplete stability model/seed/frame-count cohort')
            for v,g in stability.groupby('variant'):
                if set(g.seed)!=set(seeds):raise ValueError('Stability results omit training seeds; rerun stage 10')
        stability_cols=['repeat_agreement_with_reference','modal_consistency','mean_abs_bcs_change','expected_bcs_sd','mean_total_variation','repeated_accuracy','mean_unique_frames']
        output=[]
        for (v,k),g in stability.groupby(['variant','frame_count']):
            # Collapse repeat-derived quantities across seeds within episode BEFORE cluster resampling.
            e=g.groupby(['bag_id','animal_id','fold_id'])[stability_cols].mean().reset_index()
            if not a.allow_partial and set(e.bag_id)!=set(cohort):raise ValueError('Incomplete stability cohort')
            rng=np.random.default_rng(a.bootstrap_seed);data=e[stability_cols].to_numpy(float);dist=[]
            for _ in range(a.bootstrap):dist.append(data[cluster_indices(e,rng)].mean(0))
            dist=np.asarray(dist)
            for j,key in enumerate(stability_cols):
                lo,hi,n=interval(dist[:,j]);output.append(dict(variant=v,frame_count=k,metric=key,estimate=data[:,j].mean(),ci_low=lo,ci_high=hi,n_episodes=len(e),valid_bootstrap=n))
        pd.DataFrame(output).to_csv(out/'episode_stability_summary.csv',index=False)
        half=pd.read_csv(rel/'disjoint_halves.csv',dtype={'bag_id':str,'animal_id':str});hrows=[]
        if half.duplicated(['variant','seed','bag_id','half']).any():raise ValueError('Duplicate half predictions')
        if not a.allow_partial:
            for v,g in half.groupby('variant'):
                if set(g.seed)!=set(seeds):raise ValueError('Half results omit training seeds; rerun stage 10')
        raw=pd.read_csv(a.results/'all_episodes.csv')
        for (v,s),g in half.groupby(['variant','seed']):
            if not a.allow_partial and set(g.bag_id)!=set(cohort):raise ValueError('Incomplete split-half cohort')
            common=g[g.eligible==1].groupby('bag_id').size();paired=set(common[common==2].index)
            f=g[(g.half=='first') & g.bag_id.isin(paired)].sort_values('bag_id')
            l=g[(g.half=='second') & g.bag_id.isin(paired)].sort_values('bag_id')
            report=dict(variant=v,seed=s,n_full_eligible=g.bag_id.nunique(),n_both_halves_eligible=len(paired),
                paired_fraction_of_full_eligible=len(paired)/g.bag_id.nunique(),
                paired_fraction_of_all_episodes=len(paired)/len(raw))
            if paired:
                yp=f.pred_label5.to_numpy(int);yl=l.pred_label5.to_numpy(int)
                report.update(exact_half_agreement=float(np.mean(yp==yl)),within_one_half_agreement=float(np.mean(np.abs(yp-yl)<=1)),
                    mean_absolute_half_bcs_difference=float(np.abs(yp-yl).mean()))
                matched_full=groups[(v,int(s))]
                matched_full=matched_full[matched_full.bag_id.isin(paired)]
                for name,sub in [('first',f),('second',l),('full_matched',matched_full)]:
                    report.update({name+'_'+key:val for key,val in metrics(sub).items()})
                # Cluster-bootstrap the agreement; halves are paired within episode.
                paired_meta=f[['animal_id','fold_id']].reset_index(drop=True);rng=np.random.default_rng(a.bootstrap_seed)
                agr=(yp==yl).astype(float);d=[]
                for _ in range(a.bootstrap):d.append(agr[cluster_indices(paired_meta,rng)].mean())
                report['agreement_ci_low'],report['agreement_ci_high'],_=interval(d)
            hrows.append(report)
        pd.DataFrame(hrows).to_csv(out/'disjoint_half_summary.csv',index=False)
        rel_status='available'
    json_write(out/'analysis_record.json',dict(n_animals=int(meta.animal_id.nunique()),n_episodes=len(meta),seeds=[int(x) for x in seeds],
        bootstrap=a.bootstrap,bootstrap_seed=a.bootstrap_seed,allow_partial=a.allow_partial,reliability=rel_status,
        prediction_sha256={str(f.relative_to(a.results)):digest(f) for f in files},
        uncertainty='95% percentile intervals; animals resampled within test folds; same draw for all paired models/seeds; fixed fitted models',
        primary_contrast='angle_given_quality on Acc5',secondary='Other metrics/contrasts are exploratory; no multiplicity-adjusted significance claim'))
    print('Tables saved:',out)
    print('Reliability:',rel_status,'; these are NEW runs, not replacements for original Table 8 scores.')

if __name__=='__main__':main()
