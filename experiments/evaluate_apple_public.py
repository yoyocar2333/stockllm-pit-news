"""Frozen chronological E0-E4 exploratory evaluation on real public articles.

All preprocessing/model fitting uses 2023 only. Neither validation nor test is
used for hyperparameter selection. Date-only news uses the next market session.
"""
import argparse
import hashlib
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from experiments.infer_apple_public import EVENT_COLUMNS

SENT=['sent_positive','sent_negative','sent_neutral']
MARKET=['return_1','return_5','close_sma_5','close_sma_20','volatility_5','volatility_20','intraday_return','range_ratio','volume_ratio']


def validate_market(market):
    df=market.copy()
    df['Date']=pd.to_datetime(df.Date).dt.normalize()
    df=df.sort_values('Date').reset_index(drop=True)
    if df.Date.duplicated().any() or not np.isfinite(df[['Open','High','Low','Close','Volume']]).all().all():
        raise ValueError('Invalid market rows')
    if (df.Close<=0).any() or (df.Open<=0).any():
        raise ValueError('Prices must be positive')
    expected=xcals.get_calendar('XNYS').sessions_in_range(df.Date.min(),df.Date.max()).tz_localize(None)
    missing=expected.difference(pd.DatetimeIndex(df.Date))
    extra=pd.DatetimeIndex(df.Date).difference(expected)
    if len(missing) or len(extra):
        raise ValueError(f'Exchange-session mismatch: missing={list(missing)} extra={list(extra)}')
    return df


def market_features(market):
    df=market.copy()
    close=df.Close.astype(float)
    returns=close.pct_change()
    df['return_1']=returns
    df['return_5']=close.pct_change(5)
    df['close_sma_5']=close/close.rolling(5).mean()-1
    df['close_sma_20']=close/close.rolling(20).mean()-1
    df['volatility_5']=returns.rolling(5).std()
    df['volatility_20']=returns.rolling(20).std()
    df['intraday_return']=close/df.Open-1
    df['range_ratio']=(df.High-df.Low)/close
    df['volume_ratio']=df.Volume/df.Volume.rolling(20).mean()-1
    df['target_date']=df.Date.shift(-1)
    df['target']=(df.Close.shift(-1)>close).astype(int)
    return df.dropna(subset=MARKET+['target_date']).reset_index(drop=True)


def availability_positions(publication_dates, sessions):
    dates=pd.to_datetime(publication_dates).to_numpy(dtype='datetime64[ns]')
    # Strictly greater publication date. Do not pretend date-only means 00:00 UTC.
    return np.searchsorted(sessions.to_numpy(dtype='datetime64[ns]'),dates,side='right')


def daily_features(features, sessions, source, weighted=False, half_life=3, max_age=20):
    x=features[features.source==source].copy()
    if x.article_id.duplicated().any():
        raise ValueError('Duplicate feature article IDs')
    positions=availability_positions(x.publication_date,sessions)
    values=x[SENT+EVENT_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite news features')
    neutral=np.zeros(values.shape[1]); neutral[2]=1
    result=np.tile(neutral,(len(sessions),1))
    counts=np.zeros(len(sessions),dtype=int)
    usable=positions<len(sessions)
    positions,values=positions[usable],values[usable]
    weights=x.article_weight.to_numpy(dtype=float)[usable]
    for index in range(len(sessions)):
        same=positions==index
        counts[index]=int(same.sum())
        if not weighted:
            if same.any():
                result[index]=values[same].mean(axis=0)
        else:
            ages=index-positions
            active=(ages>=0)&(ages<max_age)
            if active.any():
                w=weights[active]*2.0**(-ages[active]/half_life)
                # Finite neutral prior ensures isolated stale news decays to zero
                # signal instead of cancelling decay through normalization.
                result[index]=neutral+(w[:,None]*(values[active]-neutral)).sum(axis=0)/(1.0+w.sum())
    result=pd.DataFrame(result,columns=SENT+EVENT_COLUMNS)
    result['Date']=sessions
    result['has_news']=(counts>0).astype(float)
    result['log_news_count']=np.log1p(counts)
    return result


def split_masks(df,config):
    masks={}
    for split in ('train','validation','test'):
        start=pd.Timestamp(config[split+'_start']); end=pd.Timestamp(config[split+'_end'])
        # A sample's next-session label must be available within its own split.
        masks[split]=(df.Date>=start)&(df.Date<=end)&(df.target_date<=end)
    if any(not mask.any() for mask in masks.values()):
        raise ValueError('Empty temporal split')
    if sum(mask.astype(int) for mask in masks.values()).max()>1:
        raise ValueError('Overlapping splits')
    return masks


def metrics(y,p):
    pred=(p>=.5).astype(int)
    return {'n':len(y),'up_days':int(y.sum()),'accuracy':float(accuracy_score(y,pred)),
            'f1':float(f1_score(y,pred,zero_division=0)),
            'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
            'ap':float(average_precision_score(y,p)) if len(np.unique(y))>1 else None,
            'brier':float(brier_score_loss(y,p))}


def bootstrap_indices(n,draws,block,seed):
    if n<block:
        raise ValueError('Test set smaller than bootstrap block')
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n-block+1,size=(draws,int(np.ceil(n/block))))
    return (starts[:,:,None]+np.arange(block)).reshape(draws,-1)[:,:n]


def paired_contrasts(predictions,config):
    test=predictions[predictions.split=='test']
    truth=test[test.arm=='E0'].sort_values('Date').target.to_numpy(dtype=int)
    probs={arm:rows.sort_values('Date').probability.to_numpy() for arm,rows in test.groupby('arm')}
    for arm in probs:
        arm_dates=test[test.arm==arm].sort_values('Date')[['Date','target_date','target']].reset_index(drop=True)
        reference=test[test.arm=='E0'].sort_values('Date')[['Date','target_date','target']].reset_index(drop=True)
        if not arm_dates.equals(reference):
            raise ValueError('Unpaired evaluation population')
    indices=bootstrap_indices(len(truth),config['bootstrap_draws'],config['bootstrap_block_sessions'],config['seed'])
    comparisons=[tuple(value.split('-')) for value in config['primary_contrasts']]
    rows=[]
    rng=np.random.default_rng(config['seed']+1)
    for arm,reference in comparisons:
        correct_a=((probs[arm]>=.5)==truth).astype(float)
        correct_b=((probs[reference]>=.5)==truth).astype(float)
        accuracy_delta=correct_a-correct_b
        brier_delta=(probs[arm]-truth)**2-(probs[reference]-truth)**2
        boot_acc=accuracy_delta[indices].mean(axis=1)
        boot_brier=brier_delta[indices].mean(axis=1)
        block=config['bootstrap_block_sessions']
        blocks=np.array([accuracy_delta[start:start+block].sum() for start in range(0,len(truth),block)])
        signs=rng.choice([-1,1],size=(config['bootstrap_draws'],len(blocks)))
        permuted=(signs*blocks).sum(axis=1)
        observed=abs(accuracy_delta.sum())
        p=(1+np.count_nonzero(np.abs(permuted)>=observed-1e-12))/(1+len(permuted))
        rows.append({'contrast':arm+'-'+reference,'n':len(truth),'accuracy_delta':float(accuracy_delta.mean()),
                     'accuracy_ci_low':float(np.quantile(boot_acc,.025)), 'accuracy_ci_high':float(np.quantile(boot_acc,.975)),
                     'brier_delta':float(brier_delta.mean()),'brier_ci_low':float(np.quantile(boot_brier,.025)),
                     'brier_ci_high':float(np.quantile(boot_brier,.975)), 'approx_block_sign_p':float(p)})
    order=np.argsort([row['approx_block_sign_p'] for row in rows]); running=0
    for rank,index in enumerate(order):
        running=max(running,(len(rows)-rank)*rows[index]['approx_block_sign_p'])
        rows[index]['holm_p']=min(1.0,running)
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--market',default='data/raw/apple_public_v1/aapl_market.csv')
    parser.add_argument('--features',default='outputs/apple_public_v1/article_features.csv')
    parser.add_argument('--output-dir',default='outputs/apple_public_v1')
    parser.add_argument('--config',default='experiments/apple_public_v1.json')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    market=validate_market(pd.read_csv(args.market))
    base=market_features(market)
    sessions=pd.DatetimeIndex(market.Date)
    features=pd.read_csv(args.features)
    if set(features.source)!={'title','full'}:
        raise ValueError('Both sources required')
    title_ids=set(features[features.source=='title'].article_id)
    full_ids=set(features[features.source=='full'].article_id)
    if title_ids!=full_ids:
        raise ValueError('Title/full article population differs')
    article_dates=features.pivot(index='article_id',columns='source',values='publication_date')
    if not article_dates.title.equals(article_dates.full):
        raise ValueError('Title/full publication dates differ')
    variants={
        'title':daily_features(features,sessions,'title'),
        'full':daily_features(features,sessions,'full'),
        'weighted':daily_features(features,sessions,'full',True,
                                  config['e4_half_life_sessions'],config['e4_max_age_sessions'])}
    assignments={'Market':'title','E0':'title','E1':'full','E2':'title','E3':'full','E4':'weighted'}
    masks=split_masks(base,config)
    all_predictions=[]; summaries=[]; fitted=[]
    y_train=base.loc[masks['train'],'target'].to_numpy(dtype=int)
    if len(np.unique(y_train))!=2:
        raise ValueError('Training target needs both classes')
    for arm,variant in assignments.items():
        dataset=base.merge(variants[variant],on='Date',how='left',validate='one_to_one')
        columns=MARKET.copy()
        if arm!='Market':
            columns+=SENT+['has_news','log_news_count']
        if arm in {'E2','E3','E4'}:
            columns+=EVENT_COLUMNS
        pipeline=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=config['seed']))
        pipeline.fit(dataset.loc[masks['train'],columns],y_train)
        probability=pipeline.predict_proba(dataset[columns])[:,1]
        scaler=pipeline.named_steps['standardscaler']; classifier=pipeline.named_steps['logisticregression']
        fitted.append({'arm':arm,'columns':columns,'train_scaler_mean':scaler.mean_.tolist(),
                       'train_scaler_scale':scaler.scale_.tolist(),'coefficients':classifier.coef_[0].tolist(),
                       'intercept':float(classifier.intercept_[0]),'train_rows':len(y_train)})
        for split,mask in masks.items():
            rows=dataset.loc[mask,['Date','target_date','target','has_news']].copy()
            rows['probability']=probability[mask.to_numpy()]; rows['arm']=arm; rows['split']=split
            all_predictions.append(rows)
            summaries.append({'arm':arm,'split':split,**metrics(rows.target.to_numpy(),rows.probability.to_numpy())})
    for arm,p in [('Always_up',1.0),('Always_down',0.0),('Neutral',.5),('Train_prior',float(y_train.mean()))]:
        for split,mask in masks.items():
            rows=base.loc[mask,['Date','target_date','target']].copy()
            rows['has_news']=variants['title'].set_index('Date').loc[rows.Date,'has_news'].to_numpy()
            rows['probability']=p; rows['arm']=arm; rows['split']=split
            all_predictions.append(rows)
            summaries.append({'arm':arm,'split':split,**metrics(rows.target.to_numpy(),rows.probability.to_numpy())})
    predictions=pd.concat(all_predictions,ignore_index=True)
    summary=pd.DataFrame(summaries)
    contrasts=paired_contrasts(predictions,config)
    baseline_config={**config,'primary_contrasts':config['secondary_baseline_contrasts']}
    baseline_contrasts=paired_contrasts(predictions,baseline_config)
    summary.to_csv(output/'metrics.csv',index=False)
    predictions.to_csv(output/'predictions.csv',index=False)
    contrasts.to_csv(output/'paired_contrasts.csv',index=False)
    baseline_contrasts.to_csv(output/'baseline_contrasts.csv',index=False)
    for variant,df in variants.items():
        df.to_csv(output/('daily_'+variant+'.csv'),index=False)
    (output/'fitted_models.json').write_text(json.dumps(fitted,indent=2),encoding='utf-8')
    split_summary={split:{'n':int(mask.sum()),'first':str(base.loc[mask,'Date'].min().date()),
                          'last':str(base.loc[mask,'Date'].max().date()),
                          'news_days':int(variants['title'].set_index('Date').loc[base.loc[mask,'Date'],'has_news'].sum())}
                   for split,mask in masks.items()}
    audit={'experiment_id':config['experiment_id'],'config_sha256':hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
           'market_sha256':hashlib.sha256(Path(args.market).read_bytes()).hexdigest(),
           'features_sha256':hashlib.sha256(Path(args.features).read_bytes()).hexdigest(),
           'article_count':len(title_ids),'calendar_validation':'all observed dates equal XNYS sessions',
           'paired_population':'identical cutoffs, targets and article IDs','splits':split_summary,
           'test_used_for_tuning':False,'refitted_on_validation':False,
           'fitted_feature_dimensions':{row['arm']:len(row['columns']) for row in fitted},
           'independent_human_gold_available':False,'study_status':config['status'],'limitations':config['limitations']}
    (output/'evaluation_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(summary[summary.split=='test'].to_string(index=False),flush=True)
    print(contrasts.to_string(index=False),flush=True)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    main()
