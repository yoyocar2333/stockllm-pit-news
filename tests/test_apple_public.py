import json
import numpy as np
import pandas as pd
import pytest
from experiments.collect_apple_newsroom import parse_article
from experiments.evaluate_apple_public import availability_positions, daily_features, market_features, split_masks, validate_market
from experiments.infer_apple_public import EVENT_COLUMNS, event_features


def feature_row(article_id,date,source='title',sentiment=1.0):
    return {'article_id':article_id,'publication_date':date,'source':source,
            'sent_positive':sentiment,'sent_negative':0.0,'sent_neutral':1-sentiment,
            'article_weight':1.0,**{column:0.0 for column in EVENT_COLUMNS}}


def test_date_only_news_excludes_same_day_and_maps_holiday():
    sessions=pd.DatetimeIndex(pd.to_datetime(['2024-07-03','2024-07-05','2024-07-08']))
    assert availability_positions(['2024-07-03','2024-07-04','2024-07-06'],sessions).tolist()==[1,1,2]


def test_missing_exchange_session_is_not_silently_skipped():
    market=pd.DataFrame({'Date':['2024-07-03','2024-07-08'],'Open':[100,101],'High':[102,103],
                         'Low':[98,99],'Close':[100,101],'Volume':[1000,1000]})
    with pytest.raises(ValueError,match='Exchange-session mismatch'):
        validate_market(market)


def test_news_does_not_arrive_early_or_carry_in_unweighted_arms():
    sessions=pd.bdate_range('2024-01-02',periods=4)
    features=pd.DataFrame([feature_row('a','2024-01-02')])
    daily=daily_features(features,sessions,'title')
    assert daily.sent_positive.tolist()==[0,1,0,0]
    assert daily.has_news.tolist()==[0,1,0,0]


def test_weighted_stale_news_decays_and_expires():
    sessions=pd.bdate_range('2024-01-02',periods=8)
    features=pd.DataFrame([feature_row('a','2024-01-01')])
    daily=daily_features(features,sessions,'title',weighted=True,half_life=1,max_age=3)
    assert .5==daily.loc[0,'sent_positive']>daily.loc[1,'sent_positive']>daily.loc[2,'sent_positive']>0
    assert daily.loc[3:,'sent_positive'].eq(0).all()
    assert daily.loc[3:,'sent_neutral'].eq(1).all()


def test_future_news_does_not_change_past_features():
    sessions=pd.bdate_range('2024-01-02',periods=5)
    base=pd.DataFrame([feature_row('a','2024-01-01')])
    future=pd.concat([base,pd.DataFrame([feature_row('b','2024-01-05',sentiment=0)])],ignore_index=True)
    old=daily_features(base,sessions,'title',True)
    new=daily_features(future,sessions,'title',True)
    pd.testing.assert_frame_equal(old.iloc[:4],new.iloc[:4])


def test_future_prices_do_not_change_past_market_inputs():
    market=pd.DataFrame({'Date':pd.bdate_range('2023-01-02',periods=60),
                         'Open':np.arange(100,160),'High':np.arange(102,162),
                         'Low':np.arange(99,159),'Close':np.arange(101,161),'Volume':1000})
    from experiments.evaluate_apple_public import MARKET
    first=market_features(market)
    changed=market.copy(); changed.loc[50:,'Close']=changed.loc[50:,'Close']*10
    second=market_features(changed)
    pd.testing.assert_frame_equal(first[first.Date<market.loc[49,'Date']][MARKET],
                                  second[second.Date<market.loc[49,'Date']][MARKET])


def test_cross_split_labels_are_purged():
    config={split+'_start':value for split,value in [('train','2023-01-01'),('validation','2024-01-01'),('test','2025-01-01')]}
    config.update({split+'_end':value for split,value in [('train','2023-12-31'),('validation','2024-12-31'),('test','2025-12-31')]})
    frame=pd.DataFrame({'Date':pd.to_datetime(['2023-12-28','2023-12-29','2024-12-30','2024-12-31','2025-01-02']),
                        'target_date':pd.to_datetime(['2023-12-29','2024-01-02','2024-12-31','2025-01-02','2025-01-03'])})
    masks=split_masks(frame,config)
    assert masks['train'].tolist()==[True,False,False,False,False]
    assert masks['validation'].tolist()==[False,False,True,False,False]


def test_event_hashes_are_stable_and_role_importance_is_text_only():
    event={'actor':'Apple','action':'sue','object':'Samsung','event_type':'litigation','target_role':'actor','extraction_status':'ok'}
    first,weight,count=event_features([event])
    second,_,_=event_features([event,dict(event)])
    assert weight==2 and count==1
    assert first['target_role_actor']==1
    assert all(first[k]==second[k] for k in EVENT_COLUMNS if k!='event_log_count')


def test_collector_extracts_prose_without_press_contacts_or_navigation():
    html='''<h1 class="hero-headline">Synthetic parser fixture</h1><span class="category-eyebrow__date">February 1, 2024</span>
    <article class="article"><div class="pagebody-copy">'''+'A factual body paragraph. '*12+'''</div><div class="presscontacts">contact@example.com</div></article>
    <footer>Navigation not body</footer>'''
    row=parse_article(html,'https://www.apple.com/newsroom/2024/02/fixture/')
    assert row['publication_date']=='2024-02-01'
    assert row['timestamp_precision']=='date_only'
    assert 'contact@example.com' not in row['content'] and 'Navigation' not in row['content']
