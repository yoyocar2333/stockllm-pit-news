"""Collect English public Apple articles and raw daily AAPL quotes, with caches.

Full article text and HTTP responses stay under ignored data/raw/. The public
manifest contains URLs and hashes, not redistributed article bodies.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import pandas as pd
import requests


def get_cached(session, url, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (hashlib.sha256(url.encode()).hexdigest() + '.txt')
    if path.exists():
        return path.read_text(encoding='utf-8')
    for attempt in range(3):
        try:
            response = session.get(url, timeout=35)
            response.raise_for_status()
            text = response.content.decode('utf-8')
            path.write_text(text, encoding='utf-8')
            time.sleep(.2)
            return text
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def parse_article(html, url):
    soup = BeautifulSoup(html, 'html.parser')
    headline = soup.select_one('h1.hero-headline')
    date = soup.select_one('.category-eyebrow__date')
    if not headline or not date:
        raise ValueError('Missing article headline/date')
    published = pd.to_datetime(date.get_text(' ', strip=True), format='%B %d, %Y').strftime('%Y-%m-%d')
    paragraphs = []
    for node in soup.select('article .pagebody-copy, .article .pagebody-copy'):
        text = node.get_text(' ', strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) >= 30 and text not in paragraphs:
            paragraphs.append(text)
    body = '\n\n'.join(paragraphs)
    if len(body) < 200:
        raise ValueError('Missing or insufficient article body')
    category = soup.select_one('.category-eyebrow__category')
    title = re.sub(r'\s+', ' ', headline.get_text(' ', strip=True)).strip()
    return dict(article_id='apple_' + hashlib.sha256(url.encode()).hexdigest()[:16],
                ticker='AAPL', target_name='Apple', publication_date=published,
                timestamp_precision='date_only', title=title, content=body,
                source='Apple Newsroom English', url=url,
                category=category.get_text(' ',strip=True) if category else 'unknown',
                text_sha256=hashlib.sha256((title+'\n\n'+body).encode()).hexdigest(),
                html_sha256=hashlib.sha256(html.encode()).hexdigest(),
                historical_text_revision_verified=False)


def collect_news(session, start, end, raw, output, max_pages):
    cache = raw / 'apple_http_cache'
    urls = set()
    archive_rows = []
    for page in range(1, max_pages + 1):
        archive_url = f'https://www.apple.com/newsroom/archive/?page={page}'
        html = get_cached(session, archive_url, cache)
        soup = BeautifulSoup(html, 'html.parser')
        links = [urljoin('https://www.apple.com', a['href']) for a in soup.select('a[href]')
                 if re.fullmatch(r'/newsroom/\d{4}/\d{2}/[^/?]+/?',a['href'])]
        if not links:
            raise ValueError('Archive page has no article links: ' + str(page))
        years = [int(re.search(r'/newsroom/(\d{4})/', url)[1]) for url in links]
        archive_rows.append(dict(page=page, url=archive_url, article_links=len(links),
                                 min_year=min(years), max_year=max(years),
                                 html_sha256=hashlib.sha256(html.encode()).hexdigest()))
        urls.update(url for url, year in zip(links, years) if start.year <= year <= end.year)
        if page % 10 == 0:
            print('Archive pages:',page,'candidate URLs:',len(urls),flush=True)
        if max(years) < start.year:
            break
    else:
        raise ValueError('Archive scan reached page limit before covering the start year')
    rows, failures = [], []
    for i, url in enumerate(sorted(urls), 1):
        try:
            row = parse_article(get_cached(session,url,cache),url)
            if start <= pd.Timestamp(row['publication_date']) <= end:
                rows.append(row)
        except (ValueError, requests.RequestException) as exc:
            failures.append(dict(url=url, reason=str(exc)))
        if i % 25 == 0:
            print('Articles:',i,'/',len(urls),'accepted:',len(rows),'excluded:',len(failures),flush=True)
    df=pd.DataFrame(rows).sort_values(['publication_date','article_id']).reset_index(drop=True)
    if df.empty or df.article_id.duplicated().any():
        raise ValueError('Invalid collected article population')
    df.to_csv(raw/'apple_newsroom_articles.csv',index=False,encoding='utf-8')
    manifest=df.drop(columns=['content','title'])
    manifest.to_csv(output/'article_manifest.csv',index=False)
    pd.DataFrame(archive_rows).to_csv(output/'archive_scan.csv',index=False)
    (output/'collection_exclusions.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
    return df


def collect_market(session, raw, output):
    begin=int(pd.Timestamp('2022-11-01',tz='UTC').timestamp())
    end=int(pd.Timestamp('2026-01-04',tz='UTC').timestamp())
    url=f'https://query1.finance.yahoo.com/v8/finance/chart/AAPL?period1={begin}&period2={end}&interval=1d&events=div%2Csplits'
    text=get_cached(session,url,raw/'market_http_cache')
    result=json.loads(text)['chart']['result'][0]
    quote=result['indicators']['quote'][0]
    dates=pd.to_datetime(result['timestamp'],unit='s',utc=True).tz_convert('America/New_York').strftime('%Y-%m-%d')
    df=pd.DataFrame({'Date':dates,**{key.title():values for key,values in quote.items()}})
    df=df[['Date','Open','High','Low','Close','Volume']].dropna().sort_values('Date')
    if df.Date.duplicated().any() or (df.Close <= 0).any():
        raise ValueError('Invalid market observations')
    df.to_csv(raw/'aapl_market.csv',index=False)
    provenance={'url':url,'sha256':hashlib.sha256(text.encode()).hexdigest(),
                'rows':len(df),'start':df.Date.min(),'end':df.Date.max(),
                'price_policy':'raw OHLC, unadjusted; no adjusted-close target',
                'events':result.get('events',{})}
    (output/'market_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    return df


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',default='data/raw/apple_public_v1')
    parser.add_argument('--output-dir',default='outputs/apple_public_v1')
    parser.add_argument('--start',default='2023-01-01')
    parser.add_argument('--end',default='2025-12-31')
    parser.add_argument('--max-pages',type=int,default=100)
    args=parser.parse_args()
    raw,output=Path(args.raw_dir),Path(args.output_dir)
    raw.mkdir(parents=True,exist_ok=True); output.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    session.headers.update({'User-Agent':'StockLLM university research reproduction (public articles and quotes)'})
    articles=collect_news(session,pd.Timestamp(args.start),pd.Timestamp(args.end),raw,output,args.max_pages)
    market=collect_market(session,raw,output)
    summary={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'articles':len(articles),
             'articles_by_year':articles.publication_date.str[:4].value_counts().sort_index().to_dict(),
             'market_rows':len(market), 'source_scope':'Apple corporate newsroom only',
             'timestamp_precision':'date only; earliest cutoff is next trading session after published date',
             'historical_text_revision_verified':False}
    (output/'collection_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
