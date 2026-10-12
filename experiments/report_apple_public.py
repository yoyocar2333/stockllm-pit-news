"""Generate an evidence-linked Chinese report and exportable comparison figure."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='outputs/apple_public_v1')
    args=parser.parse_args()
    folder=Path(args.output_dir)
    collection=json.loads((folder/'collection_summary.json').read_text())
    audit=json.loads((folder/'evaluation_audit.json').read_text())
    inference=json.loads((folder/'inference_audit.json').read_text())
    registry=json.loads((folder/'model_registry.json').read_text())
    metrics=pd.read_csv(folder/'metrics.csv')
    test=metrics[metrics.split=='test'].set_index('arm')
    primary=pd.read_csv(folder/'paired_contrasts.csv')
    baseline=pd.read_csv(folder/'baseline_contrasts.csv')
    rows=[]
    for arm in ['Market','E0','E1','E2','E3','E4','Always_up','Always_down','Neutral','Train_prior']:
        m=test.loc[arm]
        rows.append(f"| {arm} | {m.accuracy:.2%} | {m.f1:.4f} | {m.roc_auc:.4f} | {m.ap:.4f} | {m.brier:.4f} |")
    comparisons=[]
    for row in primary.itertuples():
        comparisons.append(f"| {row.contrast} | {100*row.accuracy_delta:+.2f} | [{100*row.accuracy_ci_low:+.2f}, {100*row.accuracy_ci_high:+.2f}] | {row.brier_delta:+.4f} | {row.holm_p:.4f} |")
    split_rows=[]
    for split,record in audit['splits'].items():
        split_rows.append(f"| {split} | {record['first']} 至 {record['last']} | {record['n']} | {record['news_days']} |")
    n=int(test.loc['E0','n'])
    body_vs_market=baseline[baseline.contrast=='E1-Market'].iloc[0]
    event_brier=primary[primary.contrast=='E3-E1'].iloc[0]
    report=f'''# AAPL 真實公開資料：E0–E4 第一輪結果

這輪已完成真實文章的收集、本地情緒／事件推論與固定設定的回溯預測評估。
它是探索性實驗，尚未具備獨立人工 gold 標註或當年內文版本的證明。

## 結果與解讀

2025 年的 {n} 個相同測試交易日中，E1（標題加擷取內文、無事件特徵）點估計最高，
Accuracy={test.loc['E1','accuracy']:.2%}；固定每天預測上漲為 {test.loc['Always_up','accuracy']:.2%}，
相差 {100*(test.loc['E1','accuracy']-test.loc['Always_up','accuracy']):.2f} 個百分點。
相對純量價 Market，E1 高 {100*body_vs_market.accuracy_delta:.2f} 個百分點，
95% 配對區塊 bootstrap 區間為 [{100*body_vs_market.accuracy_ci_low:.2f}, {100*body_vs_market.accuracy_ci_high:.2f}] 個百分點，
此基準比較家族的 Holm 校正 p={body_vs_market.holm_p:.4f}。目前不能主張有可靠的新增預測優勢。

事件特徵未帶來穩健改善：E2 的 Accuracy 與 E0 相同；E3 低於 E1；
E4 相對 E3 的改善區間涵蓋零。E3 的 Brier 比 E1 高 {event_brier.brier_delta:.4f}
（越低越好），95% 區間 [{event_brier.brier_ci_low:.4f}, {event_brier.brier_ci_high:.4f}]。
這是應保留的負面／不確定結果，不能用測試年重新調權重來改寫本輪結論。

| 組別 | Accuracy | F1 | ROC-AUC | AP | Brier（低較好） |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Neutral 的 probability=0.5，固定閾值 >=0.5，因此分類結果也是全部預測上漲。
Train_prior 的機率只取訓練年上漲比例，並未取測試年比例。

## 真實資料與切分

- [Apple Newsroom 英文公開存檔](https://www.apple.com/newsroom/archive/)：{collection['articles']} 篇，
  2023／2024／2025 年分別為 {collection['articles_by_year']['2023']}／{collection['articles_by_year']['2024']}／{collection['articles_by_year']['2025']}。
- Yahoo AAPL 日 OHLCV：{collection['market_rows']} 筆，含暖身期與下一年度邊界；
  來源 URL 與回應雜湊保存在 `market_provenance.json`。
- 全部價格日期通過 XNYS 交易日曆檢查。目標為下一個交易日原始收盤價是否高於當日收盤價。
- Apple 只有發布日期，因此最早使用日為發布日之後的第一個交易日，不把日期虛構成精確時刻。
- 每個切分都排除目標收盤價落在下一年度的樣本。所有組別使用相同文章與交易日。

| 切分 | 預測截止日範圍 | 交易日數 | 有新文章日數 |
|---|---|---:|---:|
{chr(10).join(split_rows)}

完整 train／validation／test 指標在 `metrics.csv`；validation 只作診斷，沒有據此挑參數或重新訓練。
StandardScaler 與 LogisticRegression 僅用 2023 年擬合。C=0.1，閾值0.5，設定在評估前固定。

## 配對差異

以下 Accuracy 差異單位為百分點；區間是未作多重比較校正的 95% percentile interval。
p 是近似的 5 日區塊 sign-permutation 檢定，再對四個主要比較作 Holm 校正。
區間與檢定使用不同近似方法，應共同解讀，不能只挑區間或單一點估計宣稱成功。

| 比較 | Accuracy 差異 | 95% 區間 | Brier 差異 | Holm p |
|---|---:|---:|---:|---:|
{chr(10).join(comparisons)}

E0–E4 相對 Market 的五個次要比較，另列於 `baseline_contrasts.csv` 並在該家族內校正。

## 抽取與文字覆蓋稽核

- FinBERT revision：`{registry['revision']}`。
- 450 篇 × 2 種來源，產生 {inference['feature_rows']} 列文章特徵。
- 共 {inference['total_body_and_title_chunks']} 個 FinBERT 段；
  {inference['full_articles_over_single_512_token_window']} 篇全文超過單段容量，使用所有分段的 token 加權平均。
  截斷 token 數={inference['truncated_tokens']}；分段之間沒有無限長的上下文能力。
- 標題中有 {inference['title_articles_without_event']} 篇未抽到事件。
  語法事件紀錄並非去重後的金融事件，全文可能產生很多普通動詞紀錄。
- 事件特徵共 53 維；加上固定量價、情緒和新聞數後，E2–E4 各有 67 維。
- 沒有獨立人工答案，所以**本輪沒有真實抽取 Precision／Recall／F1**。
  先前四筆合成示例的 1.0 分數不能套用在這批文章。

## 限制與接下來的研究

Apple 官方文章有公司來源偏差，未涵蓋獨立媒體的法律／負面新聞。
現今擷取的內文可能曾修改，現今預訓練模型亦可能含歷史資訊；
保守日期對齊不能替代當年的版本存檔。因此結果限於這個資料集與設定。

下一階段先做固定 100–200 個真實新聞項目的獨立盲標，分開檢查主受詞、事件類別、
否定／推測、公司別名與重複提及，再决定事件特徵的修訂。新版本應在開發資料上調整，
並使用新的未查看時段作最後驗證；2025 年本輪結果已被查看，不能再當作全新的測試集。

本輪完整規格與重跑命令見 `docs/APPLE_PUBLIC_V1_PROTOCOL.md`。
`predictions.csv`、`fitted_models.json` 與 `evaluation_audit.json` 可追查每筆結果與固定設定。
公開結果包只含衍生資料、URL／雜湊、程式與報告，不配送新聞全文或模型權重。
'''
    (folder/'RESULTS_ZH.md').write_text(report,encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    arms=['Market','E0','E1','E2','E3','E4']
    axes[0].bar(arms,[100*test.loc[arm,'accuracy'] for arm in arms],color=['#9ca3af']+['#2b6cb0']*5)
    axes[0].axhline(100*test.loc['Always_up','accuracy'],color='#a53a3a',ls='--',lw=1.5,label='Always-up baseline')
    axes[0].set_ylim(0,65); axes[0].set_ylabel('Test accuracy (%)')
    axes[0].set_title(f'AAPL 2025: {n} matched sessions'); axes[0].legend(frameon=False,fontsize=9)
    axes[0].spines[['top','right']].set_visible(False)
    x=100*primary.accuracy_delta.to_numpy()
    low=100*primary.accuracy_ci_low.to_numpy(); high=100*primary.accuracy_ci_high.to_numpy()
    axes[1].errorbar(x,np.arange(len(primary)),xerr=[x-low,high-x],fmt='o',color='#2b6cb0',capsize=4)
    axes[1].axvline(0,color='#555',ls='--',lw=1)
    axes[1].set_yticks(np.arange(len(primary)),primary.contrast)
    axes[1].invert_yaxis(); axes[1].set_xlabel('Accuracy difference (percentage points)')
    axes[1].set_title('Paired 95% block-bootstrap intervals')
    axes[1].spines[['top','right']].set_visible(False)
    fig.savefig(folder/'comparison.png',dpi=180)
    plt.close(fig)
    print('Saved report and comparison.png')


if __name__=='__main__':
    main()
