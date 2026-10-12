# StockLLM V2：從執行到人工驗證

本指南依實際 `stockllm_v2_event_extraction.zip` 與本次實跑整理。原包 SHA256：
`E1DD000B29AA712609B46F44BD5D262A21469A0F5973772B403AD25B64175765`。

## 1. 這個包已經做了什麼

原包根目錄是 `stockllm_v2_proto/`：

```text
stockllm_v2_proto/
  README.md
  requirements.txt
  docs/E0_E4_PROTOCOL.md
  event_extraction/
    __init__.py
    extractors.py
    run.py
    sample_news.csv
  tests/test_event_extractors.py
```

還包含 `.pytest_cache/`、`__pycache__/` 與 `.pyc`；這些是執行暫存，無須上傳。
原包 README 的預定分支寫成 `experiment/event-extraction-e0-e4`，實際遠端原先只有
`main` 和 `research/event-extraction-e0-e4`，後者起初與 main 指向同一個提交
`fa0c6055d62920da1659274da468dec0e44e5afe`。該儲存庫當時沒有 `stock2` 分支。
本次將原型整合到既有 `research/event-extraction-e0-e4` 分支；查看 GitHub 時須選它。

程式目前完成的是「新聞文字 → 事件欄位」。spaCy 是語法分析加固定規則；
GLiNER2 是選用的本地結構抽取模型。事件類別仍由本包的固定規則產生。
兩者都不是已訓練好的股價預測器。E0–E4 文件是研究設計，完整的五組實驗尚未實作。
示例只有 4 筆合成英文新聞；原有 52 篇 AAPL 實驗資料不在這個包中。

新增 `event_extraction/annotate.py`、本指南與 Colab notebook。原包 README 原文保存在
`docs/EVENT_EXTRACTION_ORIGINAL_README.md`。本次修正全文模式把標題／內文連成一句的問題、
`Regulators investigate Apple` 被標成 other，以及「啟動調查」被判為產品發布的規則問題。

## 2. Google Colab：依序執行

可直接開啟 `notebooks/event_extraction_colab.ipynb`，或建立空白 Colab，依序貼下列儲存格。
先使用 CPU，硬體加速器選「無」。免費 Colab 資源與執行時間有限，不能承諾無限大量執行；
參見 [Colab 官方 FAQ](https://research.google.com/colaboratory/faq.html)。

**儲存格 1：取得正確分支。** 新 runtime 只執行一次。

```python
!git clone --branch research/event-extraction-e0-e4 --single-branch https://github.com/yoyocar2333/stockllm-pit-news.git /content/stockllm-pit-news
%cd /content/stockllm-pit-news
```

**儲存格 2：安裝 CPU 依賴及模型。** 首次下載需要網路；無須 API key。

```python
%pip install -r requirements-event.txt
!python -m spacy download en_core_web_sm
```

若 Colab 提示已載入套件需要重新啟動，選「重新啟動工作階段」後先重跑
`%cd /content/stockllm-pit-news`，再繼續。不要重複 clone 到已存在的目錄。

**儲存格 3：跑 4 筆示例。**

```python
!python -m event_extraction.run --input event_extraction/sample_news.csv --backend spacy --source title --output outputs/events_title.csv
!python -m event_extraction.run --input event_extraction/sample_news.csv --backend spacy --source full --output outputs/events_full.csv
!python -m pytest tests/test_event_extractors.py tests/test_event_annotation.py tests/test_event_run.py -q
```

成功時會看到 `Saved: outputs/events_title.csv rows: 4` 與全文檔同樣 4 列。
`source=full` 表示標題和內文分別分析，再合併抽取結果，避免跨段落連成一句。

**儲存格 4：產生可用試算表開啟的人工檢查表。**

```python
!python -m event_extraction.annotate prepare --input outputs/events_title.csv --output outputs/annotation_title.csv
!python -m event_extraction.annotate prepare --input outputs/events_full.csv --output outputs/annotation_full.csv
from google.colab import files
files.download('outputs/annotation_title.csv')
files.download('outputs/annotation_full.csv')
```

用 Excel 或 Google Sheets 編輯這些 CSV，依第 5 節填完後另存為 UTF-8 CSV。
重新執行 prepare 時若同名標註表已存在，程式會拒絕覆蓋；換檔名即可。

**儲存格 5：上傳已完成的標題標註表。** 僅選一個 CSV。

```python
from google.colab import files
from pathlib import Path
uploaded = files.upload()
assert len(uploaded) == 1, '請只選一個已完成的標題標註 CSV'
Path('outputs/annotation_title_reviewed.csv').write_bytes(next(iter(uploaded.values())))
!python -m event_extraction.annotate validate --predictions outputs/events_title.csv --annotations outputs/annotation_title_reviewed.csv --output outputs/extraction_metrics_title.json
```

驗證全文時將 predictions 改為 `outputs/events_full.csv`，annotations 改為全文標註表，
output 改為 `outputs/extraction_metrics_full.json`，不要交叉配對。

**儲存格 6：保存所有結果。** Colab 結束後本機檔案可能消失，跑完就下載。

```python
import shutil
from google.colab import files
shutil.make_archive('/content/event_extraction_outputs', 'zip', 'outputs')
files.download('/content/event_extraction_outputs.zip')
```

## 3. Windows 本地：可複製的 PowerShell 指令

需要 Python 3.12 與 Git。以下在你想存放專案的資料夾執行；新目錄只 clone 一次。
直接指定虛擬環境的 Python，因此不必修改 PowerShell 執行原則。

```powershell
git clone --branch research/event-extraction-e0-e4 --single-branch https://github.com/yoyocar2333/stockllm-pit-news.git stockllm-pit-news
Set-Location stockllm-pit-news
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-event.txt
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
.\.venv\Scripts\python.exe -m event_extraction.run --input event_extraction/sample_news.csv --backend spacy --source title --output outputs/events_title.csv
.\.venv\Scripts\python.exe -m event_extraction.run --input event_extraction/sample_news.csv --backend spacy --source full --output outputs/events_full.csv
.\.venv\Scripts\python.exe -m event_extraction.annotate prepare --input outputs/events_title.csv --output outputs/annotation_title.csv
.\.venv\Scripts\python.exe -m event_extraction.annotate prepare --input outputs/events_full.csv --output outputs/annotation_full.csv
.\.venv\Scripts\python.exe -m pytest tests/test_event_extractors.py tests/test_event_annotation.py tests/test_event_run.py -q
```

若 `py -3.12` 找不到版本，先安裝 Python 3.12；不要把未驗證的 Python 版本當作已保證相容。
填完標註表後執行：

```powershell
.\.venv\Scripts\python.exe -m event_extraction.annotate validate --predictions outputs/events_title.csv --annotations outputs/annotation_title_reviewed.csv --output outputs/extraction_metrics_title.json
```

Linux/macOS 的環境建立方式：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-event.txt
python -m spacy download en_core_web_sm
python -m event_extraction.run --backend spacy --source title --output outputs/events_title.csv
```

如果只想用原 ZIP，在 PowerShell 執行以下指令，先解壓到一個新的位置：

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\stockllm_v2_event_extraction.zip" -DestinationPath .\stockllm_zip_original
Set-Location .\stockllm_zip_original\stockllm_v2_proto
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
.\.venv\Scripts\python.exe -m event_extraction.run --backend spacy --source title --output outputs/events_title.csv
.\.venv\Scripts\python.exe -m pytest -q
```

注意：原 ZIP 沒有 annotate 工具，也沒有本次全文與類別修正；建議使用 GitHub 分支版本。

## 4. 換成你的真實新聞

輸入 UTF-8 CSV 必須包含這五欄，欄名完全一致：

```csv
article_id,ticker,target_name,title,content
news0001,AAPL,Apple,Apple sues Samsung over patents,Apple filed a lawsuit against Samsung.
```

`article_id` 不可重複或空白；`ticker` 與 `target_name` 不可空白。
`target_name` 是原文中可匹配的公司名稱，AAPL 通常填 Apple，不要只填股票代號。
程式目前以整個 actor/object 字串是否出現公司名稱決定 target_role；它沒有完整的公司別名連結、
代名詞解析或跨句共指辨識。抽取器內部接受 aliases，但此 CLI 尚未讀入別名清單。
英文模型只適用本指南的英文流程；中文新聞需要另選模型並獨立驗證。
有逗號、換行的欄位應由 CSV 工具正確加引號，不要手動拆行。

把檔案放在 `data/raw/my_news.csv`（該位置已被 Git 忽略，不會當作程式上傳），然後：

```powershell
.\.venv\Scripts\python.exe -m event_extraction.run --input data/raw/my_news.csv --backend spacy --source title --output outputs/my_events_title.csv
.\.venv\Scripts\python.exe -m event_extraction.run --input data/raw/my_news.csv --backend spacy --source full --output outputs/my_events_full.csv
```

Colab 可用 `files.upload()` 上傳 CSV，再指定 `--input` 為實際上傳的檔名。
初次先用固定抽樣的 100–200 篇，不要直接跑整個歷史庫。
比較 E0–E4 的標題／全文條件時，必須使用同一批具有內文的 article IDs；
抽取器允許空白 content，但空內文不能作為「全文效果」證據。每個輸出的 `.meta.json`
會記錄輸入雜湊、套件版本、缺內文篇數及模型名稱。

PIT 股價實驗還需要發布時間、時區、實際可取得時間、交易日曆與 OHLCV。
這些目前不由事件抽取程式處理；可在 CSV 保留額外欄位，程式會帶到輸出。

## 5. 人工標註：你實際需要做的事

`events_title.csv` 每列代表一篇新聞，`events_json` 中可含多個事件。
`actor/action/object` 是文字／语法角色；`target_role` 不是利多、利空或股價方向。
`event_count` 是抽取成功狀態的事件紀錄數，並不等於去重後的真實事件數。
兩後端的 action 會盡量使用同一 spaCy 語境動詞原形：例如 GLiNER2 原先輸出的 sues 改為 sue。
若抽出的片段無法對齊原文中的動詞則保留原值，人工檢查時需註明；不把未對齊結果當作已標準化。
同一事件在標題和內文重複出現時會保留兩次，目前尚未做跨句事件合併。

標註 CSV 將事件拆成一列一筆，並顯示原始文字與 predicted 欄位：

| 要填的欄位 | 填法 |
|---|---|
| `review_status` | 完成填 `reviewed`；無法判定填 `uncertain`；勿留下 `pending` |
| `decision` | 原抽取可修正為正確事件填 `keep`；虛假的多餘事件填 `delete`；漏掉的事件新增一列填 `add`；整篇確實無事件填 `no_event` |
| `gold_actor` | 行動者，使用原文跨度；被動句需還原行動者 |
| `gold_action` | 動詞原形，例如 sue，不要一份寫 sues、另一份寫 sue |
| `gold_object` | 原文受詞／受影響跨度；不可隨意省略限定詞或另換公司別名 |
| `gold_event_type` | `litigation / earnings / product_launch / merger_acquisition / regulation / other` |
| `gold_target_role` | `actor / object / both / other / unknown` |
| `evidence` | 從本條件可見的文字逐字複製證據，必須是原文的連續片段 |
| `notes` | 歧義、否定、推測、共指、跨度選擇、重複事件等說明 |

標題範例：`Samsung sues Apple over patents`、target_name=Apple。
填 actor=Samsung、action=sue、object=Apple、event_type=litigation、target_role=object。
不能因為「Apple 是你關注的股票」就把 Apple 填為 actor。

檢查 `Apple may sue Samsung` 或 `Apple did not sue Samsung` 時，原型可能仍抽出 sue。
它沒有專門的否定、事實性與推測欄位。依事先約定的標註範圍判定是否收錄；把相關資訊記在 notes。
本版評分僅涵蓋五個欄位，不能宣稱已評估事實性或金融影響。

若缺一筆事件，複製該篇一列，保留 article_id/source，改 `row_id` 為唯一值
（例如 `news0001:added1`），填 `decision=add`、全部 gold 欄位與 evidence。
新增列的 predicted 欄位留白。不要為了讓分數漂亮而刪掉模型抽錯的列。
`delete/no_event` 列的 gold 欄位必須留白。若原先是 no-event 佔位列，實際有事件，改填 add。
每篇都必須留下至少一列；一篇中任何列填 uncertain，整篇會排除並明列在報告中。
標題條件只看標題；匯出的標題標註表已隱藏內文。全文條件看標題與內文。

**獨立驗證**請改用盲標表：

```powershell
.\.venv\Scripts\python.exe -m event_extraction.annotate prepare --input outputs/my_events_title.csv --output outputs/gold_title_blind.csv --blind
```

盲標每篇初始只有一個空白列，不顯示模型事件數或答案。人工自行從原文填全部事件，
有事件填 add，多事件複製列並換 row_id；沒有事件填 no_event。
保留 `annotation_mode=blind`，填完仍用相同 validate 指令。
先看模型答案的 assisted 表適合除錯，不能當作獨立的抽取準確率證據。
正式研究宜由兩位標註者獨立盲標固定測試集，之後解決分歧並保存定稿。
先用開發集約定語法受詞與語意受影響實體的取捨、完整跨度和重複提及規則，再凍結測試集。

## 6. 驗證輸出怎麼看

可先用套件附的 `event_extraction/sample_gold_title.csv` 試跑 validate：

```powershell
.\.venv\Scripts\python.exe -m event_extraction.annotate validate --predictions outputs/events_title.csv --annotations event_extraction/sample_gold_title.csv --output outputs/demo_metrics.json
```

它是 4 筆合成新聞的示範答案，並不是你的真實新聞標註或獨立人工測試集。

validate 會拒絕未完成的列、遺失／多出的 article IDs、重複 row_id、錯配 source、
錯誤的類別／角色值，以及不在原文的 evidence。這些檢查不能取代人工判斷內容是否正確。

輸出 JSON 的 `triple` 比較 actor/action/object；`all_fields` 同時比較五個欄位。
使用逐篇的 multiset 比對，保留重複紀錄；僅忽略大小寫及重複空白，
不自動合併別名、同義詞或同一事件的不同提及。跨度略有差異也會算未命中。

| 指標 | 意義 |
|---|---|
| `precision` | 模型抽出的事件中，有多少完整符合人工答案 |
| `recall` | 人工標出的事件中，有多少被模型完整抽出 |
| `f1` | 兩者的調和平均 |
| `false_positive` | 模型多抽或未符合答案的事件數 |
| `false_negative` | 人工答案中漏抽或未符合答案的事件數 |

沒有分母的值輸出 null；查看 `scored_articles` 及 `excluded_uncertain_articles`，
避免只報分數卻忽略排除比例。這是嚴格的抽取一致性評分，不是股價 Accuracy、投資報酬或主客體的寬鬆跨度 F1。
真實語意品質還應檢查錯誤樣本與否定／共指等案例。

## 7. 選用 GLiNER2

先跑通 CPU spaCy，再選擇這個比較。官方仍支援
`GLiNER2.from_pretrained('fastino/gliner2-base-v1')` 與 `extract_json`；
本包不會呼叫 Fastino 雲端 API。
此比較也使用已安裝的 spaCy 英文模型統一動詞原形，不是單獨安裝 gliner2 就足夠。
參見 [官方程式庫](https://github.com/fastino-ai/GLiNER2) 與
[模型卡](https://huggingface.co/fastino/gliner2-base-v1)。

```powershell
.\.venv\Scripts\python.exe -m pip install "gliner2[local]==2.1.0"
.\.venv\Scripts\python.exe -m event_extraction.run --input event_extraction/sample_news.csv --backend gliner2 --source title --output outputs/gliner2_title.csv
```

Colab 對應指令：

```python
%pip install "gliner2[local]==2.1.0"
!python -m event_extraction.run --backend gliner2 --source title --output outputs/gliner2_title.csv
```

首次下載需要網路、磁碟空間和等待時間；無按篇 API 費用，仍需自己的運算資源。
權重未隨 ZIP 或 GitHub 程式包配送。模型 revision 未鎖定；正式實驗應另外記錄實際權重 revision。
長內文需先驗證模型輸入長度限制及截斷，不要假設 full 就代表任意長文章每個字都已處理。
比較兩種後端時，同一來源條件用同一份盲標 gold 表與同一批文章，避免為不同模型改答案。

## 8. 能自動化到哪裡

| 工作 | 目前狀態 |
|---|---|
| CSV 檢查、批次推論、匯出事件與版本紀錄 | 已有 CLI，自動執行 |
| 產生標註表、欄位與證據驗證、抽取一致性指標 | 本次新增，可自動執行 |
| 建立可信人工答案、处理主受詞歧義、判斷語意／事實性 | 需要人工；不是逐篇手寫程式 |
| 大庫分片、斷點續跑、單篇錯誤重試、批次加速 | 尚未實作；目前一次讀全檔，出錯會停止，成功結束才保存 CSV |
| 新聞擷取、發布時間驗證、PIT、情緒特徵、E0–E4 模型訓練與回測 | 尚未由此原型實作 |

先完成一份固定的 100–200 篇獨立標註集，確認規則的錯誤類型，再決定大量推論。
後續 E0–E4 須按 `docs/E0_E4_PROTOCOL.md` 凍結文章集合、PIT、模型、標籤與評估方式。
原型跑出 CSV 或測試通過，均不代表新聞特徵能提升股價預測。
