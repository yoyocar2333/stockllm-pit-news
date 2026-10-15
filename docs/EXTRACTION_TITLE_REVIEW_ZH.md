# 固定 100 篇標題：AI 預審與人類核對

這份規格補充前一階段的獨立標註指引。日期為 2026-10-15，版本為
`title_semantic_v1_20261015`。只用固定 2023 年開發標題，不讀內文或價格，未重新查看 2025 年。

## 目前的研究狀態

已完成 100 篇的 AI 語意草稿與三種後端的差異比較。草稿來自本次 Codex 助理，
沒有呼叫付費推論 API；助理在此前已看過開發案例，不能聲稱盲標或獨立人工標註。
草稿有 104 筆事件，5 篇判為無符合事件，尚未由人類核對。
精確模型身分與訓練資料截止日未經獨立驗證。

本輪不會產生「獨立人工 Precision／Recall／F1」。
V2 存在可核對的過濾風險，暫未接入股價驗證。

## 語意標註的邊界

舊版 V1 是語法動詞紀錄；本次草稿是語意事件標註，不能把兩者差異全部當作程式錯誤。
指引是在同一開發樣本上細化，並非事前註冊的未見資料準確率評估。

- 標題必須明示 Apple、任務品牌或其關聯。不能只因來源是 Apple Newsroom 就補成 Apple 事件。
  明示產品作為工具、服務上的第三方事件可收錄；未明示 Apple 關聯的第三方故事不收錄。
- 品牌／關聯字典包括原有 Apple 品牌，以及 Mac／iMac、tvOS、Siri、FaceTime、Final Cut Pro、
  Logic Pro、Apple Card、Apple News、WWDC、Swift Student Challenge、Apple 門市與明示所有權的計畫／研究。
  這是任務字典，並不讓上述實體自動成為 Apple 公司本體。
- 抽主要事件。明確的並列動詞拆開，可繼承已明示主詞。產品修飾語、目的、用途、能力補語
  不另計已實現事件。同一動詞的並列受詞通常保留完整；語意類別不同時拆開並記錄原因。
- `Join`／`Meet` 這類導讀祈使句與沒有明確述詞的回顧標題可標為無事件。
- 發布、預告、可用性均可收錄；`is available` 的 action 記 `be`，有明確可用性述語但省略 be 時
  也如此正規化。這不是說 be 在任何句子都是金融事件。
- 可用性／狀態句以明示主詞作 actor、明示述語作 object；不及物動作可讓 object 空白。
  被動句的受事放 object；主詞未明示時不從發行者補入。action 用原形。
- actor/object 保留完整相關參數與並列名稱，可保留結果／位置補語。
  有多種合理跨度時記錄 notes；嚴格五欄一致性會懲罰這些跨度差異。
- 財報發布與明示財務結果歸 earnings；開發者收入不能解讀成 Apple 公司收入。
  新產品、軟體、服務、功能、預告與首次可用性歸 product_launch。
  榜單、獎項、播報人員、門市開幕、合作供應協議通常歸 other；合作不是併購。
- `target_role` 只標公司本體的 actor/object/both/other/unknown。
  店名、產品、公司所有物或員工不是公司本體。品牌關聯本身也不等於公司是受詞。
- 未來計畫、能力、預告與否定可收錄，但需在 notes 標明。AI 草稿另保存文章級 factuality 提示；
  它不是已驗證的逐事件事實性標籤，五欄比較不評分這個欄位。

## 如何使用已填草稿

本機產物在 `outputs/apple_extraction_review_v1/`，不提交原文或完整答案到公開 GitHub。

1. 用瀏覽器開啟 `review_titles_ai_assisted.html`。100 篇已預填，但全部仍是待核對。
2. 填核對者代號，逐篇讀標題、改正事件欄位與備註，按「完成本篇」。
3. 按「下一篇」或用篇號跳轉；瀏覽器會保存進度，也應定期按「匯出標註／備份」。
4. 匯出 `reviewed_title_annotations.jsonl`。未做完的列會保留 pending，不能拿去當完成的參考答案。

這條路線是 AI 輔助核對，即使人類按過確認，也不會變成獨立人工標註。
若需要獨立人類參考，應由未看過答案的標註者使用 `review_titles.html` 空白版。
獨立標註頁不包含預測；來源網址只供核對，若網頁已修改，以頁面中的凍結標題為準。
正式研究宜兩位標註者獨立作業並處理分歧；本階段尚未達成。

兩種頁面的瀏覽器儲存區分開。看過 AI 頁面的人再開空白頁仍不具有實際獨立性；
介面無法偵測外部接觸歷史，標註者必須如實聲明。
匯入草稿或帶有 AI 接觸標記的備份，會維持 `human_ai_assisted`，不允許升格為獨立人類。
瀏覽器資料可能被清除，請以匯出的檔案備份。

## 重現本輪比較

需要原來 corpus、三種後端的私有輸出與 AI 草稿。本次沒有重新訓練或重跑模型。
以下命令在儲存庫根目錄執行；既有本機環境可將 `python` 換成 `..\.event-venv\Scripts\python.exe`。

```powershell
python -m experiments.review_extraction_gate --ai-draft outputs/apple_extraction_review_v1/private_ai_title_draft.jsonl
python -m pytest -q --basetemp outputs/pytest_title_review
```

比較程式驗證來源檔 SHA256、100 個唯一樣本代號、標題雜湊、網址、AI 來源聲明與事件格式。
不會從預測自動生成獨立標準答案。公開統計包含草稿雜湊；完整草稿與核對頁只留本機。
從 GitHub 重新取得程式並不會自動取得私有語料或草稿，必須使用原本保存的本機檔案。

真正的獨立標註完成後，仍使用原來的嚴格評分入口：

```powershell
python -m experiments.score_independent_extraction --gold path/to/reviewed_title_annotations.jsonl
```

AI 草稿、AI 接觸標記或 pending 都會被獨立評分入口拒絕。
程式不能驗證真人身分或完整接觸歷史，也不會因評分成功就自動宣告 V2 可以放行。

## 進入下一個實驗前

需先核對被排除事件、可用性句、品牌／門市角色，明確處理 `add` 的多義性，
再在新的版本保存抽取規則。不能在同一批開發標題上改完規則後宣稱泛化提升。
固定特徵後才進行 2024 驗證；2025 年已看過，不再當成全新測試。
