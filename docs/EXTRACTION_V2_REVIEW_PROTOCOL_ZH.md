# 真實事件抽取：開發診斷與獨立標註規格

本階段只使用 2023 年的 152 篇開發資料，不重跑或調整已查看的 2025 年測試結果。
V1 的抽取器、FinBERT 特徵、股價預測程式與結果都保留；V2 在獨立模組中試作。

## 本階段能證明什麼

結構診斷可確認缺欄位、輔助動詞、普通引述、品牌字串混淆、重複紀錄等具體行為。
它不能把「沒有 Apple 字串」直接等同不相關，也不能由保留筆數推論 Precision／Recall。
產品／服務、公司員工、代名詞與跨句關係都可能帶有真實資訊。
這些診斷只能提供上一輪未改善的可能原因，不能證明因果。

固定種子 20261014，先從按 article_id 排序的 2023 母體中不放回抽 100 篇；
保存 IDs、來源 URL、日期和文字雜湊。模型輸出在私有檔中，盲標範本不含預測或股價。
這批來自開發年，適合修訂與錯誤分析；V2 已在同年文字上開發，不能把日後此批分數稱為未見資料泛化證據。

## V2 候選規則

- 排除純 AUX、say／tell 等引述／心理動詞，過長參數及 Apple 只出現在複雜從句的情況暫列拒絕。
  原始紀錄仍保留，以便人工檢查是否誤刪。
- 把 Apple Watch、Apple Pencil、Apple Pay 等已列出的品牌與公司本體分開。
  `target_reference=brand` 表示有品牌關聯，`target_role` 不因此自動變成公司 object 或 both。
- 名詞所有權與員工關係不等於公司本體；未解決代名詞／共指時不擅自補成 Apple。
- 原句有明確並列動詞及可繼承主詞時，保留繼承標記。這是有限規則，不是完整共指解析。
- 財報發布標題的 quarter results 補為 earnings；含產品／服務詞的 introduce 等補為 product_launch。
- 保存直接附著的否定與情態動詞。並列或跨句的作用範圍仍可能不完整。
- 缺主詞或受詞記為 `partial_requires_review`；有完整參數也只記 `candidate`，不宣稱已人工驗證。
- 同一篇內精確重複去重；沒有做語意等價事件合併。

它尚未接入股價模型。保留較少紀錄也可能降低 Recall，必須用標註檢查。

## 獨立標註：固定 100 篇標題

範本：`outputs/apple_extraction_audit_v1/independent_gold_template.jsonl`。
逐一依 URL 查看標題，先不看模型答案或價格。公司目標為 AAPL／Apple。
標題條件只能用標題；內文能補出的主詞不能當作標題直接觀察到的主詞。
如果線上標題已修改，須回到原收集的私有快照確認，不能靜默換成新版本。

每列維持 article_id、URL、source=title，填：

- `review_status=reviewed`、非空的 reviewer_id。
- `reviewer_type=human_independent` 只供實際獨立人類標註者使用；AI 協助結果不能填這個值。
- `no_event` 明確填 true 或 false。true 對應空 events；false 必須有至少一筆事件。
- events 每筆有 actor/action/object/event_type/target_role 五個字串。
- action 使用動詞原形；保留完整參數跨度及並列實體，記錄跨度歧義於 notes。
- 缺少主詞時 actor 可空白；不要從發行者資訊猜主詞。每筆須有 action 及至少一個參數。
- 類別固定為 litigation、earnings、product_launch、merger_acquisition、regulation、other。
  product_launch 包含新產品／軟體／服務發布；財報發布為 earnings，獎項或商店開幕通常是 other。
- target_role 是公司本體的 actor/object/both/other/unknown。產品名稱含 Apple 不等於公司受詞。
- 否定／推測事件是否收錄及如何處理，先在開發標註指引中一致約定並在 notes 說明；
  目前嚴格五欄評分不評估事實性，不能宣稱已驗證這些屬性。

獨立人類標註尚未完成時不得報抽取 F1。程式只能檢查標註者聲明，無法驗證真人身分或工作過程。
正式研究建議兩人獨立標註、處理分歧並保存定稿；之後另取未查看資料評估泛化。

## 重跑

使用已安裝依賴與先前真實 corpus，從儲存庫根目錄執行：

```bash
python -m experiments.audit_apple_extraction
python -m experiments.compare_extraction_candidates
```

選用 GLiNER2 標題比較：

```bash
python -m pip install -r requirements-extraction-audit.txt
python -m experiments.compare_extraction_candidates --compare-gliner
```

若模型已下載，可設定 HF_HOME 為原本的快取路徑及 HF_HUB_OFFLINE=1。
GLiNER2 僅比較同一份 100 篇標題，不處理長內文。它與 V1 共用原始類別規則；
換後端不會自動修好類別規則。嚴格欄位不一致也可能只是名詞跨度不同，不等於錯誤。

完成真正的獨立標註後才執行：

```bash
python -m experiments.score_independent_extraction --gold path/to/completed_gold.jsonl
```

目前範本有 100 筆 pending，這個命令預期會拒絕評分，避免空答案被當成成功。
評分是五個欄位經大小寫／空白正規化後的嚴格 multiset 一致性，不是寬鬆跨度 F1。
使用尚未標註的樣本或 AI 輔助樣本時，不會自動產生可信的真實抽取準確率。

重新執行 audit 會保留既有範本；若更換樣本或來源，程式會要求改用新的輸出目錄。
人工已填的檔案另存為 completed_gold，並保留原始空白範本。
只公開統計、樣本 URL 與空白標註範本；原文證據、完整句子與私有模型輸出不放入公開結果包。
