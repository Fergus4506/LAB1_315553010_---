# Lab 1 Report Spec 對照

學號 315553010，楊敦傑。對照 lab1_spec.pdf 第 2、4、13–15 頁。

正式實驗：`results/corrected_group_split_20261002/`；原始影像符合指定 Kaggle 第 2 版，修正逐張切分造成的驗證洩漏後已重跑三模型。

| PDF 要求 | 專案內容 | 狀態 |
| --- | --- | --- |
| 自訂 Dataset/DataLoader、NORMAL/PNEUMONIA 分類 | train.py 的 ChestXrayDataset、DataLoader | 完成 |
| 至少兩種 ResNet | ResNet18、ResNet50、ResNet101，各 12 輪 | 完成 |
| ImageNet 預訓練可用，重新初始化最後層 | fc 重新初始化為 2 類；pretrained_verification.json 三者通過 | 完成 |
| 每輪訓練／測試 accuracy、F1 比較 | 36 筆 CSV；comparison_accuracy.png、comparison_f1.png | 完成 |
| 驗證曲線加分項 | comparison_validation.png | 完成 |
| 最終測試混淆矩陣 | comparison_final_test_confusion.png，權重由驗證 F1 選擇 | 完成 |
| Introduction | DOCX 研究目標 | 完成 |
| Experiment setups：模型、DataLoader、增強 | DOCX 實驗設定，包括修正切分及 venv/GPU | 完成 |
| 最高測試 accuracy、F1 和 screenshot | 成績表、comparison_results_screenshot.jpg 實際結果頁截圖，完整 training.log 保留 | 完成 |
| 訓練／測試 accuracy 曲線、測試 F1 曲線、最高準確率熱圖 | DOCX 圖 2、3、5 | 完成 |
| Discussion 35% | 依修正結果分析切分、模型成本、曲線、混淆矩陣、限制與改善 | 完成 |
| GitHub Link 5% | DOCX 待填欄，依本人要求自行推送後補入 | 待本人處理 |
| 規定 PDF 檔名 | 可編輯 LAB1_315553010_楊敦傑.docx，由本人微調後匯出同名 PDF | 待本人處理 |

## 實驗有效性與交付檢查

- 官方 ZIP 的 5856 張影像與本機路徑、大小、CRC32 全部吻合。
- 原始 30 組重複影像全部保留；修正切分的檔名組、原檔及解碼像素跨區重疊為 0，官方 test 624 張未更動。
- D 槽內建 venv、RTX 4070；三模型 ImageNet 預訓練核對通過。
- 每輪指標與選模規則已核對；GPU 重新評估三模型交付權重重現驗證／測試 confusion counts、accuracy、precision、recall、F1。
- 最高觀察測試值與驗證選模成績分列，不把最高值當成權重選擇依據。
- 尚未取得可核實臨床病患 ID；單次切分／種子、預訓練 V1/V2 差異與多次測試觀察的限制已寫入討論。
- DOCX 結構、4 表格與 6 嵌入圖已核對。LibreOffice 不可用、Word COM 未登錄，實際分頁尚無法自動驗證。
- 未建立 .git；本人確認實驗與報告後才建立。PDF 匯出與 GitHub 連結由本人處理。
