"""Write source/split findings and a Git-friendly evidence summary."""
import json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data'/'source_verification'
OUT=ROOT/'results'/'corrected_group_split_20261002'
OUT.mkdir(parents=True,exist_ok=True)
verified=json.loads((SOURCE/'official_image_verification.json').read_text(encoding='utf-8'))
manifest=json.loads((SOURCE/'corrected_split_manifest.json').read_text(encoding='utf-8'))
diag=json.loads((SOURCE/'split_diagnosis.json').read_text(encoding='utf-8'))
meta=json.loads((SOURCE/'kaggle_metadata.json').read_text(encoding='utf-8'))
shutil.copy2(SOURCE/'split_diagnosis.json',OUT/'split_diagnosis.json')
summary={k:v for k,v in verified.items() if k!='files'}
summary['official_metadata']={k:meta.get(k) for k in ('id','ref','title','currentVersionNumber','lastUpdated','totalBytes')}
summary['excluded_from_image_inventory']='macOS __MACOSX and ._ resource metadata; identical nested ZIP image copies collapsed by canonical relative path'
(OUT/'source_verification_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
text='''# 資料來源與切分問題核對

學號 315553010，楊敦傑。核對日期：2026 年 10 月 2 日。

## 結論

本機資料集正確：指定 Kaggle 第 2 版的 5856 張影像，與本機相對路徑、檔案大小及 ZIP CRC32 全部吻合，無缺少或多出的影像。資料卡文字寫 5863 張，與官方可下載的影像數不同，不能據此判定本機缺檔。此項核對驗證來源與檔案內容，沒有重新審核醫學標籤。

官方來源：https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## 原始資料與本專案的問題

| 檢查 | 官方原分區跨區重疊 | 舊切分跨區重疊 | 修正切分跨區重疊 |
| --- | ---: | ---: | ---: |
| 原始檔 SHA256 相同群組 | 0 | 9 | 0 |
| 解碼 RGB 像素 SHA256 相同群組 | 0 | 9 | 0 |
| 完整檔名組 | 0 | 270 | 0 |

原始資料共有 30 組相同檔案／像素影像，出現在同一官方分區內。先前 `stratified_split` 按個別影像隨機抽取驗證資料，未先按影像身份分組，造成其中 9 組相同影像及 270 組檔名組跨訓練與驗證；這是本專案切分程式的疏漏。不能只歸咎原始資料，保留洩漏切分作正式結果。

先前僅按 person 數字計算的 train/test 編號重疊，混合了 bacteria 與 virus 命名範圍。加入病原類型前綴後，官方 train、val、test 的檔名組沒有重疊，因此裸數字重疊不能證明同一病患跨區。

## 修正方法

`split_dataset.py` 以三種關係建立連通分組：

- 肺炎影像使用 `person編號_bacteria` 或 `person編號_virus`；正常影像使用 `IM-序號` 或 `NORMAL2-IM-序號`。
- 原始檔 SHA256 相同。
- 解碼後 RGB 的尺寸和像素 SHA256 相同。

同組全部置於同一分區。官方 val 保留在驗證集，官方 train 按類別抽取約 15% 作驗證；官方 test 的 624 張、標籤及檔案完全保留。沒有刪除或改寫任何原始影像，原始重複影像留在同一區。切分清單載入時會重新核對所有影像 SHA256、總清單、分組不重疊及官方 test 成員。

| 修正分區 | NORMAL | PNEUMONIA | 總數 |
| --- | ---: | ---: | ---: |
| 訓練 | 1140 | 3294 | 4434 |
| 驗證 | 209 | 589 | 798 |
| 測試 | 234 | 390 | 624 |

張數與舊切分相同，影像成員不同。修正後檔名組、原始檔與解碼像素雜湊跨區重疊均為 0。這不排除近似影像，也不等同經臨床病患識別碼確認的病患層級獨立。

## 重新實驗與保存位置

- 修正實驗：`results/corrected_group_split_20261002/`，ResNet18、ResNet50、ResNet101 各 12 輪，D 槽 `.venv` 與 RTX 4070。
- 舊數值／圖／權重仍在 `results/` 和 `results/resnet101/`，僅作歷史紀錄。
- 修改前程式與 Word 報告：`report_backups/before_split_correction_20261002/`。
- 官方完整 ZIP、官方 API metadata、逐檔驗證清單：`data/source_verification/`。
- Git 可保存的來源摘要、診斷與完整切分清單：修正結果資料夾中的 `source_verification_summary.json`、`split_diagnosis.json`、`split_manifest.json`。

每模型重新設定相同種子，停用 cuDNN benchmark 並啟用 deterministic。這些設定及切分變動同時發生，新舊成績差異不能直接量化為資料洩漏造成的差值。超參數沿用原設定；所有權重依驗證 F1 選擇，測試集只依作業要求逐輪記錄。
'''
(ROOT/'DATASET_SPLIT_REVIEW_20261002.md').write_text(text,encoding='utf-8')
print('Saved dataset/split review and evidence summary inside D project.')
