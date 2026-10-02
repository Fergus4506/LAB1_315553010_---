"""Refresh README and the requirement checklist after all corrected audits pass."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'corrected_group_split_20261002'
ORDER=('resnet18','resnet50','resnet101')
if not (OUT/'experiment_completed.json').exists():raise RuntimeError('Corrected experiments must finish first')
summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
meta=json.loads((OUT/'run_metadata.json').read_text(encoding='utf-8'))
def pct(v):return f'{100*v:.2f}%'
lines=['| 模型 | 驗證選定輪次 | Val F1 | 選定 Test Acc | 選定 Test F1 | 最高觀察 Test Acc | 最高觀察 Test F1 |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for n in ORDER:
    item=summary[n];test=item['final_test_at_validation_selected_epoch'];acc=item['highest_observed_test_accuracy'];f1=item['highest_observed_test_f1']
    lines.append(f"| {n} | {item['validation_selected_epoch']} | {item['best_validation_f1']:.4f} | {pct(test['accuracy'])} | {test['f1']:.4f} | {pct(acc['accuracy'])}（{acc['epoch']} 輪） | {f1['f1']:.4f}（{f1['epoch']} 輪） |")
text=r'''# Lab 1 胸部 X 光肺炎分類

學號：315553010　姓名：楊敦傑

使用自訂 `ChestXrayDataset` 與 PyTorch `DataLoader` 做 NORMAL／PNEUMONIA 二元分類，比較 ImageNet 預訓練 ResNet18、ResNet50、ResNet101。三模型均已在 D 槽 Python 內建 `.venv`、RTX 4070 上各訓練 12 輪，保存的權重已重新評估並重現混淆矩陣、accuracy 及 F1。

**目前正式結果位於 `results/corrected_group_split_20261002/`。** 舊實驗在 `results/` 與 `results/resnet101/` 保留供追查；舊切分有驗證洩漏，不作此次正式結果。所有程式、下載、快取、影像、權重及可編輯 Word 報告均在 `D:\人工智慧醫學影像應用\lab1\LAB1_315553010_楊敦傑`。

## 資料來源與切分修正

資料來源：[Kaggle Chest X Ray Images Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)，第 2 版。已重新下載官方 ZIP，逐檔比較相對路徑、大小及 CRC32：本機 5856 張影像全部相符。資料卡文字的 5863 張與官方實際影像數不同；本機沒有缺檔。ZIP 中重複的巢狀影像副本只計一次，macOS 資源資訊不當作 X 光影像。

原資料有 30 組相同影像，未跨官方分區。先前逐張隨機切分將其中 9 組分到訓練與驗證集。此次以檔名組、原始檔 SHA256 及解碼 RGB SHA256 建立連通分組，整組放在同一區；官方 val 保留在驗證集，官方 train 按類別抽出約 15% 作驗證，官方 test 624 張完整保留。**沒有刪除或改寫原始影像，原始重複資料也保留在同一分區。** 固定種子為 315553010。

肺炎檔名組包含 `person編號_bacteria` 或 `person編號_virus`，避免不同命名範圍重用数字造成錯判；正常影像依 `IM-序號` 或 `NORMAL2-IM-序號` 分組。修正後檔名組、原始檔雜湊及解碼像素雜湊的跨區重疊均為 0。臨床病患識別碼尚未核實，不能據此宣稱病患層級已完全獨立。

| 分區 | NORMAL | PNEUMONIA | 總數 |
| --- | ---: | ---: | ---: |
| 訓練 | 1140 | 3294 | 4434 |
| 驗證 | 209 | 589 | 798 |
| 測試 | 234 | 390 | 624 |

張數與舊切分相同，影像成員不同。詳細診斷在 `DATASET_SPLIT_REVIEW_20261002.md`；修正結果資料夾中的 `source_verification_summary.json`、`split_diagnosis.json` 與 `split_manifest.json` 可隨程式納入 Git。完整官方 ZIP 及原始核對清單在 `data/source_verification/`，由 `.gitignore` 排除。

## 為何選擇這三種 ResNet

殘差捷徑有助深層網路最佳化。ResNet18 採 basic block、參數較少，提供較低成本的基準；ResNet50 採 bottleneck，提供成本與表徵能力的另一個比較點；ResNet101 明顯增加深度和容量，讓本實驗能檢查更深網路是否帶來實際收益。三者二類版本總參數為 11,177,538、23,512,130、42,504,258。更大容量也可能增加過度擬合與計算成本，因此不保證比較準確。

使用 torchvision 的 `ResNet18_Weights.IMAGENET1K_V1`、`ResNet50_Weights.IMAGENET1K_V2`、`ResNet101_Weights.IMAGENET1K_V2`，**最後 `fc` 層重新初始化為二類輸出**。凍結 conv1、bn1、layer1、layer2（其中 BatchNorm 也固定為 eval），微調 layer3、layer4、fc。預訓練提供已有的影像特徵，可能降低小型資料集從零訓練的成本；本次未做從零訓練對照，未量化它的改善幅度。三模型交付權重的 conv1 已與官方權重逐元素核對相同，最後層皆輸出 2 類，見正式結果中的 `pretrained_verification.json`。

若要改善準確率，可先檢視錯誤影像及標籤，再只透過分組驗證集比較學習率、解凍範圍、輸入解析度、保守的增強、類別權重和分類閾值，並同時查看 precision、recall、specificity。重複多個種子可檢查模型差異是否穩定；有可信病患資料時，應再作病患分組及外部驗證。上述改善尚未實測，不能承諾增幅。

## 環境與訓練方法

Python 3.12.14 內建 venv，PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、CUDA 12.8、NVIDIA GeForce RTX 4070。每模型 12 輪、batch size 32，輸入 224 × 224 RGB，ImageNet 均值與標準差正規化。訓練增強包含 ±7° 旋轉、最多 3% 平移、0.95–1.05 倍縮放及最多 10% 亮度／對比變动；驗證和測試只縮放。

加權交叉熵的類別權重為 `N / (2 × 該類別訓練張數)`；AdamW 學習率、weight decay 均為 1e-4，cosine 衰減，CUDA 混合精度。每模型開始前重設相同種子，停用 cuDNN benchmark、啟用 deterministic；不保證跨硬體／版本逐位元相同。**每個模型只以驗證集 PNEUMONIA F1 選保存權重，平手取較早輪次；跨模型推薦也只以驗證 F1 比較。** 測試每輪評估是 PDF 的曲線要求，逐輪最高值只是描述性結果，未用於選權重或此次調參。這裡 F1 是肺炎正類的二元 F1，並非 macro F1。

## 在 D 槽重現

先於指定專案資料夾開啟 PowerShell。現有 `.venv` 已建立；在新的環境可依序執行：

```powershell
New-Item -ItemType Directory -Force data\temp, data\pip_cache | Out-Null
$env:TEMP = (Join-Path (Get-Location) 'data\temp')
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = (Join-Path (Get-Location) 'data\pip_cache')
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe download_data.py --output-dir data
.\.venv\Scripts\python.exe verify_dataset_source.py
.\.venv\Scripts\python.exe split_dataset.py
.\.venv\Scripts\python.exe diagnose_split.py
```

`verify_dataset_source.py` 固定下載官方第 2 版，保存在 `data/source_verification/`。資料需有 `data/chest_xray/train`、`val`、`test` 的兩類子資料夾。建立切分後，訓練程式每次重新驗證清單、原始檔 SHA256 及官方測試成員；不符就停止。沒有 CUDA 也會停止。

以新的結果資料夾重跑，避免覆蓋歷史紀錄：

```powershell
.\.venv\Scripts\python.exe train.py --data-dir data\chest_xray --split-manifest data\source_verification\corrected_split_manifest.json --output-dir results\my_group_run --models resnet18 resnet50 resnet101 --epochs 12 --batch-size 32 --num-workers 2
.\.venv\Scripts\python.exe compare_models.py --results-dir results\my_group_run
.\.venv\Scripts\python.exe verify_pretrained.py --results-dir results\my_group_run
.\.venv\Scripts\python.exe audit_corrected_experiment.py --results-dir results\my_group_run
```

若要改種子／驗證比例，須同步修改 `split_dataset.py` 的建置參數並重新產生切分。`--legacy-image-split` 僅供重現舊實驗，不用於正式報告。`run_corrected_experiments.py` 是本次完整執行入口，目標資料夾固定，已有完成結果時會拒絕覆蓋。

## 修正實驗結果

RESULT_TABLE

數值由 `summary.json` 和 36 筆 `combined_epoch_metrics.csv` 產生。`audit_corrected.json` 已核對每輪 confusion-derived accuracy／precision／recall／F1、選模輪次，並重新以 GPU 載入三份交付權重重現驗證及測試指標。新舊切分與隨機執行設定同時有變動，不能把差值完全歸因於移除洩漏。

正式結果含：

- `training.log`、`run_metadata.json`、`summary.json`、`comparison_summary.json`。
- `comparison_accuracy.png`、`comparison_f1.png`、`comparison_validation.png`。
- `comparison_highest_test_confusion.png` 與 `comparison_final_test_confusion.png`。
- `comparison_results_screenshot.jpg`：本機結果頁的實際截圖，用於 Report Spec 要求。
- `comparison_log_excerpt.png`：保存日誌中最高觀察成績原始行的排版圖，完整日誌也一併保留。
- 三模型 `*_best_validation.pt`（大型權重不納入 Git）。

可編輯報告：`LAB1_315553010_楊敦傑.docx`。此次保留既有版面，更新切分說明、數值、圖與依據新結果撰寫的討論。若要依記錄重新產生完整報告，可用 `.\.venv\Scripts\python.exe build_docx_report.py`，但會覆蓋自行修改的全文；`update_corrected_report.py` 用於更新實驗相關段落。報告微調後由本人匯出規格要求的 `LAB1_315553010_楊敦傑.pdf`。

## 單張影像推論

```powershell
.\.venv\Scripts\python.exe inference.py --checkpoint results\corrected_group_split_20261002\resnet101_best_validation.pt --image path\to\image.jpeg
```

輸出兩類 softmax 機率。

## Git 與歷史紀錄

目前未建立 `.git`，依本人要求先確認實驗與報告，再建立並自行推送 GitHub。GitHub 網址可於推送後補入 Word 的待填欄。`.gitignore` 排除 `.venv/`、`data/`、大型 `.pt`、封存、報告備份與 QA 檔；數值、切分清單、訓練日誌與圖可納入 Git。

`archive_from_c/` 保留先前 C 槽產出；`report_backups/before_split_correction_20261002/` 保存此次修正前程式與 Word；較早的實驗核對在 `EXPERIMENT_REVIEW_20261002.md`，此次原因與處理則以 `DATASET_SPLIT_REVIEW_20261002.md` 為準。Word 結構、表格、圖片及數值已檢查，環境無可用的 LibreOffice／Word COM，未能自動驗證實際分頁；本人微調及匯出 PDF 時仍須檢查版面。
'''
(ROOT/'README.md').write_text(text.replace('RESULT_TABLE','\n'.join(lines)).replace('数字','數字').replace('變动','變動'),encoding='utf-8')
check='''# Lab 1 Report Spec 對照

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
- 未建立 .git；本人确认實驗與報告後才建立。PDF 匯出與 GitHub 連結由本人處理。
'''
(ROOT/'REPORT_SPEC_CHECKLIST.md').write_text(check.replace('确认','確認'),encoding='utf-8')
print('Updated README and Report Spec checklist using corrected results.')
