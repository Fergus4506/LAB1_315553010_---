# Lab 1：胸部 X 光肺炎分類

學號：315553010　姓名：楊敦傑

本專案以自訂 PyTorch Dataset／DataLoader，使用 ImageNet 預訓練 ResNet18、ResNet50、ResNet101，分類 NORMAL（0）與 PNEUMONIA（1）。三模型各完成 12 輪 GPU 訓練；交付權重已重新載入，重現驗證與測試指標。

**正式實驗唯一位置：`results/corrected_group_split_20261002/`。** Word、README 與圖表均引用這次修正切分後的結果。專案根目錄為 `D:\人工智慧醫學影像應用\lab1\LAB1_315553010_楊敦傑`。

## 1. 專案結構

```text
LAB1_315553010_楊敦傑/
├─ README.md                           專案說明與重現步驟
├─ requirements.txt                    非 PyTorch 相依套件
├─ .gitignore                          排除環境、資料、快取與大型權重
├─ LAB1_315553010_楊敦傑.docx             可編輯報告
├─ DATASET_SPLIT_REVIEW_20261002.md      資料來源與切分診斷
├─ REPORT_SPEC_CHECKLIST.md             PDF 報告規格對照
├─ download_data.py                     固定官方第 2 版、擷取唯一影像
├─ verify_dataset_source.py             與官方 ZIP 逐檔核對
├─ split_dataset.py                     建立與驗證分組切分清單
├─ train.py                            GPU 訓練與逐輪紀錄
├─ run_corrected_experiments.py         三模型訓練、圖表及權重稽核入口
├─ compare_models.py                   整合結果與繪圖
├─ verify_pretrained.py                 核對官方預訓練特徵與二類輸出
├─ audit_corrected_experiment.py        核對 CSV、選模與 GPU 權重重評估
├─ inference.py                        單張影像推論
├─ build_docx_report.py                 由正式結果重建完整 Word
├─ update_corrected_report.py           更新既有 Word 的實驗段落並備份
├─ corrected_report_content.py          結果導向的討論內容
├─ build_result_page.py                 建立結果展示 HTML
├─ check_corrected_report.py            Word 表格、圖片與數值核對
├─ tools/cleanup_project.ps1            舊檔清理：預覽後可明確執行
├─ .venv/                              Python 內建 venv（不納入 Git）
├─ data/                               原始影像、核對清單、官方權重快取
│  ├─ chest_xray/{train,val,test}/      原始 5856 張影像，保持不變
│  ├─ source_verification/              來源驗證及 corrected_split_manifest.json
│  └─ torch_cache/                     官方 ImageNet 權重
└─ results/corrected_group_split_20261002/
   ├─ summary.json、run_metadata.json   成績與環境設定
   ├─ combined_epoch_metrics.csv        三模型共 36 筆逐輪指標
   ├─ training.log                     完整訓練日誌
   ├─ split_manifest.json              本次實驗切分清單副本
   ├─ source_verification_summary.json、split_diagnosis.json
   ├─ audit_corrected.json、pretrained_verification.json
   ├─ report_verification.json          Word 內容核對與檔案雜湊
   ├─ comparison_*.png、comparison_results_screenshot.jpg
   └─ *_best_validation.pt             三份交付權重（不納入 Git）
```

快取、報告更新備份及 QA 目錄會由相關程式按需建立。若尚有 `archive_from_c/`、舊的 `results/` 根目錄檔案或 `results/resnet101/`，它們是待清理的舊產物，不能當作正式結果；清理方式見第 7 節。

## 2. 資料來源與切分

來源：[Kaggle Chest X Ray Images Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)，固定第 2 版。本機 **5856 張**影像的相對路徑、大小及 CRC32 全部符合官方 ZIP，沒有缺少或額外影像。資料卡文字的 5863 張與官方實際影像數不同；ZIP 的重複巢狀副本只擷取一次，`__MACOSX`／`._` 是資源資訊，不當作 X 光影像。

原始資料有 30 組完全相同的檔案／像素影像，均未跨官方分區。舊程式按單張影像抽驗證集，造成 9 組相同影像及 270 組完整檔名組跨訓練與驗證。此次修正屬於專案切分錯誤，並已重新訓練全部三模型。

修正程式以檔名組、原檔 SHA256、解碼 RGB 尺寸與像素 SHA256 建立連通分組，整組放同一區。肺炎檔名組包含 `person編號_bacteria`／`person編號_virus`，正常影像使用 `IM-序號`／`NORMAL2-IM-序號`。官方 val 保留於驗證集，從官方 train 按類別選約 15% 作驗證，官方 test 完全保持原樣。原始重複影像也全部保留於同區。

| 分區 | NORMAL | PNEUMONIA | 總數 |
| --- | ---: | ---: | ---: |
| 訓練 | 1140 | 3294 | 4434 |
| 驗證 | 209 | 589 | 798 |
| 測試 | 234 | 390 | 624 |

修正後檔名組、原始檔雜湊、像素雜湊與連通分組的跨區重疊均為 **0**。每次訓練重新核對影像 SHA256、清單及官方 test 成員，資料不符就停止。未取得可信臨床病患識別碼，因此不能宣稱已證明病患獨立，也未排除所有近似影像。詳見資料診斷文件。

## 3. 模型選擇與訓練方法

### 為何選 ResNet18、50、101

殘差捷徑有助於深層網路最佳化。ResNet18 的 basic block 與較少參數提供小模型基準；ResNet50 使用 bottleneck，提供容量與成本的比較點；ResNet101 明顯增加深度，檢查更大容量是否有實際收益。更深網路也可能增加過度擬合與運算成本，不能保證更準確。

| 模型 | 二類總參數 | 預訓練權重 |
| --- | ---: | --- |
| ResNet18 | 11,177,538 | ResNet18_Weights.IMAGENET1K_V1 |
| ResNet50 | 23,512,130 | ResNet50_Weights.IMAGENET1K_V2 |
| ResNet101 | 42,504,258 | ResNet101_Weights.IMAGENET1K_V2 |

三者均**使用 pretrained model**；最後 `fc` 重新初始化為二類輸出，符合 PDF 限制。凍結 conv1、bn1、layer1、layer2，並固定其中 BatchNorm 為 eval；微調 layer3、layer4、fc。交付權重 conv1 與對應官方權重逐元素相同，預訓練核對通過。預訓練可利用既有影像特徵降低從零訓練成本，本次沒有從零訓練對照，未量化改善幅度。V1／V2 權重不同，也使三者的差異無法完全歸因於深度。

### 共同設定與選模規則

- Python 3.12.14 內建 venv；PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、CUDA 12.8；GPU 為 RTX 4070。
- 每模型 12 輪，batch size 32、DataLoader workers 2；224 × 224 RGB、ImageNet 正規化。
- 訓練增強：±7° 旋轉、最多 3% 平移、0.95–1.05 倍縮放、10% 亮度／對比變動；驗證／測試僅縮放及正規化。
- 加權交叉熵：`N / (2 × 該類別訓練張數)`；AdamW lr 1e-4、weight decay 1e-4、cosine 衰減、CUDA AMP。
- 每模型重設種子 315553010；停用 cuDNN benchmark、啟用 deterministic。不同硬體／版本仍可能有數值差異。
- 各模型按**最高驗證集肺炎 F1**保存權重，精確平手取較早輪次；跨模型推薦也只看驗證 F1。
- 按 PDF 要求每輪記錄測試 accuracy／F1。最高觀察測試值只用於展示，沒有用於選權重或此次調參。F1 是肺炎正類二元 F1，並非 macro F1。

## 4. 建置與重現（Windows PowerShell）

### 建立 venv 與確認 GPU

請先在專案根目錄開啟 PowerShell。以下安裝需 NVIDIA 驅動支援所用 CUDA wheel；現有專案已具備可用環境，不需重建它。

```powershell
Set-Location -LiteralPath 'D:\人工智慧醫學影像應用\lab1\LAB1_315553010_楊敦傑'
New-Item -ItemType Directory -Force data\temp, data\pip_cache | Out-Null
$env:TEMP = (Join-Path (Get-Location) 'data\temp')
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = (Join-Path (Get-Location) 'data\pip_cache')
$env:TORCH_HOME = (Join-Path (Get-Location) 'data\torch_cache')
$env:MPLCONFIGDIR = (Join-Path (Get-Location) 'data\mplcache')
$env:CUDA_CACHE_PATH = (Join-Path (Get-Location) 'data\cuda_cache')
$env:PYTHONIOENCODING = 'utf-8'

# 只在尚未建立 .venv 的新環境執行
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
```

訓練必須使用 CUDA，沒有 GPU 就停止。直接呼叫 venv 的 python，不需啟動 activate，也不會使用全域 Python 套件。

### 下載、核對、建立切分

```powershell
.\.venv\Scripts\python.exe download_data.py --output-dir data
.\.venv\Scripts\python.exe verify_dataset_source.py
.\.venv\Scripts\python.exe split_dataset.py
```

使用固定官方版本下載，不需 Kaggle 帳戶。ZIP 保存在 `data/source_verification/`；兩個來源工具均會重用已存在的 ZIP。清理 ZIP 後再次來源核對會重新下載約 2.46 GB。已有影像經驗證相符時不會重複擷取。

`split_dataset.py` 使用固定預設比例 0.15 和種子 315553010；若改動它們，必須重新產生清單，並讓訓練設定一致。正式實驗的切分副本已保存在結果資料夾。

### 完整重跑三種模型

```powershell
.\.venv\Scripts\python.exe run_corrected_experiments.py --output-dir results\reproduction_run
```

此入口依序訓練三模型、產生比較圖、核對 pretrained、以 GPU 重新評估三份交付權重。完成後寫入 `experiment_completed.json`。結果目錄必須位於專案內；已有完成成績的目錄會拒絕覆蓋，第二次重跑請改用新的目錄名稱。

如只需檢查現有正式結果：

```powershell
.\.venv\Scripts\python.exe verify_pretrained.py --results-dir results\corrected_group_split_20261002
.\.venv\Scripts\python.exe audit_corrected_experiment.py --results-dir results\corrected_group_split_20261002
```

重新 GPU 稽核需要 `data/chest_xray/`、切分清單及三份 `.pt`。這些大型檔案不納入 Git，其他人取得原始碼後需依上述步驟下載並訓練。正式圖表、CSV 和 JSON 可直接閱讀。

## 5. 實驗結果與討論

### 由驗證 F1 選定權重的正式測試成績

| 模型 | 選定輪次 | 驗證 F1 | 測試 accuracy | 測試 precision | 測試 recall | 測試 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 9 | 0.9889 | 85.58% | 0.8138 | 0.9974 | 0.8963 |
| ResNet50 | 10 | 0.9880 | 87.66% | 0.8380 | 0.9949 | 0.9097 |
| ResNet101 | 7 | 0.9897 | 89.26% | 0.8549 | 0.9974 | 0.9207 |

依驗證 F1 推薦 ResNet101；其測試正確 557／624 張。相較 ResNet50，總錯誤從 77 降為 67，假陽性從 75 降為 66、假陰性從 2 降為 1，accuracy 增加 1.60 個百分點。三模型均以較高肺炎 recall 換來較多正常影像誤判為肺炎。這是單次切分與種子的觀察，不能推論更深模型普遍較好。

### PDF 要求的逐輪最高觀察值

| 模型 | 最高測試 accuracy | 最高測試 F1 | 各模型整段耗時 |
| --- | ---: | ---: | ---: |
| ResNet18 | 92.79%（第 2 輪） | 0.9441（第 2 輪） | 522.8 秒 |
| ResNet50 | 93.27%（第 2 輪） | 0.9475（第 2 輪） | 522.6 秒 |
| ResNet101 | 91.35%（第 3 輪） | 0.9349（第 9 輪） | 524.1 秒 |

這些最高值與正式交付權重成績不同，最高 accuracy 和最高 F1 也可能來自不同輪次。耗時包括資料讀取、評估及輸出，不能用這次近似耗時宣稱三者純 GPU 運算成本相同。驗證 F1 近 0.99，但正式測試 F1 約 0.90–0.92，表示該驗證分區無法完全代表官方測試資料。

`audit_corrected.json` 核對全部 36 筆指標、混淆矩陣及選模規則，並以 GPU 重評估三份保存權重，驗證與測試成績全部重現。完整討論在 Word；結果頁 `report_results.html`、實際截圖、曲線與兩種混淆矩陣均位於正式結果資料夾。

### 可以如何提升準確率

先檢視假陽性影像及標籤，再只用分組驗證集比較學習率、解凍範圍、輸入解析度、保守增強、類別權重與分類閾值，同時查看 precision、recall 和 specificity。增加種子重複試驗可檢查差異是否穩定；若取得可信病患資料，應採病患分組與外部驗證。這些是後續實驗建議，尚未實測，不能宣稱有確定增幅。

## 6. 推論、Word 與報告規格

```powershell
.\.venv\Scripts\python.exe inference.py --checkpoint results\corrected_group_split_20261002\resnet101_best_validation.pt --image 'data\chest_xray\test\NORMAL\IM-0001-0001.jpeg'
```

若使用重新訓練的權重，替換為 `results/reproduction_run/resnet101_best_validation.pt`。推論輸出預測類別及兩類 softmax 機率；推論可使用 CPU，訓練與實驗 GPU 稽核需要 CUDA。

Word 可直接微調；依 Report Spec 含 Introduction、Experiment setups、Results 與結果導向的 Discussion，圖表及數字均來自正式結果。其他工具：

```powershell
# 檢查既有報告的表格、圖片和主要討論數字
.\.venv\Scripts\python.exe check_corrected_report.py
# 更新既有版型中的實驗內容，會先在 report_backups 建立備份
.\.venv\Scripts\python.exe update_corrected_report.py
# 重新建立全文；會覆蓋自行修改內容，需先備份
.\.venv\Scripts\python.exe build_docx_report.py
```

報告生成工具固定引用本次正式結果，不會自動把其他重跑資料夾取代為正式實驗。Word 的 4 個表格、6 張嵌入圖及數值核對通過；環境缺少 LibreOffice／可用 Word COM，**實際分頁未能自動驗證**。請本人微調、檢查分頁後匯出 `LAB1_315553010_楊敦傑.pdf`，並在自行推送 GitHub 後補入報告連結。完整對照見 `REPORT_SPEC_CHECKLIST.md`。

## 7. 清理與 Git

本次 Word 已完成。工具的自動核准審核拒絕刪除操作，因此舊檔案目前尚未由工具刪除。清理腳本已列出固定的舊目錄、舊模型與不再使用的程式，保護原始影像、venv、官方預訓練快取、正式結果與 Word；遇到重新解析點或超出專案的路徑會停止。

請在專案根目錄執行，第一行只預覽，第二行才實際刪除：

```powershell
& .\tools\cleanup_project.ps1
& .\tools\cleanup_project.ps1 -Execute
```

清理會刪除 C 槽舊產物副本、舊實驗、舊報告備份／QA、過期程式、可再生快取及冗餘 ZIP；保留全部 5856 張原始影像與三模型正式權重。執行前核對 Word SHA256 與通過的內容稽核，完成後記錄 `results/corrected_group_split_20261002/cleanup_summary.json`，不會建立 `.git`。

依本人要求，確認實驗與報告後才建立 Git，推送由本人處理。`.gitignore` 排除 `.venv/`、`data/`、`.pt`、快取、舊產物與報告備份；保留正式 CSV、JSON、圖、訓練日誌及 Word。Git 儲存庫是程式與實驗紀錄，取得後重建環境、下載資料並重訓即可產生被排除的大型檔案。


