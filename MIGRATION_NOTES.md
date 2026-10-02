# 資料搬移與 Git 整理紀錄

目前使用的程式、資料、虛擬環境、三模型結果和可編輯 Word 報告位於本專案主目錄。先前在 C 槽產生的內容按原樣放入 `archive_from_c/outputs` 與 `archive_from_c/work`，供追溯；這份封存已由 `.gitignore` 排除，避免建立 Git 儲存庫時提交重複資料、下載快取和舊版報告。

| 來源 | D 槽封存 | 核對結果 |
| --- | --- | --- |
| C 槽 `outputs` | `archive_from_c/outputs` | 29 個可讀檔案，約 0.131 GiB；來源目錄已移走 |
| C 槽 `work` | `archive_from_c/work` | 42,758 個可讀檔案，約 6.835 GiB；與搬移前的可讀檔案數及大小相同 |

**待處理的 Windows 權限項目：** C 槽 `work/download_helper` 的套件子目錄（例如 `bin`、`kagglehub`）拒絕目前執行帳號讀取或移動。一般檔案權限已申請，但 Windows ACL 仍拒絕存取，因此該來源目錄尚留在 C 槽。它是先前下載資料時用的輔助套件副本；目前 D 槽 `.venv` 已安裝專案所需套件，模型訓練、推論及報告無須讀取這個舊目錄。請由擁有該目錄權限的 Windows 帳號搬移或清理，或授予目前執行帳號存取權後再請我完成搬移。`archive_from_c/work/download_helper` 目前僅有上次搬移建立的空目錄，**不代表原套件已完整封存**。

建立 `.git` 前可先檢查 `.gitignore` 與待提交檔案。依本人先前要求，目前尚未建立 `.git`，GitHub 推送由本人處理。提交報告前，仍需在 DOCX 中補 GitHub 連結，並自行匯出符合命名規格的 PDF。
