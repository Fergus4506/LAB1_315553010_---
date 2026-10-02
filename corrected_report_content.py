"""Generate result-grounded text for the corrected grouped-split report."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
ORDER=('resnet18','resnet50','resnet101')
NAMES={n:n.replace('resnet','ResNet') for n in ORDER}
def pct(v):return f'{v*100:.2f}%'
def discussion_content(summary,metadata,rows):
    diagnosis=json.loads((ROOT/'results'/'corrected_group_split_20261002'/'split_diagnosis.json').read_text(encoding='utf-8'))
    scores={n:summary[n]['final_test_at_validation_selected_epoch'] for n in ORDER}
    selected_rows={n:next(r for r in rows if r['model']==n and int(r['epoch'])==summary[n]['validation_selected_epoch']) for n in ORDER}
    last_rows={n:max((r for r in rows if r['model']==n),key=lambda r:int(r['epoch'])) for n in ORDER}
    out=[('Heading 1','討論'),('Heading 2','資料來源與切分修正')]
    exact=diagnosis['identity_audit']['sha256'];decoded=diagnosis['identity_audit']['decoded_sha256']
    out.append(('Normal',f"本次重新下載指定 Kaggle 資料集第 2 版，逐檔核對相對路徑、檔案大小及 ZIP CRC32；本機 5856 張影像全部吻合，沒有漏檔或多餘影像。資料卡文字寫 5863 張，與實際可下載的 5856 張影像不同，不能據此判定本機資料不足。原始資料中有 {exact['repeated_groups']} 組位元組完全相同的影像，均未跨官方 train／val／test 分區；解碼為 RGB 後另查得 {decoded['repeated_groups']} 組相同像素影像。"))
    out.append(('Normal','原始訓練集的重複影像屬資料本身的問題，但先前逐張分層隨機切分，將其中 9 組完全相同影像分到訓練與驗證集，是本專案切分方法的疏漏。此次保留全部原始影像，將同一檔名組、相同檔案雜湊及相同解碼像素以連通分組合併，再依類別抽取驗證組；官方 val 保留在驗證集，官方 test 的 624 張影像完整保留。修正後三分區之間的檔名組、檔案雜湊與解碼像素雜湊重疊均為 0。'))
    out.append(('Normal','肺炎檔名的 person 數字會在 bacteria 與 virus 命名範圍重用，所以單看數字重疊不能判定同一病患。本次以 person 編號加病原類型為檔名組；正常影像則依 IM 或 NORMAL2-IM 序號分組。這可避免同一已知影像序列跨訓練與驗證，但資料沒有經核實的臨床病患識別碼，因此仍不能宣稱已證實病患層級完全獨立。'))
    out.append(('Heading 2','模型表現與計算成本'))
    out.append(('Normal','修正切分後，各模型依驗證 F1 選定權重的測試結果為：'+ '；'.join(f"{NAMES[n]} 第 {summary[n]['validation_selected_epoch']} 輪，accuracy {pct(scores[n]['accuracy'])}、F1 {scores[n]['f1']:.4f}" for n in ORDER)+'。逐輪最高測試準確率則為 '+ '、'.join(f"{NAMES[n]} {pct(summary[n]['highest_observed_test_accuracy']['accuracy'])}（第 {summary[n]['highest_observed_test_accuracy']['epoch']} 輪）" for n in ORDER)+'。最高觀察值反映多次評估中的最大值，交付權重仍只依驗證 F1 決定，不能把兩者當成同一成績。'))
    best=max(ORDER,key=lambda n:scores[n]['accuracy']);runner=sorted(ORDER,key=lambda n:scores[n]['accuracy'],reverse=True)[1]
    difference=(scores[best]['accuracy']-scores[runner]['accuracy'])*100
    image_diff=(scores[best]['tn']+scores[best]['tp'])-(scores[runner]['tn']+scores[runner]['tp'])
    out.append(('Normal',f"以各模型驗證選定權重的測試結果描述，{NAMES[best]} 準確率最高，比次高的 {NAMES[runner]} 多判對 {image_diff} 張，差 {difference:.2f} 個百分點。三者總參數約 1118 萬、2351 萬與 4250 萬，本次 12 輪耗時分別為 "+'、'.join(f"{summary[n]['elapsed_seconds']/60:.1f} 分鐘" for n in ORDER)+'。本次三者耗時差距很小；此為包含資料讀取、訓練、驗證、測試與圖表輸出的整體時間，尚未分解純 GPU 運算與其他成本，不能據此認定更深網路具有相同計算成本。成績也只來自單次實驗。ResNet18 使用 ImageNet V1 權重，ResNet50 與 ResNet101 使用 V2，且可微調參數量不同，因此無法將差異完全歸因於深度；本次沒有從零訓練對照，不能量化預訓練本身的改善幅度。'))
    out.append(('Heading 2','曲線與驗證選模的限制'))
    out.append(('Normal','在驗證選定輪次，訓練與測試準確率的差距分別為 '+ '、'.join(f"{NAMES[n]} {(float(selected_rows[n]['train_accuracy'])-scores[n]['accuracy'])*100:.2f} 個百分點" for n in ORDER)+'；第 12 輪的訓練／測試準確率分別為 '+ '、'.join(f"{NAMES[n]} {pct(float(last_rows[n]['train_accuracy']))}／{pct(float(last_rows[n]['test_accuracy']))}" for n in ORDER)+'。訓練指標是在隨機增強與逐 batch 更新過程中累計，測試則以輪末權重在未增強影像上計算，兩者推論條件不同。若訓練曲線持續改善而測試沒有同步改善，與過度擬合或資料分布差異相容，但本次未做控制實驗，無法分辨各原因的影響。'))
    out.append(('Normal','修正後各模型最高驗證 F1 為 '+ '、'.join(f"{NAMES[n]} {summary[n]['best_validation_f1']:.4f}" for n in ORDER)+'，對應測試 F1 為 '+ '、'.join(f"{scores[n]['f1']:.4f}" for n in ORDER)+'。驗證 F1 與官方測試 F1 仍有落差，顯示驗證改善未完全轉移到官方測試集。此次同時修正切分成員與隨機執行設定，沒有只改切分的控制實驗，因此無法量化資料洩漏造成的影響。正式報告以本次分組切分實驗為準。'))
    out.append(('Heading 2','混淆矩陣與類別辨識'))
    out.append(('Normal','驗證選模後的錯誤為 '+ '；'.join(f"{NAMES[n]}：FP {scores[n]['fp']}、FN {scores[n]['fn']}，肺炎 recall {pct(scores[n]['recall'])}，正常 specificity {pct(scores[n]['tn']/(scores[n]['tn']+scores[n]['fp']))}" for n in ORDER)+f"。FP 表示正常影像被判為肺炎，FN 表示肺炎被漏判。相較 {NAMES[runner]}，{NAMES[best]} 的 FP 從 {scores[runner]['fp']} 變為 {scores[best]['fp']}，FN 從 {scores[runner]['fn']} 變為 {scores[best]['fn']}，總錯誤從 {scores[runner]['fp']+scores[runner]['fn']} 變為 {scores[best]['fp']+scores[best]['fn']}，與多判對 {image_diff} 張的 accuracy 差異一致。結合 precision、recall 與 specificity，可判斷較高 accuracy 是否改善兩類辨識，或只是改變兩種錯誤的取捨。"))
    vn=metadata['validation_counts']['0']/sum(metadata['validation_counts'].values())
    out.append(('Normal',f"驗證集 NORMAL 比例為 {pct(vn)}，官方測試集則為 37.50%。當正常與肺炎的錯誤率不同，類別比例改變會改變整體 accuracy；類別比例本身並不證明假陽性率增加。此處 F1 以 PNEUMONIA 為正類，並非 macro F1，較高的肺炎 F1 不代表正常影像也同樣容易辨識。加權交叉熵依訓練類別比例設定權重，但沒有類別權重的對照實驗，尚不能認定它是錯誤取捨的原因。"))
    out.append(('Heading 2','後續改善方向'))
    dominant='正常影像的假陽性' if sum(s['fp'] for s in scores.values())>=sum(s['fn'] for s in scores.values()) else '肺炎影像的漏判'
    out.append(('Normal',f"本次錯誤主要集中在{dominant}，後續可先檢視錯誤影像的曝光、尺寸、邊框與標籤，再在分組驗證集比較學習率、解凍範圍、較保守的增強及類別權重。分類閾值應僅由驗證集選擇，並同時檢查 precision、recall 與 specificity，以了解降低 FP 是否增加 FN。重複多個訓練種子及分組切分，可用平均值與變異判斷目前僅數張影像的模型差距是否穩定；若有可信病患資料與外部資料集，應再檢查病患獨立性與跨來源泛化。這些改善尚未實測，不能承諾提升幅度。"))
    out.append(('Normal','本次依作業要求保留每輪測試曲線，測試資料沒有參與梯度更新、模型權重選擇或本次超參數調整。不過，反覆觀察同一測試集及報告最高值仍有多次比較的樂觀偏差；本研究採單次種子、單一公開來源，也不能作出統計顯著性或臨床效能的結論。'))
    return out

def add_discussion(doc,summary,metadata,rows):
    for style,text in discussion_content(summary,metadata,rows):doc.add_paragraph(text,style=style)
