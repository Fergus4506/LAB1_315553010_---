"""Generate a local results page from completed summaries and original log lines."""
import html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'corrected_group_split_20261002'
ORDER=('resnet18','resnet50','resnet101')
summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
if set(summary)!=set(ORDER):raise RuntimeError('All three model summaries required')
logs=(OUT/'training.log').read_text(encoding='utf-8').splitlines()
records=[];selected=[];excerpts=[]
for name in ORDER:
    item=summary[name];acc=item['highest_observed_test_accuracy'];f1=item['highest_observed_test_f1'];score=item['final_test_at_validation_selected_epoch']
    records.append(f"<tr><td>{name}</td><td>{100*acc['accuracy']:.2f}%</td><td>{acc['epoch']}</td><td>{f1['f1']:.4f}</td><td>{f1['epoch']}</td></tr>")
    selected.append(f"<tr><td>{name}</td><td>{item['validation_selected_epoch']}</td><td>{item['best_validation_f1']:.4f}</td><td>{100*score['accuracy']:.2f}%</td><td>{score['f1']:.4f}</td></tr>")
    for epoch in sorted({acc['epoch'],f1['epoch']}):
        matches=[line for line in logs if line.startswith(f'{name} epoch {epoch:02d}/')]
        assert len(matches)==1
        excerpts.append(html.escape(matches[0]))
page='''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>Lab1 修正實驗結果</title><style>
body{margin:0;background:white;color:#18212b;font-family:"Microsoft JhengHei",Arial,sans-serif;font-size:20px}main{max-width:950px;margin:25px 0;padding:0 24px}h1{font-size:28px;color:black;margin:0 0 12px}h2{font-size:22px;color:black;margin:24px 0 10px}p{line-height:1.6;margin:8px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border:1px solid #ccd3d9;padding:9px 12px;text-align:center}th{background:#e8eef3}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f4f6;border:1px solid #ccd3d9;padding:14px;font:18px/1.55 Consolas,monospace}small{font-size:17px;color:#444}
</style></head><body><main><h1>Lab1 分組切分修正實驗結果</h1><p>315553010 楊敦傑　ResNet18／50／101　各 12 輪　RTX 4070　Python venv</p>
<p>資料來源核對：官方第 2 版 5856 張全部相符。訓練／驗證／測試：4434／798／624；檔名組、原始檔及解碼像素雜湊跨區重疊均為 0。</p>
<h2>逐輪最高觀察測試成績</h2><table><thead><tr><th>模型</th><th>最高 Test Accuracy</th><th>輪次</th><th>最高 Test F1</th><th>輪次</th></tr></thead><tbody>HIGHEST</tbody></table>
<h2>最高成績所在輪次的原始訓練日誌</h2><pre>LOGS</pre>
<h2>僅依驗證 F1 選定權重的結果</h2><table><thead><tr><th>模型</th><th>選定輪次</th><th>Val F1</th><th>Test Accuracy</th><th>Test F1</th></tr></thead><tbody>SELECTED</tbody></table>
<p><small>最高觀察值用於作業要求的呈現，未用來選權重或此次調參。F1 為 PNEUMONIA 正類的二元 F1。數值直接讀取保存的 summary.json 與 training.log。</small></p></main></body></html>'''
(OUT/'report_results.html').write_text(page.replace('HIGHEST',''.join(records)).replace('LOGS','\n'.join(excerpts)).replace('SELECTED',''.join(selected)),encoding='utf-8')
print('Local results page saved.')
