from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=ROOT/'corrected_report_content.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace("    old=json.loads((ROOT/'results'/'comparison_summary.json').read_text(encoding='utf-8'))\n",'')
a="+'。舊實驗的最高驗證 F1 則為 '+ '、'.join(f\"{old[n]['best_validation_f1']:.4f}\" for n in ORDER)+'。此次同時改變切分成員、每模型重設隨機種子，並停用 cuDNN benchmark、啟用 deterministic 設定；因此新舊差異不能當成資料洩漏影響的單一因果估計。舊結果保留供追查，正式報告以此次修正實驗為準。'"
b="+'。驗證 F1 與官方測試 F1 仍有落差，顯示驗證改善未完全轉移到官方測試集。此次同時修正切分成員與隨機執行設定，沒有只改切分的控制實驗，因此無法量化資料洩漏造成的影響。正式報告以本次分組切分實驗為準。'"
assert a in s;s=s.replace(a,b);p.write_text(s,encoding='utf-8')
p=ROOT/'train.py';s=p.read_text(encoding='utf-8-sig');a=s.index('def stratified_split(');b=s.index('def make_transforms(',a);s=s[:a]+s[b:]
s='\n'.join(line for line in s.split('\n') if 'parser.add_argument("--legacy-image-split"' not in line)
s=s.replace('bool(args.legacy_image_split)','False').replace('not args.legacy_image_split','True')
a=s.index('    train_all = collect_samples(');b=s.index('    train_tf, eval_tf = make_transforms',a)
s=s[:a]+'''    splits, split_info = load_samples(args.data_dir, args.split_manifest)
    if split_info['seed'] != args.seed or split_info['validation_fraction'] != args.validation_fraction:
        raise ValueError("Manifest seed/fraction differ from training arguments")
    train_samples, validation_samples, test_samples = (splits[s] for s in ('train','validation','test'))
    (args.output_dir / 'split_manifest.json').write_bytes(args.split_manifest.read_bytes())
'''+s[b:]
s=s.replace('(hashlib.sha256(args.split_manifest.read_bytes()).hexdigest() if True else None)','hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()')
s=s.replace('        if True:\n            random.seed(args.seed)\n            np.random.seed(args.seed % (2**32))\n            torch.manual_seed(args.seed)\n            torch.cuda.manual_seed_all(args.seed)','        random.seed(args.seed)\n        np.random.seed(args.seed % (2**32))\n        torch.manual_seed(args.seed)\n        torch.cuda.manual_seed_all(args.seed)')
assert 'legacy_image' not in s;p.write_text(s,encoding='utf-8')
p=ROOT/'compare_models.py';s=p.read_text(encoding='utf-8-sig');a=s.index('    logs = {',s.index('def plot_log_excerpt'));b=s.index('    display_lines = []',a);s=s[:a]+s[b:]
a=s.index('        generated_log = RESULTS / "training.log"');b=s.index('        lines = log_path.read_text',a);s=s[:a]+'        log_path = RESULTS / "training.log"\n'+s[b:]
s=s.replace('    if not any(row["model"] == "resnet101" for row in rows):\n        rows += load_rows(RESULTS / "resnet101" / "epoch_metrics.csv")\n','')
s=s.replace('    if "resnet101" not in summary:\n        summary.update(json.loads((RESULTS / "resnet101" / "summary.json").read_text(encoding="utf-8")))\n','')
s=s.replace('default=RESULTS)', 'default=ROOT / "results" / "corrected_group_split_20261002")');p.write_text(s,encoding='utf-8')
p=ROOT/'verify_pretrained.py';s=p.read_text(encoding='utf-8-sig').replace('"resnet101/resnet101_best_validation.pt"','"resnet101_best_validation.pt"')
s=s.replace('default=ROOT/"results")','default=ROOT/"results"/"corrected_group_split_20261002")')
s=s.replace('        if name=="resnet101" and (args.results_dir/"resnet101_best_validation.pt").exists():\n            checkpoint_path=args.results_dir/"resnet101_best_validation.pt"\n','');p.write_text(s,encoding='utf-8')
p=ROOT/'build_docx_report.py';s=p.read_text(encoding='utf-8-sig');a=s.index('    metadata101 = metadata');b=s.index('    with (RESULTS / "combined_epoch_metrics.csv")',a);s=s[:a]+s[b:]
a=s.index('    if any(metadata[key] != metadata101[key]');b=s.index('    return summary, metadata, rows',a);s=s[:a]+s[b:];p.write_text(s,encoding='utf-8')
p=ROOT/'run_corrected_experiments.py';s=p.read_text(encoding='utf-8-sig').replace('import json,os,subprocess,sys','import argparse,json,os,subprocess,sys')
s=s.replace("OUT=ROOT/'results'/'corrected_group_split_20261002'", "parser=argparse.ArgumentParser(description='Train and audit three ResNets with the verified group split')\nparser.add_argument('--output-dir',type=Path,default=ROOT/'results'/'reproduction_run')\nargs=parser.parse_args()\nOUT=args.output_dir.resolve()\nif not OUT.is_relative_to(ROOT.resolve()):parser.error('Output must stay inside this project')")
p.write_text(s,encoding='utf-8')
print('Removed legacy source dependencies; current grouped training unchanged.')
