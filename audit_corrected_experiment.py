"""Recompute confusion-derived histories and GPU-evaluate audited corrected checkpoints."""
import argparse,csv,hashlib,json,math
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from split_dataset import load_samples
from train import ChestXrayDataset,counts_to_metrics,create_model,make_transforms,run_epoch
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results'/'corrected_group_split_20261002')
    args=parser.parse_args();out=args.results_dir
    meta=json.loads((out/'run_metadata.json').read_text(encoding='utf-8'))
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    rows=list(csv.DictReader((out/'epoch_metrics.csv').open(encoding='utf-8')))
    splits,manifest=load_samples(ROOT/'data'/'chest_xray',out/'split_manifest.json')
    digest=hashlib.sha256((out/'split_manifest.json').read_bytes()).hexdigest()
    assert digest==meta['split_manifest_sha256']
    for split,samples in splits.items():
        assert dict(Counter(str(label) for _,label in samples))==meta[f'{split}_counts']
    for row in rows:
        for split in splits:
            counts=[int(row[f'{split}_{k}']) for k in ('tn','fp','fn','tp')]
            assert sum(counts)==len(splits[split])
            assert counts[0]+counts[1]==meta[f'{split}_counts']['0']
            assert counts[2]+counts[3]==meta[f'{split}_counts']['1']
            for k,v in counts_to_metrics(*counts).items():
                assert math.isclose(v,float(row[f'{split}_{k}']),abs_tol=1e-12)
    assert torch.cuda.is_available()
    device=torch.device('cuda');_,transform=make_transforms(meta['args']['image_size'])
    loaders={s:DataLoader(ChestXrayDataset(samples,transform),batch_size=32,num_workers=2,pin_memory=True) for s,samples in splits.items() if s!='train'}
    counts=meta['train_counts'];n=sum(counts.values())
    criterion=nn.CrossEntropyLoss(weight=torch.tensor([n/(2*counts[str(i)]) for i in (0,1)],device=device))
    evals={}
    for name,item in summary.items():
        model_rows=[r for r in rows if r['model']==name]
        assert [int(r['epoch']) for r in model_rows]==list(range(1,meta['args']['epochs']+1))
        selected=max(model_rows,key=lambda r:float(r['validation_f1']))
        assert int(selected['epoch'])==item['validation_selected_epoch']
        for k in ('accuracy','f1'):
            best=max(model_rows,key=lambda r:float(r[f'test_{k}']))
            assert int(best['epoch'])==item[f'highest_observed_test_{k}']['epoch']
            assert float(best[f'test_{k}'])==item[f'highest_observed_test_{k}'][k]
        checkpoint=torch.load(out/f'{name}_best_validation.pt',map_location='cpu',weights_only=True)
        assert checkpoint['split_manifest_sha256']==digest and checkpoint['epoch']==item['validation_selected_epoch']
        model,_=create_model(name,False);model.load_state_dict(checkpoint['state_dict']);model.to(device)
        scores={s:run_epoch(model,loader,criterion,device) for s,loader in loaders.items()}
        for split in scores:
            for k in ('tn','fp','fn','tp','accuracy','precision','recall','f1'):
                assert scores[split][k]==float(selected[f'{split}_{k}'])
        evals[name]={'checkpoint_epoch':checkpoint['epoch'],'evaluation':scores,'matches_history':True,
                     'parameters':{'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}}
        print(name,'checkpoint reproduced:',scores['test'],flush=True)
        del model;torch.cuda.empty_cache()
    audit={'history_rows':len(rows),'all_confusion_metrics_and_selections_match':True,
           'split_manifest_sha256':digest,'counts':manifest['counts'],'cross_split_group_counts':manifest['cross_split_group_counts'],
           'official_test_unchanged':True,'gpu':torch.cuda.get_device_name(0),'checkpoint_reevaluation':evals}
    (out/'audit_corrected.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Corrected experiment audit passed',flush=True)
if __name__=='__main__':main()
