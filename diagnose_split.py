"""Describe the source defect separately from the superseded split implementation."""
import json
from collections import defaultdict,Counter
from pathlib import Path
from train import collect_samples,stratified_split
from split_dataset import filename_group
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data'/'source_verification'
verified=json.loads((OUT/'official_image_verification.json').read_text(encoding='utf-8'))
manifest=json.loads((OUT/'corrected_split_manifest.json').read_text(encoding='utf-8'))
data=ROOT/'data'/'chest_xray'
train,val=stratified_split(collect_samples(data,'train'),.15,315553010)
val+=collect_samples(data,'val')
old_splits={p.relative_to(data).as_posix():s for s,samples in [('train',train),('validation',val),('test',collect_samples(data,'test'))] for p,_ in samples}
identities={key:defaultdict(list) for key in ('sha256','decoded_sha256','filename_group')}
for record in manifest['files']:
    for key,index in identities.items():index[record[key]].append(record)
info={}
for key,index in identities.items():
    repeats=[g for g in index.values() if len(g)>1]
    raw_cross=[g for g in repeats if len({r['source_split'] for r in g})>1]
    old_cross=[g for g in repeats if len({old_splits[r['path']] for r in g})>1]
    new_cross=[g for g in repeats if len({r['split'] for r in g})>1]
    info[key]={'repeated_groups':len(repeats),'original_cross_split_groups':len(raw_cross),
               'legacy_cross_split_groups':len(old_cross),'corrected_cross_split_groups':len(new_cross),
               'legacy_cross_split_examples':[[r['path'] for r in g] for g in old_cross]}
assert info['sha256']['legacy_cross_split_groups']==9
assert all(v['corrected_cross_split_groups']==0 for v in info.values())
result={'source_version':2,'local_matches_official_zip':verified['all_match_official_size_crc32'],
        'official_unique_images':verified['official_unique_images'],
        'raw_counts':dict(Counter(r['source_split'] for r in manifest['files'])),
        'identity_audit':info,'corrected_counts':manifest['counts'],
        'conclusion':'Original dataset repeats images within its train set. The old individual-image split introduced train-validation leakage. The corrected grouped split preserves every raw image and the official test set.',
        'patient_identity_verified':False}
(OUT/'split_diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print({key:{k:v for k,v in value.items() if k!='legacy_cross_split_examples'} for key,value in info.items()},flush=True)
