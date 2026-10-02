"""Keep official test intact; split training data by image identity and filename groups."""
from __future__ import annotations
import hashlib, json, random, re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/'data'/'source_verification'/'corrected_split_manifest.json'

def filename_group(path: str):
    name=Path(path).stem
    if '/PNEUMONIA/' in path:
        m=re.fullmatch(r'(person\d+_(?:bacteria|virus))(?:_\d+)+',name)
    else:
        m=re.fullmatch(r'((?:NORMAL2-)?IM-\d+)-\d+(?:-\d+)?',name)
    if not m:
        raise ValueError(f'Unrecognized filename; explicit grouping needed: {path}')
    return ('PNEUMONIA:' if '/PNEUMONIA/' in path else 'NORMAL:')+m[1]

def build_manifest(fraction=.15, seed=315553010):
    verified=json.loads((MANIFEST.parent/'official_image_verification.json').read_text(encoding='utf-8'))
    assert verified['all_match_official_size_crc32'] and not verified['extra_local_paths']
    records=verified['files']
    def enrich(item):
        item=dict(item)
        item['label']=0 if '/NORMAL/' in item['path'] else 1
        item['source_split']=item['path'].split('/')[0]
        item['filename_group']=filename_group(item['path'])
        with Image.open(ROOT/'data'/'chest_xray'/item['path']) as image:
            image=image.convert('RGB')
            item['decoded_sha256']=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        return item
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(enrich,records))
    parent=list(range(len(records)))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for key in ('filename_group','sha256','decoded_sha256'):
        seen={}
        for i,item in enumerate(records):
            if item[key] in seen:union(i,seen[item[key]])
            else:seen[item[key]]=i
    groups=defaultdict(list)
    for i,item in enumerate(records):groups[find(i)].append(item)
    # Preserve official test rather than silently deriving a new test set.
    conflicts=[g for g in groups.values() if any(x['source_split']=='test' for x in g) and any(x['source_split']!='test' for x in g)]
    if conflicts:
        (MANIFEST.parent/'official_test_overlap.json').write_text(json.dumps(conflicts,indent=2),encoding='utf-8')
        raise ValueError('Official test overlaps with training pool; review original dataset evidence first')
    for group in groups.values():
        if len({x['label'] for x in group})!=1:raise ValueError('Conflicting class labels for an identity group')
    rng=random.Random(seed)
    for label in (0,1):
        candidates=[g for g in groups.values() if g[0]['label']==label and g[0]['source_split']!='test']
        forced=[g for g in candidates if any(x['source_split']=='val' for x in g)]
        remaining=[g for g in candidates if g not in forced]
        original_train_count=sum(x['source_split']=='train' and x['label']==label for x in records)
        original_val_count=sum(x['source_split']=='val' and x['label']==label for x in records)
        target=round(original_train_count*fraction)+original_val_count
        current=sum(len(g) for g in forced)
        rng.shuffle(remaining)
        selected={g[0]['path'] for g in forced}
        for group in remaining:
            if abs(current+len(group)-target)<abs(current-target):
                selected.add(group[0]['path']);current+=len(group)
        for group in candidates:
            split='validation' if group[0]['path'] in selected else 'train'
            for item in group:item['split']=split
    for group in groups.values():
        for item in group:
            if item['source_split']=='test':item['split']='test'
            item['component_id']=group[0]['path']
    overlaps={}
    for key in ('filename_group','sha256','decoded_sha256','component_id'):
        memberships=defaultdict(set)
        for item in records:memberships[item[key]].add(item['split'])
        overlaps[key]=sum(len(v)>1 for v in memberships.values())
        assert overlaps[key]==0
    duplicates={}
    for key in ('sha256','decoded_sha256'):
        matches=defaultdict(list)
        for item in records:matches[item[key]].append(item['path'])
        duplicates[key]=[v for v in matches.values() if len(v)>1]
    counts={s:dict(Counter(str(r['label']) for r in records if r['split']==s)) for s in ('train','validation','test')}
    manifest={'source':verified['source'],'version':2,'seed':seed,'validation_fraction':fraction,
              'method':'connected components of filename namespace (person+etiology / normal series), byte SHA256 and decoded RGB SHA256; official val forced into validation; official test unchanged',
              'patient_id_verified':False,'counts':counts,'cross_split_group_counts':overlaps,
              'original_duplicate_groups':duplicates,'files':records}
    MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('files','original_duplicate_groups')},indent=2),flush=True)
    print('Original duplicate groups:',{k:len(v) for k,v in duplicates.items()},flush=True)
    return manifest

def load_samples(data_dir:Path, manifest_path:Path):
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    samples={s:[] for s in ('train','validation','test')}
    identities={key:defaultdict(set) for key in ('filename_group','sha256','decoded_sha256','component_id')}
    paths=set()
    for item in manifest['files']:
        rel=Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe manifest path')
        if len(rel.parts)!=3 or rel.parts[0] not in {'train','val','test'} or rel.parts[1] not in {'NORMAL','PNEUMONIA'}:
            raise ValueError('Invalid source split/class path')
        expected_label=0 if rel.parts[1]=='NORMAL' else 1
        if item['label']!=expected_label or item['source_split']!=rel.parts[0] or item['filename_group']!=filename_group(item['path']):
            raise ValueError('Manifest labels or filename group disagree with source path')
        if item['path'] in paths:raise ValueError('Repeated manifest path')
        paths.add(item['path'])
        path=data_dir/rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError(f'Dataset changed since audit: {path}')
        samples[item['split']].append((path,item['label']))
        for key in identities:identities[key][item[key]].add(item['split'])
        if (item['source_split']=='test')!=(item['split']=='test'):raise ValueError('Official test modified')
    actual={p.relative_to(data_dir).as_posix() for p in data_dir.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpeg','.jpg','.png'}}
    if actual!=paths:raise ValueError('Dataset inventory differs from audited manifest')
    for key,memberships in identities.items():
        if any(len(v)>1 for v in memberships.values()):raise ValueError(f'Split leakage: {key}')
    assert len(samples['test'])==624
    return samples,manifest

if __name__=='__main__':build_manifest()
