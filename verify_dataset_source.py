"""Verify canonical images against the public Kaggle version-2 ZIP without extracting."""
import json, time, zipfile, zlib, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'source_verification'
OUT.mkdir(parents=True, exist_ok=True)
ZIP = OUT / 'chest-xray-pneumonia-v2.zip'
url = 'https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia?datasetVersionNumber=2'
if not ZIP.exists():
    partial = ZIP.with_suffix('.zip.partial')
    print('Downloading public Kaggle version 2 into', OUT, flush=True)
    with requests.get(url, stream=True, timeout=(30,120)) as response:
        response.raise_for_status()
        total = int(response.headers.get('Content-Length',0))
        received = 0
        last = time.monotonic()
        with partial.open('wb') as f:
            for chunk in response.iter_content(1024*1024):
                if chunk:
                    f.write(chunk); received += len(chunk)
                if time.monotonic()-last > 15:
                    print(f'Downloaded {received/1024**2:.1f} MiB / {total/1024**2:.1f} MiB',flush=True)
                    last=time.monotonic()
    partial.rename(ZIP)
with zipfile.ZipFile(ZIP) as archive:
    records = []
    by_path = {}
    for entry in archive.infolist():
        parts = entry.filename.split('/')
        if '__MACOSX' in parts or parts[-1].startswith('._'):
            continue
        if len(parts) < 3 or Path(parts[-1]).suffix.lower() not in {'.jpeg','.jpg','.png'}:
            continue
        # ZIP contains duplicate nested copies. Match only canonical split/class/image suffix.
        split, label, filename = parts[-3:]
        if split not in {'train','val','test'} or label not in {'NORMAL','PNEUMONIA'}:
            continue
        path = '/'.join((split,label,filename))
        record = {'path':path,'size':entry.file_size,'crc32':entry.CRC}
        if path in by_path:
            assert by_path[path]['size']==entry.file_size and by_path[path]['crc32']==entry.CRC
        by_path[path]=record
    def inspect(record):
        path = ROOT/'data'/'chest_xray'/record['path']
        if not path.exists(): return {'path':record['path'],'error':'missing'}
        contents=path.read_bytes()
        return {**record,'sha256':hashlib.sha256(contents).hexdigest(),
                'matches_official':len(contents)==record['size'] and zlib.crc32(contents)==record['crc32']}
    with ThreadPoolExecutor(max_workers=8) as pool:
        records=list(pool.map(inspect, sorted(by_path.values(),key=lambda x:x['path'])))
local={str(p.relative_to(ROOT/'data'/'chest_xray')).replace('\\','/') for p in (ROOT/'data'/'chest_xray').rglob('*') if p.is_file() and p.suffix.lower() in {'.jpeg','.jpg','.png'}}
result={'source':'paultimothymooney/chest-xray-pneumonia','version':2,
        'zip_bytes':ZIP.stat().st_size,'official_unique_images':len(records),'local_images':len(local),
        'all_match_official_size_crc32':all(r.get('matches_official',False) for r in records),
        'extra_local_paths':sorted(local-set(by_path)),'files':records}
(OUT/'official_image_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print({k:v for k,v in result.items() if k!='files'},flush=True)
assert result['all_match_official_size_crc32'] and not result['extra_local_paths']
