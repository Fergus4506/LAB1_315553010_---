"""Download public Kaggle version 2 and extract one canonical copy of each image."""
from __future__ import annotations
import argparse,os,time,zipfile,zlib
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data'/'source_verification'
URL='https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia?datasetVersionNumber=2'

def ensure_official_zip():
    SOURCE.mkdir(parents=True,exist_ok=True)
    archive_path=SOURCE/'chest-xray-pneumonia-v2.zip'
    if archive_path.exists():return archive_path
    partial=archive_path.with_suffix('.zip.partial')
    with requests.get(URL,stream=True,timeout=(30,120)) as response:
        response.raise_for_status();received=0;last=time.monotonic()
        with partial.open('wb') as output:
            for chunk in response.iter_content(1024*1024):
                if chunk:output.write(chunk);received+=len(chunk)
                if time.monotonic()-last>15:
                    print(f'Downloaded {received/1024**2:.1f} MiB',flush=True);last=time.monotonic()
    # Require a valid central directory before committing the download.
    with zipfile.ZipFile(partial):pass
    partial.rename(archive_path)
    return archive_path

def canonical_members(archive):
    members={}
    for item in archive.infolist():
        parts=item.filename.split('/')
        if '__MACOSX' in parts or parts[-1].startswith('._') or len(parts)<3:continue
        split,label,filename=parts[-3:]
        if split not in {'train','val','test'} or label not in {'NORMAL','PNEUMONIA'}:continue
        if Path(filename).suffix.lower() not in {'.jpeg','.jpg','.png'}:continue
        if filename!=Path(filename).name:raise ValueError('Unsafe archive filename')
        relative=Path(split)/label/filename
        if relative in members:
            previous=members[relative]
            if (previous.file_size,previous.CRC)!=(item.file_size,item.CRC):raise ValueError('Nested ZIP copies disagree')
        else:members[relative]=item
    if len(members)!=5856:raise ValueError(f'Unexpected version-2 image inventory: {len(members)}')
    return members

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'data')
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if not output.is_relative_to(ROOT.resolve()):parser.error('Output must stay inside this D-drive project')
    data=output/'chest_xray'
    written=skipped=0
    with zipfile.ZipFile(ensure_official_zip()) as archive:
        for relative,item in sorted(canonical_members(archive).items()):
            destination=data/relative
            if destination.exists():
                contents=destination.read_bytes()
                if len(contents)!=item.file_size or zlib.crc32(contents)!=item.CRC:
                    raise ValueError(f'Existing image differs from official source; preserved unchanged: {destination}')
                skipped+=1;continue
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(archive.read(item));written+=1
    print(f'Canonical images: 5856; extracted {written}; existing verified/preserved {skipped}',flush=True)
    print('Use --data-dir',data,flush=True)
if __name__=='__main__':main()
