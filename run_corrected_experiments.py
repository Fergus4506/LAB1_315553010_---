"""Run all corrected models and required post-run checks inside this project."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Train and audit three ResNets with the verified group split')
parser.add_argument('--output-dir',type=Path,default=ROOT/'runs'/'reproduction_run')
args=parser.parse_args()
OUT=args.output_dir.resolve()
if not OUT.is_relative_to(ROOT.resolve()):parser.error('Output must stay inside this project')
for name in ('temp','mplcache','cuda_cache','torch_cache'):(ROOT/'data'/name).mkdir(parents=True,exist_ok=True)
env=dict(os.environ, PYTHONIOENCODING='utf-8',TEMP=str(ROOT/'data'/'temp'),TMP=str(ROOT/'data'/'temp'),
         TORCH_HOME=str(ROOT/'data'/'torch_cache'),MPLCONFIGDIR=str(ROOT/'data'/'mplcache'),CUDA_CACHE_PATH=str(ROOT/'data'/'cuda_cache'))
commands=[['train.py','--data-dir',str(ROOT/'data'/'chest_xray'),'--output-dir',str(OUT),
          '--models','resnet18','resnet50','resnet101','--epochs','12','--batch-size','32','--num-workers','2'],
          ['compare_models.py','--results-dir',str(OUT)],
          ['verify_pretrained.py','--results-dir',str(OUT)],
          ['audit_corrected_experiment.py','--results-dir',str(OUT)]]
for command in commands:
    print('Running:',command[0],flush=True)
    subprocess.run([sys.executable,*command],cwd=ROOT,env=env,check=True)
(OUT/'experiment_completed.json').write_text(json.dumps({'all_three_models_complete':True,'gpu_checkpoint_audit_passed':True,'pretrained_verification_passed':True},indent=2),encoding='utf-8')
print('All corrected training and audits completed.',flush=True)
