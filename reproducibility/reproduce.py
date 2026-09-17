"""Recompute fixed-output results in a disposable copy; CPU only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,subprocess,sys,time

ROOT=Path(__file__).resolve().parent
CHECKS=[
 'check_manifest.py','verify.py','verify_level_audit.py','verify_public_rms.py',
 'refinement/verify_refinement.py','refinement/verify_all_predictors.py',
 'refinement/verify_scale.py','musicnet/reproduce_musicnet.py',
 'template_preservation/audit_template_preservation.py','transfer_shape/audit_transfer_shape.py',
 'duration_weights/audit_duration_weights.py','final_clarifications/audit_final_clarifications.py',
 'round7/audit_round7.py','round7/verify_round7.py',
 'acceptance/corpus_analysis.py','acceptance/public_directions.py',
]
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,help='A new directory; default: runs/UTC timestamp')
    ap.add_argument('--quick',action='store_true',help='Manifest plus core metrics/paired bootstrap only')
    args=ap.parse_args()
    out=(args.output or ROOT/'runs'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    out.mkdir(parents=True,exist_ok=False)
    work=out/'analysis';shutil.copytree(ROOT/'analysis',work)
    results=[];start=time.monotonic()
    for name in CHECKS[:2] if args.quick else CHECKS:
        print('RUN',name,flush=True);tick=time.monotonic()
        run=subprocess.run([sys.executable,'-u',str(work/name)],cwd=work,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=out/(name.replace('/','_')+'.log');log.write_bytes(run.stdout)
        results.append(dict(script=name,exit_code=run.returncode,seconds=round(time.monotonic()-tick,3),log=log.name))
        report=dict(status='running' if run.returncode==0 else 'failed',python=sys.version,checks=results,seconds=round(time.monotonic()-start,3))
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if run.returncode:
            print(run.stdout.decode('utf-8',errors='replace')[-5000:]);raise SystemExit(run.returncode)
        print('PASS',name,results[-1]['seconds'],'s',flush=True)
    report['status']='passed'
    report['core_results']=json.loads((work/'verification.json').read_text())
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS:',len(results),'checks. Report:',out/'report.json',flush=True)

if __name__=='__main__':main()
