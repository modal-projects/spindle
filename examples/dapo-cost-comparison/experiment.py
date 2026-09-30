"""Deploy one trainer layout and run eight staggered standard Tinker SDK clients."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
ENVIRONMENT='lilo-deploy'
FRONTEND='spindle-dapo-comparison-20260930'
BASE_URL='https://modal-labs-lilo-deploy--spindle-dapo-comparison-20260930-server.us-west.modal.run'
NAME='dapo-instruct-tp1-dp8-20260930'
APPS=['spindle-lora-'+NAME,'spindle-trainer-'+NAME,'spindle-inference-'+NAME,FRONTEND]


def save(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2));temporary.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--deploy-python',required=True)
    args=p.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'PYTHONPATH':str(REPO/'src')}
    state=dict(started=time.time(),pid=os.getpid(),state='deploying',tensor_parallel=1,data_parallel=8)
    save(root/'experiment.json',state)
    try:
        with (root/'deploy.log').open('w') as log:
            subprocess.run([args.deploy_python,'-m','spindle.deployment_cli','deploy',str(ROOT/'deployment.py')],
                env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        state['state']='training';save(root/'experiment.json',state)
        with (root/'supervisor.log').open('w') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'supervise.py'),'--root',str(root/'training'),
                '--base-url',BASE_URL,'--clients','8','--stagger-seconds','150',
                '--owned-apps',*APPS],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=13*3600)
        state.update(state='completed',finished=time.time());save(root/'experiment.json',state)
    except Exception as exc:
        state.update(state='failed',error=f'{type(exc).__name__}: {exc}',finished=time.time())
        save(root/'experiment.json',state)
        for app in APPS:
            subprocess.run([args.deploy_python,'-m','modal','app','stop','-y',app,'--env',ENVIRONMENT],
                capture_output=True,text=True,timeout=120)
        raise


if __name__=='__main__':main()
