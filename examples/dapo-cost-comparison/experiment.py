"""Compare two disposable trainer layouts, then launch the 30-update experiment."""
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


def apps(name):
    return ['spindle-lora-'+name,'spindle-trainer-'+name,'spindle-inference-'+name,FRONTEND]


def save(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2));temporary.replace(path)


def cleanup(python,name):
    results={}
    for app in apps(name):
        result=subprocess.run([python,'-m','modal','app','stop','-y',app,'--env',ENVIRONMENT],
            capture_output=True,text=True,timeout=120)
        # The profiler never creates a sampling pool, so that app can be absent.
        results[app]=dict(returncode=result.returncode,output=(result.stdout+result.stderr)[-500:])
        if result.returncode and not any(s in result.stdout+result.stderr for s in ['No App with name','already stopped']):
            raise RuntimeError(f'Could not stop owned app {app}')
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--deploy-python',required=True)
    args=p.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'PYTHONPATH':str(REPO/'src')}
    state=dict(started=time.time(),pid=os.getpid(),state='profiling',profiles={})
    save(root/'experiment.json',state)
    candidates={}
    try:
        for tp in [1,2]:
            name=f'dapo-instruct-tp{tp}-dp{8//tp}-20260930'
            config=root/f'tp{tp}.py'
            config.write_text('from deployment import Config\nconfig = Config(name='+repr(name)+
                ', overrides={"miles_cfg.tensor_model_parallel_size": '+str(tp)+'})\n')
            deploy_env={**env,'PYTHONPATH':str(ROOT)+os.pathsep+env['PYTHONPATH']}
            state.update(state='deploying_profile',tensor_parallel=tp);save(root/'experiment.json',state)
            try:
                with (root/f'tp{tp}-deploy.log').open('w') as log:
                    subprocess.run([args.deploy_python,'-m','spindle.deployment_cli','deploy',str(config)],
                        env=deploy_env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
                state.update(state='profiling',tensor_parallel=tp);save(root/'experiment.json',state)
                with (root/f'tp{tp}-profile.log').open('w') as log:
                    subprocess.run([sys.executable,'-u',str(ROOT/'profile.py'),'--base-url',BASE_URL,
                        '--output',str(root/f'tp{tp}.json')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
                rows=json.loads((root/f'tp{tp}.json').read_text())
                measured=[r for r in rows if not r['warmup']]
                candidates[tp]=sum(r['seconds'] for r in measured)
                state['profiles'][str(tp)]=dict(measured_seconds=candidates[tp],status='completed')
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
                state['profiles'][str(tp)]=dict(status='failed',error=str(exc))
            finally:
                state['cleanup']=cleanup(args.deploy_python,name)
                save(root/'experiment.json',state)
        if not candidates:raise RuntimeError('Neither trainer layout completed profiling')
        best=min(candidates,key=candidates.get)
        name=f'dapo-instruct-tp{best}-dp{8//best}-20260930'
        state.update(state='deploying_training',selected_tensor_parallel=best,
            selection='Lowest total measured latency on fixed 64-sequence synthetic batches at 2k/8k/16k; not a global throughput optimum')
        save(root/'experiment.json',state)
        with (root/'training-deploy.log').open('w') as log:
            subprocess.run([args.deploy_python,'-m','spindle.deployment_cli','deploy',str(root/f'tp{best}.py')],
                env=deploy_env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        state['state']='training';save(root/'experiment.json',state)
        with (root/'training-supervisor.log').open('w') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'supervise.py'),'--root',str(root/'training'),
                '--base-url',BASE_URL,'--clients','8','--stagger-seconds','150',
                '--owned-apps',*apps(name)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=13*3600)
        state.update(state='completed',finished=time.time());save(root/'experiment.json',state)
    except Exception as exc:
        state.update(state='failed',error=f'{type(exc).__name__}: {exc}',finished=time.time())
        save(root/'experiment.json',state)
        if 'name' in locals():cleanup(args.deploy_python,name)
        raise


if __name__=='__main__':main()
