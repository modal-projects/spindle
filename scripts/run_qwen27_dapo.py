"""Validate, then run the matched 27B DAPO comparison with independent clients."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/dapo-cost-comparison'
PYTHON = str(ROOT / '.venv/bin/python')
TINKER_PYTHON = str(ROOT / '.venv-tinker/bin/python')
ENVIRONMENT = 'lilo-deploy'


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def events(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def check_pilot(folder):
    summaries = []
    for client in sorted(folder.glob('client-*')):
        if not client.is_dir():
            continue
        trace = events(client / 'events.jsonl')
        updates = [r for r in trace if r['event'] == 'update_done']
        if len(updates) != 2 or not any(r['event'] == 'completed' for r in trace):
            raise RuntimeError(f'{client.name}: pilot did not finish two updates')
        starts = {r['update']: r['time'] for r in trace if r['event'] == 'train_start'}
        ends = {r['update']: r['time'] for r in trace if r['event'] == 'train_done'}
        rollout = {r['update']: r for r in trace if r['event'] == 'rollout_start'}
        finished = {r['update']: r for r in trace if r['event'] == 'rollout_done'}
        overlap = max(0, min(ends[1], finished[2]['time']) - max(starts[1], rollout[2]['time']))
        if overlap <= 0:
            raise RuntimeError(f'{client.name}: second rollout did not overlap first update')
        if any(r['groups_sampled'] != 8 or r['groups_filtered'] for r in finished.values()):
            raise RuntimeError('Pilot workload differs from the fixed 8x8 groups')
        summaries.append({'client': client.name, 'overlap_seconds': overlap,
                          'policy_lags': [r['policy_lag'] for r in updates]})
    if len(summaries) != 2:
        raise RuntimeError('Expected two independent pilot clients')
    save(folder.parent / 'validation.json', summaries)


class ExistingTinker:
    def __init__(self, pid, folder):
        self.pid, self.folder = pid, folder
        self.returncode = None

    def poll(self):
        trace = events(self.folder / 'events.jsonl')
        if any(row['event'] == 'completed' for row in trace):
            self.returncode = 0
        elif any(row['event'] == 'failed' for row in trace):
            self.returncode = 1
        else:
            try:
                os.kill(self.pid, 0)
            except ProcessLookupError:
                self.returncode = 1
        return self.returncode


class ExistingStage:
    def __init__(self, folder):
        self.folder = folder
        self.pid = json.loads((folder/'status.json').read_text())['pid']
        self.returncode = None

    def poll(self):
        status = json.loads((self.folder/'status.json').read_text())
        if status['state'] in ('completed', 'failed'):
            self.returncode = int(status['state'] == 'failed')
        else:
            try:
                os.kill(self.pid, 0)
            except ProcessLookupError:
                self.returncode = 1
        return self.returncode

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.folder), timeout)
            time.sleep(.25)
        return self.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-existing', action='store_true', help='Preserve the existing Tinker run and resume self-hosted stages')
    parser.add_argument('--spindle-env-file', type=Path, default=Path.home() / 'spindle/.env')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume_existing)
    os.chmod(output, 0o700)
    config = json.loads((EXAMPLE / 'config_27b.json').read_text())
    self_env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'MODAL_ENVIRONMENT': ENVIRONMENT,
                'SPINDLE_REQUEST_TIMING': '1'}
    # The parent's key is for real Tinker; only Spindle children receive this file's key.
    for line in args.spindle_env_file.read_text().splitlines():
        key, separator, value = line.removeprefix('export ').partition('=')
        if separator and key.strip() == 'TINKER_API_KEY':
            self_env[key.strip()] = value.strip().strip('"\'')
    tinker_env = dict(os.environ)
    tinker_env.pop('TINKER_BASE_URL', None)
    if not args.resume_existing and not tinker_env.get('TINKER_API_KEY'):
        raise RuntimeError('Real Tinker key must be supplied in the parent environment')
    save(output / 'config.json', config)
    save(output / 'source.json', {'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                                 'model_revision': config['tokenizer_revision']})
    (output / 'source.diff').write_bytes(subprocess.check_output(['git','diff'],cwd=ROOT))
    for source in [EXAMPLE/'client.py', EXAMPLE/'supervise.py', EXAMPLE/'config_27b.json',
                   ROOT/'scripts/qwen27_dapo_tp1.py', ROOT/'scripts/qwen27_dapo_tp2.py',
                   ROOT/'scripts/run_qwen27_dapo.py', ROOT/'scripts/report_qwen27_dapo.py']:
        target = output/'source-files'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    state = {'started': time.time(), 'pid': os.getpid(), 'state': 'starting', 'stages': []}
    if args.resume_existing:
        state = json.loads((output / 'status.json').read_text())
        state.update(pid=os.getpid(), state='resuming')
        state.pop('error', None)
    save(output / 'status.json', state)
    current = None
    tinker = None
    log_process = None
    owned = []

    def interrupted(*_):
        raise KeyboardInterrupt('Stop requested')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def stop_owned():
        for app in owned:
            subprocess.run([PYTHON,'-m','modal','app','stop','-y',app,'--env',ENVIRONMENT],
                           env=self_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)

    def stop_child(child):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    def wait(child, deadline, training=None):
        while child.poll() is None:
            if (output / 'STOP_NOW').exists():
                raise RuntimeError('Stop requested')
            now = time.time()
            if now > deadline:
                raise TimeoutError('Stage exceeded its deadline')
            if training and training.exists():
                trainer_log = training.parent / 'trainer.log'
                if trainer_log.exists():
                    with trainer_log.open('rb') as handle:
                        handle.seek(max(0, trainer_log.stat().st_size - 131072))
                        tail = handle.read().decode(errors='replace')
                    if 'out of memory' in tail.lower() or 'ActorDiedError' in tail:
                        raise RuntimeError('Trainer failed; inspect trainer.log')
                status = training / 'status.json'
                if status.exists():
                    progress = json.loads(status.read_text())
                    state['client_updates'] = progress.get('updates', [])
                for path in training.glob('client-*/events.jsonl'):
                    trace = events(path)
                    if trace and trace[-1]['event'] != 'completed' and now-trace[-1]['time'] > 3600:
                        raise TimeoutError(f'{path.parent.name} has made no progress for one hour')
            state['heartbeat'] = now
            if tinker is not None:
                state['tinker_returncode'] = tinker.poll()
                tinker_log = output/'tinker.log'
                if tinker_log.exists():
                    with tinker_log.open('rb') as log:
                        log.seek(max(0,tinker_log.stat().st_size-4096))
                        if b'paused due to billing status' in log.read():
                            state['tinker_state'] = 'billing_paused'
                        else:
                            state['tinker_state'] = 'running' if tinker.poll() is None else 'finished'
                state['tinker_updates'] = sum(e['event']=='update_done' for e in events(output/'tinker/events.jsonl'))
            save(output / 'status.json', state)
            time.sleep(15)
        if child.returncode:
            raise RuntimeError(f'Stage process exited {child.returncode}; inspect stage logs')

    try:
        # A single paid Tinker run. No retry or additional Tinker validation run.
        if args.resume_existing:
            tinker = ExistingTinker(state['tinker_pid'], output / 'tinker')
            if tinker.poll() not in (None, 0):
                state['tinker_state'] = 'validation_only'
        else:
            cfg = {**config, 'updates': 20}
            save(output / 'tinker-config.json', cfg)
            with (output / 'tinker.log').open('w') as log:
                tinker = subprocess.Popen([TINKER_PYTHON,'-u',str(EXAMPLE/'client.py'),
                    '--config',str(output/'tinker-config.json'),'--output',str(output/'tinker')],
                    env=tinker_env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['tinker_pid'] = tinker.pid
        for tp in (1, 2):
            name = f'qwen27-dapo-tp{tp}-20261008'
            frontend = f'spindle-{name}'
            owned[:] = [f'spindle-lora-{name}', f'spindle-trainer-{name}', f'spindle-inference-{name}', frontend]
            url = f'https://modal-labs-lilo-deploy--{frontend}-server.us-west.modal.run'
            for phase, clients, updates in [('pilot', 2, 2), ('full', 8, 30)]:
                folder = output / f'tp{tp}-{phase}'
                if any(s['tp'] == tp and s['phase'] == phase and s['state'] == 'completed' for s in state['stages']):
                    continue
                active = None
                status_file = folder/'training/status.json'
                if status_file.exists():
                    candidate = ExistingStage(folder/'training')
                    if candidate.poll() is None:
                        active = candidate
                if active:
                    stage = next(s for s in reversed(state['stages']) if s['tp'] == tp and s['phase'] == phase)
                    current = active
                else:
                    if folder.exists():
                        folder.rename(output / f'tp{tp}-{phase}-failed-{int(time.time())}')
                    folder.mkdir()
                    stage = {'tp':tp,'phase':phase,'started':time.time(),'state':'deploying','apps':list(owned)}
                    state['stages'].append(stage)
                    state.update(state='running', stage=f'tp{tp}-{phase}', client_updates=[])
                    save(output / 'status.json', state)
                    cfg = {**config,'updates':updates}
                    save(folder / 'config.json', cfg)
                    with (folder/'deploy.log').open('w') as log:
                        current = subprocess.Popen([PYTHON,'-m','spindle.deployment_cli','deploy',
                            str(ROOT/f'scripts/qwen27_dapo_tp{tp}.py'),'--refresh-trainer',name,'--refresh-inference',name],
                            env=self_env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    wait(current,time.time()+2400)
                with (folder/'trainer.log').open('a') as log:
                    log_process = subprocess.Popen([PYTHON,'-m','modal','app','logs',f'spindle-trainer-{name}',
                        '--env',ENVIRONMENT,'--follow'],env=self_env,cwd=ROOT,stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True)
                state['log_pid'] = log_process.pid
                stage['state']='training'
                if not active:
                    with (folder/'supervisor.log').open('w') as log:
                        current = subprocess.Popen([PYTHON,'-u',str(EXAMPLE/'supervise.py'),
                            '--root',str(folder/'training'),'--base-url',url,'--config',str(folder/'config.json'),
                            '--clients',str(clients),'--stagger-seconds','150','--hours','14',
                            '--resource-since',str(stage['started']),'--owned-apps',*owned],
                            env=self_env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(state='running', stage=f'tp{tp}-{phase}')
                wait(current,time.time()+15*3600,folder/'training')
                stop_child(log_process)
                if phase == 'pilot':
                    check_pilot(folder/'training')
                else:
                    subprocess.run([PYTHON,str(EXAMPLE/'trainer_activity.py'),str(folder/'training'),
                        '--trainer-log',str(folder/'trainer.log'),'--through',str(time.time())],cwd=ROOT,check=True)
                stage.update(state='completed',finished=time.time())
                save(output/'status.json',state)
        billing_blocked = 'paused due to billing status' in (output/'tinker.log').read_text()[-32768:]
        if not billing_blocked and tinker.poll() is None:
            wait(tinker,time.time()+3600)
        subprocess.run([PYTHON,str(ROOT/'scripts/report_qwen27_dapo.py'),str(output),
                        '--allow-partial-tinker'],cwd=ROOT,check=True)
        state.update(state='completed',finished=time.time())
    except BaseException as exc:
        state.update(state='failed',error=f'{type(exc).__name__}: {exc}',finished=time.time())
        stop_child(current)
        stop_child(log_process)
        stop_owned()
        # The single paid baseline continues independently of self-hosted failures.
        raise
    finally:
        save(output/'status.json',state)


if __name__=='__main__':
    main()
