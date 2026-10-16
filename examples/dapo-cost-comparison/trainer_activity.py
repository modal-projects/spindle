"""Plot backend call occupancy and client request windows from an existing run.

No changes to the Tinker-compatible client or running deployment are required.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

COLORS = {'forward_backward': '#27898e', 'optim_step': '#d58a25',
          'save_sampler_weights': '#8763b4'}
LABELS = {'forward_backward': 'Forward/backward', 'optim_step': 'Optimizer',
          'save_sampler_weights': 'Snapshot capture'}
ACTOR = re.compile(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) actor_cell\d+_rank0\].*?fn=forward_backward phase=(start|end)')


def events(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass  # A live writer may not have finished the final line.
    return rows


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def seconds(intervals, start, end):
    return sum(max(0, min(b, end) - max(a, start)) for a, b in merge(intervals))


def collect(root, log, end):
    clients = {p.parent.name: events(p) for p in sorted(root.glob('client-*/events.jsonl'))}
    start = min(r['time'] for rows in clients.values() for r in rows if r['event'] == 'ready')
    if end <= start:
        raise ValueError('Observation end must follow first client readiness')
    intervals = []; seen = set(); pending = None
    for line in log.splitlines():
        offset = line.find('{"event": "spindle_step_timing"')
        if offset >= 0:
            row, _ = json.JSONDecoder().raw_decode(line[offset:])
            key = (row['ts'], row['phase'], row['model_id'])
            if row['phase'] in COLORS and key not in seen:
                seen.add(key)
                a, b = max(start, row['ts'] - row['seconds']), min(end, row['ts'])
                if b > a:
                    intervals.append(dict(start=a, end=b, operation=row['phase'],
                                          first_model_id=row['model_id']))
        match = ACTOR.search(line)
        if match:
            stamp = datetime.fromisoformat(match[1]).replace(tzinfo=timezone.utc).timestamp()
            if stamp <= end:
                pending = stamp if match[2] == 'start' else None
    requests = []
    for client, rows in clients.items():
        for row in rows:
            if row['event'] != 'train_start' or row['time'] > end:
                continue
            done = next((r['time'] for r in rows if r['event'] == 'train_done' and r['update'] == row['update']), None)
            requests.append(dict(client=client, update=row['update'], start=row['time'],
                                 end=min(end, done) if done else end, completed=done is not None and done <= end))
    busy = [(r['start'], r['end']) for r in intervals]
    return dict(start=start, end=end, clients=list(clients), intervals=intervals,
                requests=requests, unfinished_forward_backward_start=pending,
                completed_call_seconds=seconds(busy, start, end),
                completed_call_occupancy=seconds(busy, start, end)/(end-start),
                caveats=['Host backend call time, not hardware GPU utilization; includes synchronization and compilation.',
                         'Completed calls only in occupancy; hatched interval is an unfinished rank-zero call.',
                         'Backend model ID identifies first participant only; client lanes show requests including queue time.',
                         'Log coverage must span the run; missing logs must not be interpreted as idle.'])


def render(run, output):
    start, end = run['start'], run['end']; duration = (end-start)/60
    fig, (ax, clients) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                      gridspec_kw={'height_ratios': [1, 2]}, layout='constrained')
    fig.suptitle('DAPO · shared 8×H200 trainer', fontsize=16)
    for operation, color in COLORS.items():
        bars = [((r['start']-start)/60, (r['end']-r['start'])/60)
                for r in run['intervals'] if r['operation'] == operation]
        ax.broken_barh(bars, (0, 1), facecolors=color)
    pending = run['unfinished_forward_backward_start']
    if pending is not None:
        ax.axvspan(max(0, (pending-start)/60), duration, facecolor='#27898e', alpha=.25, hatch='///')
    busy = [(r['start'], r['end']) for r in run['intervals']]
    if end-start >= 60:
        centers = np.linspace(start+30, end-30, 600)
        ax.plot((centers-start)/60, [seconds(busy, t-30, t+30)/60 for t in centers], color='black', lw=1)
    ax.set(ylim=(0, 1.05), yticks=[0, .5, 1], yticklabels=['0%', '50%', '100%'],
           title=f"Backend call occupancy: {run['completed_call_occupancy']:.1%}")
    for index, client in enumerate(run['clients']):
        for row in run['requests']:
            if row['client'] == client:
                clients.broken_barh([((row['start']-start)/60, (row['end']-row['start'])/60)],
                                    (index-.3, .6), facecolors=plt.cm.tab10(index),
                                    hatch=None if row['completed'] else '///', alpha=.7)
    clients.set(yticks=range(len(run['clients'])), yticklabels=run['clients'],
                xlim=(0, duration), xlabel='Minutes since first client ready',
                title='Client training request latency')
    clients.invert_yaxis()
    for axis in (ax, clients):
        axis.grid(axis='x', alpha=.15)
    ax.legend(handles=[Patch(color=c, label=LABELS[k]) for k, c in COLORS.items()], loc='upper left')
    fig.savefig(output/'trainer-activity.png', dpi=160)
    fig.savefig(output/'trainer-activity.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--trainer-log', required=True, type=Path)
    parser.add_argument('--through', type=float, required=True, help='Unix timestamp through which logs were collected')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.run/'activity'; output.mkdir(parents=True, exist_ok=True)
    run = collect(args.run, args.trainer_log.read_text(), args.through)
    (output/'trainer-activity.json').write_text(json.dumps(run, indent=2)+'\n')
    render(run, output)
    print(json.dumps({k: run[k] for k in ('start', 'end', 'completed_call_seconds', 'completed_call_occupancy')}, indent=2))


if __name__ == '__main__':
    main()
