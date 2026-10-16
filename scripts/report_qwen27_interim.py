"""Report active-window throughput and GPU/token-price estimates for a live run."""
import argparse
import datetime
import gzip
import json
from pathlib import Path

from report_qwen27_dapo import GPU_SECOND_USD, TINKER_RATES, load_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--cutoff', type=float, required=True)
    args = parser.parse_args()
    directories = sorted((args.run / 'tp1-full/training').glob('client-??'))
    traces = [load_events(d / 'events.jsonl') for d in directories]
    start = max(next(e['time'] for e in events if e['event'] == 'rollout_start') for events in traces)
    seconds = args.cutoff - start
    if seconds <= 0:
        raise ValueError('All clients must have started before cutoff')
    client_tokens = {}
    for directory in directories:
        tokens = 0
        for file in directory.glob('group-*.json.gz'):
            with gzip.open(file, 'rt') as stream:
                group = json.load(stream)
            if start <= group['end'] <= args.cutoff:
                tokens += sum(len(sequence['tokens']) for sequence in group['sequences'])
        client_tokens[directory.name] = tokens
    total = sum(client_tokens.values())
    lifetimes = json.loads((args.run / 'tp1-full/training/gpu-lifetimes.json').read_text())
    gpu_seconds = sum(row['gpu_count'] * max(0, min(row['finished_at'] or args.cutoff, args.cutoff) - max(row['started_at'], start)) for row in lifetimes if row['gpu_count'])
    spindle = dict(start=start, end=args.cutoff, seconds=seconds, output_tokens=total,
                   aggregate_output_tps=total/seconds, mean_output_tps_per_client=total/seconds/len(directories),
                   per_client_output_tps={key: value/seconds for key, value in client_tokens.items()},
                   gpu_seconds=gpu_seconds, gpu_cost_usd=gpu_seconds*GPU_SECOND_USD,
                   gpu_cost_per_million_output_tokens=gpu_seconds*GPU_SECOND_USD/total*1e6,
                   completed_updates=[sum(e['event']=='update_done' and e['time']<=args.cutoff for e in events) for events in traces])
    events = load_events(args.run / 'tinker/events.jsonl')
    completed = [e for e in events if e['event']=='update_done' and e['update']>=4 and e['time']<=args.cutoff]
    last = max(e['update'] for e in completed)
    rollouts = [e for e in events if e['event']=='rollout_done' and 4<=e['update']<=last]
    t_start = next(e['time'] for e in events if e['event']=='rollout_start' and e['update']==4)
    t_end = max(e['time'] for e in completed)
    output = sum(e['output_tokens'] for e in rollouts)
    inputs = sum(e['sampling_input_tokens'] for e in rollouts)
    training = sum(e['training_input_tokens'] for e in rollouts)
    cost_low = (output*TINKER_RATES['output'] + inputs*TINKER_RATES['cached_input'] + training*TINKER_RATES['training'])/1e6
    cost_high = cost_low + inputs*(TINKER_RATES['input']-TINKER_RATES['cached_input'])/1e6
    tinker = dict(updates=[4,last], start=t_start, end=t_end, seconds=t_end-t_start,
                  output_tokens=output, sampling_input_tokens=inputs, training_tokens=training,
                  aggregate_output_tps=output/(t_end-t_start), rollout_only_tps=output/sum(e['seconds'] for e in rollouts),
                  estimated_cost_usd=[cost_low,cost_high],
                  cost_per_million_output_tokens=[cost_low/output*1e6,cost_high/output*1e6])
    result = dict(as_of_utc=datetime.datetime.fromtimestamp(args.cutoff, datetime.timezone.utc).isoformat(),
                  spindle=spindle, tinker=tinker,
                  notes=['Interim completed-group throughput: boundary-crossing work can make short windows noisy.',
                         'Spindle window starts when the eighth client starts; startup and initial staggering excluded.',
                         'Spindle cost includes all trainer and inference GPU time in that window; CPU/memory/storage excluded.',
                         'Tinker uses updates 4 onward after first billing pause, including final training/publication time.',
                         'Tinker prices include generation, prompts, and training; cache status unknown; not an invoice.',
                         'Tokens already generated but not yet trained can cross the Spindle measurement boundary.'],
                  rates=dict(h200_gpu_second=GPU_SECOND_USD,tinker_per_million=TINKER_RATES))
    destination = args.run / 'interim-throughput-cost.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
