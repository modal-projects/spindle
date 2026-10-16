"""Summarize measured rollout windows and price estimates for the matched 27B runs."""
import argparse
import gzip
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GPU_SECOND_USD = .001261
TINKER_RATES = {'input':1.86,'cached_input':.372,'output':5.595,'training':4.103}
ROOT = Path(__file__).resolve().parents[1]


def load_events(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize(path, provider, allow_partial=False):
    folders = [path] if (path/'manifest.json').exists() else sorted(p for p in path.glob('client-*') if p.is_dir())
    clients=[]
    for folder in folders:
        trace=load_events(folder/'events.jsonl')
        rollout=[r for r in trace if r['event']=='rollout_done']
        updates=[r for r in trace if r['event']=='update_done']
        warmup=[r for r in trace if r['event']=='sampling_warmup']
        complete = any(r['event']=='completed' for r in trace)
        if allow_partial and not complete:
            done = {r['update'] for r in updates}
            rollout = [r for r in rollout if r['update'] in done]
        if not rollout or not updates or (not complete and not allow_partial):
            raise ValueError(f'{folder} is incomplete; final cost comparison withheld')
        excluded_path = folder/'measurement-exclusions.json'
        excluded = json.loads(excluded_path.read_text())['updates'] if excluded_path.exists() else {}
        measured_rollouts = [r for r in rollout if str(r['update']) not in excluded]
        latency_path = folder/'latency-exclusions.json'
        latency_excluded = json.loads(latency_path.read_text())['updates'] if latency_path.exists() else {}
        measured_steps = [r for r in updates if str(r['update']) not in excluded and str(r['update']) not in latency_excluded]
        starts={r['update']:r['time'] for r in trace if r['event']=='rollout_start'}
        clients.append(dict(name=folder.name,updates=len(updates),complete=complete,measured_updates=len(measured_steps),
            excluded_updates=excluded,latency_excluded_updates=latency_excluded,excluded_intervals=[(starts[r['update']],r['time']) for r in rollout if str(r['update']) in excluded],
            rollout_start=min(starts.values()),rollout_end=max(r['time'] for r in rollout),
            output_tokens=sum(r['output_tokens'] for r in measured_rollouts),
            rollout_seconds=sum(r['seconds'] for r in measured_rollouts),
            mean_rollout_seconds=statistics.mean(r['seconds'] for r in measured_rollouts),
            mean_step_seconds=statistics.mean(r['seconds'] for r in measured_steps),
            total_output_tokens=sum(r['output_tokens'] for r in rollout+warmup),
            input_tokens=sum(r['sampling_input_tokens'] for r in rollout+warmup),
            training_tokens=sum(r['training_input_tokens'] for r in rollout),
            reward=statistics.mean(r['reward'] for r in rollout),
            truncated_fraction=sum(r['truncated'] for r in rollout)/sum(len(r['output_lengths']) for r in rollout)))
    # Count exact completed group tokens while all clients have active workloads.
    start=max(c['rollout_start'] for c in clients)
    end=min(c['rollout_end'] for c in clients)
    if end<=start:
        raise ValueError('No common active window for throughput measurement')
    exclusions = [interval for c in clients for interval in c['excluded_intervals']]
    if exclusions and len(clients) != 1:
        raise ValueError('Aggregate exclusions require a single-client baseline')
    measured_window_seconds = end-start-sum(max(0,min(b,end)-max(a,start)) for a,b in exclusions)
    tokens=0
    for folder in folders:
        for file in sorted(folder.glob('group-*.json.gz')):
            with gzip.open(file,'rt') as f:
                group=json.load(f)
            if start<=group['end']<=end and not any(a<=group['end']<=b for a,b in exclusions):
                tokens+=sum(len(s['tokens']) for s in group['sequences'])
    result=dict(provider=provider,clients=clients,active_window_start=start,active_window_end=end,
                active_window_completed_output_tokens=tokens,aggregate_output_tps=tokens/measured_window_seconds,measured_window_seconds=measured_window_seconds,
                per_client_output_tps=statistics.mean(c['output_tokens']/c['rollout_seconds'] for c in clients),
                mean_rollout_s=statistics.mean(c['mean_rollout_seconds'] for c in clients),
                mean_step_s=statistics.mean(c['mean_step_seconds'] for c in clients))
    count=sum(c['updates'] for c in clients)
    if provider=='Tinker':
        output=sum(c['total_output_tokens'] for c in clients)
        inputs=sum(c['input_tokens'] for c in clients)
        training=sum(c['training_tokens'] for c in clients)
        base=(output*TINKER_RATES['output']+training*TINKER_RATES['training'])/1e6
        low=base+inputs*TINKER_RATES['cached_input']/1e6
        high=base+inputs*TINKER_RATES['input']/1e6
        result.update(cost_basis='token-price range; input cache hits unknown; not an invoice',
                      cost_usd_low=low,cost_usd_high=high,rates_per_million=TINKER_RATES)
    else:
        lifetimes=json.loads((path/'gpu-lifetimes.json').read_text())
        gpu=[r for r in lifetimes if r['gpu_count'] and r['started_at']]
        if not gpu or any(not r['finished_at'] for r in gpu):
            raise ValueError('GPU lifetime coverage incomplete; cost calculation withheld')
        if any(r['gpu_type']!='H200' for r in gpu):
            raise ValueError('Unexpected GPU type; rate must be checked')
        seconds=sum(r['gpu_count']*(r['finished_at']-r['started_at']) for r in gpu)
        result.update(gpu_seconds=seconds,cost_basis='GPU-only list-price estimate; includes startup and staggering',
                      cost_usd_low=seconds*GPU_SECOND_USD,cost_usd_high=seconds*GPU_SECOND_USD)
    result['validation_only'] = any(not c['complete'] for c in clients)
    result['cost_per_update_low']=result['cost_usd_low']/count
    result['cost_per_update_high']=result['cost_usd_high']/count
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--allow-partial-tinker', action='store_true')
    args=parser.parse_args()
    results=[summarize(args.run/'tp1-full/training','8 trainer + 8 inference H200'),
             summarize(args.run/'tp2-full/training','8 trainer + 16 inference H200'),
             summarize(args.run/'tinker','Tinker',allow_partial=args.allow_partial_tinker)]
    (args.run/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=['## Qwen3.8-27B rollout throughput','',
        'DAPO math, rank 32, thinking enabled, 32k context and 16,384 maximum output tokens. '
        'Eight fixed groups of eight responses per update, no dynamic filtering. '
        'Each self-hosted run has eight independent clients, 150-second startup spacing, '
        'and one batch prefetched during training on a shared TP1/DP8 8×H200 trainer. '
        f"The single Tinker client completed {results[-1]['clients'][0]['updates']} updates of the same workload.",'',
        '| Configuration | Clients × updates | Mean rollout | Output TPS/client | Aggregate output TPS | Mean step |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        lines.append(f"| {r['provider']} | {len(r['clients'])} × {r['clients'][0]['updates']} | {r['mean_rollout_s']:.0f}s | {r['per_client_output_tps']:,.0f} | {r['aggregate_output_tps']:,.0f} | {r['mean_step_s']:.0f}s |")
    lines += ['', 'Per-client TPS divides generated output tokens by measured rollout batch duration. '
        'Aggregate TPS counts completed group output tokens during the common active window of all clients. '
        'Batch duration includes inference queueing and response processing.','',
        '| Configuration | Run cost estimate | Cost/client/update | Cost/client for 30 updates |',
        '| --- | ---: | ---: | ---: |']
    def money(a,b):
        return f'${a:.2f}' if abs(a-b)<.005 else f'${a:.2f}–${b:.2f}'
    for r in results:
        if r['validation_only']:
            lines.append(f"| {r['provider']} (validation only) | Incomplete | — | — |")
        else:
            lines.append(f"| {r['provider']} | {money(r['cost_usd_low'],r['cost_usd_high'])} | {money(r['cost_per_update_low'],r['cost_per_update_high'])} | {money(30*r['cost_per_update_low'],30*r['cost_per_update_high'])} |")
    lines += ['', 'Self-hosted costs use measured GPU lifetimes at the '
        '[Modal H200 list price](https://modal.com/pricing); CPU, memory and storage are excluded. '
        'Tinker costs apply [published token prices](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/) '
        'to this actual run’s tokens, with a range for unknown prompt-cache hits. '
        'These are estimates, not invoices. Pilot costs are separate.','']
    excluded = results[-1]['clients'][0]['excluded_updates']
    if excluded:
        lines += [f"Tinker updates {', '.join(excluded)} were interrupted by a billing pause and are excluded from latency/TPS. "
                  'All completed updates remain in token-cost accounting.','']
    if results[-1]['validation_only']:
        lines += ['Tinker was blocked by billing during the baseline. Its throughput uses only completed updates; '
                  'the incomplete run is excluded from the cost comparison.','']
    report='\n'.join(lines)
    (args.run/'comparison.md').write_text(report)
    # Update only this isolated checkout after all three measured runs complete.
    doc=ROOT/'docs/lora_validation.md'
    begin='<!-- qwen27-dapo-throughput:start -->';finish='<!-- qwen27-dapo-throughput:end -->'
    old=doc.read_text()
    section=begin+'\n'+report+'\n'+finish
    if begin in old:
        old=old[:old.index(begin)]+section+old[old.index(finish)+len(finish):]
    else:
        old+='\n'+section+'\n'
    doc.write_text(old)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    labels=['8 inference GPUs\n8 clients','16 inference GPUs\n8 clients','Tinker\n1 client']
    colors=['#258b90','#277ba0','#d78a31']
    metrics=[('aggregate_output_tps','Aggregate rollout throughput','Output tokens / second'),
             ('per_client_output_tps','Rollout throughput per client','Output tokens / second / client'),
             ('mean_step_s','Client update time','Seconds / update'),
             ('cost_per_update_high','Cost per client update','USD / update')]
    for ax,(key,title,ylabel) in zip(axes.flat,metrics):
        measured=[(label,r,color) for label,r,color in zip(labels,results,colors)
                  if not (key.startswith('cost') and r['validation_only'])]
        values=[r[key] for _,r,_ in measured]
        bars=ax.bar([label for label,_,_ in measured],values,color=[color for _,_,color in measured])
        ax.bar_label(bars,labels=[f'${v:.2f}' if key.startswith('cost') else f'{v:,.0f}' for v in values],padding=4)
        ax.set(ylabel=ylabel,title=title,ylim=(0,max(values)*1.15))
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Qwen3.8-27B · DAPO math · rank 32 · 16k output cap')
    for ext in ['png','pdf']:
        fig.savefig(args.run/f'comparison.{ext}',dpi=180)
    print(report)


if __name__=='__main__':
    main()
