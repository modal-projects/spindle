"""Report all generation and GPU lifetime costs; accept actual billed totals."""
import argparse
import json
from pathlib import Path
import statistics


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(root):
    folders=sorted(root.glob('client-*')) if not (root/'manifest.json').exists() else [root]
    clients=[];manifests=[]
    for folder in folders:
        if not folder.is_dir() or not (folder/'manifest.json').exists():continue
        manifests.append(json.loads((folder/'manifest.json').read_text()))
        events=read_events(folder/'events.jsonl')
        rollouts=[r for r in events if r['event']=='rollout_done']
        updates=[r for r in events if r['event']=='update_done']
        warmups=[r for r in events if r['event']=='sampling_warmup']
        lengths=[length for r in rollouts for length in r['output_lengths']]
        clients.append(dict(client=folder.name,completed=any(r['event']=='completed' for r in events),
            updates=len(updates),output_tokens=sum(r['output_tokens'] for r in rollouts+warmups),
            used_output_tokens=sum(r['used_output_tokens'] for r in rollouts),
            sampling_input_tokens=sum(r['sampling_input_tokens'] for r in rollouts+warmups),
            training_input_tokens=sum(r['training_input_tokens'] for r in rollouts),
            mean_output_length=statistics.mean(lengths) if lengths else None,
            truncated_fraction=sum(r['truncated'] for r in rollouts)/len(lengths) if lengths else None,
            mean_batch_seconds=statistics.mean(r['seconds'] for r in updates) if updates else None,
            mean_reward=statistics.mean(r['reward'] for r in rollouts) if rollouts else None))
    result=dict(clients=clients,manifests=manifests,total_output_tokens=sum(c['output_tokens'] for c in clients))
    path=root/'gpu-lifetimes.json'
    if path.exists():
        lifetimes=json.loads(path.read_text());complete=all(r['finished_at'] for r in lifetimes if r['gpu_count'])
        result['gpu_lifetimes_complete']=complete
        if complete:
            result['gpu_seconds']=sum(r['gpu_count']*(r['finished_at']-r['started_at']) for r in lifetimes if r['gpu_count'])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('--compare',type=Path)
    p.add_argument('--billed-usd',type=float)
    p.add_argument('--comparison-billed-usd',type=float)
    args=p.parse_args();result=summarize(args.run)
    if args.billed_usd is not None:
        result['billed_usd']=args.billed_usd
        result['billed_usd_per_client']=args.billed_usd/len(result['clients'])
    if args.compare:
        other=summarize(args.compare)
        keys=['config_sha256','prompts_sha256','dataset_sha256','stop_token_ids','packages']
        manifests=result['manifests']+other['manifests']
        if not manifests:raise ValueError('No manifests')
        differences=[k for k in keys if any(m[k]!=manifests[0][k] for m in manifests)]
        result['comparison']=other;result['configuration_differences']=differences
        if args.comparison_billed_usd is not None:other['billed_usd']=args.comparison_billed_usd
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
