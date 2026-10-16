"""Capture one live rank-zero forward/backward without changing training inputs."""
import argparse
import json
import time
from pathlib import Path

import ray


def install(actor, output):
    import json
    import time
    from pathlib import Path
    import torch

    original = actor._profiled
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    actor._spindle_diagnostic = {'state': 'armed', 'armed_at': time.time()}

    def measured(operation, *args, **kwargs):
        if operation != 'forward_backward':
            return original(operation, *args, **kwargs)
        actor._profiled = original
        started = time.time()
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
        )
        with profiler:
            result = original(operation, *args, **kwargs)
        ended = time.time()
        try:
            averages = profiler.key_averages()
            (directory/'operators.txt').write_text(averages.table(sort_by='self_cuda_time_total', row_limit=80))
            rows = [{'name': e.key, 'count': e.count, 'cpu_us': e.cpu_time_total,
                     'self_cpu_us': e.self_cpu_time_total, 'device_us': e.device_time_total,
                     'self_device_us': e.self_device_time_total} for e in averages]
            (directory/'operators.json').write_text(json.dumps(rows))
            profiler.export_chrome_trace(str(directory/'trace.json'))
            status = {'state': 'completed', 'started': started, 'ended': ended,
                      'seconds': ended-started, 'output': str(directory)}
        except Exception as exc:
            status = {'state': 'export_failed', 'started': started, 'ended': ended, 'error': str(exc)}
        actor._spindle_diagnostic = status
        (directory/'status.json').write_text(json.dumps(status))
        return result

    actor._profiled = measured
    return actor._spindle_diagnostic


def status(actor):
    return actor._spindle_diagnostic


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--namespace', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    ray.init(address='auto', logging_level='ERROR')
    manager = ray.get_actor('ray_worker_manager', namespace=args.namespace)
    cells = ray.get(manager.get_cell_infos.remote(pool_ids=['trainer-actor']))
    infos = ray.get(manager.get_worker_infos.remote(cell_id=next(iter(cells))))
    worker = next(info for info in infos if info.gpu_ids == [0]).handle._actor_handle
    print(json.dumps(ray.get(worker.__ray_call__.remote(install, args.output))), flush=True)
    deadline = time.monotonic()+600
    while time.monotonic()<deadline:
        time.sleep(5)
        current = ray.get(worker.__ray_call__.remote(status))
        if current['state'] != 'armed':
            print(json.dumps(current), flush=True)
            break
    ray.shutdown()


if __name__ == '__main__':
    main()
