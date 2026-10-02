"""Time concurrent base-model clients against an already deployed frontend.

Set TINKER_API_KEY, then run with --base-url, --model, and --output. Use
--warmup to exclude the initial asset download and inference startup.
Requires tokenizers in addition to Spindle's client dependencies.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import statistics
import time

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import tinker
from tinker import types


async def benchmark(args):
    tokenizer = Tokenizer.from_file(hf_hub_download(args.model, "tokenizer.json"))
    prompt = types.ModelInput.from_ints(
        tokenizer.encode("The capital of France is").ids
    )
    params = types.SamplingParams(max_tokens=32, temperature=0, seed=42)
    service = tinker.ServiceClient(
        base_url=args.base_url,
        api_key=os.environ["TINKER_API_KEY"],
        timeout=1200,
        max_retries=0,
    )
    service.holder.get_session_id()
    report = dict(
        model=args.model,
        clients=args.clients,
        warmup=args.warmup,
        rounds=args.rounds,
        status="running",
        requests=[],
        phases=[],
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    async def sample(phase, index, sampling=None):
        record = dict(phase=phase, index=index, started=time.time())
        report["requests"].append(record)
        try:
            if sampling is None:
                start = time.perf_counter()
                sampling = await service.create_sampling_client_async(
                    base_model=args.model
                )
                record["create_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            result = await sampling.sample_async(
                prompt=prompt, num_samples=1, sampling_params=params
            )
            record["sample_seconds"] = time.perf_counter() - start
            record["output_tokens"] = len(result.sequences[0].tokens)
            record["text"] = tokenizer.decode(result.sequences[0].tokens)
        except BaseException as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            save()
        return sampling

    async def wave(phase, clients=None):
        start = time.perf_counter()
        clients = await asyncio.gather(
            *(
                sample(phase, i, clients[i] if clients else None)
                for i in range(args.clients)
            )
        )
        records = [r for r in report["requests"] if r["phase"] == phase]
        result = dict(
            phase=phase,
            wall_seconds=time.perf_counter() - start,
            p50_seconds=statistics.median(
                r.get("create_seconds", 0) + r["sample_seconds"] for r in records
            ),
        )
        report["phases"].append(result)
        save()
        print(json.dumps(result), flush=True)
        return clients

    try:
        async with asyncio.timeout(args.timeout):
            if args.warmup:
                await sample("warmup", 0)
            for index in range(args.rounds):
                clients = await wave(f"new_clients_{index}")
                await wave(f"existing_clients_{index}", clients)
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = type(exc).__name__
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--clients", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(benchmark(parser.parse_args()))
