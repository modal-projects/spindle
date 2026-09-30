"""Disposable trainer-only comparison; never include these updates in RL results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np
import tinker
from tinker import types

from client import create_trainer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-url',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lengths',type=int,nargs='+',default=[2048,8192,16384])
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    service=tinker.ServiceClient(base_url=args.base_url)
    print('Creating eight disposable profiler adapters',flush=True)
    with ThreadPoolExecutor(8) as pool:
        trainers=list(pool.map(lambda i:create_trainer(service,'Qwen/Qwen3.5-9B',32),range(8)))
    results=[]
    for length in args.lengths:
        # A fixed synthetic token batch isolates backend compute from generation.
        # Valid repeated tokens, constant advantages; this is not a quality run.
        datum=types.Datum(model_input=types.ModelInput.from_ints([100]*length),loss_fn_inputs={
            'target_tokens':[100]*length,'logprobs':[-5.]*length,
            'advantages':[1./(64*length)]*length})
        data=[datum]*64
        for repeat in range(3):
            trainer=trainers[repeat%8];start=time.time()
            f=trainer.forward_backward(data,'ppo',loss_fn_config={'clip_low_threshold':.8,'clip_high_threshold':1.28})
            o=trainer.optim_step(types.AdamParams(learning_rate=1e-5,grad_clip_norm=1.))
            result=f.result(timeout=1800);opt=o.result(timeout=1800)
            row=dict(length=length,repeat=repeat,warmup=repeat==0,seconds=time.time()-start,
                tokens=length*64,metrics=result.metrics,optimizer_metrics=opt.metrics)
            results.append(row)
            args.output.write_text(json.dumps(results,indent=2))
            print(json.dumps(row),flush=True)
    print('PROFILE_COMPLETE',flush=True)


if __name__=='__main__':main()
