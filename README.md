# Spindle

> This README mirrors the [Spindle guide](https://modal.com/docs/guide/spindle) in the Modal docs.

Spindle is an open-source Tinker-compatible API with support for
[multi-tenant LoRA training](docs/multi-lora.md) and
[single-tenant full-parameter training](docs/full-fine-tunes.md).

Why use Spindle?

[Owning](https://modal.com/blog/introducing-auto-endpoints) your training stack is incredibly valuable, especially if
you don't have to manage the underlying infrastructure.

Out of the box, we provide defaults that work for the vast majority of use cases. But when you need to incorporate custom forks into the backend/trainer layer,
tweak compute parameters for optimal price and performance, or even control the entire training/scheduling runtime, you have the power to do so.

Below, we detail how to setup your own Spindle server. See [here](https://modal.com/docs/examples/swe_gym) for a complete example.

## Supported training modes

- **Multi-tenant LoRA training.** Several LoRA adapters share a base model on a Miles/Megatron backend, each with its own gradients and optimizer state. This path works with the Tinker SDK out of the box via `create_lora_training_client`. See [Working with Multi-LoRA](docs/multi-lora.md).
- **Single-tenant full-parameter training (FFT).** Each run gets dedicated training containers in your Modal workspace. Tinker (v0.25) doesn't natively support FFT, so clients are created with a Spindle helper, but the rest follows the same abstractions. See [Working with Full Fine-Tunes](docs/full-fine-tunes.md).

## Install Spindle

Install the library:

```bash
uv add 'modal-spindle @ git+https://github.com/modal-projects/spindle.git'
```

## Set up the server

Generate an API key for your Spindle server and store it as a [Modal Secret](https://modal.com/docs/guide/secrets):

```bash
export TINKER_API_KEY="$(openssl rand -hex 32)"
modal secret create spindle-api TINKER_API_KEY="$TINKER_API_KEY"
```

The Spindle server requires a [Modal Proxy Token](https://modal.com/docs/guide/webhook-proxy-auth):

```bash
modal workspace proxy-tokens create
modal workspace proxy-tokens allow <token-id> "$MODAL_ENVIRONMENT"
```

Store this token under the `spindle-proxy` secret in the same environment:

```bash
modal secret create spindle-proxy \
  MODAL_PROXY_TOKEN_ID="$MODAL_PROXY_TOKEN_ID" \
  MODAL_PROXY_TOKEN_SECRET="$MODAL_PROXY_TOKEN_SECRET"
```

To deploy the Spindle server, simply run:

```bash
spindle deploy
```

You'll want to save the `server` URL from the above deployment output.

A single deployment can serve multiple concurrent independent training jobs across different base models, training parameterizations, and experiment scales.
GPUs are allocated on demand upon the first training/inference requests, so the first step may incur longer cold start and model compilation times.

## Run a Tinker script

Any Tinker-compatible script can be run out of the box against a Spindle server just by changing the base URL and API key.

First, set these two variables from above:

```bash
export TINKER_BASE_URL='https://your-server-url.modal.run'
export TINKER_API_KEY='your-spindle-api-key'
```

As an example, the following script implements a single step RL update, which exercises the full generation/training/sampler publication path.

```python
import os

import tinker
from tinker import types

service = tinker.ServiceClient(
    base_url=os.environ["TINKER_BASE_URL"],
    api_key=os.environ["TINKER_API_KEY"],
)
training = service.create_lora_training_client(
    base_model="Qwen/Qwen3.5-9B-Base",
    rank=16,
)
tokenizer = training.get_tokenizer()
prompt_tokens = tokenizer.encode(
    "What is 2 + 2? Answer with only the number.",
    add_special_tokens=True,
)
prompt = types.ModelInput.from_ints(prompt_tokens)
params = types.SamplingParams(max_tokens=16, temperature=1.0)

sampling = training.save_weights_and_get_sampling_client()
response = (
    sampling.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=params,
    )
    .result(timeout=3600)
    .sequences[0]
)

answer_tokens = list(response.tokens)
logprobs = list(response.logprobs or [])
assert answer_tokens and len(logprobs) == len(answer_tokens)
answer = tokenizer.decode(answer_tokens).strip()
reward = 1.0 if answer == "4" else -1.0

prompt_targets = len(prompt_tokens) - 1
datum = types.Datum(
    model_input=types.ModelInput.from_ints(prompt_tokens + answer_tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": prompt_tokens[1:] + answer_tokens,
        "logprobs": [0.0] * prompt_targets + logprobs,
        "advantages": [0.0] * prompt_targets + [reward] * len(answer_tokens),
    },
)

forward = training.forward_backward([datum], loss_fn="importance_sampling")
optimizer = training.optim_step(types.AdamParams(learning_rate=1e-5))

print(f"Answer: {answer!r}, reward: {reward}")
print("Training metrics:", forward.result(timeout=3600).metrics)
print("Optimizer metrics:", optimizer.result(timeout=3600).metrics)

updated_sampling = training.save_weights_and_get_sampling_client()
updated = (
    updated_sampling.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=params,
    )
    .result(timeout=3600)
    .sequences[0]
)
print("After update:", tokenizer.decode(updated.tokens))
```

## Customize the Spindle server

Since the server is just a [Modal App](https://modal.com/docs/guide/apps), everything from the compute allocation, training backend details, and inference settings is fully customizable.

For example, this Qwen3.5-9B config demonstrates some settings you might change based on your training workload.

```python
# config.py

from spindle.configuration import BaseConfig

class Config(BaseConfig):
    model = "Qwen/Qwen3.5-9B"
    name = "qwen35-9b-lora-16k"
    max_context_length = 16_384
    backend = "miles"


    trainer_gpu = "H100"
    trainer_gpus_per_node = 4
    trainer_cpu = 16
    trainer_memory_mib = 65_536
    trainer_max_clients_per_instance = 6

    miles_cfg = {
        "model_type": "qwen3.5-9B",
        "tensor_model_parallel_size": 4,
        "max_lora_slots": 6,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
        "target_modules": [
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
            "output_layer",
        ],
        "max_tokens_per_gpu": 16_384,
        "cli_options": {
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
    }

    inference_gpu = "H200"
    inference_min_replicas = 2
    inference_max_replicas = 8
    sglang_cfg = {
        "tp_size": 1,
        "mem_fraction_static": 0.8,
        "max_running_requests": 32,
        "max_queued_requests": 8,
        "max_loaded_loras": 64,
        "max_loras_per_batch": 8,
    }


config = Config()
```

To deploy the above config, it's just:

```bash
spindle deploy config.py
```

## Next steps

Refer to the docs for design and for more advanced features when working with either the full-parameter or LoRA paths: 

Read [Working with Full Fine-Tunes](docs/full-fine-tunes.md) for full training,
or [Working with Multi-LoRA](docs/multi-lora.md) for shared Miles adapters, batch
submission, scheduling, and sampling. See [scoped runs](docs/scoped-runs.md) for
recovery and custom engines on the full-parameter path.

See the [raw Tinker RL example](scripts/rl_example.py) for sampling and a toy
policy update. Copy examples you want to run into your project; repository
`scripts/` are not installed with the package.

The [W&B RL example](scripts/wandb_rl_example.py) extends it to a multi-step
loop that logs reward, response length, and Spindle's training metrics to Weights
& Biases from the client side; tinker-cookbook users can instead set
`wandb_project`/`wandb_name` on the cookbook `Config`.

See [Design](docs/design.md) for the control-plane, training-engine, and sampling
architecture.

See [Profiling](docs/profiling.md) for how to enable the `torch.profiler` trace of
a training step and read it in Perfetto.

See [Observability](docs/observability.md) for OTLP export to Datadog or a custom
destination, experiment labels, and the complete span/metric inventory.

## Validation

See [FFT validation](docs/validation.md) and [LoRA validation](docs/lora_validation.md)
for end-to-end training runs we've done with both parameterizations. The [Codeforces codegolf](examples/codeforces-codegolf/README.md) example provides a larger-scale e2e code-RL training run, which trains Qwen3.5-9B
with GRPO or TailRL advantages for correctness and short solutions. It includes
a sandboxed judge, checkpoint recovery, and commands to continue a checkpoint
with a different reward or advantage estimator, as well as pass@k and best-of-k evaluation. 
