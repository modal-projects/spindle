import shlex

import modal

from .image_dependencies import (
    CORE_PACKAGES,
    MEGATRON_RUNTIME_CHECK,
    MEGATRON_RUNTIME_PACKAGES,
    STITCH_PACKAGE,
    ignore_config_source,
)

# The release tag pins Miles together with the Megatron-LM and Megatron-Bridge
# revisions its image was built and tested with. Update all three by moving to a
# newer release tag, digest, and commit together, then refresh the trainer app.
MILES_RELEASE = "v0.1.1"
BASE_IMAGE = (
    f"radixark/miles:{MILES_RELEASE}"
    "@sha256:6355834f16bacd35d5d40c43f142e3758376f7b2e8d678bccfe870c092bd96bf"
)
MILES_COMMIT = "2806267d060d51b1d3b62f85a1f9b145047aeef9"
MILES_PATH = "/root/miles"
MEGATRON_PATH = "/root/Megatron-LM"
RELEASE_CHECK = "; ".join(
    [
        "import json, re, subprocess",
        "from importlib.metadata import distribution",
        (
            "head = lambda path: subprocess.check_output("
            "['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()"
        ),
        f"assert head('{MILES_PATH}') == '{MILES_COMMIT}', 'Miles is not at MILES_COMMIT'",
        f"lock = json.load(open('{MILES_PATH}/release-lock.json'))",
        (
            f"assert head('{MEGATRON_PATH}') == lock['megatron_commit'], "
            "'Megatron-LM differs from release-lock.json'"
        ),
        f"dockerfile = open('{MILES_PATH}/docker/Dockerfile').read()",
        "bridge = re.search(r'Megatron-Bridge[.]git@([0-9a-f]{40})', dockerfile)[1]",
        "url = json.loads(distribution('megatron-bridge').read_text('direct_url.json'))",
        (
            "assert url['vcs_info']['commit_id'] == bridge, "
            "'Megatron-Bridge differs from the Miles Dockerfile'"
        ),
        "print('megatron-lm', lock['megatron_commit'], 'megatron-bridge', bridge)",
    ]
)

# The release validator assumes gated experts; ungated MLPs have no gate_proj.
MILES_UNGATED_EXPERT_PATCH = (
    "--- a/miles/backends/megatron_utils/lora/bridge.py\n"
    "+++ b/miles/backends/megatron_utils/lora/bridge.py\n"
    "@@ -105,7 +105,9 @@\n"
    '     served = {target.rsplit(".", 1)[-1] for target in args.hf_lora_targets}\n'
    '     if "gate_up_proj" in served:\n'
    '         served.update(("gate_proj", "up_proj"))\n'
    '-    expert_pair = {"gate_proj", "up_proj", "down_proj"}\n'
    '+    expert_pair = {"up_proj", "down_proj"}\n'
    "+    if provider.gated_linear_unit:\n"
    '+        expert_pair.add("gate_proj")\n'
    "     if served & expert_pair:\n"
    "         assert expert_pair <= served, (\n"
    '             f"Multi-LoRA on MoE experts requires all of {sorted(expert_pair)} in "\n'
)

# Bridge-built models bypass Megatron's CLI-to-config conversion.
MILES_BRIDGE_CP_PATCH = (
    "--- a/miles/backends/megatron_utils/model_provider.py\n"
    "+++ b/miles/backends/megatron_utils/model_provider.py\n"
    "@@ -45,3 +45,6 @@\n"
    "     provider.sequence_parallel = args.sequence_parallel\n"
    "     provider.context_parallel_size = args.context_parallel_size\n"
    "+    provider.cp_comm_type = (\n"
    "+        args.cp_comm_type[0] if len(args.cp_comm_type) == 1 else args.cp_comm_type\n"
    "+    )\n"
    " \n"
    "--- a/miles/backends/megatron_utils/lora/bridge.py\n"
    "+++ b/miles/backends/megatron_utils/lora/bridge.py\n"
    "@@ -154,3 +154,6 @@\n"
    "     provider.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size\n"
    "     provider.context_parallel_size = args.context_parallel_size\n"
    "+    provider.cp_comm_type = (\n"
    "+        args.cp_comm_type[0] if len(args.cp_comm_type) == 1 else args.cp_comm_type\n"
    "+    )\n"
    "     provider.gradient_accumulation_fusion = args.gradient_accumulation_fusion\n"
)

BRIDGE_IMPORT_CACHE_PATCH = (
    "--- a/megatron/bridge/models/conversion/model_bridge.py\n"
    "+++ b/megatron/bridge/models/conversion/model_bridge.py\n"
    "@@ -21,6 +21,7 @@\n"
    " import math\n"
    " import re\n"
    " import warnings\n"
    "+from collections import Counter\n"
    " from dataclasses import dataclass, field, fields, is_dataclass\n"
    " from pathlib import Path\n"
    " from typing import (\n"
    "@@ -1470,6 +1471,11 @@\n"
    "         self.unquantized_state_dict = None\n"
    " \n"
    "         _hf_import_cache: Dict[str, torch.Tensor] = {}\n"
    "+        remaining_grouped_uses = Counter(\n"
    "+            str(task.mapping.hf_param)\n"
    "+            for task in hf_to_megatron_tasks\n"
    '+            if task.megatron_module is not None and getattr(task.mapping, "is_grouped_export", False)\n'
    "+        )\n"
    "         for task in self._with_progress_tracking(hf_to_megatron_tasks, description):\n"
    "             # None means megatron module not on current rank, skip if this task is not going to happen\n"
    "             if task.megatron_module is None:\n"
    "@@ -1483,6 +1489,12 @@\n"
    "                 hf_weights = self.maybe_modify_loaded_hf_weight(task.mapping.hf_param, hf_state_dict)\n"
    "                 if is_grouped:\n"
    "                     _hf_import_cache[hf_param_key] = hf_weights\n"
    "+            if is_grouped:\n"
    "+                remaining_grouped_uses[hf_param_key] -= 1\n"
    "+                if remaining_grouped_uses[hf_param_key] == 0:\n"
    "+                    # The local reference keeps this tensor alive through conversion.\n"
    "+                    # Retaining earlier layers would cache a full expanded model per rank.\n"
    "+                    del _hf_import_cache[hf_param_key]\n"
    " \n"
    "             # 2) Delegate conversion & distribution to the bridge\n"
    "             converted_weights = self._convert_loaded_hf_weight(task, hf_weights)\n"
)

# Stage adapter snapshots on CPU during conversion, before materializing the list.
MILES_ADAPTER_SNAPSHOT_CPU_PATCH = (
    "--- a/miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py\n"
    "+++ b/miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py\n"
    "@@ -17,10 +17,21 @@\n"
    " class HfWeightIteratorBridge(MegatronHfWeightIteratorBase):\n"
    "     def __init__(self, *args, **kwargs):\n"
    "         super().__init__(*args, **kwargs)\n"
    "+        self._adapter_export_cpu = False\n"
    " \n"
    "         from megatron.bridge import AutoBridge\n"
    " \n"
    "         self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)\n"
    "+\n"
    "+    def materialize_adapter(self, adapter, *, materialize=True):\n"
    "+        # Disk snapshots need CPU tensors. Stage each converted tensor before\n"
    "+        # the exporter builds its full list; copying the final list is too late.\n"
    "+        previous = self._adapter_export_cpu\n"
    "+        self._adapter_export_cpu = True\n"
    "+        try:\n"
    "+            return super().materialize_adapter(adapter, materialize=materialize)\n"
    "+        finally:\n"
    "+            self._adapter_export_cpu = previous\n"
    " \n"
    "     def _iter_hf_param_units(self, weights, *, materialize):\n"
    "         renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in weights.items()}\n"
    "@@ -74,7 +85,11 @@\n"
    "                 **self._source_name_kwargs(self._bridge.export_adapter_weights),\n"
    "             )\n"
    '             named_weights = self._postprocess_and_quantize(named_weights, "lora")\n'
    "-            return [(h, w) for h, w, _m in named_weights if is_lora_weight_name(h)]\n"
    "+            return [\n"
    '+                (h, w.detach().to("cpu", copy=True) if self._adapter_export_cpu else w)\n'
    "+                for h, w, _m in named_weights\n"
    "+                if is_lora_weight_name(h)\n"
    "+            ]\n"
    " \n"
    "     @staticmethod\n"
    "     def _source_name_kwargs(export_fn) -> dict:\n"
)

# The release has the DSA backend hooks but omits the TileLang implementation.
# Upstream #5049 adds the kernels and their packed/context-parallel tests.
DSA_KERNEL_COMMIT = "4f657171b150510cafc48f429673fb1b2c4c098d"
DSA_KERNEL_SHA256 = "416f29e8b9456a9377707066963bd924f68f38a155e8c9474bcaabb658dac182"
DSA_KERNEL_INSTALL = (
    "import hashlib, subprocess, urllib.request; "
    f"url = 'https://github.com/NVIDIA/Megatron-LM/commit/{DSA_KERNEL_COMMIT}.diff'; "
    "patch = urllib.request.urlopen(url, timeout=60).read(); "
    f"assert hashlib.sha256(patch).hexdigest() == '{DSA_KERNEL_SHA256}'; "
    f"subprocess.run(['git', '-C', '{MEGATRON_PATH}', 'apply', '-'], input=patch, check=True)"
)


image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .env(
        {
            "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "SPINDLE_MILES_COMMIT": MILES_COMMIT,
        }
    )
    .run_commands(
        f"python -c {shlex.quote(RELEASE_CHECK)}",
        "cd /root/miles && git apply - <<'PATCH'\n"
        + MILES_UNGATED_EXPERT_PATCH
        + "PATCH\n",
        "python - <<'PY'\n"
        "from importlib.metadata import distribution\n"
        "import subprocess\n"
        "root = distribution('megatron-bridge').locate_file('')\n"
        "subprocess.run(['patch', '-p1', '-d', str(root)], input="
        + repr(BRIDGE_IMPORT_CACHE_PATCH)
        + ", text=True, check=True)\nPY\n",
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE)
    .pip_install(
        *MEGATRON_RUNTIME_PACKAGES,
        "xxhash==3.7.1",
        "transformers==5.12.1",
        "pyyaml>=6.0.2",
        "opentelemetry-exporter-otlp==1.43.0",
        "opentelemetry-exporter-otlp-proto-grpc==1.43.0",
        "opentelemetry-exporter-otlp-proto-http==1.43.0",
    )
    .run_commands(
        "pip install --no-deps 'peft>=0.18.1'",
        MEGATRON_RUNTIME_CHECK,
        'python -c "from miles.ray.train.group import TrainerController as T; '
        "assert all(hasattr(T, name) for name in "
        "('load_slot', 'unload_slot', 'forward_backward', "
        "'forward_only', 'optim_step', 'save_slot', 'export_slot'))\"",
        "cd /root/miles && git apply - <<'PATCH'\n"
        + MILES_BRIDGE_CP_PATCH
        + MILES_ADAPTER_SNAPSHOT_CPU_PATCH
        + "PATCH\n",
    )
    .run_commands(f"python -c {shlex.quote(DSA_KERNEL_INSTALL)}")
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
