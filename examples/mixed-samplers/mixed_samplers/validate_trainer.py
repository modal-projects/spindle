"""Bounded native-loss/optimizer/publication check; not a training result."""

import argparse
import json
import math
import os
from pathlib import Path

import httpx
import modal
import tinker
from tinker import types

import spindle

from .config import MIS_HIGH, MIS_LOW, config
from .train import tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-file", type=Path)
    args = parser.parse_args()
    url = modal.Function.from_name(
        config.platform["frontend"], "server", environment_name="kailash-dev"
    ).get_web_url()
    key = (
        args.api_key_file.read_text().strip()
        if args.api_key_file
        else os.environ["TINKER_API_KEY"]
    )
    service = tinker.ServiceClient(base_url=url, api_key=key)
    print("Creating the B200 trainer for native-loss validation", flush=True)
    training = spindle.create_full_training_client(service, config.model)
    print(json.dumps({"model_id": training.model_id}), flush=True)
    try:
        rows = []
        n = 128
        for sign in (1.0, -1.0):
            target = 2 if sign > 0 else 3
            rows.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(
                        [100, 101, 102, 103] * (n // 4)
                    ),
                    loss_fn_inputs={
                        "target_tokens": tensor([target] * n, "int64"),
                        "weights": tensor([1 / (2 * n)] * n),
                        "advantages": tensor([sign] * n),
                        "logprobs": tensor([math.log(0.3 if target == 2 else 0.6)] * n),
                        "sampling_mask_ids": tensor([0, 2, 3] * n, "int64"),
                        "sampling_mask_offsets": tensor(
                            list(range(0, n * 3 + 1, 3)), "int64"
                        ),
                        "sampling_mask_logprobs": tensor(
                            [math.log(q) for q in (0.1, 0.3, 0.6)] * n
                        ),
                    },
                )
            )
        result = training.forward_backward(
            rows, "score_centered_mis", {"mis_low": MIS_LOW, "mis_high": MIS_HIGH}
        ).result(timeout=3600)
        print(json.dumps({"forward_backward": result.metrics}), flush=True)
        update = training.optim_step(
            types.AdamParams(learning_rate=1e-6, grad_clip_norm=1.0)
        ).result(timeout=1800)
        print(json.dumps({"optimizer": update.metrics}), flush=True)
        if (
            not all(
                math.isfinite(v) for v in {**result.metrics, **update.metrics}.values()
            )
            or update.metrics["update_successful:mean"] != 1
        ):
            raise RuntimeError("native-loss validation produced an invalid update")
        publication = training.save_weights_for_sampler(
            "native-loss-validation"
        ).result(timeout=3600)
        report = {
            "forward_backward": result.metrics,
            "optimizer": update.metrics,
            "publication": publication.path,
            "model_id": training.model_id,
        }
        Path("/tmp/spindle-native-trainer-validation.json").write_text(
            json.dumps(report, indent=2)
        )
        print(json.dumps(report), flush=True)
    finally:
        response = httpx.post(
            f"{url}/api/v1/unload_model",
            headers={"X-API-Key": key},
            json={"model_id": training.model_id},
            timeout=300,
        )
        response.raise_for_status()


if __name__ == "__main__":
    main()
