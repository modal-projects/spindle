from types import SimpleNamespace

import torch

from spindle.backends.miles_runtime import allgather_cp_lora


class Rows:
    rollout_data = {"n_adapters": 3}

    def get_next(self, keys):
        return {"tokens": torch.zeros(2, 5), "adapter_slots": [2, 0]}


def get_batch(data_iterator, keys, *args, allgather_cp=False, **kwargs):
    batch = data_iterator.get_next(keys)
    if allgather_cp and "adapter_slots" in batch:
        raise AssertionError("Miles rejects adapter slots under all-gather CP")
    return batch


def test_bshd_rows_count_their_local_tokens_per_adapter():
    model = SimpleNamespace(get_batch=get_batch)
    allgather_cp_lora.install(model)
    once = model.get_batch
    allgather_cp_lora.install(model)
    assert model.get_batch is once

    batch = model.get_batch(Rows(), ["tokens"], None, "bshd", allgather_cp=True)
    assert batch["adapter_slots"] == [2, 0]
    assert batch["adapter_token_counts"].tolist() == [5, 0, 5]

    batch = model.get_batch(Rows(), ["tokens"], None, "bshd")
    assert "adapter_token_counts" not in batch
