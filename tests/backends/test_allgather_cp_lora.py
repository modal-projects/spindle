from types import SimpleNamespace

import torch

from spindle.backends.miles_runtime import allgather_cp_lora


class Row:
    rollout_data = {"n_adapters": 3}

    def get_next(self, keys):
        return {"tokens": torch.zeros(1, 5), "adapter_slots": [2]}


def get_batch(data_iterator, keys, *args, allgather_cp=False, **kwargs):
    batch = data_iterator.get_next(keys)
    if allgather_cp and "adapter_slots" in batch:
        raise AssertionError("Miles rejects adapter slots under all-gather CP")
    return batch


def test_a_bshd_row_counts_its_local_chunk_for_its_adapter():
    model = SimpleNamespace(get_batch=get_batch)
    allgather_cp_lora.install(model)

    batch = model.get_batch(Row(), ["tokens"], None, "bshd", allgather_cp=True)
    assert batch["adapter_slots"] == [2]
    assert batch["adapter_token_counts"].tolist() == [0, 0, 5]

    batch = model.get_batch(Row(), ["tokens"], None, "bshd")
    assert "adapter_token_counts" not in batch
