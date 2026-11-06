"""Load the article-era task set and its published image mapping."""

import json

from huggingface_hub import hf_hub_download
from pyarrow import parquet

from .config import DATASET, DATASET_REVISION


def load_tasks():
    paths = {
        name: hf_hub_download(
            DATASET, name, repo_type="dataset", revision=DATASET_REVISION
        )
        for name in ("code.parquet", "image-mapping.jsonl")
    }
    with open(paths["image-mapping.jsonl"]) as source:
        mappings = [json.loads(line) for line in source]
    images = {row["dataset_image"]: row["dockerhub_image"] for row in mappings}
    tasks = []
    for row in parquet.read_table(paths["code.parquet"]).to_pylist():
        task = json.loads(row["extra_info"]["instance_json"])
        task["docker_image"] = images[task["docker_image"]]
        tasks.append(task)
    if len(tasks) != 2698 or len({t["instance_id"] for t in tasks}) != 2698:
        raise ValueError("the article-era dataset must contain 2698 distinct tasks")
    return tasks
