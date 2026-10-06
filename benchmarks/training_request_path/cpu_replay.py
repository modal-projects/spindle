"""Replay exact archived DAPO tensors through both submission codecs on CPU.

Times SDK encoding through backend payload decoding, excluding routing, queues,
HTTP, GPU work and result delivery. These are not network latency measurements.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx
import zstandard
from replay import batches
from tinker import types
from tinker.lib._pydantic_conv import to_pydantic_request
from tinker.proto.request_conv import forward_backward_request_to_proto

from spindle.encoding import fingerprint
from spindle.engine.api import Command
from spindle.engine.ingress import decode_forward_backward
from spindle.engine.operations import (
    parse_operation_payload,
    serialize_operation_payload,
)
from spindle.engine.training_transport import decode_batch, encode_batch, encode_payload
from spindle.proto import tinker_public_pb2


def checksum(payload):
    return hashlib.sha256(
        json.dumps(
            serialize_operation_payload(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def baseline(request):
    start = time.perf_counter()
    raw = to_pydantic_request(request).model_dump(
        mode="json", exclude_unset=False, exclude_none=True
    )
    body = httpx.Request("POST", "http://unused", json=raw).content
    json.loads(body)["model_id"]
    model, seq, kind, payload = decode_forward_backward(body, "application/json")
    fingerprint(kind.value, serialize_operation_payload(payload))
    encoded = json.dumps(
        {
            "executions": [
                {
                    "model_id": model,
                    "kind": kind.value,
                    "payload": serialize_operation_payload(payload),
                }
            ]
        }
    ).encode()
    restored = parse_operation_payload(
        kind, json.loads(encoded)["executions"][0]["payload"]
    )
    return restored, time.perf_counter() - start, len(body), len(encoded)


def optimized(request):
    start = time.perf_counter()
    body = forward_backward_request_to_proto(request).SerializeToString()
    compressed = zstandard.ZstdCompressor().compress(body)
    body = zstandard.ZstdDecompressor().decompress(compressed)
    route = tinker_public_pb2.ForwardBackwardRequest()
    route.ParseFromString(body)
    model, seq, kind, payload = decode_forward_backward(body, "application/x-protobuf")
    encoded = encode_payload(payload)
    hashlib.sha256(kind.value.encode() + b"\0" + encoded).hexdigest()
    wire = encode_batch((Command(model, kind, payload, encoded),))
    restored = decode_batch(wire)[0].payload
    return restored, time.perf_counter() - start, len(compressed), len(wire)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w") as output:
        for client in range(8):
            for update in range(8, 14):
                data = batches(a.archive, client, update)
                request = types.ForwardBackwardRequest(
                    model_id="cpu-replay",
                    seq_id=1,
                    forward_backward_input=types.ForwardBackwardInput(
                        data=data,
                        loss_fn="ppo",
                        loss_fn_config={
                            "clip_low_threshold": 0.8,
                            "clip_high_threshold": 1.28,
                        },
                    ),
                )
                methods = [("before", baseline), ("after", optimized)]
                if (client + update) % 2:
                    methods.reverse()
                record = {
                    "client": client,
                    "archive_update": update,
                    "examples": len(data),
                    "tokens": sum(d.model_input.length for d in data),
                }
                hashes = []
                for name, method in methods:
                    result, seconds, upload, internal = method(request)
                    digest = checksum(result)
                    hashes.append(digest)
                    record[name] = {
                        "seconds": seconds,
                        "upload_bytes": upload,
                        "internal_bytes": internal,
                        "sha256": digest,
                    }
                    del result
                assert hashes[0] == hashes[1], record
                output.write(json.dumps(record) + "\n")
                output.flush()
                print(
                    client,
                    update,
                    record["before"]["seconds"],
                    record["after"]["seconds"],
                    flush=True,
                )


if __name__ == "__main__":
    main()
