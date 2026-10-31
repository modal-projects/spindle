"""Opt-in sampler replay for Miles LoRA and full-parameter score centering."""

from __future__ import annotations

import base64
import math
import struct
import time

from pydantic import BaseModel, Field
from tinker import SamplingClient, types
from tinker.lib.api_future_impl import _APIFuture
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture

REPLAY_FIELDS = frozenset(
    {
        "routed_experts",
        "sampling_mask_ids",
        "sampling_mask_offsets",
        "sampling_mask_logprobs",
    }
)


class SamplingReplay(BaseModel):
    prompt_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0, allow_inf_nan=False)
    routed_experts: str | None = None  # SGLang's base64 little-endian int32 buffer
    sampling_mask_ids: list[int] | None = None
    sampling_mask_offsets: list[int] | None = None
    sampling_mask_logprobs: list[float] | None = None
    sampling_logprobs: list[float] | None = None

    def loss_fn_config(self) -> dict[str, float]:
        # Greedy capture has singleton supports, whose normalized logprob is zero.
        return {"sampling_temperature": self.temperature or 1.0}

    def training_inputs(
        self,
        response_tokens: int,
        *,
        num_layers: int | None = None,
        experts_per_token: int | None = None,
    ) -> dict[str, types.TensorData]:
        """Replay tensors for a datum with (prompt + response)[:-1] as input.

        Router dimensions are required because SGLang returns an unshaped buffer.
        Sampling supports cover generated targets; empty prompt rows mean unmasked.
        """
        if response_tokens < 1:
            raise ValueError("replay requires at least one generated token")
        n = self.prompt_tokens + response_tokens - 1
        result = {}
        if self.routed_experts is not None:
            if (
                not num_layers
                or not experts_per_token
                or min(num_layers, experts_per_token) < 1
            ):
                raise ValueError(
                    "router replay requires num_layers and experts_per_token"
                )
            raw = base64.b64decode(self.routed_experts, validate=True)
            stride = num_layers * experts_per_token
            # SGLang can also forward the final token at a stop boundary.
            if len(raw) == (n + 1) * stride * 4:
                raw = raw[: n * stride * 4]
            if len(raw) != n * stride * 4:
                raise ValueError(
                    "router replay shape does not match prompt + response - 1"
                )
            values = [v for (v,) in struct.iter_unpack("<i", raw)]
            result["routed_experts"] = types.TensorData(
                data=values, dtype="int64", shape=[n, num_layers, experts_per_token]
            )
        if self.sampling_mask_ids is not None:
            offsets = self.sampling_mask_offsets
            logprobs = self.sampling_logprobs
            if (
                offsets is None
                or len(offsets) != response_tokens + 1
                or logprobs is None
                or len(logprobs) != response_tokens
            ):
                raise ValueError(
                    "sampling replay length does not match generated tokens"
                )
            result["sampling_mask_ids"] = _tensor(self.sampling_mask_ids)
            result["sampling_mask_offsets"] = _tensor(
                [0] * (self.prompt_tokens - 1) + offsets
            )
            result["logprobs"] = types.TensorData(
                data=[0.0] * (self.prompt_tokens - 1) + logprobs,
                dtype="float32",
                shape=[n],
            )
            if self.sampling_mask_logprobs is not None:
                if len(self.sampling_mask_logprobs) != len(self.sampling_mask_ids):
                    raise ValueError(
                        "sampling support logprobs do not match the recorded mask"
                    )
                result["sampling_mask_logprobs"] = types.TensorData(
                    data=self.sampling_mask_logprobs,
                    dtype="float32",
                    shape=[len(self.sampling_mask_logprobs)],
                )
        return result


def _tensor(values: list[int]) -> types.TensorData:
    return types.TensorData(data=values, dtype="int64", shape=[len(values)])


class ReplaySequence(BaseModel):
    tokens: list[int]
    logprobs: list[float] | None = None
    stop_reason: str
    replay: SamplingReplay


class ReplaySampleResponse(BaseModel):
    sequences: list[ReplaySequence]
    prompt_cache_hit_tokens: int = 0


def sample_with_replay(
    client: SamplingClient,
    prompt: types.ModelInput,
    *,
    sampling_params: types.SamplingParams,
    num_samples: int = 1,
    return_routed_experts: bool = False,
    return_sampling_mask: bool = False,
    return_sampling_probabilities: bool = False,
) -> AwaitableConcurrentFuture[ReplaySampleResponse]:
    """Like SamplingClient.sample(), with a .replay object on every sequence.

    Returns a future supporting result() and result_async(). Ordinary Tinker
    sampling decoders do not preserve these additional response fields.
    """
    if return_sampling_probabilities:
        return_sampling_mask = True
    if not (return_routed_experts or return_sampling_mask):
        raise ValueError("request at least one replay type")

    async def sample():
        seq_id = client._request_id_counter
        client._request_id_counter += 1
        started = time.time()
        request = types.SampleRequest(
            sampling_session_id=client._sampling_session_id,
            seq_id=seq_id,
            num_samples=num_samples,
            prompt=prompt,
            sampling_params=sampling_params,
        )
        estimated_bytes = client.holder.estimate_bytes_count_in_model_input(prompt)
        async with client.holder.sample_dispatch_rate_limit(estimated_bytes):
            with client.holder.aclient(ClientConnectionPoolType.SAMPLE) as api:
                # Post the sample request directly because Tinker SDK 0.26
                # removed extra_body, which used to carry the replay flags.
                future = await api.post(
                    "/api/v1/asample",
                    body={
                        **request.model_dump(exclude_none=True, mode="json"),
                        "return_routed_experts": return_routed_experts,
                        "return_sampling_mask": return_sampling_mask,
                        "return_sampling_probabilities": return_sampling_probabilities,
                    },
                    cast_to=types.UntypedAPIFuture,
                )
        return await _APIFuture(
            ReplaySampleResponse,
            client.holder,
            future,
            request_start_time=started,
            request_type="Sample",
            queue_state_observer=client,
        ).result_async()

    return client.holder.run_coroutine_threadsafe(sample())


def capture_replay(
    meta: dict,
    tokens: list[int],
    *,
    prompt_tokens: int,
    temperature: float,
    routes: bool,
    mask: bool,
    probabilities: bool = False,
) -> dict:
    """Validate requested capture instead of silently training without replay."""
    replay = SamplingReplay(prompt_tokens=prompt_tokens, temperature=temperature)
    if routes:
        encoded = meta.get("routed_experts")
        if not isinstance(encoded, str) or (tokens and not encoded):
            raise ValueError(
                "sampler did not return routed_experts; enable route capture on the rollout server"
            )
        if len(base64.b64decode(encoded, validate=True)) % 4:
            raise ValueError("invalid routed_experts int32 buffer")
        replay.routed_experts = encoded
    if mask or probabilities:
        supports = meta.get("output_token_sampling_mask")
        logprobs = meta.get("output_token_sampling_logprobs")
        if (
            not isinstance(supports, list)
            or not isinstance(logprobs, list)
            or len(supports) != len(tokens)
            or len(logprobs) != len(tokens)
        ):
            raise ValueError(
                "sampler did not return token-aligned sampling masks and sampling logprobs"
            )
        ids, offsets = [], [0]
        for token, support in zip(tokens, supports, strict=True):
            if (
                not isinstance(support, list)
                or not support
                or any(type(i) is not int or i < 0 for i in support)
                or token not in support
                or len(set(support)) != len(support)
            ):
                raise ValueError(
                    "invalid sampling support: every emitted token must be in its mask"
                )
            ids.extend(support)
            offsets.append(len(ids))
        replay.sampling_mask_ids, replay.sampling_mask_offsets = ids, offsets
        if probabilities:
            flat, chosen = [], []
            for token, support, row in zip(tokens, supports, logprobs, strict=True):
                if (
                    not isinstance(row, list)
                    or len(row) != len(support)
                    or any(
                        not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value > 0
                        for value in row
                    )
                    or not math.isclose(
                        sum(math.exp(value) for value in row), 1, abs_tol=1e-4
                    )
                ):
                    raise ValueError(
                        "sampling support logprobs must be a normalized distribution over the mask"
                    )
                flat.extend(float(value) for value in row)
                chosen.append(float(row[support.index(token)]))
            replay.sampling_mask_logprobs = flat
            replay.sampling_logprobs = chosen
        else:
            if any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value > 0
                for value in logprobs
            ):
                raise ValueError("invalid sampling logprobs")
            replay.sampling_logprobs = [float(value) for value in logprobs]
    return replay.model_dump(exclude_none=True)
