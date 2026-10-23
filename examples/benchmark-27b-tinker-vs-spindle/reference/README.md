# Measured Tinker baseline

`tinker-events.jsonl` is an extract from the actual October 8, 2026 run
`run-20261008-194808/tinker/events.jsonl`. It preserves the `rollout_start`,
`rollout_done` and `update_done` timestamps and token counts for updates 4–9.
Only the fields consumed by the reporter are retained; model IDs, generated
text/tokens/logprobs and unrelated events are omitted. This is observed client
telemetry, not an invoice or an entire successful 20-update run.

The run stopped after nine completed updates, with billing pauses outside the
selected six-update window. Its manifest recorded Tinker 0.30.4,
Transformers 5.17.0, NumPy 2.4.6 and the same config as `../config.json` except
`updates=20`. Tokenizer revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

- Dataset SHA-256: `57c1b3bbaf38abb049eb1474805d7f7427b93392ef99279161b2a41988e26d42`
- Rendered-prompt SHA-256: `004262185897d63da3af2d31b8a7142593b8ba2f95301f5448f4c5cfc72f22f6`
- Config SHA-256: `a0739340fa306bfb44eaa5eb26a37183e7d7f4b6e7c45fd9743d29bdb27e970a`

The reporter should recover 4,628,189 output tokens over 1,272.714909 seconds,
or **3,636.469541 output tokens/s**, and a historical token-price estimate of
**$9.768050–$9.791456 per million output tokens**, including prompt and training
tokens. Unknown prompt-cache hits explain the range.
