"""Call the deployed engine probe directly, without creating an ephemeral App."""

import argparse
import contextlib
import json
import time
from pathlib import Path

import modal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", default="spindle-qwen9b-engine-snapshot-1106")
    parser.add_argument("--mode", choices=("baseline", "snapshot"), default="snapshot")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--settle-seconds", type=float, default=90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.settle_seconds < 0:
        parser.error("repeats must be positive and settle-seconds nonnegative")
    cls = modal.Cls.from_name(args.app, args.mode.capitalize())
    rows = []
    for i in range(args.repeats):
        if i and args.mode == "snapshot":
            time.sleep(args.settle_seconds)
        sent = time.time()
        start = time.monotonic()
        call = cls().benchmark.spawn(sent)
        print("DIRECT_ENGINE_CALL", call.object_id, flush=True)
        try:
            row = call.get(timeout=2400)
        except BaseException as exc:
            with contextlib.suppress(Exception):
                call.cancel(terminate_containers=True)
            args.output.write_text(
                json.dumps(
                    {"mode": args.mode, "rows": rows, "failure": str(exc)}, indent=2
                )
            )
            raise
        row["client_to_result_s"] = time.monotonic() - start
        rows.append(row)
        args.output.write_text(json.dumps({"mode": args.mode, "rows": rows}, indent=2))
        print(
            "DIRECT_ENGINE_RESULT",
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "task_id",
                        "creator_task_id",
                        "restored_from_different_container",
                        "client_to_engine_ready_wall_s",
                        "hook_to_engine_ready_s",
                        "resume_s",
                        "producer_publication_hold_s",
                    )
                }
            ),
            flush=True,
        )
        if args.mode == "snapshot" and row["restored_from_different_container"]:
            break
    if args.mode == "snapshot" and not any(
        row["restored_from_different_container"] for row in rows
    ):
        raise RuntimeError("No separate-container restore observed")


if __name__ == "__main__":
    main()
