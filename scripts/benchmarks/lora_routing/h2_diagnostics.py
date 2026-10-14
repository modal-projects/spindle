"""Record only HTTP/2 GOAWAY metadata during the benchmark, never headers."""

import json
import time

from h2.connection import H2Connection


enabled = False


def enable():
    global enabled
    if enabled:
        return
    enabled = True
    original = H2Connection._receive_goaway_frame

    def receive(connection, frame):
        print(
            json.dumps(
                dict(
                    event="benchmark_h2_goaway",
                    time=time.time(),
                    error_code=frame.error_code,
                    last_stream_id=frame.last_stream_id,
                    additional_data_bytes=len(frame.additional_data),
                )
            ),
            flush=True,
        )
        return original(connection, frame)

    H2Connection._receive_goaway_frame = receive
