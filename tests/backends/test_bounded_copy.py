import os

from spindle.backends.miles_runtime import bounded_copy


def test_bounded_copy_reproduces_the_file_and_syncs_periodically(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "bucket.bin"
    data = os.urandom(10 * 1024 + 7)
    source.write_bytes(data)
    syncs = []
    monkeypatch.setattr(bounded_copy, "_fdatasync", lambda fd: syncs.append(fd))

    target = tmp_path / "nested" / "copy.bin"
    bounded_copy.copy_file_bounded(
        source, target, chunk_bytes=1024, sync_every_bytes=4096
    )

    assert target.read_bytes() == data
    # Two interior syncs after 4 KiB each, plus the final one.
    assert len(syncs) == 3
