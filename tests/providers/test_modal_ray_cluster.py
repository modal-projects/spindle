from spindle.providers.modal import ray_cluster
from spindle.providers.modal.ray_cluster import ClusterTopology


def test_start_trainer_cluster_runs_asset_hooks_in_order(monkeypatch) -> None:
    head = ClusterTopology(nodes=2, rank=0, head_addr="head", node_ip="head-ip")
    worker = ClusterTopology(nodes=2, rank=1, head_addr="head", node_ip="worker-ip")
    topologies = iter((head, worker))
    events: list[str] = []

    monkeypatch.setattr(ray_cluster, "discover_topology", lambda _: next(topologies))
    monkeypatch.setattr(
        ray_cluster,
        "_start_head",
        lambda topology, *, join_timeout: events.append("start_head"),
    )
    monkeypatch.setattr(
        ray_cluster,
        "_wait_for_head",
        lambda topology, *, timeout: events.append("wait_for_head"),
    )
    monkeypatch.setattr(
        ray_cluster,
        "_start_worker",
        lambda topology: events.append("start_worker"),
    )
    monkeypatch.setattr(
        ray_cluster,
        "_idle_until_head_exits",
        lambda topology: events.append("idle"),
    )

    assert (
        ray_cluster.start_trainer_cluster(
            2,
            before_head=lambda: events.append("before_head"),
            before_worker_join=lambda: events.append("before_worker_join"),
        )
        == "head:6379"
    )
    assert events == ["before_head", "start_head"]

    assert (
        ray_cluster.start_trainer_cluster(
            2,
            before_head=lambda: events.append("before_head"),
            before_worker_join=lambda: events.append("before_worker_join"),
        )
        is None
    )
    assert events == [
        "before_head",
        "start_head",
        "wait_for_head",
        "before_worker_join",
        "start_worker",
        "idle",
    ]


def test_cluster_interface_follows_the_node_address(monkeypatch) -> None:
    import socket
    import sys
    import types

    psutil = types.ModuleType("psutil")

    def entry(family, address):
        return types.SimpleNamespace(family=family, address=address)

    psutil.net_if_addrs = lambda: {
        "eth0": [entry(socket.AF_INET, "172.20.5.133")],
        "overlay0": [
            entry(socket.AF_INET6, "fe80::1"),
            entry(socket.AF_INET, "10.100.0.2"),
        ],
    }
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    assert ray_cluster.cluster_interface("10.100.0.2") == "overlay0"
    assert ray_cluster.cluster_interface("10.1.1.1") == "eth0"
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert ray_cluster.cluster_interface("10.100.0.2") == "eth0"
