"""Read-only native cache export and CPU counters for all three experiment arms."""

from pathlib import Path
import sys


def replace(path, old, new):
    source = path.read_text()
    assert source.count(old) == 1, (str(path), old[:70], source.count(old))
    path.write_text(source.replace(old, new))


def install(root, spindle):
    rust = root / "rust/sglang-radix-tree/src"
    replace(
        rust / "unified_tree_core.rs",
        "    pub fn root_node_handle(&self, _extra_key: Option<&str>) -> NodeId {",
        """    // Benchmark observation only: no access updates, splits, or tensor reads.
    pub fn routing_snapshot(&self) -> Vec<(usize, usize, Option<String>, Vec<i64>, bool, i64)> {
        self.collect_all_nodes_().into_iter().filter_map(|idx| {
            let node = self.arena.node(idx);
            if node.is_root() || !node.has_device_value(FULL) || K::IS_BIGRAM || node.namespace.cache_salt().is_some() {
                return None;
            }
            Some((node.id, self.arena.node(node.parent()).id,
                node.namespace.extra_key().map(str::to_string),
                K::raw_token_ids(node.key.as_ref()).into_owned(),
                node.has_device_value(MAMBA), node.last_access_counter))
        }).collect()
    }

    pub fn root_node_handle(&self, _extra_key: Option<&str>) -> NodeId {""",
    )
    replace(
        rust / "python_bindings.rs",
        "            /// Drop the entire tree and reinitialize empty state.",
        """            fn routing_snapshot(&self, py: Python<'_>) -> PyResult<Vec<(usize, usize, Option<String>, Vec<i64>, bool, i64)>> {
                catch_native_panic(|| Ok(py.allow_threads(|| self.inner.core().routing_snapshot())))
            }

            /// Drop the entire tree and reinitialize empty state.""",
    )
    engine = root / "python/sglang"
    sys.path.insert(0, str(spindle.parent))
    from spindle.inference.sglang_routing_observer import install as observer_install

    observer_install(engine, modern=True)
    manager = engine / "srt/lora/lora_manager.py"
    replace(
        manager,
        "import logging\n",
        "import logging\nimport time\nfrom sglang.srt import spindle_routing_observer as routing_observer\n",
    )
    replace(
        manager,
        "        self.memory_pool.prepare_lora_batch(\n",
        "        routing_before = set(self.memory_pool.uid_to_buffer_id)\n        routing_started = time.perf_counter()\n        self.memory_pool.prepare_lora_batch(\n",
    )
    replace(
        manager,
        "        if new_uids:\n",
        "        routing_observer.loaded(routing_before, set(self.memory_pool.uid_to_buffer_id), time.perf_counter() - routing_started)\n        if new_uids:\n",
    )
    scheduler = engine / "srt/managers/scheduler.py"
    replace(
        scheduler,
        '        """Run a batch."""\n',
        '        """Run a batch."""\n        if self.enable_lora:\n            spindle_routing_observer.batch(batch, self.tp_worker.model_runner.lora_manager)\n',
    )


if __name__ == "__main__":
    install(Path("/sgl-workspace/sglang"), Path("/opt/rl-measurement/spindle"))
