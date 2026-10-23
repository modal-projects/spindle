"""Install read-only incremental observations into the pinned SGLang engine."""

from pathlib import Path


def replace(path, old, new):
    source = path.read_text()
    assert source.count(old) == 1, (str(path), old[:70], source.count(old))
    path.write_text(source.replace(old, new))


def install(root, spindle):
    rust = root / "rust/sglang-radix-tree/src"
    replace(
        rust / "unified_tree_core.rs",
        "    pub fn root_node_handle(&self, _extra_key: Option<&str>) -> NodeId {",
        """    /// Export all residency metadata, but token bytes only for changed keys.
    /// Node handles survive arena-slot reuse and reset. In this pinned core a
    /// live key changes only by a split, which strictly shortens it. Therefore
    /// (handle, key length) identifies its immutable token segment. The consumer
    /// supplies its acknowledged inventory; this method holds no export cursor.
    /// No tensors, match operations, access counters or eviction state are touched.
    pub fn routing_snapshot_delta(
        &self,
        known: &std::collections::HashMap<usize, usize>,
    ) -> (usize, Vec<(usize, usize, Option<String>, Option<Vec<u8>>, bool, i64, usize)>) {
        let rows = self.collect_all_nodes_().into_iter().filter_map(|idx| {
            let node = self.arena.node(idx);
            if node.is_root() || !node.has_device_value(FULL) || K::IS_BIGRAM
                || node.namespace.cache_salt().is_some() {
                return None;
            }
            let length = node.key.atom_len();
            let tokens = (known.get(&node.id) != Some(&length)).then(|| {
                K::raw_token_ids(node.key.as_ref()).iter()
                    .flat_map(|token| token.to_le_bytes()).collect()
            });
            Some((node.id, self.arena.node(node.parent()).id,
                node.namespace.extra_key().map(str::to_string), tokens,
                node.has_device_value(MAMBA), node.last_access_counter, length))
        }).collect();
        (self.arena.node(self.arena.root()).id, rows)
    }

"""
        + "    pub fn root_node_handle(&self, _extra_key: Option<&str>) -> NodeId {",
    )
    replace(
        rust / "python_bindings.rs",
        "            /// Drop the entire tree and reinitialize empty state.",
        """            fn routing_snapshot_delta(&self, py: Python<'_>, known: HashMap<usize, usize>)
                -> PyResult<(usize, Vec<(usize, usize, Option<String>, Option<Py<PyBytes>>, bool, i64, usize)>)> {
                catch_native_panic(|| {
                    let (root, rows) = py.allow_threads(|| self.inner.core().routing_snapshot_delta(&known));
                    Ok((root, rows.into_iter().map(|(id, parent, adapter, tokens, mamba, accessed, length)| {
                        (id, parent, adapter, tokens.map(|bytes| PyBytes::new_bound(py, &bytes).unbind()),
                            mamba, accessed, length)
                    }).collect()))
                })
            }

"""
        + "            /// Drop the entire tree and reinitialize empty state.",
    )
    engine = root / "python/sglang"
    scheduler = engine / "srt/managers/scheduler.py"
    replace(
        scheduler,
        "import logging\n",
        "import logging\nfrom sglang.srt import spindle_routing_observer\n",
    )
    hook = "        self.process_pending_chunked_abort()\n"
    replace(
        scheduler,
        hook,
        "        spindle_routing_observer.snapshot(self, running_batch, last_batch)\n"
        + hook,
    )
    observer = (spindle / "inference/sglang_routing_observer.py").read_text()
    observer = observer.replace(
        "import xxhash\n",
        'import xxhash\nif __name__ != "__main__":\n    from sglang.srt.runtime_context import get_parallel\n',
    )
    observer = observer.replace("scheduler.ps.", "get_parallel().")
    (engine / "srt/spindle_routing_observer.py").write_text(observer)


if __name__ == "__main__":
    install(Path("/sgl-workspace/sglang"), Path("/opt/spindle-routing/spindle"))
