//! Group-scoped affinity with bounded waiting; hints never count as cache hits.
use crate::{Candidate, PendingPrefix, Replica, Request};
use std::cmp::Reverse;

// Experimental caps: one observer period plus polling/transport grace, not a
// claim that waiting two seconds is profitable for every workload.
const PREFIX_GRACE: f64 = 2.0;
const MAX_CACHE_WAIT: f64 = 2.0;

pub struct CacheWait {
    pub absolute: usize,
    pub relative: f64,
    pub threshold: f64,
}
pub enum Selection {
    Wait(&'static str, f64),
    Route(Option<usize>, &'static str),
}
impl CacheWait {
    #[allow(clippy::too_many_arguments)]
    pub fn select(
        &self,
        all: &[Candidate],
        available: &[Candidate],
        replicas: &[Replica],
        request: &Request,
        pending: Option<&PendingPrefix>,
        prefill_rate: f64,
        elapsed: f64,
    ) -> Selection {
        let Some(minimum) = available.iter().map(|c| c.active).min() else {
            // Capacity waiting does not start the affinity budget.
            return Selection::Route(None, "capacity");
        };
        let balanced = |c: &Candidate| {
            !(c.active.saturating_sub(minimum) > self.absolute
                && c.active as f64 > minimum as f64 * self.relative)
        };
        let best_available = available.iter().map(|c| c.cached_tokens).max().unwrap();
        let aligned_prompt = request.hashes.len() * 64;
        if let Some(p) = pending {
            if let Some(target) = all
                .iter()
                .find(|c| replicas[c.replica_index].replica == p.replica)
            {
                let remaining =
                    (PREFIX_GRACE - p.started.elapsed().as_secs_f64()).min(PREFIX_GRACE - elapsed);
                if aligned_prompt > best_available
                    && target.cached_tokens < aligned_prompt
                    && balanced(target)
                    && remaining > 0.0
                {
                    return Selection::Wait("pending_prefix", remaining);
                }
            }
        }
        let best = all
            .iter()
            .filter(|c| balanced(c))
            .max_by_key(|c| c.cached_tokens);
        if let Some(target) = best {
            let saved = target.cached_tokens.saturating_sub(best_available) as f64 / prefill_rate;
            let remaining = saved.min(MAX_CACHE_WAIT) - elapsed;
            if target.cached_tokens as f64 > request.prompt_tokens as f64 * self.threshold
                && remaining > 0.0
            {
                return Selection::Wait("cached_capacity", remaining);
            }
        }
        let use_cache = available.iter().any(|c| {
            balanced(c) && c.cached_tokens as f64 > request.prompt_tokens as f64 * self.threshold
        });
        let selected = available
            .iter()
            .enumerate()
            .filter(|(_, c)| balanced(c))
            .min_by_key(|(_, c)| {
                (
                    if use_cache { 0 } else { c.active },
                    Reverse(if use_cache { c.cached_tokens } else { 0 }),
                    c.adapter_tier,
                    !(use_cache && c.preferred),
                    c.active,
                    Reverse(c.tie),
                )
            })
            .map(|(i, _)| i);
        Selection::Route(
            selected,
            if elapsed > 0.0 {
                "affinity_released"
            } else if use_cache {
                "cached"
            } else {
                "load"
            },
        )
    }
}
