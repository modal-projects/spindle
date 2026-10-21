//! Group-scoped affinity with bounded waiting; hints never count as cache hits.
use crate::{Candidate, PendingPrefix, Replica, Request};
use std::cmp::Reverse;

// Allow one observation period plus polling/transport grace before spillover.
const PREFIX_GRACE: f64 = 2.0;
const MAX_CACHE_WAIT: f64 = 2.0;
const SUBSTANTIAL_PREFILL_SECONDS: f64 = 0.5;
// Keep space for history between tool turns. Existing valuable histories get
// more headroom than new placement, and a small memory difference never spills.
const PLACEMENT_KV_LIMIT: f64 = 0.80;
const CACHED_KV_LIMIT: f64 = 0.90;
const KV_IMBALANCE: f64 = 0.15;

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
        // Do not use the full overload allowance to concentrate new work on
        // short/shared prefixes. Once a substantial history has accumulated,
        // preserve it with the wider band instead of undoing that placement.
        let best_observed = all.iter().map(|c| c.cached_tokens).max().unwrap_or(0);
        let substantial_cache = best_observed as f64
            > request.prompt_tokens as f64 * self.threshold
            && best_observed as f64 / prefill_rate >= SUBSTANTIAL_PREFILL_SECONDS;
        let (absolute, relative) = if substantial_cache {
            (self.absolute, self.relative)
        } else {
            (
                (self.absolute / 2).max(1),
                1.0 + (self.relative - 1.0) / 2.0,
            )
        };
        let request_balanced = |c: &Candidate| {
            !(c.active.saturating_sub(minimum) > absolute
                && c.active as f64 > minimum as f64 * relative)
        };
        // Compare only destinations that can actually accept this request and
        // satisfy the request-count guard. Otherwise the two guards could
        // exclude each other's only usable destination.
        let least_pinned = available
            .iter()
            .filter(|c| request_balanced(c))
            .filter_map(|c| c.kv_pinned_fraction)
            .reduce(f64::min);
        let kv_limit = if substantial_cache {
            CACHED_KV_LIMIT
        } else {
            PLACEMENT_KV_LIMIT
        };
        let balanced = |c: &Candidate| {
            request_balanced(c)
                && match (c.kv_pinned_fraction, least_pinned) {
                    (Some(pinned), Some(least)) => {
                        !(pinned > kv_limit && pinned - least > KV_IMBALANCE)
                    }
                    _ => true,
                }
        };
        // An excluded worker's cache is not an alternative to waiting for a
        // usable target. Estimate the benefit against actual route choices.
        let best_available = available
            .iter()
            .filter(|c| balanced(c))
            .map(|c| c.cached_tokens)
            .max()
            .unwrap();
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
                && saved > 0.0
            {
                // Moving to another engine's queue does not provide immediate
                // service. Keep the better cache while alternatives have queued
                // work; the request deadline still bounds admission. A drained
                // alternative queue restores the ordinary short spillover budget.
                if available.iter().all(|c| c.queued > 0) {
                    return Selection::Wait("cached_saturated", 0.1);
                }
                if remaining > 0.0 {
                    return Selection::Wait("cached_capacity", remaining);
                }
            }
        }
        let use_cache = available.iter().any(|c| {
            balanced(c) && c.cached_tokens as f64 > request.prompt_tokens as f64 * self.threshold
        });
        let rank = |c: &Candidate| {
            (
                Reverse(if use_cache { c.cached_tokens } else { 0 }),
                // Cache ties can retain adapter locality too: grouping must not
                // disappear once a shared prompt becomes cached on several workers.
                Reverse(c.adapter_active),
                c.active_adapter_count,
                if use_cache { 0 } else { c.active },
                c.adapter_tier,
                !(use_cache && c.preferred),
                c.active,
                Reverse(if use_cache { c.tie } else { c.adapter_tie }),
            )
        };
        let selected = available
            .iter()
            .enumerate()
            .filter(|(_, c)| balanced(c))
            .min_by_key(|(_, c)| rank(c))
            .map(|(i, _)| i);
        if let Some(index) = selected {
            if let Some(target) = all.iter().filter(|c| balanced(c)).min_by_key(|c| rank(c)) {
                // A full/busy target can still be the better adapter group.
                // Only preserve it while every alternative already queues work;
                // a drained queue or the existing imbalance guard permits spillover.
                let alternative = &available[index];
                if target.adapter_active > alternative.adapter_active
                    && (!use_cache || target.cached_tokens >= alternative.cached_tokens)
                    && available.iter().all(|c| c.queued > 0)
                {
                    return Selection::Wait("adapter_saturated", 0.1);
                }
            }
        }
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
