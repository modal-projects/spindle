//! Bounded waiting for a cached replica.
use crate::{Candidate, Request};
use std::cmp::Reverse;

const MAX_CACHE_WAIT: f64 = 2.0;
const SUBSTANTIAL_PREFILL_SECONDS: f64 = 0.5;
// Valuable histories keep more KV headroom than a new placement.
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
    pub fn select(
        &self,
        all: &[Candidate],
        available: &[Candidate],
        request: &Request,
        prefill_rate: f64,
        elapsed: f64,
    ) -> Selection {
        let Some(minimum) = available.iter().map(|c| c.active).min() else {
            // Capacity waiting does not start the affinity budget.
            return Selection::Route(None, "capacity");
        };
        // Widen the balance band only after a substantial cache hit.
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
        // Judge both guards against the same destinations.
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
        // Score cache only among workers that can accept the request.
        let best_available = available
            .iter()
            .filter(|c| balanced(c))
            .map(|c| c.cached_tokens)
            .max()
            .unwrap();
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
                // Keep the better cache while every alternative is queued, within the affinity budget.
                if available.iter().all(|c| c.queued > 0) && elapsed < MAX_CACHE_WAIT {
                    return Selection::Wait("cached_saturated", 0.1);
                }
                if remaining > 0.0 {
                    return Selection::Wait("cached_capacity", remaining);
                }
            }
        }
        // A small fraction of a long prompt can still save seconds of prefill.
        let use_cache = available.iter().any(|c| {
            balanced(c)
                && (c.cached_tokens as f64 > request.prompt_tokens as f64 * self.threshold
                    || c.cached_tokens as f64 / prefill_rate >= SUBSTANTIAL_PREFILL_SECONDS)
        });
        let rank = |c: &Candidate| {
            (
                Reverse(if use_cache { c.cached_tokens } else { 0 }),
                // Break cache ties by adapter locality.
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
