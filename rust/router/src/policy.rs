//! Policies choose among eligible replicas. Admission stays in Router.
use crate::Candidate;

pub trait Policy: Send + Sync {
    fn select(
        &self,
        candidates: &[Candidate],
        max_active: usize,
        prompt_tokens: usize,
    ) -> Option<usize>;
}

pub struct LeastLoad {
    pub sticky: bool,
}
impl Policy for LeastLoad {
    fn select(&self, candidates: &[Candidate], _: usize, _: usize) -> Option<usize> {
        candidates
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| {
                (
                    !(self.sticky && c.preferred),
                    c.active,
                    std::cmp::Reverse(c.tie),
                )
            })
            .map(|(i, _)| i)
    }
}

/// Retained unchanged as a comparison with the earlier weighted policy.
pub struct Joint;
impl Policy for Joint {
    fn select(&self, candidates: &[Candidate], _: usize, _: usize) -> Option<usize> {
        candidates
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (!a.preferred)
                    .cmp(&!b.preferred)
                    .then_with(|| {
                        (a.backlog_seconds + a.prefill_seconds + a.adapter_seconds)
                            .total_cmp(&(b.backlog_seconds + b.prefill_seconds + b.adapter_seconds))
                    })
                    .then_with(|| b.tie.cmp(&a.tie))
            })
            .map(|(i, _)| i)
    }
}

pub struct LoraRadix {
    pub absolute: usize,
    pub relative: f64,
}
impl Policy for LoraRadix {
    fn select(&self, candidates: &[Candidate], _: usize, _: usize) -> Option<usize> {
        let minimum = candidates.iter().map(|c| c.active).min()?;
        candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                !(c.active - minimum > self.absolute
                    && c.active as f64 > minimum as f64 * self.relative)
            })
            .min_by_key(|(_, c)| {
                (
                    c.adapter_tier,
                    std::cmp::Reverse(c.cached_tokens),
                    !c.preferred,
                    c.active,
                    std::cmp::Reverse(c.tie),
                )
            })
            .map(|(i, _)| i)
    }
}

/// Switch the whole pool to least-load when imbalanced. LoRA warmth only breaks ties.
pub struct CacheAware {
    pub absolute: usize,
    pub relative: f64,
    pub cache_threshold: f64,
}
impl Policy for CacheAware {
    fn select(
        &self,
        candidates: &[Candidate],
        max_active: usize,
        prompt_tokens: usize,
    ) -> Option<usize> {
        let minimum = candidates.iter().map(|c| c.active).min()?;
        // max_active includes full replicas: admission must not hide imbalance.
        let imbalanced = max_active - minimum > self.absolute
            && max_active as f64 > minimum as f64 * self.relative;
        let best_match = candidates.iter().map(|c| c.cached_tokens).max()?;
        let use_cache =
            !imbalanced && best_match as f64 > prompt_tokens as f64 * self.cache_threshold;
        candidates
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| {
                (
                    if use_cache { 0 } else { c.active },
                    std::cmp::Reverse(if use_cache || imbalanced {
                        c.cached_tokens
                    } else {
                        0
                    }),
                    c.adapter_tier,
                    c.active,
                    std::cmp::Reverse(c.tie),
                )
            })
            .map(|(i, _)| i)
    }
}
