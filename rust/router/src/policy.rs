//! Policies choose among eligible replicas. Admission stays in Router.
use crate::Candidate;

pub trait Policy: Send + Sync {
    fn select(&self, candidates: &[Candidate]) -> Option<usize>;
}

pub struct LeastLoad {
    pub sticky: bool,
}
impl Policy for LeastLoad {
    fn select(&self, candidates: &[Candidate]) -> Option<usize> {
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
    fn select(&self, candidates: &[Candidate]) -> Option<usize> {
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
    fn select(&self, candidates: &[Candidate]) -> Option<usize> {
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
