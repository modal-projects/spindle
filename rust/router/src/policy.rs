//! Policies rank eligible replicas. Admission and reservations belong to Router.
use crate::Candidate;

pub trait Policy: Send + Sync {
    fn score(&self, candidate: &Candidate) -> f64;
}

pub struct LeastLoad;
impl Policy for LeastLoad {
    fn score(&self, c: &Candidate) -> f64 {
        c.active as f64
    }
}

/// Compare outstanding work and setup in seconds; KV and LoRA use exact versions.
pub struct Joint;
impl Policy for Joint {
    fn score(&self, c: &Candidate) -> f64 {
        c.backlog_seconds + c.prefill_seconds + c.adapter_seconds
    }
}
