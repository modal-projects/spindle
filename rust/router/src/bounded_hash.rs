//! Rendezvous-hash adapter groups with conversation stickiness and bounded load.
//! Uses only discovered workers and request counts, never cache observations.
use std::cmp::Reverse;
use xxhash_rust::xxh3::xxh3_64;

pub const DEFAULT_LOAD_FACTOR: f64 = 1.25;

pub struct BoundedHash {
    /// Replicas per run. Zero sizes groups from the number of recent runs.
    pub group_size: usize,
    pub load_factor: f64,
}

fn score(key: &str, replica: &str) -> u64 {
    xxh3_64(format!("{key}\0{replica}").as_bytes())
}

impl BoundedHash {
    /// Give each run about its share of replicas, but at least two so a burst
    /// from one run has a second warm replica before it spills further.
    pub fn group(&self, replicas: usize, runs: usize) -> usize {
        if self.group_size > 0 {
            return self.group_size.min(replicas);
        }
        let share = (replicas as f64 / runs.max(1) as f64).round() as usize;
        share.clamp(replicas.min(2), replicas)
    }

    /// The run's group ordered for this conversation, then the other replicas
    /// in the run's order. Adding or removing a replica moves few runs.
    pub fn order(&self, run: &str, affinity: &str, replicas: &[&str], runs: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..replicas.len()).collect();
        order.sort_by_key(|&i| (Reverse(score(run, replicas[i])), replicas[i]));
        let k = self.group(replicas.len(), runs);
        order[..k].sort_by_key(|&i| (Reverse(score(affinity, replicas[i])), replicas[i]));
        order
    }

    /// The first usable replica within the load bound. Otherwise, when the
    /// replicas within the bound are failed or busy, the least loaded usable one.
    pub fn select(
        &self,
        order: &[usize],
        group: usize,
        loads: &[usize],
        usable: &[bool],
    ) -> Option<(usize, &'static str)> {
        let mean = loads.iter().sum::<usize>() as f64 / loads.len().max(1) as f64;
        let bound = (self.load_factor * mean).ceil() as usize + 1;
        let reason = |position: usize| match position {
            0 => "sticky",
            p if p < group => "group_spill",
            _ => "spill",
        };
        order
            .iter()
            .enumerate()
            .find(|(_, &i)| usable[i] && loads[i] <= bound)
            .map(|(position, &i)| (i, reason(position)))
            .or_else(|| {
                order
                    .iter()
                    .filter(|&&i| usable[i])
                    .min_by_key(|&&i| loads[i])
                    .map(|&i| (i, "spill"))
            })
    }
}
