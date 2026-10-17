//! Version-scoped radix trees over cumulative 64-token prefix fingerprints.
use std::collections::{HashMap, HashSet};

pub type Row = (String, String, Option<String>, bool);
#[derive(Default)]
struct Node {
    path: Vec<u128>,
    owners: Vec<Vec<usize>>,
    children: HashMap<u128, Node>,
}
impl Node {
    fn insert(&mut self, path: &[u128], reusable: &[bool], replica: usize) {
        if path.is_empty() {
            return;
        }
        let child = self.children.entry(path[0]).or_default();
        let shared = child
            .path
            .iter()
            .zip(path)
            .take_while(|(a, b)| a == b)
            .count();
        if shared < child.path.len() {
            let tail = Node {
                path: child.path.split_off(shared),
                owners: child.owners.split_off(shared),
                children: std::mem::take(&mut child.children),
            };
            child.children.insert(tail.path[0], tail);
        }
        if child.path.is_empty() {
            child.path = path.to_vec();
            child.owners = reusable
                .iter()
                .map(|yes| if *yes { vec![replica] } else { vec![] })
                .collect();
            return;
        }
        for (owners, yes) in child.owners.iter_mut().zip(reusable) {
            if *yes && !owners.contains(&replica) {
                owners.push(replica);
            }
        }
        child.insert(&path[shared..], &reusable[shared..], replica);
    }
}
#[derive(Default)]
pub struct Index {
    roots: HashMap<String, Node>,
}
impl Index {
    pub fn add(&mut self, replica: usize, rows: &[Row]) -> Result<(), String> {
        if rows.len() > 8192 {
            return Err("prefix catalog exceeds 8192 nodes".into());
        }
        let mut nodes = HashMap::new();
        let mut parents = HashSet::new();
        for (adapter, key, parent, reusable) in rows {
            let hash = u128::from_str_radix(key, 16).map_err(|_| "invalid prefix hash")?;
            if nodes
                .insert(key, (adapter, parent, *reusable, hash))
                .is_some()
            {
                return Err("duplicate prefix node".into());
            }
            if let Some(p) = parent {
                parents.insert(p);
            }
        }
        let mut reached = HashSet::new();
        for (key, (adapter, _, _, _)) in &nodes {
            if parents.contains(*key) {
                continue;
            }
            let mut path = Vec::new();
            let mut reusable = Vec::new();
            let mut seen = HashSet::new();
            let mut cursor = Some(*key);
            while let Some(k) = cursor {
                if !seen.insert(k) {
                    return Err("cyclic prefix ancestry".into());
                }
                let (owner, parent, yes, hash) = nodes.get(k).ok_or("missing prefix parent")?;
                if owner != adapter {
                    return Err("prefix parent crosses adapter versions".into());
                }
                reached.insert(k);
                path.push(*hash);
                reusable.push(*yes);
                cursor = parent.as_ref();
            }
            path.reverse();
            reusable.reverse();
            self.roots
                .entry((*adapter).clone())
                .or_default()
                .insert(&path, &reusable, replica);
        }
        if reached.len() != rows.len() {
            return Err("unreachable prefix nodes".into());
        }
        Ok(())
    }
    pub fn matches(&self, adapter: &str, hashes: &[u128], replicas: usize) -> Vec<usize> {
        let mut result = vec![0; replicas];
        let Some(mut node) = self.roots.get(adapter) else {
            return result;
        };
        let mut depth = 0;
        while depth < hashes.len() {
            let Some(child) = node.children.get(&hashes[depth]) else {
                break;
            };
            for (hash, owners) in child.path.iter().zip(&child.owners) {
                if hashes.get(depth) != Some(hash) {
                    return result;
                }
                depth += 1;
                for replica in owners {
                    result[*replica] = depth * 64;
                }
            }
            node = child;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rows(adapter: &str, path: &[u128], valid: &[bool]) -> Vec<Row> {
        path.iter()
            .enumerate()
            .map(|(i, h)| {
                (
                    adapter.into(),
                    format!("{h:x}"),
                    i.checked_sub(1).map(|j| format!("{:x}", path[j])),
                    valid[i],
                )
            })
            .collect()
    }
    #[test]
    fn splits_edges_and_merges_replica_owners() {
        let mut tree = Index::default();
        tree.add(0, &rows("v1", &[1, 2, 3], &[true, true, true]))
            .unwrap();
        tree.add(1, &rows("v1", &[1, 2, 4], &[true, true, true]))
            .unwrap();
        tree.add(2, &rows("v1", &[1, 2], &[true, true])).unwrap();
        assert_eq!(tree.matches("v1", &[1, 2, 3], 3), vec![192, 128, 128]);
        assert_eq!(tree.matches("v1", &[1, 2, 4], 3), vec![128, 192, 128]);
        assert_eq!(tree.matches("v2", &[1, 2, 3], 3), vec![0, 0, 0]);
    }
    #[test]
    fn hybrid_structural_nodes_are_not_cache_hits() {
        let mut tree = Index::default();
        tree.add(0, &rows("v", &[1, 2, 3], &[false, false, true]))
            .unwrap();
        assert_eq!(tree.matches("v", &[1, 2], 1), vec![0]);
        assert_eq!(tree.matches("v", &[1, 2, 3, 4], 1), vec![192]);
    }
    #[test]
    fn rejects_broken_ancestry() {
        let mut tree = Index::default();
        assert!(tree
            .add(0, &[("v".into(), "1".into(), Some("2".into()), true)])
            .is_err());
        assert!(tree
            .add(0, &[("v".into(), "1".into(), Some("1".into()), true)])
            .is_err());
        let mut r = rows("v", &[1, 2], &[true, true]);
        r[1].0 = "other".into();
        assert!(tree.add(0, &r).is_err());
    }
}
