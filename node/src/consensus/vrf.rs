use sha2::{Digest, Sha256};

/// Verifiable Random Function output used for deterministic, leaderless work
/// assignment. Every peer computes the same output given the same inputs, so no
/// coordinator is needed.
///
/// The VRF is seeded with `round_id || sorted_peer_ids` and produces a
/// deterministic permutation used to assign data shards and ring positions.
#[derive(Debug, Clone)]
pub struct VrfOutput {
    /// The raw 256-bit hash output.
    pub hash: [u8; 32],
}

impl VrfOutput {
    /// Compute a VRF output from a training round ID and the sorted list of
    /// participating peer IDs.
    pub fn compute(round_id: &str, sorted_peer_ids: &[String]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"openclaw-vrf-v1:");
        hasher.update(round_id.as_bytes());
        hasher.update(b":");
        for pid in sorted_peer_ids {
            hasher.update(pid.as_bytes());
            hasher.update(b",");
        }
        let hash: [u8; 32] = hasher.finalize().into();
        Self { hash }
    }

    /// Derive a deterministic permutation of indices [0..n) from the VRF
    /// output. Used to assign peers to data shards or ring positions.
    pub fn permutation(&self, n: usize) -> Vec<usize> {
        if n == 0 {
            return vec![];
        }

        // Fisher-Yates shuffle seeded by successive hashes.
        let mut indices: Vec<usize> = (0..n).collect();
        let mut seed = self.hash;

        for i in (1..n).rev() {
            // Derive next random bytes.
            let mut hasher = Sha256::new();
            hasher.update(seed);
            hasher.update(&(i as u64).to_le_bytes());
            seed = hasher.finalize().into();

            // Extract a usize from the first 8 bytes.
            let rand_val =
                u64::from_le_bytes(seed[..8].try_into().unwrap()) as usize;
            let j = rand_val % (i + 1);
            indices.swap(i, j);
        }

        indices
    }

    /// Assign `n_peers` to `n_shards` data partitions. Returns a mapping from
    /// shard index to the peer index responsible for it.
    pub fn assign_shards(&self, n_peers: usize, n_shards: usize) -> Vec<usize> {
        let perm = self.permutation(n_peers);
        (0..n_shards).map(|s| perm[s % n_peers]).collect()
    }

    /// Compute ring all-reduce topology: returns the ordered ring of peer
    /// indices.
    pub fn ring_order(&self, n_peers: usize) -> Vec<usize> {
        self.permutation(n_peers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_output() {
        let peers = vec!["peerA".into(), "peerB".into(), "peerC".into()];
        let a = VrfOutput::compute("round-1", &peers);
        let b = VrfOutput::compute("round-1", &peers);
        assert_eq!(a.hash, b.hash);
    }

    #[test]
    fn different_rounds_differ() {
        let peers = vec!["peerA".into(), "peerB".into()];
        let a = VrfOutput::compute("round-1", &peers);
        let b = VrfOutput::compute("round-2", &peers);
        assert_ne!(a.hash, b.hash);
    }

    #[test]
    fn permutation_is_valid() {
        let peers = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let vrf = VrfOutput::compute("test", &peers);
        let perm = vrf.permutation(4);
        assert_eq!(perm.len(), 4);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn shard_assignment_covers_all() {
        let peers = vec!["a".into(), "b".into(), "c".into()];
        let vrf = VrfOutput::compute("test", &peers);
        let assignments = vrf.assign_shards(3, 6);
        assert_eq!(assignments.len(), 6);
        // Each peer should appear at least once.
        for p in 0..3 {
            assert!(assignments.contains(&p));
        }
    }
}
