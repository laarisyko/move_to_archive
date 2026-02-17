pub mod work_queue;

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::consensus::shard_map::{SharedReputationTable, SharedShardMap, ShardMap};
use crate::network::gossip;
use crate::network::{SwarmCommand, SwarmEvent2};

/// Scheduler state tracking ongoing training rounds and peer health.
struct SchedulerState {
    /// Known live peers and their last heartbeat time.
    live_peers: HashMap<String, u64>,
    /// Active training rounds this node is participating in.
    active_rounds: HashMap<String, RoundState>,
}

#[derive(Debug, Clone)]
struct RoundState {
    round_id: String,
    model_id: String,
    participants: Vec<String>,
    phase: RoundPhase,
}

#[derive(Debug, Clone, PartialEq)]
enum RoundPhase {
    Proposed,
    Joining,
    Computing,
    Aggregating,
    Checkpointing,
    Complete,
}

/// Main scheduler loop. Reacts to network events and dispatches work.
pub async fn run(
    mut event_rx: mpsc::Receiver<SwarmEvent2>,
    command_tx: mpsc::Sender<SwarmCommand>,
    shard_map: SharedShardMap,
    reputation: SharedReputationTable,
) {
    let mut state = SchedulerState {
        live_peers: HashMap::new(),
        active_rounds: HashMap::new(),
    };

    info!("Scheduler started");

    while let Some(event) = event_rx.recv().await {
        match event {
            SwarmEvent2::PeerDiscovered(peer_id) => {
                let pid = peer_id.to_string();
                let now = now_ms();
                state.live_peers.insert(pid.clone(), now);

                if let Ok(mut rep) = reputation.write() {
                    rep.record_heartbeat(&pid, now);
                }
                info!("Peer discovered: {} (total: {})", pid, state.live_peers.len());
            }

            SwarmEvent2::PeerExpired(peer_id) => {
                let pid = peer_id.to_string();
                state.live_peers.remove(&pid);

                // Remove peer's shards from the map.
                if let Ok(mut map) = shard_map.write() {
                    map.remove_peer(&pid);
                }
                info!("Peer expired: {} (total: {})", pid, state.live_peers.len());
            }

            SwarmEvent2::GossipMessage { source, topic, data } => {
                let domain = gossip::topic_domain(&topic);
                debug!(
                    "Gossip from {} on {}: {} bytes",
                    source,
                    domain,
                    data.len()
                );

                match domain {
                    "heartbeat" => {
                        handle_heartbeat(&mut state, &reputation, &source.to_string(), &data);
                    }
                    "shard_map" => {
                        handle_shard_map_update(&shard_map, &data);
                    }
                    "training" => {
                        handle_training_message(&mut state, &command_tx, &data).await;
                    }
                    "gradient" => {
                        debug!("Gradient message from {}", source);
                    }
                    "checkpoint" => {
                        debug!("Checkpoint message from {}", source);
                    }
                    _ => {
                        warn!("Unknown gossip topic: {}", topic);
                    }
                }
            }
        }
    }

    info!("Scheduler stopped");
}

fn handle_heartbeat(
    state: &mut SchedulerState,
    reputation: &SharedReputationTable,
    peer_id: &str,
    data: &[u8],
) {
    if let Ok(hb) = serde_json::from_slice::<serde_json::Value>(data) {
        let ts = hb
            .get("timestamp_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(now_ms);
        state.live_peers.insert(peer_id.to_string(), ts);

        if let Ok(mut rep) = reputation.write() {
            rep.record_heartbeat(peer_id, ts);
        }
    }
}

fn handle_shard_map_update(shard_map: &SharedShardMap, data: &[u8]) {
    if let Some(remote) = ShardMap::from_bytes(data) {
        if let Ok(mut local) = shard_map.write() {
            local.merge(&remote);
            debug!(
                "Shard map merged, now {} entries",
                local.entries().len()
            );
        }
    }
}

async fn handle_training_message(
    state: &mut SchedulerState,
    _command_tx: &mpsc::Sender<SwarmCommand>,
    data: &[u8],
) {
    // Parse training protocol messages.
    if let Ok(msg) = serde_json::from_slice::<serde_json::Value>(data) {
        let msg_type = msg
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        match msg_type {
            "proposal" => {
                let round_id = msg
                    .get("round_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let model_id = msg
                    .get("model_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                info!(
                    "Received training proposal: round={}, model={}",
                    round_id, model_id
                );
                state.active_rounds.insert(
                    round_id.clone(),
                    RoundState {
                        round_id,
                        model_id,
                        participants: vec![],
                        phase: RoundPhase::Proposed,
                    },
                );
            }
            "join" => {
                let round_id = msg
                    .get("round_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let peer_id = msg
                    .get("peer_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if let Some(round) = state.active_rounds.get_mut(round_id) {
                    round.participants.push(peer_id.clone());
                    info!(
                        "Peer {} joined round {} ({} participants)",
                        peer_id,
                        round_id,
                        round.participants.len()
                    );
                }
            }
            _ => {
                debug!("Unknown training message type: {}", msg_type);
            }
        }
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
