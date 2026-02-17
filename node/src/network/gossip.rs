use anyhow::Result;
use libp2p::gossipsub::{self, IdentTopic, TopicHash};
use tracing::info;

// Well-known gossipsub topic names for the OpenClaw protocol.
pub const TOPIC_HEARTBEAT: &str = "openclaw/heartbeat";
pub const TOPIC_SHARD_MAP: &str = "openclaw/shard-map";
pub const TOPIC_TRAINING: &str = "openclaw/training";
pub const TOPIC_GRADIENT: &str = "openclaw/gradient";
pub const TOPIC_CHECKPOINT: &str = "openclaw/checkpoint";

/// All topics the node subscribes to.
pub fn all_topics() -> Vec<IdentTopic> {
    vec![
        IdentTopic::new(TOPIC_HEARTBEAT),
        IdentTopic::new(TOPIC_SHARD_MAP),
        IdentTopic::new(TOPIC_TRAINING),
        IdentTopic::new(TOPIC_GRADIENT),
        IdentTopic::new(TOPIC_CHECKPOINT),
    ]
}

/// Subscribe to all OpenClaw gossipsub topics. Returns the topic hashes.
pub fn subscribe_all(gossipsub: &mut gossipsub::Behaviour) -> Result<Vec<TopicHash>> {
    let mut hashes = Vec::new();
    for topic in all_topics() {
        gossipsub.subscribe(&topic)?;
        info!("Subscribed to gossipsub topic: {}", topic);
        hashes.push(topic.hash());
    }
    Ok(hashes)
}

/// Determine which protocol domain a topic belongs to.
pub fn topic_domain(topic: &str) -> &'static str {
    match topic {
        TOPIC_HEARTBEAT => "heartbeat",
        TOPIC_SHARD_MAP => "shard_map",
        TOPIC_TRAINING => "training",
        TOPIC_GRADIENT => "gradient",
        TOPIC_CHECKPOINT => "checkpoint",
        _ => "unknown",
    }
}
