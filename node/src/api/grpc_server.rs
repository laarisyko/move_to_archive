use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;
use tracing::info;

use crate::consensus::shard_map::{SharedShardMap, ShardMap};
use crate::network::SwarmCommand;

/// Simple JSON-over-HTTP API server for the OpenClaw node.
///
/// In a full implementation this would use tonic gRPC with the protobuf
/// service definitions. For the initial implementation we use a lightweight
/// HTTP/JSON API with tokio.
pub async fn run(
    port: u16,
    command_tx: mpsc::Sender<SwarmCommand>,
    shard_map: SharedShardMap,
) -> Result<()> {
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("API server listening on {}", addr);

    loop {
        let (stream, peer_addr) = listener.accept().await?;
        let cmd_tx = command_tx.clone();
        let smap = shard_map.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, cmd_tx, smap).await {
                tracing::debug!("API connection from {} error: {}", peer_addr, e);
            }
        });
    }
}

async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    command_tx: mpsc::Sender<SwarmCommand>,
    shard_map: SharedShardMap,
) -> Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf).await?;
    let request = String::from_utf8_lossy(&buf[..n]);

    // Parse the first line to get method + path.
    let first_line = request.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let (method, path) = if parts.len() >= 2 {
        (parts[0], parts[1])
    } else {
        ("GET", "/")
    };

    let (status, body) = match (method, path) {
        ("GET", "/health") => ("200 OK", serde_json::json!({"status": "ok"}).to_string()),

        ("GET", "/peers") => {
            // Report shard map entries as a proxy for known peers.
            let map = shard_map.read().unwrap();
            let entries: Vec<_> = map.entries().values().collect();
            ("200 OK", serde_json::to_string(&entries).unwrap_or_default())
        }

        ("GET", "/shards") => {
            let map = shard_map.read().unwrap();
            ("200 OK", serde_json::to_string(&map).unwrap_or_default())
        }

        ("POST", "/publish") => {
            // Expect JSON body: {"topic": "...", "data": "base64..."}
            let body_start = request.find("\r\n\r\n").map(|i| i + 4).unwrap_or(n);
            let body_str = &request[body_start..];
            match serde_json::from_str::<PublishRequest>(body_str) {
                Ok(req) => {
                    let data = base64_decode(&req.data);
                    let _ = command_tx
                        .send(SwarmCommand::Publish {
                            topic: req.topic,
                            data,
                        })
                        .await;
                    ("200 OK", serde_json::json!({"published": true}).to_string())
                }
                Err(e) => (
                    "400 Bad Request",
                    serde_json::json!({"error": e.to_string()}).to_string(),
                ),
            }
        }

        ("POST", "/infer") => {
            // Placeholder for inference endpoint.
            let body_start = request.find("\r\n\r\n").map(|i| i + 4).unwrap_or(n);
            let body_str = &request[body_start..];
            match serde_json::from_str::<InferRequest>(body_str) {
                Ok(req) => {
                    let response = serde_json::json!({
                        "request_id": req.request_id,
                        "model_id": req.model_id,
                        "text": "[inference not yet connected to engine]",
                        "status": "pending_engine_integration"
                    });
                    ("200 OK", response.to_string())
                }
                Err(e) => (
                    "400 Bad Request",
                    serde_json::json!({"error": e.to_string()}).to_string(),
                ),
            }
        }

        _ => ("404 Not Found", serde_json::json!({"error": "not found"}).to_string()),
    };

    let response = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).await?;
    Ok(())
}

#[derive(Deserialize)]
struct PublishRequest {
    topic: String,
    data: String,
}

#[derive(Deserialize)]
struct InferRequest {
    request_id: String,
    model_id: String,
    prompt: String,
}

fn base64_decode(input: &str) -> Vec<u8> {
    // Simple base64 decode; in production use the `base64` crate.
    input.as_bytes().to_vec()
}
