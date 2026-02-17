"""Network client -- communicates with the local openclaw-node via its API."""

from __future__ import annotations

import json
import logging
import socket
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NetworkClient:
    """HTTP client for the local openclaw-node API.

    Talks to the Rust node's gRPC/HTTP endpoint to publish messages,
    query peers, and submit inference/training requests.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:50051"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        """Check if the local node is healthy."""
        return self._get("/health")

    def peers(self) -> List[Dict[str, Any]]:
        """List known peers."""
        result = self._get("/peers")
        if isinstance(result, list):
            return result
        return []

    def shards(self) -> Dict[str, Any]:
        """Get the current shard map."""
        return self._get("/shards")

    def publish(self, topic: str, data: Any):
        """Publish a message to a gossipsub topic."""
        payload = {"topic": topic, "data": json.dumps(data)}
        return self._post("/publish", payload)

    def dial(self, multiaddr: str):
        """Instruct the node to dial a peer."""
        return self._post("/dial", {"address": multiaddr})

    def submit_inference(
        self,
        model_id: str,
        prompt: str,
        request_id: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Submit an inference request to the local node."""
        payload = {
            "request_id": request_id,
            "model_id": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return self._post("/infer", payload)

    def _get(self, path: str) -> Any:
        """Perform an HTTP GET request."""
        try:
            return self._http_request("GET", path)
        except Exception as e:
            logger.debug("GET %s failed: %s", path, e)
            return {"error": str(e)}

    def _post(self, path: str, data: Any) -> Any:
        """Perform an HTTP POST request with JSON body."""
        try:
            return self._http_request("POST", path, json.dumps(data))
        except Exception as e:
            logger.debug("POST %s failed: %s", path, e)
            return {"error": str(e)}

    def _http_request(self, method: str, path: str, body: str = "") -> Any:
        """Minimal HTTP client using raw sockets (no external deps)."""
        from urllib.parse import urlparse

        parsed = urlparse(self.base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80

        headers = f"{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n"
        if body:
            headers += f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        headers += "\r\n"

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.sendall((headers + body).encode())

            response = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk

        # Parse response body (skip HTTP headers).
        response_str = response.decode(errors="replace")
        body_start = response_str.find("\r\n\r\n")
        if body_start >= 0:
            body_str = response_str[body_start + 4 :]
            try:
                return json.loads(body_str)
            except json.JSONDecodeError:
                return {"raw": body_str}

        return {"raw": response_str}
