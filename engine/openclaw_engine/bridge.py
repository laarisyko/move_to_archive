"""Bridge between the Rust P2P node and the Python ML engine.

In a full implementation, this would use PyO3 (Rust -> Python FFI) so the
Rust node can invoke the Python engine directly. For the initial version,
we use a gRPC / HTTP bridge where the Rust node calls the Python engine
over localhost.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Dict, Optional

from .model.shard import ModelShard
from .inference.server import InferenceServer, InferenceRequest
from .training.trainer import LocalTrainer, TrainingConfig

logger = logging.getLogger(__name__)


class NodeBridge:
    """Bridges the Rust P2P node (network layer) with the Python ML engine.

    The bridge exposes methods that the node can call to:
    - Run inference on a local shard
    - Execute training steps
    - Get/set gradients for all-reduce
    - Compute Merkle roots for weight verification
    """

    def __init__(
        self,
        inference_server: InferenceServer,
        trainer: Optional[LocalTrainer] = None,
    ):
        self.inference_server = inference_server
        self.trainer = trainer
        self._callbacks: Dict[str, Callable] = {}

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for node events (e.g. 'peer_joined', 'round_started')."""
        self._callbacks[event] = callback

    async def handle_node_message(self, msg_type: str, payload: bytes) -> bytes:
        """Handle a message from the Rust node.

        This is the main entry point called by the Rust node (via FFI or HTTP).
        """
        if msg_type == "infer":
            return await self._handle_infer(payload)
        elif msg_type == "train_step":
            return await self._handle_train_step(payload)
        elif msg_type == "get_gradients":
            return self._handle_get_gradients()
        elif msg_type == "set_gradients":
            return self._handle_set_gradients(payload)
        elif msg_type == "merkle_root":
            return self._handle_merkle_root(payload)
        elif msg_type == "health":
            return self._handle_health()
        else:
            return json.dumps({"error": f"unknown message type: {msg_type}"}).encode()

    async def _handle_infer(self, payload: bytes) -> bytes:
        data = json.loads(payload)
        request = InferenceRequest(
            model_id=data.get("model_id", ""),
            prompt=data.get("prompt", ""),
            max_tokens=data.get("max_tokens", 256),
            temperature=data.get("temperature", 0.7),
        )
        response = await self.inference_server.infer_async(request)
        return json.dumps({
            "request_id": response.request_id,
            "text": response.text,
            "latency_ms": response.latency_ms,
        }).encode()

    async def _handle_train_step(self, payload: bytes) -> bytes:
        if self.trainer is None:
            return json.dumps({"error": "no trainer configured"}).encode()

        data = json.loads(payload)
        import torch

        # Deserialize input activations.
        # In production this would be a proper tensor format.
        input_shape = data.get("input_shape", [1, 128, 512])
        input_tensor = torch.randn(*input_shape)

        metrics = self.trainer.train_step(input_tensor)
        return json.dumps(metrics).encode()

    def _handle_get_gradients(self) -> bytes:
        if self.trainer is None:
            return json.dumps({"error": "no trainer configured"}).encode()

        grads = self.trainer.get_gradients()
        # Serialize gradient shapes and sizes (not the full data over JSON).
        grad_info = {
            name: {"shape": list(t.shape), "numel": t.numel()}
            for name, t in grads.items()
        }
        return json.dumps(grad_info).encode()

    def _handle_set_gradients(self, payload: bytes) -> bytes:
        return json.dumps({"status": "ok"}).encode()

    def _handle_merkle_root(self, payload: bytes) -> bytes:
        data = json.loads(payload)
        model_id = data.get("model_id", "")

        shard = None
        if model_id in self.inference_server._models:
            shard = self.inference_server._models[model_id]

        if shard:
            root = shard.merkle_root()
            return json.dumps({"merkle_root": root.hex()}).encode()
        else:
            return json.dumps({"error": "model not found"}).encode()

    def _handle_health(self) -> bytes:
        stats = self.inference_server.stats()
        return json.dumps({"status": "ok", **stats}).encode()


class HttpBridgeServer:
    """Simple HTTP server that the Rust node talks to over localhost.

    This is the fallback bridge when PyO3 is not available.
    """

    def __init__(self, bridge: NodeBridge, port: int = 50052):
        self.bridge = bridge
        self.port = port

    async def start(self):
        """Start the HTTP bridge server."""
        server = await asyncio.start_server(
            self._handle_connection, "127.0.0.1", self.port
        )
        logger.info("Bridge server listening on 127.0.0.1:%d", self.port)
        async with server:
            await server.serve_forever()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        try:
            data = await reader.read(65536)
            request = data.decode()

            # Parse minimal HTTP.
            lines = request.split("\r\n")
            first_line = lines[0] if lines else ""
            parts = first_line.split()
            path = parts[1] if len(parts) >= 2 else "/"

            body_start = request.find("\r\n\r\n")
            body = request[body_start + 4 :].encode() if body_start >= 0 else b""

            # Route to bridge.
            msg_type = path.strip("/")
            response_body = await self.bridge.handle_node_message(msg_type, body)

            response = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(response_body)}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode() + response_body

            writer.write(response)
            await writer.drain()
        finally:
            writer.close()


class DirectBridgeHandler:
    """In-process bridge handler for direct Python-to-Python communication.

    Bypasses HTTP/serialization overhead for local simulation and testing.
    Each simulated peer gets its own handler wrapping its own shard + trainer.
    """

    def __init__(
        self,
        shard: ModelShard,
        trainer: Optional[LocalTrainer] = None,
        training_config: Optional[TrainingConfig] = None,
    ):
        import torch
        import torch.nn as nn

        self.shard = shard
        self.trainer = trainer or LocalTrainer(
            shard, training_config or TrainingConfig()
        )
        self._last_gradients: Dict[str, "torch.Tensor"] = {}

    def train_step(self, input_shape: list = None) -> Dict:
        """Execute a training step and cache gradients."""
        import torch
        import torch.nn as nn

        input_shape = input_shape or [1, 16]
        x = torch.randn(*input_shape, requires_grad=True)

        target = None
        loss_fn = None
        if self.shard.config.is_last:
            target = torch.randn(input_shape[0], input_shape[-1])
            loss_fn = nn.MSELoss()

        metrics = self.trainer.train_step(x, target, loss_fn)
        self._last_gradients = self.trainer.get_gradients()
        return metrics

    def get_gradients(self) -> Dict[str, "torch.Tensor"]:
        """Return cached gradient tensors from last training step."""
        return self._last_gradients

    def set_gradients(self, gradients: Dict[str, "torch.Tensor"]):
        """Apply aggregated gradients and update model weights."""
        self.trainer.set_gradients(gradients)
        self.trainer.apply_gradients()

    def merkle_root(self) -> str:
        """Compute the Merkle root of current weights."""
        return self.shard.merkle_root().hex()
