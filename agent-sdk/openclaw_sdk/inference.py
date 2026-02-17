"""Inference client API -- send prompts to the decentralized network."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

from .network import NetworkClient

logger = logging.getLogger(__name__)


class InferenceClient:
    """Client for running inference on models in the OpenClaw network.

    Sends requests to the local node, which routes them through the
    pipeline if the model is sharded across peers.
    """

    def __init__(self, network: NetworkClient):
        self.network = network

    def infer(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Run synchronous inference.

        Args:
            model_id: Model to use (e.g. "llama-7b").
            prompt: Input text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated text.
        """
        request_id = str(uuid.uuid4())
        result = self.network.submit_inference(
            model_id=model_id,
            prompt=prompt,
            request_id=request_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if "error" in result:
            logger.error("Inference failed: %s", result["error"])
            return f"[error: {result['error']}]"

        return result.get("text", "[no text in response]")

    async def infer_async(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Async inference (runs sync call in executor to avoid blocking)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.infer(model_id, prompt, max_tokens, temperature),
        )

    def list_models(self) -> list:
        """List available models on the network."""
        result = self.network.shards()
        if isinstance(result, dict) and "entries" in result:
            model_ids = set()
            for entry in result["entries"].values():
                if "model_id" in entry:
                    model_ids.add(entry["model_id"])
            return sorted(model_ids)
        return []
