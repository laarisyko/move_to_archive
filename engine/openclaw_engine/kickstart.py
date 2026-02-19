"""Kickstart: bootstrap an LLM from scratch on a decentralized network.

This is the top-level entry point for "start from nothing." A peer that joins
the network for the first time runs this to:

    1. Create a fresh model (random weights, architecture from genome)
    2. Load local training data (any text files the peer has)
    3. Tokenize and prepare batches
    4. Run local training steps
    5. Collect gradients for decentralized aggregation

After aggregation, all peers converge to the same weights -- even though each
peer trained on different data and started with different random seeds. This
is the magic of averaging: E[gradient_i] converges to the true gradient.

Bootstrapping protocol:
    Round 0: All peers init from the same architecture genome (but random weights).
             After gradient averaging, weights converge to a shared starting point.
    Round 1+: Normal training. Each round, peers train on their local data,
              aggregate gradients, checkpoint, and repeat.

The model is intentionally small at first (e.g. 6 layers, 256 hidden dim, ~5M params).
As the network grows and the model improves, the architecture governance system
can propose mutations (add layers, increase width) and peers vote on them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .data.tokenizer import Tokenizer, TokenizerConfig
from .data.pipeline import TextDataPipeline, DataConfig
from .model.lm import LanguageModel, LMConfig, create_from_scratch

logger = logging.getLogger(__name__)


@dataclass
class KickstartConfig:
    """Configuration for bootstrapping a new LLM from scratch."""

    # Model.
    model_id: str = "openclaw-v0"
    hidden_dim: int = 256
    n_layers: int = 6
    n_heads: int = 4
    vocab_size: int = 260  # 256 bytes + 4 special tokens
    max_seq_length: int = 128
    dropout: float = 0.1

    # Training.
    learning_rate: float = 3e-4
    batch_size: int = 4
    steps_per_round: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Data.
    data_paths: List[str] = field(default_factory=list)


@dataclass
class KickstartResult:
    """Result of a kickstart training round."""

    round_id: str
    steps_completed: int = 0
    avg_loss: float = 0.0
    final_loss: float = 0.0
    tokens_processed: int = 0
    time_ms: float = 0.0
    gradients: Optional[Dict[str, torch.Tensor]] = None
    sample_text: str = ""  # Generated sample to track progress


class Kickstart:
    """Bootstrap an LLM from scratch.

    Usage:
        ks = Kickstart(config)
        ks.load_data("path/to/texts")     # or ks.load_text("raw text...")
        result = ks.train_round("round-0") # local training, returns gradients

        # After decentralized aggregation:
        ks.apply_aggregated_gradients(aggregated_grads)
        ks.checkpoint("round-0")

        # Generate sample text to track progress:
        print(ks.generate("The "))
    """

    def __init__(self, config: KickstartConfig):
        self.config = config

        # Tokenizer.
        self.tokenizer = Tokenizer(TokenizerConfig(
            mode="byte",
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_seq_length,
        ))

        # Model.
        lm_config = LMConfig(
            model_id=config.model_id,
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            max_seq_length=config.max_seq_length,
            dropout=config.dropout,
        )
        self.model = create_from_scratch(lm_config)

        # Optimizer.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Data pipeline.
        self.data = TextDataPipeline(
            tokenizer=self.tokenizer,
            config=DataConfig(
                seq_length=config.max_seq_length,
                batch_size=config.batch_size,
                batches_per_round=config.steps_per_round,
            ),
        )

        self.total_steps = 0
        self.round_count = 0

        logger.info(
            "Kickstart initialized: %s (%d params, %d layers, hidden=%d, vocab=%d)",
            config.model_id,
            self.model.num_parameters,
            config.n_layers,
            config.hidden_dim,
            config.vocab_size,
        )

    def load_text(self, text: str):
        """Load raw text into the training data pipeline."""
        self.data.load_text(text)

    def load_file(self, path: str):
        """Load a text file into the training data pipeline."""
        self.data.load_file(path)

    def load_directory(self, directory: str):
        """Load all text files from a directory."""
        self.data.load_directory(directory)

    def train_round(
        self,
        round_id: str = "",
        peer_id: str = "local",
    ) -> KickstartResult:
        """Run a local training round on the peer's data.

        This is one peer's contribution to a training round. The gradients
        returned should be sent to the decentralized aggregation system.

        Returns:
            KickstartResult with gradients ready for aggregation.
        """
        if self.data.total_tokens < self.config.max_seq_length + 1:
            return KickstartResult(
                round_id=round_id,
                avg_loss=float("inf"),
            )

        start = time.monotonic()
        self.model.train()

        losses = []
        tokens = 0

        for input_ids, target_ids, mask in self.data.iter_batches(round_id, peer_id):
            self.optimizer.zero_grad()

            logits, loss = self.model(input_ids, target_ids)
            loss.backward()

            # Gradient clipping.
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            self.optimizer.step()

            losses.append(loss.item())
            tokens += input_ids.numel()
            self.total_steps += 1

        elapsed = (time.monotonic() - start) * 1000
        self.round_count += 1

        # Collect gradients for aggregation.
        # We need to do one more forward-backward to get fresh gradients
        # (the optimizer step already consumed the previous ones).
        gradients = self._collect_gradients(round_id, peer_id)

        avg_loss = sum(losses) / len(losses) if losses else float("inf")

        result = KickstartResult(
            round_id=round_id,
            steps_completed=len(losses),
            avg_loss=avg_loss,
            final_loss=losses[-1] if losses else float("inf"),
            tokens_processed=tokens,
            time_ms=elapsed,
            gradients=gradients,
        )

        # Generate a sample to track progress.
        result.sample_text = self.generate("The ", max_tokens=30)

        logger.info(
            "Round %s: %d steps, avg_loss=%.4f, %d tokens, %.0fms",
            round_id, result.steps_completed, avg_loss, tokens, elapsed,
        )

        return result

    def _collect_gradients(self, round_id: str, peer_id: str) -> Dict[str, torch.Tensor]:
        """Run one forward-backward pass to collect fresh gradients."""
        self.model.train()

        # Get one batch.
        batch_iter = self.data.iter_batches(round_id, peer_id)
        try:
            input_ids, target_ids, mask = next(batch_iter)
        except StopIteration:
            return {}

        self.optimizer.zero_grad()
        _, loss = self.model(input_ids, target_ids)
        loss.backward()

        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone().detach()
        return grads

    def apply_aggregated_gradients(self, gradients: Dict[str, torch.Tensor]):
        """Apply aggregated gradients from the decentralized network.

        After all peers submit their gradients and the coordinator aggregates
        them, each peer applies the result to converge on shared weights.
        """
        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            if name in gradients:
                param.grad = gradients[name].to(param.device)
        self.optimizer.step()

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.8) -> str:
        """Generate text from a prompt."""
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long)

        output_ids = self.model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature)
        return self.tokenizer.decode(output_ids[0].tolist())

    def state_dict(self) -> Dict:
        """Get model state for checkpointing."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict):
        """Load model state from checkpoint."""
        self.model.load_state_dict(state_dict)

    def stats(self) -> Dict:
        return {
            "model_id": self.config.model_id,
            "parameters": self.model.num_parameters,
            "total_steps": self.total_steps,
            "rounds": self.round_count,
            "data_tokens": self.data.total_tokens,
            "data_sequences": self.data.total_sequences,
            "data_batches": self.data.total_batches,
        }
