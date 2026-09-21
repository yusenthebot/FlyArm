"""Common action interface for fixed connectomes and parameter-matched controls."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from flyarm.graph import Graph


class Policy(nn.Module):
    obs_mean: torch.Tensor
    obs_scale: torch.Tensor
    adjacency: torch.Tensor

    def __init__(self, kind: str, graph: Graph, seed: int = 0) -> None:
        super().__init__()
        if kind not in {"connectome", "shuffled", "mlp", "gru", "disconnected"}:
            raise ValueError(f"Unknown policy {kind}")
        self.kind = kind
        graph.validate()
        n = len(graph.ids)
        torch.manual_seed(seed)
        self.register_buffer("obs_mean", torch.zeros(20))
        self.register_buffer("obs_scale", torch.ones(20))
        if kind in {"connectome", "shuffled", "disconnected"}:
            self.encoder = nn.Linear(20, n)
            self.readout = nn.Linear(n, 3)
            indices = torch.tensor(np.stack([graph.post, graph.pre]), dtype=torch.int64)
            values = torch.from_numpy(graph.normalized_weights())
            if kind == "disconnected":
                values = torch.zeros_like(values)
            self.register_buffer(
                "adjacency",
                torch.sparse_coo_tensor(indices, values, (n, n), check_invariants=True).coalesce(),
            )
            self.state_size = n
        elif kind == "mlp":
            # Exact parameter match to reservoir's input/readout layers.
            self.encoder = nn.Linear(20, n)
            self.readout = nn.Linear(n, 3)
            self.state_size = n
        else:
            budget = (20 + 1) * n + (n + 1) * 3
            hidden = min(
                range(4, 512), key=lambda h: abs(3 * h * (20 + h + 2) + 3 * h + 3 - budget)
            )
            self.cell = nn.GRUCell(20, hidden)
            self.readout = nn.Linear(hidden, 3)
            self.state_size = hidden

    def set_normalization(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.obs_scale.copy_(torch.as_tensor(scale, dtype=torch.float32))

    def initial_state(self, batch: int) -> torch.Tensor:
        return self.obs_mean.new_zeros(batch, self.state_size)

    def forward(self, obs: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (obs - self.obs_mean) / self.obs_scale
        if self.kind in {"connectome", "shuffled", "disconnected"}:
            recurrent = torch.sparse.mm(self.adjacency, state.T).T
            state = 0.5 * state + 0.5 * torch.tanh(self.encoder(x) + 0.8 * recurrent)
        elif self.kind == "mlp":
            state = torch.tanh(self.encoder(x))
        else:
            state = self.cell(x, state)
        return torch.tanh(self.readout(state)), state

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Controller:
    """Single-episode inference state; every rollout calls reset()."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy.eval()
        self.state = policy.initial_state(1)

    def reset(self) -> None:
        self.state = self.policy.initial_state(1)

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        action, self.state = self.policy(torch.as_tensor(observation).float()[None], self.state)
        return action[0].cpu().numpy()
