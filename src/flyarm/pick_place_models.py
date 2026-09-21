"""Policies for causal constrained-interface pick-and-place experiments."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from flyarm.graph import Graph
from flyarm.interfaces import NeuralInterface


class PickPlacePolicy(nn.Module):
    """Only declared input nodes receive observations and output nodes emit actions."""

    obs_mean: torch.Tensor
    obs_scale: torch.Tensor
    adjacency: torch.Tensor
    input_indices: torch.Tensor
    output_indices: torch.Tensor
    encoder: nn.Module
    readout: nn.Linear
    cell: nn.GRUCell

    def __init__(
        self,
        kind: str,
        graph: Graph,
        interface: NeuralInterface,
        *,
        seed: int = 0,
        obs_dim: int = 30,
        action_dim: int = 4,
        internal_steps: int = 3,
    ) -> None:
        super().__init__()
        if kind not in {
            "restricted_connectome",
            "restricted_shuffled",
            "restricted_disconnected",
            "mlp",
            "gru",
        }:
            raise ValueError(f"Unknown pick-place policy {kind}")
        if obs_dim < 1 or action_dim < 1 or internal_steps < 1:
            raise ValueError("obs_dim, action_dim, and internal_steps must all be positive")
        graph.validate()
        input_indices, output_indices = interface.resolve_indices(graph)
        self.kind = kind
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.internal_steps = internal_steps
        self.interface_fingerprint = interface.fingerprint
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_scale", torch.ones(obs_dim))
        self.register_buffer("input_indices", torch.from_numpy(input_indices))
        self.register_buffer("output_indices", torch.from_numpy(output_indices))

        # Isolate deterministic initialization from callers' global random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if kind.startswith("restricted_"):
                self.encoder = nn.Linear(obs_dim, len(input_indices))
                self.readout = nn.Linear(len(output_indices), action_dim)
                indices = torch.as_tensor(np.stack((graph.post, graph.pre)), dtype=torch.int64)
                weights = torch.from_numpy(graph.normalized_weights())
                if kind == "restricted_disconnected":
                    weights = torch.zeros_like(weights)
                self.register_buffer(
                    "adjacency",
                    torch.sparse_coo_tensor(
                        indices,
                        weights,
                        (len(graph.ids), len(graph.ids)),
                        check_invariants=True,
                    ).coalesce(),
                )
                self.state_size = len(graph.ids)
            elif kind == "mlp":
                target = self._restricted_parameter_budget(
                    obs_dim, action_dim, len(input_indices), len(output_indices)
                )
                first_width, second_width = self._mlp_widths(target, obs_dim, action_dim)
                if second_width is None:
                    self.encoder = nn.Sequential(nn.Linear(obs_dim, first_width), nn.Tanh())
                    self.readout = nn.Linear(first_width, action_dim)
                    self.state_size = first_width
                else:
                    self.encoder = nn.Sequential(
                        nn.Linear(obs_dim, first_width),
                        nn.Tanh(),
                        nn.Linear(first_width, second_width),
                        nn.Tanh(),
                    )
                    self.readout = nn.Linear(second_width, action_dim)
                    self.state_size = second_width
            else:
                # The seven-unit GRU is within 5% of the constrained model's parameter budget.
                self.cell = nn.GRUCell(obs_dim, 7)
                self.readout = nn.Linear(7, action_dim)
                self.state_size = 7

    @staticmethod
    def _restricted_parameter_budget(
        obs_dim: int, action_dim: int, input_count: int, output_count: int
    ) -> int:
        return (obs_dim + 1) * input_count + (output_count + 1) * action_dim

    @staticmethod
    def _mlp_widths(target: int, obs_dim: int, action_dim: int) -> tuple[int, int | None]:
        """Find an ordinary MLP with the exact constrained-interface parameter budget."""
        single_coefficient = obs_dim + action_dim + 1
        if (target - action_dim) % single_coefficient == 0:
            width = (target - action_dim) // single_coefficient
            if width >= 1:
                return width, None
        # (obs+1)a + (a+1)b + (b+1)action = target.
        candidates: list[tuple[int, int]] = []
        for first_width in range(1, target):
            remaining = target - action_dim - (obs_dim + 1) * first_width
            coefficient = first_width + action_dim + 1
            if remaining > 0 and remaining % coefficient == 0:
                second_width = remaining // coefficient
                if second_width >= 1:
                    candidates.append((first_width, second_width))
        if candidates:
            # Avoid a degenerate narrow-expansion control when a balanced exact MLP exists.
            return min(candidates, key=lambda widths: (abs(widths[0] - widths[1]), max(widths)))
        raise ValueError("Could not construct an exact parameter-matched MLP")

    def set_normalization(self, mean: np.ndarray, scale: np.ndarray) -> None:
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32)
        if mean_tensor.shape != (self.obs_dim,) or scale_tensor.shape != (self.obs_dim,):
            raise ValueError("Normalization shape does not match obs_dim")
        if torch.any(scale_tensor <= 0) or not torch.all(torch.isfinite(scale_tensor)):
            raise ValueError("Normalization scale must be finite and positive")
        self.obs_mean.copy_(mean_tensor)
        self.obs_scale.copy_(scale_tensor)

    def initial_state(self, batch: int) -> torch.Tensor:
        if batch < 1:
            raise ValueError("batch must be positive")
        return self.obs_mean.new_zeros(batch, self.state_size)

    def forward(self, obs: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
            raise ValueError(f"Expected observations shaped [batch, {self.obs_dim}]")
        if state.shape != (len(obs), self.state_size):
            raise ValueError("State shape does not match policy batch and state size")
        x = (obs - self.obs_mean) / self.obs_scale
        if self.kind.startswith("restricted_"):
            drive = state.new_zeros((len(obs), self.state_size))
            drive[:, self.input_indices] = self.encoder(x)
            for _ in range(self.internal_steps):
                recurrent = torch.sparse.mm(self.adjacency, state.T).T
                state = 0.5 * state + 0.5 * torch.tanh(drive + 0.8 * recurrent)
            command_state = state[:, self.output_indices]
        elif self.kind == "mlp":
            state = self.encoder(x)
            command_state = state
        else:
            state = self.cell(x, state)
            command_state = state
        return torch.tanh(self.readout(command_state)), state

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class PickPlaceController:
    """State is isolated by episode so history ablations are meaningful."""

    def __init__(self, policy: PickPlacePolicy) -> None:
        self.policy = policy.eval()
        self.state = self.policy.initial_state(1)

    def reset(self) -> None:
        self.state = self.policy.initial_state(1)

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)[None]
        action, self.state = self.policy(observation_tensor, self.state)
        return action[0].cpu().numpy()


# Kept as aliases while callers migrate from the early design notes.
RestrictedPolicy = PickPlacePolicy
RestrictedController = PickPlaceController
Controller = PickPlaceController
