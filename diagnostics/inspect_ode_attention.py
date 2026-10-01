import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFuncAttention(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, input_channels),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ODEAttention(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.ode_func = ODEFuncAttention(input_channels, hidden_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        pooled = self.avg_pool(x).reshape(batch_size, channels)
        times = x.new_tensor([0.0, 1.0])
        ode_output = odeint(self.ode_func, pooled, times)[-1]
        attention = ode_output.reshape(batch_size, channels, 1, 1)
        return x * attention
