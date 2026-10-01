from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch


OPTIMIZER_NAMES = ("adam", "adamw", "sgd", "rmsprop", "adagrad")
SCHEDULER_NAMES = ("none", "cosine")


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    name: str,
    learning_rate: float,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized not in {item.replace("_", "") for item in OPTIMIZER_NAMES}:
        raise ValueError("Unknown optimizer {!r}; choose from {}.".format(name, ", ".join(OPTIMIZER_NAMES)))
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0 <= momentum < 1:
        raise ValueError("momentum must be in [0, 1)")

    if normalized == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "sgd":
        return torch.optim.SGD(
            parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    if normalized == "rmsprop":
        return torch.optim.RMSprop(
            parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    return torch.optim.Adagrad(parameters, lr=learning_rate, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    num_epochs: int,
    min_learning_rate: float = 0.0,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized not in {item.replace("_", "") for item in SCHEDULER_NAMES}:
        raise ValueError("Unknown scheduler {!r}; choose from {}.".format(name, ", ".join(SCHEDULER_NAMES)))
    if normalized == "none":
        return None
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if min_learning_rate < 0:
        raise ValueError("min_learning_rate must be non-negative")
    base_lr = optimizer.param_groups[0]["lr"]
    if min_learning_rate > base_lr:
        raise ValueError("min_learning_rate cannot exceed the initial learning rate")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=min_learning_rate
    )


def build_optimizer_and_scheduler(
    parameters: Iterable[torch.nn.Parameter],
    optimizer_name: str,
    scheduler_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    num_epochs: int,
    min_learning_rate: float,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
    optimizer = build_optimizer(
        parameters,
        name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    scheduler = build_scheduler(
        optimizer,
        name=scheduler_name,
        num_epochs=num_epochs,
        min_learning_rate=min_learning_rate,
    )
    return optimizer, scheduler
