from __future__ import annotations

import importlib
from types import FunctionType, ModuleType
from typing import Union

import structlog
from structlog.stdlib import BoundLogger

from mitre.securingai.sdk.exceptions import TensorflowDependencyError
from mitre.securingai.sdk.utilities.decorators import require_package

LOGGER: BoundLogger = structlog.stdlib.get_logger()

try:
    from torch.optim import Optimizer

except ImportError:  # pragma: nocover
    LOGGER.warn(
        "Unable to import one or more optional packages, functionality may be reduced",
        package="torch",
    )

PYTORCH_OPTIMIZERS: str = "torch.optim"

def get_optimizer(optimizer_name: str) -> Optimizer:
    pytorch_optimizers: ModuleType = importlib.import_module(PYTORCH_OPTIMIZERS)
    optimizer: Optimizer = getattr(pytorch_optimizers, optimizer_name)
    return optimizer


