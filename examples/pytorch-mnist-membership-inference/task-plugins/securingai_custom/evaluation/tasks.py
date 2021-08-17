from __future__ import annotations

from types import FunctionType
from typing import Any, Dict, List, Union

from . import import_pytorch

import structlog
from mitre.securingai import pyplugs
from structlog.stdlib import BoundLogger

from mitre.securingai.sdk.utilities.decorators import require_package

LOGGER: BoundLogger = structlog.stdlib.get_logger()

try:
    from torch.optim import Optimizer

except ImportError:  # pragma: nocover
    LOGGER.warn(
        "Unable to import one or more optional packages, functionality may be reduced",
        package="torch",
    )

@pyplugs.register
def evaluate_metrics_generic(y_true, y_pred, metrics, func_kwargs) -> Dict[str, float]:
    names = []
    result = []
    for metric in metrics:
        name = metric[0]
        func = metric[1]
        
        extra_kwargs = func_kwargs.get(name, {})
        
        names += [name]
        metric_output = func(y_true.copy(), y_pred.copy(), **extra_kwargs)
        result += [metric_output]
        
    return dict(zip(names, result))

    
@pyplugs.register
def get_optimizer(
    model: Sequential,
    optimizer: str,
    learning_rate: float,
):
    return import_pytorch.get_optimizer(optimizer)(model.parameters(), learning_rate)
