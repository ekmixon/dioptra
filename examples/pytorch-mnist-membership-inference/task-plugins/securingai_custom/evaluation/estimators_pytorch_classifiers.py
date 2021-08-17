from __future__ import annotations

from types import FunctionType
from typing import Callable, Dict, List, Tuple, Union

import structlog
from structlog.stdlib import BoundLogger

from mitre.securingai import pyplugs
from mitre.securingai.sdk.exceptions import TensorflowDependencyError
from mitre.securingai.sdk.utilities.decorators import require_package

LOGGER: BoundLogger = structlog.stdlib.get_logger()

try:
    from torch.nn import Flatten, Sequential, Linear, Dropout, Conv2d, BatchNorm2d, MaxPool2d, ReLU, Softmax
    from torch import nn

except ImportError:  # pragma: nocover
    LOGGER.warn(
        "Unable to import one or more optional packages, functionality may be reduced",
        package="torch",
    )




@pyplugs.register
def init_classifier(
    model_architecture: str,
    input_shape: Tuple[int, int, int],
    n_classes: int,
) -> Sequential:
    classifier: Sequential = PYTORCH_CLASSIFIERS_REGISTRY[model_architecture](
        input_shape,
        n_classes,
    )
    return classifier


def le_net(input_shape: Tuple[int, int, int], n_classes: int) -> Sequential:
    model = Sequential(
        # first convolutional layer:
        Conv2d(1,20,5,1),
        ReLU(),
        MaxPool2d(2,2),

        # second conv layer, with pooling and dropout:
        Conv2d(20,50,5,1),
        ReLU(),
        MaxPool2d(2,2),
        Flatten(),
        
        # dense hidden layer, with dropout:
        Linear(4 * 4 * 50, 500),
        ReLU(),

        # output layer:
        Linear(500, 10),
        Softmax()
    )
    return model



PYTORCH_CLASSIFIERS_REGISTRY: Dict[
    str, Callable[[Tuple[int, int, int], int], Sequential]
] = dict(le_net=le_net)
