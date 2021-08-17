from __future__ import annotations

import datetime
from typing import Any, Dict, Optional

import mlflow
import structlog
from structlog.stdlib import BoundLogger

from mitre.securingai import pyplugs
from mitre.securingai.sdk.generics import estimator_predict, fit_estimator

LOGGER: BoundLogger = structlog.stdlib.get_logger()
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
import torch
import numpy as np

@pyplugs.register
def fit(
    estimator: Any,
    optimizer: Any,
    training_ds: Any,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:

    time_start: datetime.datetime = datetime.datetime.now()

    LOGGER.info(
        "Begin estimator fit",
        timestamp=time_start.isoformat(),
    )

    loss_fn = CrossEntropyLoss()
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(training_ds):
        #LOGGER.info("data shape:", str(data.shape))
        optimizer.zero_grad()
        x, target = Variable(x), Variable(target)
        out = estimator(x)
        loss = loss_fn(out, target)
        ave_loss = ave_loss * 0.9 + loss.data.item() * 0.1
        loss.backward()
        optimizer.step()
            
    time_end = datetime.datetime.now()

    total_seconds = (time_end - time_start).total_seconds()
    total_minutes = total_seconds / 60

    mlflow.log_metric("training_time_in_minutes", total_minutes)
    LOGGER.info(
        "pytorch model training complete",
        timestamp=time_end.isoformat(),
        total_minutes=total_minutes,
    )
    mlflow.pytorch.log_model(estimator, "model")
    return estimator


@pyplugs.register
@pyplugs.task_nout(2)
def predict(
    estimator: Any,
    testing_ds: Any = None,
    predict_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    estimator.eval()
    
    y_true = []
    y_pred = []
    
    for batch_idx, (data, target) in enumerate(testing_ds):
        t_x = Variable(data)
        t_y = Variable(target)
        outputs = estimator(t_x)
        _, predicted = torch.max(outputs.data, 1)

        y_pred += list(predicted.numpy())
        y_true += list(t_y.numpy())

    return np.array(y_true), np.array(y_pred)