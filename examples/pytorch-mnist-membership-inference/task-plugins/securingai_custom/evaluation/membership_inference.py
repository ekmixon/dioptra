#!/usr/bin/env python
# Copyright 2020 The MITRE Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Tuple, Any
from pathlib import Path

warnings.filterwarnings("ignore")



import mlflow
import numpy as np
import pandas as pd
import scipy.stats
import structlog
from art.estimators.classification import PyTorchClassifier
from art.attacks.inference.membership_inference.black_box import MembershipInferenceBlackBox
from torch.nn import CrossEntropyLoss
import numpy
from torch.autograd import Variable
from mitre.securingai import pyplugs

LOGGER = structlog.get_logger()
DISTANCE_METRICS = [
]

def wrap_torch_classifier(torch_model, loss_fn, input_shape, classes):
    return PyTorchClassifier(model=torch_model, loss=loss_fn, input_shape=input_shape, nb_classes=classes)

def init_mi(model, loss_fn, input_shape, classes, attack_type, **kwargs):
    classifier = wrap_torch_classifier(model, loss_fn, input_shape, classes, **kwargs)
    attack = MembershipInferenceBlackBox(classifier=classifier,
                                         input_type="loss",
                                         attack_model_type=attack_type,
                                         **kwargs)
    return classifier, attack

@pyplugs.register
def infer_membership(
    training_ds: Any,
    testing_ds: Any,
    model: Any,
    attack_type: str = "nn",
    split: float = 0.5,
    balance_sets: bool = True,
    image_size: Tuple[int, int, int] = (1, 28, 28),
    **kwargs,
):

    x_train = next(iter(training_ds))[0].numpy()
    y_train = next(iter(training_ds))[1].numpy()
    
    x_test = next(iter(testing_ds))[0].numpy()
    y_test = next(iter(testing_ds))[1].numpy()
    
    classes = len(numpy.unique(y_train))
    LOGGER.info("Classes:" + str(classes))

    classifier, attack = init_mi(model=model,
                                 loss_fn=CrossEntropyLoss(),
                                 input_shape=image_size,
                                 classes=classes,
                                 attack_type=attack_type,
                                 **kwargs)
    
    attack_train_size = int(len(x_train) * split)
    attack_test_size = int(len(x_test) * split)
      
    # Take the lesser of the two sizes if we want to keep it balanced
    if balance_sets:
        if attack_train_size < attack_test_size:
            attack_test_size = attack_train_size
        else:
            attack_train_size = attack_test_size

    attack.fit(
        x_train[:attack_train_size], y_train[:attack_train_size], x_test[:attack_test_size], y_test[:attack_test_size]
    )

    # infer attacked feature on remainder of data
    inferred_train = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
    inferred_test = attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])

    log_mi_metrics(inferred_train, inferred_test)
    

    return None

def log_mi_metrics(inferred_train, inferred_test):
    trainacc = (sum(inferred_train)) / len(inferred_train)
    testacc = (sum(inferred_test)) / len(inferred_test)

    accuracy = (sum(inferred_train) + sum(inferred_test)) / (len(inferred_train) + len(inferred_test))
    mlflow.log_metric(key="acc",value=accuracy)
    mlflow.log_metric(key="acc_train",value=trainacc)
    mlflow.log_metric(key="acc_test",value=testacc)

