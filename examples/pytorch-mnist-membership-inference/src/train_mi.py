#!/usr/bin/env python

import os
from pathlib import Path
from typing import Any, Dict, List

import click
import mlflow
import structlog
from prefect import Flow, Parameter
from prefect.utilities.logging import get_logger as get_prefect_logger
from structlog.stdlib import BoundLogger

from mitre.securingai import pyplugs
from mitre.securingai.sdk.utilities.contexts import plugin_dirs
from mitre.securingai.sdk.utilities.logging import (
    StderrLogStream,
    StdoutLogStream,
    attach_stdout_stream_handler,
    clear_logger_handlers,
    configure_structlog,
    set_logging_level,
)

_PLUGINS_IMPORT_PATH: str = "securingai_builtins"
_CUSTOM_PLUGINS_IMPORT_PATH: str = "securingai_custom"
CALLBACKS: List[Dict[str, Any]] = [
    {
        "name": "EarlyStopping",
        "parameters": {
            "monitor": "val_loss",
            "min_delta": 1e-2,
            "patience": 5,
            "restore_best_weights": True,
        },
    },
]
LOGGER: BoundLogger = structlog.stdlib.get_logger()
PERFORMANCE_METRICS: List[Dict[str, Any]] = [
    {"name": "CategoricalAccuracy", "parameters": {"name": "accuracy"}},
    {"name": "Precision", "parameters": {"name": "precision"}},
    {"name": "Recall", "parameters": {"name": "recall"}},
    {"name": "AUC", "parameters": {"name": "auc"}},
]


def _coerce_comma_separated_ints(ctx, param, value):
    return tuple(int(x.strip()) for x in value.split(","))


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, readable=True
    ),
    help="Root directory for NFS mounted datasets (in container)",
)
@click.option(
    "--image-size",
    type=click.STRING,
    callback=_coerce_comma_separated_ints,
    help="Dimensions for the input images",
)
@click.option(
    "--model-architecture",
    type=click.Choice(["le_net"], case_sensitive=False),
    default="le_net",
    help="Model architecture",
)
@click.option(
    "--batch-size",
    type=click.INT,
    help="Batch size to use when training a single epoch",
    default=32,
)
@click.option(
    "--register-model-name",
    type=click.STRING,
    default="",
    help=(
        "Register the trained model under the provided name. If an empty string, "
        "then the trained model will not be registered."
    ),
)
@click.option(
    "--learning-rate", type=click.FLOAT, help="Model learning rate", default=0.001
)
@click.option(
    "--optimizer",
    type=click.Choice(["Adam", "Adagrad", "RMSprop", "SGD"], case_sensitive=True),
    help="Optimizer to use to train the model",
    default="Adam",
)
@click.option(
    "--validation-split",
    type=click.FLOAT,
    help="Fraction of training dataset to use for validation",
    default=0.2,
)
@click.option(
    "--seed",
    type=click.INT,
    help="Set the entry point rng seed",
    default=-1,
)
def train(
    data_dir,
    image_size,
    model_architecture,
    batch_size,
    register_model_name,
    learning_rate,
    optimizer,
    validation_split,
    seed,
):
    LOGGER.info(
        "Execute MLFlow entry point",
        entry_point="train",
        data_dir=data_dir,
        image_size=image_size,
        model_architecture=model_architecture,
        batch_size=batch_size,
        register_model_name=register_model_name,
        learning_rate=learning_rate,
        optimizer=optimizer,
        validation_split=validation_split,
        seed=seed,
    )

    mlflow.autolog()

    with mlflow.start_run() as active_run:
        flow: Flow = init_train_flow()
        state = flow.run(
            parameters=dict(
                active_run=active_run,
                training_dir=Path(data_dir) / "training",
                testing_dir=Path(data_dir) / "testing",
                image_size=image_size,
                model_architecture=model_architecture,
                batch_size=batch_size,
                register_model_name=register_model_name,
                learning_rate=learning_rate,
                optimizer_name=optimizer,
                validation_split=validation_split,
                seed=seed,
            )
        )

    return state


def init_train_flow() -> Flow:
    with Flow("Train Model") as flow:
        (
            active_run,
            training_dir,
            testing_dir,
            image_size,
            model_architecture,
            batch_size,
            register_model_name,
            learning_rate,
            optimizer_name,
            validation_split,
            seed,
        ) = (
            Parameter("active_run"),
            Parameter("training_dir"),
            Parameter("testing_dir"),
            Parameter("image_size"),
            Parameter("model_architecture"),
            Parameter("batch_size"),
            Parameter("register_model_name"),
            Parameter("learning_rate"),
            Parameter("optimizer_name"),
            Parameter("validation_split"),
            Parameter("seed"),
        )
        
        
        seed, rng = pyplugs.call_task(
            f"{_PLUGINS_IMPORT_PATH}.random", "rng", "init_rng", seed=seed
        )
        dataset_seed = pyplugs.call_task(
            f"{_PLUGINS_IMPORT_PATH}.random", "sample", "draw_random_integer", rng=rng
        )

        log_mlflow_params_result = pyplugs.call_task(  # noqa: F841
            f"{_PLUGINS_IMPORT_PATH}.tracking",
            "mlflow",
            "log_parameters",
            parameters=dict(
                entry_point_seed=seed,
                dataset_seed=dataset_seed,
            ),
        )

        (training_ds, validation_ds) = pyplugs.call_task(
            f"{_CUSTOM_PLUGINS_IMPORT_PATH}.evaluation",
            "data_pytorch",
            "create_image_dataset",
            data_dir=training_dir,
            validation_split=validation_split,
            batch_size=batch_size,
            seed=dataset_seed,
            image_size=image_size,
            upstream_tasks=[]
        )

        (testing_ds, _) = pyplugs.call_task(
            f"{_CUSTOM_PLUGINS_IMPORT_PATH}.evaluation",
            "data_pytorch",
            "create_image_dataset",
            data_dir=testing_dir,
            validation_split=None,
            batch_size=batch_size,
            seed=dataset_seed + 1,
            image_size=image_size,
            upstream_tasks=[]
        )

        n_classes = pyplugs.call_task(
            f"{_CUSTOM_PLUGINS_IMPORT_PATH}.evaluation",
            "data_pytorch",
            "get_n_classes_from_directory_iterator",
            ds=training_ds
        )
        classifier = pyplugs.call_task(
            f"{_CUSTOM_PLUGINS_IMPORT_PATH}.evaluation",
            "estimators_pytorch_classifiers",
            "init_classifier",
            model_architecture=model_architecture,
            input_shape=image_size,
            n_classes=n_classes,
            upstream_tasks=[]
        )
        optimizer = pyplugs.call_task(
            f"{_CUSTOM_PLUGINS_IMPORT_PATH}.evaluation",
            "tasks",
            "get_optimizer",
            model=classifier,
            optimizer=optimizer_name,
            learning_rate=learning_rate,
            upstream_tasks=[],
        )
        history = pyplugs.call_task(
            f"{_CUSTOM_PLUGINS_IMPORT_PATH}.evaluation",
            "estimators_pytorch_methods",
            "fit",
            estimator=classifier,
            optimizer=optimizer,
            training_ds=training_ds,
            upstream_tasks=[training_ds, validation_ds, testing_ds]
        )
        actual, predicted = pyplugs.call_task(
            f"{_CUSTOM_PLUGINS_IMPORT_PATH}.evaluation",
            "estimators_pytorch_methods",
            "predict",
            estimator=classifier,
            testing_ds=testing_ds,
            upstream_tasks=[history] # This is required to ensure prediction does not start before the model has been fully trained
        )
        
        classifier_performance_metrics = pyplugs.call_task(  # noqa: F841
            f"{_PLUGINS_IMPORT_PATH}.metrics",
            "performance",
            "get_performance_metric_list",
            request=[
                {"name":"accuracy", "func":"accuracy"},
                {"name":"categorical_accuracy", "func":"categorical_accuracy"},
                {"name":"f1", "func":"f1"},
                {"name":"precision", "func":"precision"},
                {"name":"recall", "func":"recall"}
            ],
            upstream_tasks=[actual,predicted]
        )
        
        performance_metrics = pyplugs.call_task(
            f"{_CUSTOM_PLUGINS_IMPORT_PATH}.evaluation",
            "tasks",
            "evaluate_metrics_generic",
            y_true=actual,
            y_pred=predicted,
            metrics=classifier_performance_metrics,
            func_kwargs={
                'accuracy': {},
                'categorical_accuracy': {},
                'f1': {
                    'average':'weighted'
                },
                'precision': {
                    'average':'weighted'
                },
                'recall': {
                    'average':'weighted'
                }
            },
            upstream_tasks=[classifier_performance_metrics]
        )        
        
        log_classifier_performance_metrics_result = pyplugs.call_task(  # noqa: F841
            f"{_PLUGINS_IMPORT_PATH}.tracking",
            "mlflow",
            "log_metrics",
            metrics=performance_metrics,
        )
        
        log_classifier_performance_metrics_result = pyplugs.call_task(  # noqa: F841
            f"{_CUSTOM_PLUGINS_IMPORT_PATH}.evaluation",
            "registry_mlflow_pytorch",
            "add_model_to_registry",
            active_run=active_run,
            name=register_model_name,
            model_dir="model",
            upstream_tasks=[history]
        )
        
    return flow


if __name__ == "__main__":
    log_level: str = os.getenv("AI_JOB_LOG_LEVEL", default="INFO")
    as_json: bool = True if os.getenv("AI_JOB_LOG_AS_JSON") else False

    clear_logger_handlers(get_prefect_logger())
    attach_stdout_stream_handler(as_json)
    set_logging_level(log_level)
    configure_structlog()
    

    with plugin_dirs(), StdoutLogStream(as_json), StderrLogStream(as_json):
        _ = train()
