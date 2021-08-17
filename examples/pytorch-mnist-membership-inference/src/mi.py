#!/usr/bin/env python

import os
from pathlib import Path
from typing import Dict, List

import click
import mlflow
import numpy as np
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
DISTANCE_METRICS: List[Dict[str, str]] = [
]


LOGGER: BoundLogger = structlog.stdlib.get_logger()

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
    "--model-name",
    type=click.STRING,
    help="Name of model to load from registry",
)
@click.option(
    "--model-version",
    type=click.INT,
    help="Version of model to load from registry",
)
@click.option(
    "--batch-size",
    type=click.INT,
    help="Batch size to use when training a single epoch",
    default=32,
)
@click.option(
    "--attack-model-type",
    type=click.STRING,
    help="The type of model to be used in the attack",
    default="nn",
)
@click.option(
    "--split",
    type=click.FLOAT,
    help="The training data split for the attack's model",
    default=0.5,
)
@click.option(
    "--balance-sets",
    type=click.BOOL,
    help="True if the training data for the attack's model should have equal amounts of both classes",
    default=True,
)
@click.option(
    "--seed",
    type=click.INT,
    help="Set the entry point rng seed",
    default=-1,
)
def mi_attack(
    data_dir,
    image_size,
    model_name,
    model_version,
    batch_size,
    attack_model_type,
    split,
    balance_sets,
    seed,
):
    LOGGER.info(
        "Execute MLFlow entry point",
        entry_point="membership inference",
        data_dir=data_dir,
        image_size=image_size,
        model_name=model_name,
        model_version=model_version,
        batch_size=batch_size,
        attack_model_type=attack_model_type,
        split=split,
        balance_sets=balance_sets,
        seed=seed,
    )

    with mlflow.start_run() as active_run:  # noqa: F841
        flow: Flow = init_mi_flow()
        state = flow.run(
            parameters=dict(
                training_dir=Path(data_dir) / "training",
                testing_dir=Path(data_dir) / "testing",
                image_size=image_size,
                model_name=model_name,
                model_version=model_version,
                batch_size=batch_size,
                attack_model_type=attack_model_type,
                split=split,
                balance_sets=balance_sets,
                seed=seed,
            )
        )

    return state


def init_mi_flow() -> Flow:
    with Flow("Membership Inference") as flow:
        (
            training_dir,
            testing_dir,
            image_size,
            model_name,
            model_version,
            batch_size,
            attack_model_type,
            split,
            balance_sets,
            seed,
        ) = (
            Parameter("training_dir"),
            Parameter("testing_dir"),
            Parameter("image_size"),
            Parameter("model_name"),
            Parameter("model_version"),
            Parameter("batch_size"),
            Parameter("attack_model_type"),
            Parameter("split"),
            Parameter("balance_sets"),
            Parameter("seed"),
        )

        ##START
        
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
            validation_split=None,
            batch_size=batch_size,
            seed=dataset_seed,
            image_size=image_size,
            upstream_tasks=[]
        )
        
        (testing_ds, _) =  pyplugs.call_task(
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
        model = pyplugs.call_task(
            f"{_CUSTOM_PLUGINS_IMPORT_PATH}.evaluation",
            "registry_mlflow_pytorch",
            "load_pytorch_classifier",
            name=model_name,
            version=model_version            
        )
        
        out = pyplugs.call_task(
            f"{_CUSTOM_PLUGINS_IMPORT_PATH}.evaluation",
            "membership_inference",
            "infer_membership",
            training_ds=training_ds,
            testing_ds=testing_ds,
            model=model,
            attack_type=attack_model_type,
            split=split,
            balance_sets=balance_sets,
            image_size=image_size,
            upstream_tasks=[training_ds, testing_ds, model]
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
        _ = mi_attack()
