#!/usr/bin/env python

import os
from pathlib import Path

import click
import mlflow
import mlflow.tensorflow
import structlog
from prefect import Flow, Parameter
from prefect.utilities.logging import get_logger as get_prefect_logger
from structlog.stdlib import BoundLogger
from tasks import evaluate_metrics_tensorflow

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
LOGGER: BoundLogger = structlog.stdlib.get_logger()


def _coerce_comma_separated_ints(ctx, param, value):
    return tuple(int(x.strip()) for x in value.split(","))


@click.command()
@click.option(
    "--run-id",
    type=click.STRING,
    help="MLFlow Run ID of a successful fgm attack",
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
    type=click.STRING,
    help="Version of model to load from registry",
)
@click.option(
    "--adv-tar-name",
    type=click.STRING,
    default="testing_adversarial_fgm.tar.gz",
    help="Name of tarfile artifact containing fgm images",
)
@click.option(
    "--adv-data-dir",
    type=click.STRING,
    default="adv_testing",
    help="Directory in tarfile containing fgm images",
)
@click.option(
    "--seed",
    type=click.INT,
    help="Set the entry point rng seed",
    default=-1,
)
def infer_adversarial(
    run_id, image_size, model_name, model_version, adv_tar_name, adv_data_dir, seed
):
    LOGGER.info(
        "Execute MLFlow entry point",
        entry_point="infer",
        image_size=image_size,
        model_name=model_name,
        model_version=model_version,
        adv_tar_name=adv_tar_name,
        adv_data_dir=adv_data_dir,
        seed=seed,
    )

    with mlflow.start_run() as active_run:  # noqa: F841
        flow: Flow = init_infer_flow()
        state = flow.run(
            parameters=dict(
                image_size=image_size,
                model_name=model_name,
                model_version=model_version,
                run_id=run_id,
                adv_tar_name=adv_tar_name,
                adv_data_dir=(Path.cwd() / adv_data_dir).resolve(),
                seed=seed,
            )
        )

    return state


def init_infer_flow() -> Flow:
    with Flow("Inference") as flow:
        (
            image_size,
            model_name,
            model_version,
            run_id,
            adv_tar_name,
            adv_data_dir,
            seed,
        ) = (
            Parameter("image_size"),
            Parameter("model_name"),
            Parameter("model_version"),
            Parameter("run_id"),
            Parameter("adv_tar_name"),
            Parameter("adv_data_dir"),
            Parameter("seed"),
        )
        seed, rng = pyplugs.call_task(
            f"{_PLUGINS_IMPORT_PATH}.random", "rng", "init_rng", seed=seed
        )
        tensorflow_global_seed = pyplugs.call_task(
            f"{_PLUGINS_IMPORT_PATH}.random", "sample", "draw_random_integer", rng=rng
        )
        dataset_seed = pyplugs.call_task(
            f"{_PLUGINS_IMPORT_PATH}.random", "sample", "draw_random_integer", rng=rng
        )
        init_tensorflow_results = pyplugs.call_task(
            f"{_PLUGINS_IMPORT_PATH}.backend_configs",
            "tensorflow",
            "init_tensorflow",
            seed=tensorflow_global_seed,
        )

        log_mlflow_params_result = pyplugs.call_task(  # noqa: F841
            f"{_PLUGINS_IMPORT_PATH}.tracking",
            "mlflow",
            "log_parameters",
            parameters=dict(
                entry_point_seed=seed,
                tensorflow_global_seed=tensorflow_global_seed,
                dataset_seed=dataset_seed,
            ),
        )
        adv_tar_path = pyplugs.call_task(
            f"{_PLUGINS_IMPORT_PATH}.artifacts",
            "mlflow",
            "download_all_artifacts_in_run",
            run_id=run_id,
            artifact_path=adv_tar_name,
        )
        extract_tarfile_results = pyplugs.call_task(
            f"{_PLUGINS_IMPORT_PATH}.artifacts",
            "utils",
            "extract_tarfile",
            filepath=adv_tar_path,
        )
        adv_ds = pyplugs.call_task(
            f"{_PLUGINS_IMPORT_PATH}.data",
            "tensorflow",
            "create_image_dataset",
            data_dir=adv_data_dir,
            subset=None,
            validation_split=None,
            image_size=image_size,
            seed=dataset_seed,
            upstream_tasks=[init_tensorflow_results, extract_tarfile_results],
        )
        classifier = pyplugs.call_task(
            f"{_PLUGINS_IMPORT_PATH}.registry",
            "mlflow",
            "load_tensorflow_keras_classifier",
            name=model_name,
            version=model_version,
            upstream_tasks=[init_tensorflow_results],
        )
        classifier_performance_metrics = evaluate_metrics_tensorflow(
            classifier=classifier, dataset=adv_ds
        )
        log_classifier_performance_metrics_result = pyplugs.call_task(  # noqa: F841
            f"{_PLUGINS_IMPORT_PATH}.tracking",
            "mlflow",
            "log_metrics",
            metrics=classifier_performance_metrics,
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
        _ = infer_adversarial()
