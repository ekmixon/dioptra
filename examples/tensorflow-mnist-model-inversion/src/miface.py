#!/usr/bin/env python
# This Software (Dioptra) is being made available as a public service by the
# National Institute of Standards and Technology (NIST), an Agency of the United
# States Department of Commerce. This software was developed in part by employees of
# NIST and in part by NIST contractors. Copyright in portions of this software that
# were developed by NIST contractors has been licensed or assigned to NIST. Pursuant
# to Title 17 United States Code Section 105, works of NIST employees are not
# subject to copyright protection in the United States. However, NIST may hold
# international copyright in software created by its employees and domestic
# copyright (or licensing rights) in portions of software that were assigned or
# licensed to NIST. To the extent that NIST holds copyright in this software, it is
# being made available under the Creative Commons Attribution 4.0 International
# license (CC BY 4.0). The disclaimers of the CC BY 4.0 license apply to all parts
# of the software developed or licensed by NIST.
#
# ACCESS THE FULL CC BY 4.0 LICENSE HERE:
# https://creativecommons.org/licenses/by/4.0/legalcode

import tarfile
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import click
import mlflow
import mlflow.tensorflow
import numpy as np
import structlog
from pathlib import Path

from attacks import miface_infer
from data import create_image_dataset
from log import configure_stdlib_logger, configure_structlog_logger


LOGGER = structlog.get_logger()


def evaluate_classification_metrics(classifier, adv_ds):
    LOGGER.info("evaluating classification metrics using adversarial images")
    result = classifier.model.evaluate(adv_ds, verbose=0)
    adv_metrics = dict(zip(classifier.model.metrics_names, result))
    LOGGER.info(
        "computation of classification metrics for adversarial images complete",
        **adv_metrics,
    )
    for metric_name, metric_value in adv_metrics.items():
        mlflow.log_metric(key=metric_name, value=metric_value)


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, readable=True
    ),
    help="Root directory for NFS mounted datasets (in container)",
)
@click.option(
    "--model",
    type=click.STRING,
    help="Name of model to load from registry",
)
@click.option(
    "--model-architecture",
    type=click.Choice(["shallow_net", "le_net", "alex_net"], case_sensitive=False),
    default="le_net",
    help="Model architecture",
)
@click.option(
    "--batch-size",
    type=click.INT,
    help="Batch size to use when training a single epoch",
    default=1,
)
@click.option("--max-iter", type=click.INT, help="Max Iterations", default=10000)
@click.option("--window-length", type=click.INT, help="Window length", default=100)
@click.option("--threshold", type=click.FLOAT, help="Threshold", default=0.99)
@click.option("--learning-rate", type=click.FLOAT, help="Learning Rate", default=0.1)
@click.option(
    "--seed",
    type=click.INT,
    help="Set the entry point rng seed",
    default=-1,
)
def miface_attack(
    data_dir,
    model,
    model_architecture,
    batch_size,
    max_iter,
    window_length,
    threshold,
    learning_rate,
    seed,
):
    norm_mapping = {"inf": np.inf, "1": 1, "2": 2}
    rng = np.random.default_rng(seed if seed >= 0 else None)

    if seed < 0:
        seed = rng.bit_generator._seed_seq.entropy

    LOGGER.info(
        "Execute MLFlow entry point",
        entry_point="miface",
        data_dir=data_dir,
        model=model,
        model_architecture=model_architecture,
        batch_size=batch_size,
    )

    tensorflow_global_seed: int = rng.integers(low=0, high=2 ** 31 - 1)

    tf.random.set_seed(tensorflow_global_seed)

    with mlflow.start_run() as _:
        adv_data_dir = Path().cwd() / "adv_testing"

        adv_data_dir.mkdir(parents=True, exist_ok=True)

        image_size = (224, 224) if model_architecture == "alex_net" else (28, 28)
        distance_metrics = miface_infer(
            model=model,
            adv_data_dir=adv_data_dir.resolve(),
            batch_size=batch_size,
            max_iter=max_iter,
            window_length=window_length,
            threshold=threshold,
            learning_rate=learning_rate,
        )

        adv_testing_tar = Path().cwd() / "testing_adversarial_miface.tar.gz"

        with tarfile.open(adv_testing_tar, "w:gz") as f:
            f.add(str(adv_data_dir.resolve()), arcname=adv_data_dir.name)

        LOGGER.info("Log adversarial images", filename=adv_testing_tar.name)
        mlflow.log_artifact(str(adv_testing_tar))


if __name__ == "__main__":
    configure_stdlib_logger("INFO", log_filepath=None)
    configure_structlog_logger("console")

    miface_attack()
