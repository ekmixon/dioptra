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
"""Error handlers for the experiment endpoints."""

from flask_restx import Api


class ExperimentAlreadyExistsError(Exception):
    """The experiment name already exists."""


class ExperimentMLFlowTrackingAlreadyExistsError(Exception):
    """The experiment name already exists on the MLFlow Tracking backend."""


class ExperimentDoesNotExistError(Exception):
    """The requested experiment does not exist."""


class ExperimentMLFlowTrackingDoesNotExistError(Exception):
    """The requested experiment is in our database but is not in MLFlow Tracking."""


class ExperimentMLFlowTrackingRegistrationError(Exception):
    """Experiment registration to the MLFlow Tracking backend has failed."""


class ExperimentRegistrationError(Exception):
    """The experiment registration form contains invalid parameters."""


def register_error_handlers(api: Api) -> None:
    @api.errorhandler(ExperimentDoesNotExistError)
    def handle_experiment_does_not_exist_error(error):
        return {"message": "Not Found - The requested experiment does not exist"}, 404

    @api.errorhandler(ExperimentAlreadyExistsError)
    def handle_experiment_already_exists_error(error):
        return (
            {
                "message": "Bad Request - The experiment name on the registration form "
                "already exists. Please select another and resubmit."
            },
            400,
        )

    @api.errorhandler(ExperimentMLFlowTrackingAlreadyExistsError)
    def handle_mlflow_tracking_experiment_already_exists_error(error):
        return (
            {
                "message": "Bad Request - The experiment name on the registration form "
                "already exists on the MLFlow Tracking backend. Please select another "
                "and resubmit."
            },
            400,
        )

    @api.errorhandler(ExperimentMLFlowTrackingDoesNotExistError)
    def handle_mlflow_tracking_experiment_does_not_exist_error(error):
        return (
            {
                "message": "Bad Gateway - The requested experiment exists in "
                "our database but cannot be found on the MLFlow Tracking backend. "
                "If this happens more than once, please file a bug report."
            },
            502,
        )

    @api.errorhandler(ExperimentMLFlowTrackingRegistrationError)
    def handle_mlflow_tracking_experiment_registration_error(error):
        return (
            {
                "message": "Bad Gateway - Experiment registration to the MLFlow "
                "Tracking backend has failed. If this happens more than once, please "
                "file a bug report."
            },
            502,
        )

    @api.errorhandler(ExperimentRegistrationError)
    def handle_experiment_registration_error(error):
        return (
            {
                "message": "Bad Request - The experiment registration form contains "
                "invalid parameters. Please verify and resubmit."
            },
            400,
        )
