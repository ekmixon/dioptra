#!/bin/bash
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

# m4_ignore(
echo "This is just a script template, not the script (yet) - pass it to 'argbash' to fix this." >&2
exit 11 #)Created by argbash-init v2.8.1
# ARG_OPTIONAL_SINGLE([app-module],[],[Application module],[wsgi:app])
# ARG_OPTIONAL_SINGLE([backend],[],[Server backend],[gunicorn])
# ARG_OPTIONAL_SINGLE([conda-env],[],[Conda environment],[mitre-securing-ai])
# ARG_OPTIONAL_SINGLE([gunicorn-module],[],[Python module used to start Gunicorn WSGI server],[mitre.securingai.restapi.cli.gunicorn])
# ARG_OPTIONAL_ACTION([upgrade-db],[],[Upgrade the database schema],[upgrade_database])
# ARG_DEFAULTS_POS
# ARGBASH_SET_INDENT([  ])
# ARG_HELP([Securing AI Testbed API Entry Point\n])"
# ARGBASH_PREPARE

# [ <-- needed because of Argbash
shopt -s extglob
set -euo pipefail

###########################################################################################
# Global parameters
###########################################################################################

readonly ai_workdir="${AI_WORKDIR}"
readonly conda_dir="${CONDA_DIR}"
readonly gunicorn_module="${_arg_gunicorn_module}"
readonly logname="Container Entry Point"

set_parsed_globals() {
  readonly app_module="${_arg_app_module}"
  readonly conda_env="${_arg_conda_env}"
  readonly server_backend="${_arg_backend}"
}

###########################################################################################
# Secure the container at runtime
#
# Globals:
#   logname
# Arguments:
#   None
# Returns:
#   None
###########################################################################################

secure_container() {
  if [[ -f /usr/local/bin/secure-container.sh ]]; then
    /usr/local/bin/secure-container.sh
  else
    echo "${logname}: ERROR - /usr/local/bin/secure-container.sh script missing" 1>&2
    exit 1
  fi
}

###########################################################################################
# Upgrade the Securing AI database
#
# Globals:
#   ai_workdir
#   conda_dir
#   conda_env
#   logname
# Arguments:
#   None
# Returns:
#   None
###########################################################################################

upgrade_database() {
  echo "${logname}: INFO - Upgrading the Securing AI database"

  set_parsed_globals

  bash -c "\
  source ${conda_dir}/etc/profile.d/conda.sh &&\
  conda activate ${conda_env} &&\
  cd ${ai_workdir} &&\
  flask db upgrade -d ${ai_workdir}/migrations"
}

###########################################################################################
# Start gunicorn server
#
# Globals:
#   ai_workdir
#   app_module
#   conda_dir
#   conda_env
#   gunicorn_module
#   logname
# Arguments:
#   None
# Returns:
#   None
###########################################################################################

start_gunicorn() {
  echo "${logname}: INFO - Starting gunicorn server"

  bash -c "\
  source ${conda_dir}/etc/profile.d/conda.sh &&\
  conda activate ${conda_env} &&\
  cd ${ai_workdir} &&\
  python -m ${gunicorn_module} -c /etc/gunicorn/gunicorn.conf.py ${app_module}"
}

###########################################################################################
# Start RESTful API service
#
# Globals:
#   logname
#   server_backend
# Arguments:
#   None
# Returns:
#   None
###########################################################################################

start_restapi() {
  case ${server_backend} in
    gunicorn)
      start_gunicorn
      ;;
    *)
      echo "${logname}: ERROR - unsupported backend - ${server_backend}" 1>&2
      exit 1
      ;;
  esac
}

###########################################################################################
# Main script
###########################################################################################

parse_commandline "$@"
set_parsed_globals
secure_container
start_restapi
# ] <-- needed because of Argbash
