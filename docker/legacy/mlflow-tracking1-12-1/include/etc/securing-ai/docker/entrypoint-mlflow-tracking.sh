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

# Created by argbash-init v2.8.1
# ARG_OPTIONAL_SINGLE([conda-env],[],[Conda environment],[mitre-securing-ai])
# ARG_OPTIONAL_SINGLE([backend-store-uri],[],[URI to which to persist experiment and run data. Acceptable URIs are\nSQLAlchemy-compatible database connection strings (e.g. 'sqlite:///path/to/file.db')\nor local filesystem URIs (e.g. 'file:///absolute/path/to/directory').],[sqlite:////work/mlruns/mlflow-tracking.db])
# ARG_OPTIONAL_SINGLE([default-artifact-root],[],[Local or S3 URI to store artifacts, for new experiments. Note that this flag does\nnot impact already-created experiments. Default: Within file store, if a file:/\nURI is provided. If a sql backend is used, then this option is required.],[file:///work/artifacts])
# ARG_OPTIONAL_SINGLE([gunicorn-opts],[],[Additional command line options forwarded to gunicorn processes.],[])
# ARG_OPTIONAL_SINGLE([host],[],[The network address to listen on. Use 0.0.0.0 to bind to all addresses if you want to access the tracking server from other machines.],[0.0.0.0])
# ARG_OPTIONAL_SINGLE([port],[],[The port to listen on.],[5000])
# ARG_OPTIONAL_SINGLE([workers],[],[Number of gunicorn worker processes to handle requests.],[4])
# ARG_OPTIONAL_ACTION([upgrade-db],[],[Upgrade the database schema],[upgrade_database])
# ARG_DEFAULTS_POS()
# ARGBASH_SET_INDENT([  ])
# ARG_HELP([MLFlow Tracking Server Entry Point\n])"
# ARGBASH_PREPARE()
# needed because of Argbash --> m4_ignore([
### START OF CODE GENERATED BY Argbash v2.10.0 one line above ###
# Argbash is a bash code generator used to get arguments parsing right.
# Argbash is FREE SOFTWARE, see https://argbash.io for more info


die()
{
  local _ret="${2:-1}"
  test "${_PRINT_HELP:-no}" = yes && print_help >&2
  echo "$1" >&2
  exit "${_ret}"
}


begins_with_short_option()
{
  local first_option all_short_options='h'
  first_option="${1:0:1}"
  test "$all_short_options" = "${all_short_options/$first_option/}" && return 1 || return 0
}

# THE DEFAULTS INITIALIZATION - OPTIONALS
_arg_conda_env="mitre-securing-ai"
_arg_backend_store_uri="sqlite:////work/mlruns/mlflow-tracking.db"
_arg_default_artifact_root="file:///work/artifacts"
_arg_gunicorn_opts=
_arg_host="0.0.0.0"
_arg_port="5000"
_arg_workers="4"


print_help()
{
  printf '%s\n' "MLFlow Tracking Server Entry Point
"
  printf 'Usage: %s [--conda-env <arg>] [--backend-store-uri <arg>] [--default-artifact-root <arg>] [--gunicorn-opts <arg>] [--host <arg>] [--port <arg>] [--workers <arg>] [--upgrade-db] [-h|--help]\n' "$0"
  printf '\t%s\n' "--conda-env: Conda environment (default: 'mitre-securing-ai')"
  printf '\t%s\n' "--backend-store-uri: URI to which to persist experiment and run data. Acceptable URIs are
		SQLAlchemy-compatible database connection strings (e.g. 'sqlite:///path/to/file.db')
		or local filesystem URIs (e.g. 'file:///absolute/path/to/directory'). (default: 'sqlite:////work/mlruns/mlflow-tracking.db')"
  printf '\t%s\n' "--default-artifact-root: Local or S3 URI to store artifacts, for new experiments. Note that this flag does
		not impact already-created experiments. Default: Within file store, if a file:/
		URI is provided. If a sql backend is used, then this option is required. (default: 'file:///work/artifacts')"
  printf '\t%s\n' "--gunicorn-opts: Additional command line options forwarded to gunicorn processes. (no default)"
  printf '\t%s\n' "--host: The network address to listen on. Use 0.0.0.0 to bind to all addresses if you want to access the tracking server from other machines. (default: '0.0.0.0')"
  printf '\t%s\n' "--port: The port to listen on. (default: '5000')"
  printf '\t%s\n' "--workers: Number of gunicorn worker processes to handle requests. (default: '4')"
  printf '\t%s\n' "--upgrade-db: Upgrade the database schema"
  printf '\t%s\n' "-h, --help: Prints help"
}


parse_commandline()
{
  while test $# -gt 0
  do
    _key="$1"
    case "$_key" in
      --conda-env)
        test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
        _arg_conda_env="$2"
        shift
        ;;
      --conda-env=*)
        _arg_conda_env="${_key##--conda-env=}"
        ;;
      --backend-store-uri)
        test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
        _arg_backend_store_uri="$2"
        shift
        ;;
      --backend-store-uri=*)
        _arg_backend_store_uri="${_key##--backend-store-uri=}"
        ;;
      --default-artifact-root)
        test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
        _arg_default_artifact_root="$2"
        shift
        ;;
      --default-artifact-root=*)
        _arg_default_artifact_root="${_key##--default-artifact-root=}"
        ;;
      --gunicorn-opts)
        test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
        _arg_gunicorn_opts="$2"
        shift
        ;;
      --gunicorn-opts=*)
        _arg_gunicorn_opts="${_key##--gunicorn-opts=}"
        ;;
      --host)
        test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
        _arg_host="$2"
        shift
        ;;
      --host=*)
        _arg_host="${_key##--host=}"
        ;;
      --port)
        test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
        _arg_port="$2"
        shift
        ;;
      --port=*)
        _arg_port="${_key##--port=}"
        ;;
      --workers)
        test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
        _arg_workers="$2"
        shift
        ;;
      --workers=*)
        _arg_workers="${_key##--workers=}"
        ;;
      --upgrade-db)
        upgrade_database
        exit 0
        ;;
      -h|--help)
        print_help
        exit 0
        ;;
      -h*)
        print_help
        exit 0
        ;;
      *)
        _PRINT_HELP=yes die "FATAL ERROR: Got an unexpected argument '$1'" 1
        ;;
    esac
    shift
  done
}


# OTHER STUFF GENERATED BY Argbash

### END OF CODE GENERATED BY Argbash (sortof) ### ])
# [ <-- needed because of Argbash

shopt -s extglob
set -euo pipefail

###########################################################################################
# Global parameters
###########################################################################################

readonly conda_dir="${CONDA_DIR}"
readonly mlflow_s3_endpoint_url="${MLFLOW_S3_ENDPOINT_URL-}"
readonly logname="Container Entry Point"

set_parsed_globals() {
  readonly backend_store_uri="${_arg_backend_store_uri}"
  readonly conda_env="${_arg_conda_env}"
  readonly default_artifact_root="${_arg_default_artifact_root}"
  readonly gunicorn_opts="${_arg_gunicorn_opts}"
  readonly host_address="${_arg_host}"
  readonly port="${_arg_port}"
  readonly workers="${_arg_workers}"
}

###########################################################################################
# Create bucket on S3 storage
#
# Globals:
#   default_artifact_root
#   mlflow_s3_endpoint_url
# Arguments:
#   None
# Returns:
#   None
###########################################################################################

s3_mb() {
  local proto=$(echo ${default_artifact_root} | grep :// | sed -e "s,^\(.*://\).*,\1,g")
  local url=$(echo ${default_artifact_root/${proto}/})

  echo "${logname}: artifacts storage backend protocol is ${proto}"

  if [[ ${proto} == s3:// && ! -z ${mlflow_s3_endpoint_url} && -f /usr/local/bin/s3-mb.sh ]]; then
    local bucket=${url%%/*}
    echo "${logname}: artifacts storage path is ${default_artifact_root}, ensuring bucket ${bucket} exists"
    /usr/local/bin/s3-mb.sh --endpoint-url ${mlflow_s3_endpoint_url} ${bucket}
  elif [[ ${proto} == s3:// && -z ${mlflow_s3_endpoint_url} && -f /usr/local/bin/s3-mb.sh ]]; then
    local bucket=${url%%/*}
    echo "${logname}: artifacts storage path is ${default_artifact_root}, ensuring bucket ${bucket} exists"
    /usr/local/bin/s3-mb.sh ${bucket}
  elif [[ ! -f /usr/local/bin/s3-mb.sh ]]; then
    echo "${logname}: ERROR - /usr/local/bin/s3-mb.sh script missing" 1>&2
    exit 1
  else
    echo "${logname}: artifacts storage path is ${default_artifact_root}"
  fi
}

###########################################################################################
# Upgrade the MLFlow Tracking database
#
# Globals:
#   backend_store_uri
#   conda_dir
#   conda_env
# Arguments:
#   None
# Returns:
#   None
###########################################################################################

upgrade_database() {
  echo "${logname}: INFO - Upgrading the MLFlow Tracking database"

  set_parsed_globals

  bash -c "\
  source ${conda_dir}/etc/profile.d/conda.sh &&\
  conda activate ${conda_env} &&\
  mlflow db upgrade ${backend_store_uri}"
}

###########################################################################################
# Start MLFlow Tracking server
#
# Globals:
#   backend_store_uri
#   conda_dir
#   conda_env
#   default_artifact_root
#   gunicorn_opts
#   host_address
#   port
#   workers
# Arguments:
#   None
# Returns:
#   None
###########################################################################################

start_mlflow_server() {
  local optional_kwargs=""

  if [[ ! -z ${gunicorn_opts} ]]; then
    optional_kwargs="${optional_kwargs} --gunicorn-opts ${gunicorn_opts}"
  fi

  echo "${logname}: starting mlflow tracking server"
  echo "${logname}: mlflow server options -\
    --backend-store-uri ${backend_store_uri}\
    --default-artifact-root ${default_artifact_root}\
    --host ${host_address}\
    --port ${port}\
    --workers ${workers}\
    ${optional_kwargs}" |
    tr -s "[:blank:]"

  bash -c "\
  source ${conda_dir}/etc/profile.d/conda.sh &&\
  conda activate ${conda_env} &&\
  mlflow server\
    --backend-store-uri ${backend_store_uri}\
    --default-artifact-root ${default_artifact_root}\
    --host ${host_address}\
    --port ${port}\
    --workers ${workers}\
    ${optional_kwargs}"
}

###########################################################################################
# Main script
###########################################################################################

parse_commandline "$@"
set_parsed_globals
s3_mb
start_mlflow_server
# ] <-- needed because of Argbash
