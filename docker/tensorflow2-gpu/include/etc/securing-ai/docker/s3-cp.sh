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
# ARG_OPTIONAL_SINGLE([endpoint-url],[],[Endpoint URL for S3 storage],[])
# ARG_POSITIONAL_SINGLE([source],[URI or filepath to a file],[])
# ARG_POSITIONAL_SINGLE([destination],[URI or filepath to a file],[])
# ARG_DEFAULTS_POS()
# ARGBASH_SET_INDENT([  ])
# ARG_HELP([Copy file to/from S3 storage\n])"
# ARGBASH_GO()
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

# THE DEFAULTS INITIALIZATION - POSITIONALS
_positionals=()
_arg_source=
_arg_destination=
# THE DEFAULTS INITIALIZATION - OPTIONALS
_arg_endpoint_url=


print_help()
{
  printf '%s\n' "Copy file to/from S3 storage
"
  printf 'Usage: %s [--endpoint-url <arg>] [-h|--help] <source> <destination>\n' "$0"
  printf '\t%s\n' "<source>: URI or filepath to a file"
  printf '\t%s\n' "<destination>: URI or filepath to a file"
  printf '\t%s\n' "--endpoint-url: Endpoint URL for S3 storage (no default)"
  printf '\t%s\n' "-h, --help: Prints help"
}


parse_commandline()
{
  _positionals_count=0
  while test $# -gt 0
  do
    _key="$1"
    case "$_key" in
      --endpoint-url)
        test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
        _arg_endpoint_url="$2"
        shift
        ;;
      --endpoint-url=*)
        _arg_endpoint_url="${_key##--endpoint-url=}"
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
        _last_positional="$1"
        _positionals+=("$_last_positional")
        _positionals_count=$((_positionals_count + 1))
        ;;
    esac
    shift
  done
}


handle_passed_args_count()
{
  local _required_args_string="'source' and 'destination'"
  test "${_positionals_count}" -ge 2 || _PRINT_HELP=yes die "FATAL ERROR: Not enough positional arguments - we require exactly 2 (namely: $_required_args_string), but got only ${_positionals_count}." 1
  test "${_positionals_count}" -le 2 || _PRINT_HELP=yes die "FATAL ERROR: There were spurious positional arguments --- we expect exactly 2 (namely: $_required_args_string), but got ${_positionals_count} (the last one was: '${_last_positional}')." 1
}


assign_positional_args()
{
  local _positional_name _shift_for=$1
  _positional_names="_arg_source _arg_destination "

  shift "$_shift_for"
  for _positional_name in ${_positional_names}
  do
    test $# -gt 0 || break
    eval "$_positional_name=\${1}" || die "Error during argument parsing, possibly an Argbash bug." 1
    shift
  done
}

parse_commandline "$@"
handle_passed_args_count
assign_positional_args 1 "${_positionals[@]}"

# OTHER STUFF GENERATED BY Argbash

### END OF CODE GENERATED BY Argbash (sortof) ### ])
# [ <-- needed because of Argbash

shopt -s extglob
set -euo pipefail

###########################################################################################
# Global parameters
###########################################################################################

readonly destination="${_arg_destination}"
readonly endpoint_url="${_arg_endpoint_url}"
readonly logname="S3 Copy"
readonly source="${_arg_source}"

###########################################################################################
# Copy file to/from S3 storage
#
# Globals:
#   destination
#   endpoint_url
#   logname
#   source
# Arguments:
#   None
# Returns:
#   None
###########################################################################################

s3_cp() {
  echo "${logname}: ${source} to ${destination}"

  if [[ ! -z ${endpoint_url} ]]; then
    echo "${logname}: custom endpoint URL ${endpoint_url}"

    aws --endpoint-url ${endpoint_url} s3 cp ${source} ${destination}
  else
    echo "${logname}: default endpoint URL"

    aws s3 cp ${source} ${destination}
  fi
}

###########################################################################################
# Main script
###########################################################################################

s3_cp
# ] <-- needed because of Argbash
