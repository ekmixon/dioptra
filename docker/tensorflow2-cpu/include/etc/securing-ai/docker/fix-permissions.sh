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
#
# Script adapted from the work https://github.com/jupyter/docker-stacks/blob/0329eecec8739d1e175e1d9152b0b233c765ac30/base-notebook/fix-permissions.
# See copyright below.
#
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# Neither the name of the Jupyter Development Team nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Created by argbash-init v2.8.1
# ARG_POSITIONAL_INF([directory],[Path to a directory],[1])
# ARG_DEFAULTS_POS()
# ARGBASH_SET_INDENT([  ])
# ARG_HELP([Fix directory permissions\n])"
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
_arg_directory=('' )
# THE DEFAULTS INITIALIZATION - OPTIONALS


print_help()
{
  printf '%s\n' "Fix directory permissions
"
  printf 'Usage: %s [-h|--help] <directory-1> [<directory-2>] ... [<directory-n>] ...\n' "$0"
  printf '\t%s\n' "<directory>: Path to a directory"
  printf '\t%s\n' "-h, --help: Prints help"
}


parse_commandline()
{
  _positionals_count=0
  while test $# -gt 0
  do
    _key="$1"
    case "$_key" in
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
  local _required_args_string="'directory'"
  test "${_positionals_count}" -ge 1 || _PRINT_HELP=yes die "FATAL ERROR: Not enough positional arguments - we require at least 1 (namely: $_required_args_string), but got only ${_positionals_count}." 1
}


assign_positional_args()
{
  local _positional_name _shift_for=$1
  _positional_names="_arg_directory "
  _our_args=$((${#_positionals[@]} - 1))
  for ((ii = 0; ii < _our_args; ii++))
  do
    _positional_names="$_positional_names _arg_directory[$((ii + 1))]"
  done

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

readonly ai_gid="${AI_GID}"
readonly directories=${_arg_directory[@]}
readonly logname="Fix Permissions"

###########################################################################################
# Recursively set group ownership and permissions within a directory
#
# Globals:
#   ai_gid
#   logname
# Arguments:
#   Path to a directory
# Returns:
#   None
###########################################################################################

set_permissions() {
  local directory="${1}"

  echo "${logname}: set group ownership and permissions of ${directory} to GID=${ai_gid}"

  find "${directory}" \
    ! \( \
    -group ${ai_gid} \
    -a -perm -g+rwX \
    \) \
    -exec chgrp ${ai_gid} {} \; \
    -exec chmod g+rwX {} \;
}

###########################################################################################
# Recursively set setuid,setgid within a directory
#
# Globals:
#   logname
# Arguments:
#   Path to a directory
# Returns:
#   None
###########################################################################################

set_setuid_setgid() {
  local directory="${1}"

  echo "${logname}: set setuid,setgid permissions of ${directory} to +6000"

  find "${directory}" \
    \( \
    -type d \
    -a ! -perm -6000 \
    \) \
    -exec chmod +6000 {} \;
}

###########################################################################################
# Fix directory permissions
#
# After any installation, if a directory needs to be (human) user-writable, run this script
# on it. It will make everything in the directory owned by the group $AI_GID and writable
# by that group. Deployments that want to set a specific user id can preserve permissions
# by adding the `--group-add users` line to `docker run`.
#
# Uses find to avoid touching files that already have the right permissions, which would
# cause massive image explosion
#
# The right permissions are:
#   group=$AI_GID
#   AND permissions include group rwX (directory-execute)
#   AND directories have setuid,setgid bits set
#
# Globals:
#   directories
# Arguments:
#   Path to a directory
# Returns:
#   None
###########################################################################################

fix_permissions() {
  for directory in ${directories[@]}; do
    set_permissions ${directory}
    set_setuid_setgid ${directory}
  done
}

###########################################################################################
# Main script
###########################################################################################

fix_permissions
# ] <-- needed because of Argbash
