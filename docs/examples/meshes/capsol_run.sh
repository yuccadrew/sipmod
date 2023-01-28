#!/bin/bash

# myInvocation="$(printf %q "$BASH_SOURCE")$((($#)) && printf ' %q' "$@")"
# SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

home=`pwd`
work="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
capsol=`which capsol`

cd $work
$capsol
cd $home
