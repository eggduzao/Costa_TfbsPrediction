#!/bin/bash
__doc__="
This script simply runs the smith doctests via the xdoctest runner.

This must be run from the root of the smith repo, as it needs the path to the
smith source code.

This script is provided as a developer convenience. On the CI the doctests are
invoked in 'run_test.py'
"
# To simply list tests
# xdoctest -m smith --style=google list

# Reference: https://stackoverflow.com/questions/59895/bash-script-dir
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SMITH_MODPATH=$SCRIPT_DIR/../smith
echo "SMITH_MODPATH = $SMITH_MODPATH"

if [[ ! -d "$SMITH_MODPATH" ]] ; then
    echo "Could not find the path to the smith module"
else
    export XDOCTEST_GLOBAL_EXEC="from smith import nn\nimport smith.nn.functional as F\nimport smith"
    export XDOCTEST_OPTIONS="+IGNORE_WHITESPACE"
    # Note: google won't catch numpy style docstrings (a few exist) but it also won't fail
    # on things not intended to be doctests.
    export XDOCTEST_STYLE="google"
    xdoctest smith "$SMITH_MODPATH" --style="$XDOCTEST_STYLE" --global-exec "$XDOCTEST_GLOBAL_EXEC" --options="$XDOCTEST_OPTIONS"
fi
