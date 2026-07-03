#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

compiler=$(which $PHC_COMPILER)

# Pass the actual compiler to PHC via an environment variable
export PHC_COMPILER=$compiler
# Make sure that sccache includes the real compiler in the cache hash.
# Also add the scripts that we've written for good measure.
export SCCACHE_EXTRAFILES="$compiler:$script_dir/phc.py"
# We no longer need it now that we're passing it explicitly
shift
# Invoke sccache now.
exec sccache $script_dir/phc.py $@
