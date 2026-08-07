#!/bin/bash
# Stage the conda env onto node-local NVMe, once per node, and expose it.
#
# Why: the far H200 nodes reach /scratch at ~63 MB/s sequential but ~26 ms per
# file. The env is 7.6 GB in 74,446 files, so per-file latency alone is ~32
# minutes before a single byte of useful work - which is what made those jobs
# look hung. One archive turns 74k round-trips into a single streaming read,
# then extraction lands on local NVMe at ~830 MB/s.
#
# Source this from a job script, then use "$PY" instead of `python`:
#     source diag/stage_env.sh
#     "$PY" myscript.py
#
# Override STAGE_ROOT=/dev/shm/peint_env to extract into the 1 TB tmpfs instead
# of NVMe (faster still, but costs RAM and evaporates on reboot).

STAGE_ROOT=${STAGE_ROOT:-/tmp/peint_env}
TARBALL=${TARBALL:-/scratch/users/yufan.cao/env_stage/prot-evo.tar.gz}
ENV_LOCAL="$STAGE_ROOT/prot-evo"
MARKER="$ENV_LOCAL/.stage_complete"

mkdir -p "$STAGE_ROOT"

# Serialize stagers: several jobs can land on one node simultaneously, and two
# concurrent extractions into the same path would interleave and corrupt it.
exec 9>"$STAGE_ROOT/.stage.lock"
flock 9

if [ -f "$MARKER" ]; then
    echo "[stage] reusing $ENV_LOCAL (staged $(cat "$MARKER"))"
else
    echo "[stage] staging env to $ENV_LOCAL ..."
    t0=$(date +%s)
    rm -rf "$ENV_LOCAL"
    # Extract straight from the stream; never land the archive on local disk first.
    if pigz -dc "$TARBALL" | tar -C "$STAGE_ROOT" --numeric-owner -xf - ; then
        date -Iseconds > "$MARKER"
        echo "[stage] done in $(( $(date +%s) - t0 )) s"
    else
        echo "[stage] FAILED - falling back to the shared env" >&2
        rm -rf "$ENV_LOCAL"
    fi
fi
flock -u 9

if [ -f "$MARKER" ]; then
    export PY="$ENV_LOCAL/bin/python"
    export PATH="$ENV_LOCAL/bin:$PATH"
    export CONDA_PREFIX="$ENV_LOCAL"
    # The env was built at a different prefix, so console-script shebangs still
    # point at /scratch. Calling the interpreter directly is unaffected: CPython
    # derives sys.prefix from the executable's own path. Use "$PY" -m <module>
    # rather than the wrapper scripts.
else
    export PY=/scratch/users/yufan.cao/conda/envs/prot-evo/bin/python
    export PATH="/scratch/users/yufan.cao/conda/envs/prot-evo/bin:$PATH"
fi
echo "[stage] PY=$PY"
