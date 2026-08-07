# Running on the NFSv4.2 H200 nodes

Half the H200 nodes on `jsteinhardt` appear to hang: a job sits `RUNNING` with a GPU
held and emits nothing for 20-35 minutes. They are not broken, and they should not be
`--exclude`d.

## Why

Those nodes mount `oz.berkeley.edu:/pool0/scratch` over **NFSv4.2**; the A100 nodes
mount the *same export* over **NFSv3**. Verified with `findmnt`:

| node | mount | ms per small file | sequential |
|---|---|---|---|
| balrog (A100) | `nfs` **vers=3** | 2.1 | 112 MB/s |
| sneetches (H200) | `nfs4` **vers=4.2** | 768 (cold) | 63.7 MB/s |
| mooney (H200) | `nfs4` **vers=4.2** | **1487** (cold) | 82.8 MB/s |

Bandwidth is fine. **Per-file latency is 300-700x worse**, and the `prot-evo` conda env
is 7.6 GB across **74,446 files**. So `import torch` alone from the shared env on
sneetches takes **631 s**, and reading the whole env extrapolates to 16-41 hours.
That is the entire "hang".

Numbers move 3-20x with page-cache state - a warm re-read of the same files reported
4.0 GB/s sequential, which is obviously cache and not the network. Treat all of these
as order-of-magnitude.

## Fix: stage the env to node-local NVMe, once per node

The bottleneck is round-trips, not bytes, so collapse 74,446 of them into one. Local
`/tmp` is 347 GB of NVMe at ~830 MB/s and **persists between jobs**.

```bash
# once, ever - build the archive (5.7 GB from 7.6 GB / 74,446 files, ~685 s)
sbatch benchmarks/staging/build_env_tarball.sh

# in any job script targeting an H200 node
source benchmarks/staging/stage_env.sh   # exports $PY
"$PY" benchmarks/inference/run_workload.py --repo . --workload generate ...
```

Measured on sneetches:

| step | cost |
|---|---|
| stage archive -> node-local NVMe | **172 s** (once per node) |
| `import torch, esm, flash_attn` + CUDA init | **2 s** (was 631 s) |
| full generation benchmark, both batch sizes | 98 s |

After staging, sneetches matches a near node exactly: **32.9 seq/s at batch 64 and
245.5 at batch 1024**, against cubbins' 32.3 / 225-248. The GPUs were never the issue.

`stage_env.sh` takes an `flock` so concurrent jobs on one node cannot interleave
extractions, reuses an existing stage, and falls back to the shared env on failure.
`STAGE_ROOT=/dev/shm/peint_env` targets the 1 TB tmpfs instead, at the cost of RAM.

## Two things that will silently defeat this

- **`conda-pack` is not needed** - CPython derives `sys.prefix` from the executable
  path, so a relocated env works. But console-script shebangs still point at
  `/scratch`: use `"$PY" -m pytest`, never `pytest`.
- **`cwd` precedes `PYTHONPATH` in `sys.path`.** Launching from a scratch worktree
  re-imports the package from `/scratch` and undoes the staging without any error.

For inference the repo's `.py` files and the 219 MB checkpoint did *not* need staging
(a full benchmark ran in 98 s with them on `/scratch`), but that is a narrow test -
training imports considerably more.

## Diagnosing a suspected case

```bash
findmnt -no SOURCE,FSTYPE,OPTIONS --target /scratch/...   # vers=3 vs vers=4.2
```
then time ~60 small reads (`benchmarks/staging/mount_probe.sh` does both). Do **not**
trust `ps`'s aggregate `wchan`; read `/proc/<pid>/task/*/wchan` and ignore the `srun`
wrapper, whose idle `futex_wait`/`do_poll` looks like a network stall.

`node_probe.sh` characterises a node without touching conda; `e2e_sneetches.sh` is the
end-to-end validation (stage, import, shared-env control, real benchmark).

## Why it matters

This makes `sneetches` and `mooney` usable, taking H200 capacity on `jsteinhardt` from
2 nodes to 4. At 274 seq/s per H200 that is the largest throughput lever currently
available - larger than anything left in the model code.
