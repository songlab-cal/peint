"""Run one PEINT inference workload against a chosen copy of ``protevo``.

This single script is used for *both* the baseline and the optimized tree — the
``--repo`` flag decides which ``protevo`` package gets imported. That is what
makes the before/after numbers and the parity gate trustworthy: identical
instrumentation, identical workload construction, only the package under test
differs.

    # baseline (pristine release tree)
    python benchmarks/inference/run_workload.py --repo ../peint \
        --workload generate --mode bench --out results/base_generate

    # optimized (this tree)
    python benchmarks/inference/run_workload.py --repo . \
        --workload generate --mode bench --out results/fast_generate

``--mode dump`` writes deterministic outputs (logits / NLLs / hit lists) instead
of timings; ``parity.py`` diffs two dumps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _select_protevo(repo: str) -> str:
    """Put ``repo`` at the front of ``sys.path`` and verify ``protevo`` came from it."""
    repo = os.path.abspath(repo)
    if not os.path.isdir(os.path.join(repo, "protevo")):
        raise SystemExit(f"--repo {repo} does not contain a protevo/ package")
    sys.path.insert(0, repo)

    import protevo  # noqa: F401  (import side effect is the point)

    got = os.path.abspath(protevo.__file__)
    if not got.startswith(repo + os.sep):
        raise SystemExit(f"protevo resolved to {got}, expected it under {repo}")
    print(f"[run_workload] protevo <- {got}")
    return repo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True,
                    help="Repo root whose protevo/ package to benchmark (e.g. . or ../peint)")
    ap.add_argument("--workload", required=True,
                    choices=["generate", "likelihood", "homology", "logits"])
    ap.add_argument("--mode", default="bench", choices=["bench", "dump"],
                    help="bench = timings; dump = deterministic outputs for parity")
    ap.add_argument("--checkpoint", default=None,
                    help="Defaults to <repo>/model_checkpoints/peint.ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-flash", action="store_true",
                    help="Force the Vanilla (non-Flash) path")
    ap.add_argument("--batch-sizes", default="1,8,32,64",
                    help="Comma-separated batch sizes to sweep (bench mode)")
    ap.add_argument("--n-targets", type=int, default=2048,
                    help="Corpus size for the likelihood workload")
    ap.add_argument("--n-sequences", type=int, default=200,
                    help="Corpus size for the homology all-vs-all workload")
    ap.add_argument("--decode-steps", type=int, default=None,
                    help="Override max_decode_steps (default 2 x source length)")
    ap.add_argument("--length-jitter", type=float, default=0.0,
                    help="Ragged-length fraction for the synthetic corpus")
    ap.add_argument("--corpus-seed", type=int, default=0)
    ap.add_argument("--num-gpus", type=int, default=1,
                    help="Shard the homology workload across N GPUs (optimized tree only)")
    ap.add_argument("--seed", type=int, default=0, help="Torch RNG seed")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", required=True, help="Output path prefix (no extension)")
    args = ap.parse_args()

    repo = _select_protevo(args.repo)
    sys.path.insert(1, _HERE)

    import numpy as np
    import torch

    import harness
    import workloads

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_flash = not args.no_flash
    checkpoint = args.checkpoint or os.path.join(repo, "model_checkpoints", "peint.ckpt")
    if not os.path.exists(checkpoint):
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    x_seq, y_seq, t = workloads.load_fixture(repo)
    meta = harness.env_metadata()
    meta.update({
        "repo": repo,
        "num_gpus": args.num_gpus,
        "workload": args.workload,
        "mode": args.mode,
        "checkpoint": checkpoint,
        "use_flash": use_flash,
        "seed": args.seed,
        "corpus_seed": args.corpus_seed,
        "length_jitter": args.length_jitter,
        "src_len": len(x_seq),
        "fixture_time": t,
    })
    print(f"[run_workload] {json.dumps(meta)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b]
    rows = []

    # ---------------------------------------------------------------- logits
    if args.workload == "logits":
        # Deterministic teacher-forced logits: the strictest parity payload and
        # directly comparable to the golden protevo/tests/y_logits.npy.
        model, vocab = workloads.build_model(checkpoint, "standard", device, use_flash)
        logits = workloads.teacher_forced_logits(model, vocab, x_seq, y_seq, t, device)
        np.save(f"{args.out}.npy", logits)
        with open(f"{args.out}.meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"[run_workload] wrote {args.out}.npy shape={logits.shape}")
        return

    # ------------------------------------------------------------- generate
    if args.workload == "generate":
        model, vocab = workloads.build_model(checkpoint, "generator", device, use_flash)
        if args.mode == "dump":
            # p=0.0 selects argmax inside sampling_function -> fully deterministic.
            run, info = workloads.generation_callable(
                model, vocab, x_seq, t, batch_size=8, device=device,
                max_decode_steps=args.decode_steps, p=0.0,
            )
            seqs = run()
            with open(f"{args.out}.json", "w") as fh:
                json.dump({"metadata": meta, "info": info, "sequences": seqs}, fh, indent=2)
            print(f"[run_workload] wrote {args.out}.json ({len(seqs)} sequences)")
            return

        for bs in batch_sizes:
            run, info = workloads.generation_callable(
                model, vocab, x_seq, t, batch_size=bs, device=device,
                max_decode_steps=args.decode_steps, p=1.0,
            )
            mem = harness.peak_memory(run, device)
            timing = harness.cuda_timeit(run, args.warmup, args.iters, device)
            steps = info["decode_steps"]
            row = workloads.workload_metadata("generate", {"batch_size": bs, **info})
            row.update(timing)
            row.update(mem)
            row["seq_per_s"] = bs / (timing["mean_ms"] / 1e3)
            row["tok_per_s"] = bs * steps / (timing["mean_ms"] / 1e3)
            row["ms_per_step"] = timing["mean_ms"] / steps
            rows.append(row)
            print(f"[run_workload] generate bs={bs}: {timing['mean_ms']:.1f} ms, "
                  f"{row['tok_per_s']:.0f} tok/s, {mem['peak_reserved_mib']:.0f} MiB")

    # ----------------------------------------------------------- likelihood
    elif args.workload == "likelihood":
        model, _ = workloads.build_model(checkpoint, "evaluator", device, use_flash)
        corpus = workloads.make_corpus(
            y_seq, args.n_targets, seed=args.corpus_seed, length_jitter=args.length_jitter
        )
        targets = [s for _, s in corpus]

        if args.mode == "dump":
            run, info = workloads.likelihood_callable(model, x_seq, targets, t, 32, device)
            nlls = np.atleast_1d(run())
            np.save(f"{args.out}.npy", np.asarray(nlls, dtype=np.float64))
            with open(f"{args.out}.meta.json", "w") as fh:
                json.dump({"metadata": meta, "info": info}, fh, indent=2)
            print(f"[run_workload] wrote {args.out}.npy ({nlls.shape[0]} NLLs)")
            return

        for bs in batch_sizes:
            run, info = workloads.likelihood_callable(model, x_seq, targets, t, bs, device)
            mem = harness.peak_memory(run, device)
            timing = harness.cuda_timeit(run, args.warmup, args.iters, device)
            row = workloads.workload_metadata("likelihood", info)
            row.update(timing)
            row.update(mem)
            row["seq_per_s"] = len(targets) / (timing["mean_ms"] / 1e3)
            rows.append(row)
            print(f"[run_workload] likelihood bs={bs}: {timing['mean_ms']:.1f} ms, "
                  f"{row['seq_per_s']:.0f} seq/s, {mem['peak_reserved_mib']:.0f} MiB")

    # ------------------------------------------------------------- homology
    elif args.workload == "homology":
        corpus = workloads.make_corpus(
            y_seq, args.n_sequences, seed=args.corpus_seed, length_jitter=args.length_jitter
        )

        if args.mode == "dump":
            run, info = workloads.homology_callable(checkpoint, corpus, args.device, 32,
                                                    num_gpus=args.num_gpus)
            hits = run()
            payload = sorted(
                ({"query_id": h.query_id, "db_id": h.db_id, "score": float(h.score)}
                 for h in hits),
                key=lambda d: (d["db_id"], d["query_id"]),
            )
            with open(f"{args.out}.json", "w") as fh:
                json.dump({"metadata": meta, "info": info, "hits": payload}, fh, indent=2)
            print(f"[run_workload] wrote {args.out}.json ({len(payload)} hits)")
            return

        for bs in batch_sizes:
            run, info = workloads.homology_callable(checkpoint, corpus, args.device, bs,
                                                    num_gpus=args.num_gpus)
            if args.num_gpus > 1:
                # Memory is allocated inside the spawned ranks, so the parent's
                # allocator sees nothing. Reporting 0 here would be a lie; the
                # per-rank figure is the single-GPU number for its shard.
                mem = {"peak_allocated_mib": float("nan"),
                       "peak_reserved_mib": float("nan")}
            else:
                mem = harness.peak_memory(run, device)
            # All-vs-all is expensive; one timed pass is enough at large N.
            # No warmup for the sharded path either - a warmup call would pay the
            # process spawn and per-rank model load a second time.
            timing = harness.cuda_timeit(run, warmup=0, iters=max(1, args.iters // 3),
                                         device=device)
            n = len(corpus)
            row = workloads.workload_metadata("homology", info)
            row.update(timing)
            row.update(mem)
            row["pairs"] = n * (n - 1)
            row["pairs_per_s"] = row["pairs"] / (timing["mean_ms"] / 1e3)
            rows.append(row)
            print(f"[run_workload] homology N={n} bs={bs}: {timing['mean_ms'] / 1e3:.1f} s, "
                  f"{row['pairs_per_s']:.0f} pairs/s, {mem['peak_reserved_mib']:.0f} MiB")

    harness.save_report(rows, args.out, metadata=meta)


if __name__ == "__main__":
    main()
