#!/usr/bin/env python3
"""Unified KV-offload plan benchmark (offload-chain method).

Method (same for every candidate):
  T_base          original QLI
  T_ordered       ordered QLI (except plan_unordered, which uses T_base as upstream)
  T_chain         upstream QLI -> plan op, real data dependency
  plan-only       plan op on a fixed upstream output
Device view is collected with torch_npu.profiler (kernel P50); Event view is a
per-sample timing with state restore outside the Event window.

Every candidate .so is loaded in its own worker process (all Echo libraries
register the same torch.ops name).  Results are written per dataset:
  results/unified_glm.{json,md}
  results/unified_constructed.{json,md}

Usage:
  python3 bench_unified.py --orchestrate --device 2
  python3 bench_unified.py --orchestrate --device 2 --samples 10 --profile-steps 3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Profiler scratch on /dev/shm.
PROFILE_ROOT = Path(os.environ.get("UNION_PROFILE_ROOT")
                    or tempfile.mkdtemp(prefix="unionprof.", dir="/dev/shm"))
(PROFILE_ROOT / "tmp").mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = os.environ["TMP"] = os.environ["TEMP"] = str(PROFILE_ROOT / "tmp")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("ECHO_PATH2_ASYNC", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
import refs  # noqa: E402

METHODS = config.METHODS

PROFILE_KINDS = ("echo_k0", "echo_k1", "echo_k2", "echo_k3", "newop")


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------
def artifact_info(path: str):
    p = Path(path)
    if not p.exists():
        return {"path": path, "exists": False}
    return {
        "path": str(p),
        "exists": True,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "mtime": p.stat().st_mtime,
    }


def shape_key(batch, mtp, seq):
    return f"b{batch}_m{mtp}_s{seq}"


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------
def _classify(name: str):
    n = name.lower()
    if "echo_lru_init" in n:
        return "echo_init"
    if "echo_v2_kernel3" in n:
        return "echo_k3"
    if "echo_v2_kernel2" in n:
        return "echo_k2"
    if "echo_v2_sort" in n:
        return "echo_k0"
    if "echo_lru_kernel" in n:
        return "echo_k1"
    if "ordered_bitmap_512" in n:
        return "newop"
    if "quantlightning" in n or "quant_lightning" in n:
        return "qli"
    if "addaicore" in n or ("adds" in n and "add" in n):
        return "add"
    return None


def profile_kernels(fn, restore, steps: int, tag: str):
    """Run one profiler window and return per-category kernel stats (P50)."""
    import torch_npu
    trace_dir = PROFILE_ROOT / f"trace_{tag}_{uuid.uuid4().hex[:8]}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    schedule = torch_npu.profiler.schedule(wait=1, warmup=2, active=max(1, steps),
                                           repeat=1)
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=schedule,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir))) as prof:
        for _ in range(1 + 2 + max(1, steps)):
            if restore is not None:
                restore()
            fn()
            prof.step()
    torch_npu.npu.synchronize()

    by_name = {}
    for path in trace_dir.rglob("kernel_details.csv"):
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                name = row.get("Name") or row.get("Kernel Name") or ""
                dur = row.get("Duration(us)") or row.get("Duration(μs)")
                if not name or dur in (None, "", "N/A"):
                    continue
                try:
                    dur = float(dur)
                except ValueError:
                    continue
                by_name.setdefault(name, []).append(dur)
    shutil.rmtree(trace_dir, ignore_errors=True)

    cats = {}
    for name, vals in by_name.items():
        cat = _classify(name)
        if cat is None:
            continue
        arr = np.asarray(vals, np.float64)
        entry = cats.setdefault(cat, {"count": 0, "p50_us": 0.0,
                                      "mean_us": 0.0, "names": []})
        entry["count"] += int(arr.size)
        entry["p50_us"] += float(np.percentile(arr, 50))
        entry["mean_us"] += float(arr.mean())
        entry["names"].append(name)
    # Each category is the P50 sum over its kernels ("separate" fallback).
    for entry in cats.values():
        entry["names"] = sorted(set(entry["names"]))
    return {"by_category": cats,
            "by_name": {k: float(np.percentile(v, 50))
                        for k, v in by_name.items()}}


def plan_device_us(prof: dict):
    cats = prof.get("by_category", {})
    return sum(cats[k]["p50_us"] for k in PROFILE_KINDS if k in cats)


def qli_device_us(prof: dict):
    return prof.get("by_category", {}).get("qli", {}).get("p50_us", 0.0)


def add_device_us(prof: dict):
    return prof.get("by_category", {}).get("add", {}).get("p50_us", 0.0)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def import_ordered():
    from importlib import import_module
    return import_module(config.ORDER_MODULE).quant_lightning_indexer_ordered


def qli_fn_for(kind: str):
    import torch_npu
    if kind == "ordered":
        return import_ordered()
    return torch_npu.npu_quant_lightning_indexer


def run_worker(args):
    import torch_npu
    if not torch.npu.is_available():
        raise RuntimeError("torch_npu reports no available device")
    torch.npu.set_device(args.device)

    data = args.data
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    seqs = [int(x) for x in args.seqs.split(",") if x.strip()]
    mtps = [int(x) for x in args.mtps.split(",") if x.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    template_batch = max(batches)
    plan_op = None
    method_cfg = None
    if args.kind == "plan":
        method_cfg = METHODS[args.method]
        torch.ops.load_library(method_cfg["so"])
        if method_cfg["abi"] == "echo":
            plan_op = torch.ops.sparse_kv_plan_op.echo_lru
        else:
            plan_op = torch.ops.sparse_kv_plan_op.ordered_bitmap_512
        cores = int(torch.ops.sparse_kv_plan_op.echo_lru_core_count()) \
            if method_cfg["abi"] == "echo" else 64

    qli_kind = "original"
    if args.kind == "ordered":
        qli_kind = "ordered"
    if args.kind == "plan":
        qli_kind = method_cfg["qli"]
    qli_fn = qli_fn_for(qli_kind)
    print(f"[worker] data={data} kind={args.kind} method={args.method} "
          f"qli={qli_kind} device={args.device}", flush=True)

    out = {"data": data, "kind": args.kind, "method": args.method,
           "qli": qli_kind, "device": args.device,
           "batches": batches, "seqs": seqs, "mtps": mtps,
           "scenarios": scenarios, "artifact": None, "cases": []}
    if method_cfg is not None:
        out["artifact"] = artifact_info(method_cfg["so"])

    t_start = time.perf_counter()
    for seq in seqs:
        fixture = refs.make_fixture(data, seq, template_batch, args.device)
        print(f"[fixture:{data}] seq={seq} {fixture.key_mb:.0f} MB", flush=True)
        for batch in batches:
            for mtp in mtps:
                width = refs.WIDTH_OF_MTP[mtp]
                capacity = refs.CAP_OF_MTP[mtp]
                tokens = batch * width
                qli_kwargs, _, meta, _ = fixture.inputs(batch, width)
                skey = shape_key(batch, mtp, seq)

                if args.kind in ("base", "ordered"):
                    out_kind = qli_fn(**qli_kwargs)
                    torch.npu.synchronize()
                    refs.check_common(out_kind, tokens, meta["visible_seq_lens"])
                    if args.kind == "ordered":
                        base = torch_npu.npu_quant_lightning_indexer(**qli_kwargs)
                        torch.npu.synchronize()
                        arr = refs.check_ordered(out_kind)
                        refs.check_set_equals(arr, base[:, 0].cpu().numpy(),
                                              f"{skey} ordered vs base")
                    samples = refs.event_samples(
                        lambda: qli_fn(**qli_kwargs), None,
                        args.warmup, args.samples)
                    prof = profile_kernels(
                        lambda: qli_fn(**qli_kwargs), None, args.profile_steps,
                        f"{data}_{skey}_{args.kind}")
                    out["cases"].append({
                        "shape": skey, "batch": batch, "mtp": mtp,
                        "width": width, "seq_len": seq, "capacity": capacity,
                        "rows": tokens,
                        "event": refs.stats_us(samples),
                        "profile": prof,
                        "qli_device_us": qli_device_us(prof),
                    })
                    print(f"[{args.kind}] {skey} event_p50="
                          f"{refs.stats_us(samples)['p50_us']:.1f}us "
                          f"kernel_p50={qli_device_us(prof):.1f}us", flush=True)
                    continue

                # ---- plan worker ----
                out_up = qli_fn(**qli_kwargs)
                torch.npu.synchronize()
                refs.check_common(out_up, tokens, meta["visible_seq_lens"])
                if method_cfg["qli"] == "ordered":
                    arr = refs.check_ordered(out_up)
                else:
                    arr = out_up[:, 0].cpu().numpy()
                for scenario in scenarios:
                    tag = f"{args.method}_{data}_{skey}_{scenario}"
                    t0 = time.perf_counter()
                    if method_cfg["abi"] == "echo":
                        case, initial, pstats = refs.build_echo_case(
                            arr, meta, width, capacity, seq + 1, scenario)
                        input_hit = refs.hit_rate_of(
                            case["topk_indices"], case["slot_to_token"], width)
                        blocks = refs.pick_blocks(batch, cores)
                        state, outputs, workspace, spl, reset = \
                            refs.allocate_echo_state(torch, initial, case,
                                                     args.device, blocks)
                        pos = out_up.view(tokens, refs.TOPK) + 1
                        golden = refs.check_echo_golden(
                            plan_op, pos, spl, reset, state, outputs, workspace,
                            blocks, initial, case, args.device,
                            method_cfg["ref"])

                        def plan_only():
                            refs.echo_plan_call(plan_op, pos, spl, reset, state,
                                                outputs, workspace, blocks)

                        def chain():
                            o = qli_fn(**qli_kwargs)
                            p = o.view(tokens, refs.TOPK) + 1
                            refs.echo_plan_call(plan_op, p, spl, reset, state,
                                                outputs, workspace, blocks)

                        def restore():
                            refs.restore_echo_state(state, initial, args.device)
                    else:
                        np_state, pstats = refs.build_newop_case(
                            arr, meta, width, capacity, seq, scenario)
                        input_hit = refs.hit_rate_of(
                            np_state["topk_indices"],
                            np_state["slot_to_token"][0::width], width)
                        state, outputs, max_token_attr, block_size = \
                            refs.allocate_newop_state(torch, np_state, args.device)
                        pos_view = out_up.view(tokens, refs.TOPK)
                        golden = refs.check_newop_golden(
                            plan_op, state, outputs, max_token_attr, block_size,
                            np_state)

                        def plan_only():
                            refs.newop_plan_call(plan_op, state, outputs,
                                                 max_token_attr, block_size,
                                                 topk=pos_view)

                        def chain():
                            o = qli_fn(**qli_kwargs)
                            refs.newop_plan_call(plan_op, state, outputs,
                                                 max_token_attr, block_size,
                                                 topk=o.view(tokens, refs.TOPK))

                        def restore():
                            refs.restore_newop_state(state, np_state, args.device)

                    plan_samples = refs.event_samples(
                        plan_only, restore, args.warmup, args.samples)
                    chain_samples = refs.event_samples(
                        chain, restore, args.warmup, args.samples)
                    prof = profile_kernels(chain, restore, args.profile_steps, tag)
                    pdev = plan_device_us(prof)
                    qdev = qli_device_us(prof)
                    adev = add_device_us(prof)
                    row = {
                        "shape": skey, "batch": batch, "mtp": mtp,
                        "width": width, "seq_len": seq, "capacity": capacity,
                        "rows": tokens, "scenario": scenario,
                        "golden": golden, "stats": pstats,
                        "input_hit_rate": input_hit,
                        "plan_event": refs.stats_us(plan_samples),
                        "chain_event": refs.stats_us(chain_samples),
                        "profile": prof,
                        "plan_device_us": pdev,
                        "qli_device_us": qdev,
                        "add_device_us": adev,
                        "build_s": round(time.perf_counter() - t0, 2),
                    }
                    out["cases"].append(row)
                    print(f"[plan:{args.method}] {skey}/{scenario} "
                          f"golden={'PASS' if golden['ok'] else 'FAIL'} "
                          f"plan_k={pdev:.1f} chain_k={qdev + adev + pdev:.1f} "
                          f"plan_e={row['plan_event']['p50_us']:.1f} "
                          f"chain_e={row['chain_event']['p50_us']:.1f}", flush=True)
                    torch.npu.synchronize()
        fixture.release()
        torch.npu.empty_cache()

    out["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    part = Path(args.part_out)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_text(json.dumps(out, indent=2))
    print(f"[worker done] {part} in {out['elapsed_s']}s", flush=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _run(cmd, env_script=None):
    if env_script:
        import shlex
        inner = shlex.join(cmd)
        cmd = ["bash", "-c", f'source "{env_script}" >/dev/null 2>&1 && exec {inner}']
    print("[run]", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    subprocess.run(cmd, check=True)


def orchestrate(args):
    requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    missing = {m: METHODS[m]["so"] for m in requested
               if not Path(METHODS[m]["so"]).exists()}
    if missing:
        lines = ["missing candidate artifacts:"]
        for m, p in missing.items():
            env = {"plan": "KV_PLAN_SO", "plan_deferred": "KV_PLAN_DEFERRED_SO",
                   "plan_unordered": "KV_PLAN_UNORDERED_SO", "newop": "KV_NEWOP_SO"}[m]
            lines.append(f"  {m}: {p}   (override with {env}=...)")
        raise SystemExit("\n".join(lines))
    if not Path(config.ORDER_ENV).exists():
        raise SystemExit(f"ordered-QLE env not found: {config.ORDER_ENV}\n"
                         f"  override with KV_ORDER_ENV=...")

    results = Path(args.out_dir)
    parts = results / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    batches = args.batches
    seqs = args.seqs
    mtps = args.mtps
    scenarios = args.scenarios
    py = sys.executable

    for data in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        common = [py, str(HERE / "bench_unified.py"), "--worker",
                  "--data", data, "--device", str(args.device),
                  "--batches", batches, "--seqs", seqs, "--mtps", mtps,
                  "--scenarios", scenarios, "--warmup", str(args.warmup),
                  "--samples", str(args.samples),
                  "--profile-steps", str(args.profile_steps)]
        base_part = parts / f"{data}_base.json"
        _run(common + ["--kind", "base", "--part-out", str(base_part)])
        ordered_part = parts / f"{data}_ordered.json"
        _run(common + ["--kind", "ordered", "--part-out", str(ordered_part)],
             env_script=config.ORDER_ENV)

        for method in requested:
            part = parts / f"{data}_plan_{method}.json"
            env_script = config.ORDER_ENV if METHODS[method]["qli"] == "ordered" else None
            _run(common + ["--kind", "plan", "--method", method,
                           "--part-out", str(part)],
                 env_script=env_script)

    summarize(args)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _load_parts(data: str, out_dir: Path):
    parts = out_dir / "parts"
    base_path = parts / f"{data}_base.json"
    ordered_path = parts / f"{data}_ordered.json"
    if not (base_path.exists() and ordered_path.exists()):
        raise SystemExit(
            f"missing raw parts under {parts} for data={data}.\n"
            "  `results/parts/` is gitignored and only exists after a run.\n"
            "  Run the orchestrator first, or read the committed "
            "results/unified_*.{md,json} from the reference run.")
    base = json.loads(base_path.read_text())
    ordered = json.loads(ordered_path.read_text())
    plans = {}
    for path in sorted(parts.glob(f"{data}_plan_*.json")):
        obj = json.loads(path.read_text())
        plans[obj["method"]] = obj
    return base, ordered, plans


def _index(cases, key="shape"):
    return {c[key]: c for c in cases}


def summarize(args):
    out_dir = Path(args.out_dir)
    for data in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        base, ordered, plans = _load_parts(data, out_dir)
        base_idx = _index(base["cases"])
        ord_idx = _index(ordered["cases"])
        rows = []
        artifacts = {"base": {"path": "torch_npu.npu_quant_lightning_indexer"},
                     "ordered": {"path": "candi vendor ordered QLI"},
                     "plans": {m: p.get("artifact") for m, p in plans.items()}}
        for method, obj in plans.items():
            for case in obj["cases"]:
                skey = case["shape"]
                b = base_idx.get(skey)
                o = ord_idx.get(skey)
                t_base = b["event"]["p50_us"] if b else None
                t_ordered = o["event"]["p50_us"] if o else None
                chain_e = case["chain_event"]["p50_us"]
                offload_e = chain_e - t_base if t_base is not None else None
                offload_d = case["plan_device_us"] + case["add_device_us"]
                rows.append({
                    "method": method, **case,
                    "t_base_event_p50_us": t_base,
                    "t_ordered_event_p50_us": t_ordered,
                    "base_kernel_p50_us": b["qli_device_us"] if b else None,
                    "ordered_kernel_p50_us": o["qli_device_us"] if o else None,
                    "offload_event_us": offload_e,
                    "offload_device_us": offload_d,
                })
        result = {
            "dataset": data,
            "method": "offload_chain",
            "device": base["device"],
            "batches": base["batches"], "seqs": base["seqs"],
            "mtps": base["mtps"], "scenarios": base["scenarios"],
            "artifacts": artifacts,
            "base": base["cases"], "ordered": ordered["cases"],
            "results": rows,
        }
        (out_dir / f"unified_{data}.json").write_text(json.dumps(result, indent=2))
        (out_dir / f"unified_{data}.md").write_text(render_md(result))
        print(f"[summary] {out_dir / f'unified_{data}.json'}", flush=True)


def render_md(result):
    data = result["dataset"]
    lines = [f"# Unified KV offload benchmark — `{data}` data", "",
             "Method: offload chain (`T_base` / `T_ordered` / "
             "`QLI -> plan`), device kernel P50 from torch_npu.profiler; "
             "Event P50 with state restore outside the window.", "",
             "Artifacts:"]
    for m, a in result["artifacts"]["plans"].items():
        if a and a.get("exists"):
            lines.append(f"- `{m}`: `{a['path']}` sha256=`{a['sha256'][:16]}…`")
    scenarios = result["scenarios"]
    for scenario in scenarios:
        lines += ["", f"## {scenario}", ""]
        lines.append("| shape | method | hit% | golden | plan dev (μs) | chain dev (μs) | "
                     "offload dev (μs) | plan event (μs) | chain event (μs) | "
                     "offload event (μs) |")
        lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for r in result["results"]:
            if r["scenario"] != scenario:
                continue
            g = "PASS" if r["golden"]["ok"] else "FAIL:" + ",".join(r["golden"]["mismatch"])
            chain_d = r["qli_device_us"] + r["add_device_us"] + r["plan_device_us"]
            lines.append(
                f"| {r['shape']} | {r['method']} | {100.0 * r['input_hit_rate']:.1f} | {g} | "
                f"{r['plan_device_us']:.1f} | {chain_d:.1f} | "
                f"{r['offload_device_us']:.1f} | "
                f"{r['plan_event']['p50_us']:.1f} | "
                f"{r['chain_event']['p50_us']:.1f} | "
                f"{r['offload_event_us']:.1f} |")
        # Pivot: plan device / offload device per method.
        methods = sorted({r["method"] for r in result["results"]})
        shapes = sorted({r["shape"] for r in result["results"]})
        by = {(r["shape"], r["method"]): r for r in result["results"]
              if r["scenario"] == scenario}
        lines += ["", "Pivot (`plan_dev / offload_dev`, μs):", "",
                  "| shape | " + " | ".join(methods) + " |",
                  "|---|" + "---:|" * len(methods)]
        for skey in shapes:
            cells = []
            for m in methods:
                r = by.get((skey, m))
                cells.append("—" if r is None else
                             f"{r['plan_device_us']:.1f} / {r['offload_device_us']:.1f}")
            lines.append(f"| {skey} | " + " | ".join(cells) + " |")

    # Mean over batch x seq for each (scenario, mtp, method).
    lines += ["", "## Averages (mean over batch×seq, plan dev μs)", "",
              "| scenario | mtp | " + " | ".join(sorted({r["method"] for r in result["results"]})) + " |",
              "|---|---|" + "---:|" * len({r["method"] for r in result["results"]})]
    methods = sorted({r["method"] for r in result["results"]})
    for scenario in scenarios:
        for mtp in sorted({r["mtp"] for r in result["results"]}):
            cells = []
            for m in methods:
                vals = [r["plan_device_us"] for r in result["results"]
                        if r["scenario"] == scenario and r["mtp"] == mtp
                        and r["method"] == m]
                cells.append("—" if not vals else f"{sum(vals) / len(vals):.1f}")
            lines.append(f"| {scenario} | {mtp} | " + " | ".join(cells) + " |")
    lines += ["", "## Upstream (per shape)", "",
              "| shape | base Event (μs) | ordered Event (μs) | base kernel (μs) | "
              "ordered kernel (μs) |", "|---|---:|---:|---:|---:|"]
    base_idx = _index(result["base"])
    ord_idx = _index(result["ordered"])
    for skey in sorted(base_idx):
        b, o = base_idx[skey], ord_idx.get(skey)
        lines.append(f"| {skey} | {b['event']['p50_us']:.1f} | "
                     f"{(o['event']['p50_us'] if o else float('nan')):.1f} | "
                     f"{b['qli_device_us']:.1f} | "
                     f"{(o['qli_device_us'] if o else float('nan')):.1f} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orchestrate", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--kind", choices=("base", "ordered", "plan"))
    p.add_argument("--method", choices=tuple(METHODS))
    p.add_argument("--data", choices=("glm", "constructed"))
    p.add_argument("--device", type=int, default=2)
    p.add_argument("--datasets", default="glm,constructed")
    p.add_argument("--methods", default="plan,plan_deferred,plan_unordered")
    p.add_argument("--batches", default="8,16")
    p.add_argument("--seqs", default="102400,358400")
    p.add_argument("--mtps", default="0,1,3")
    p.add_argument("--scenarios", default="cold_reset,full_cache_90pct")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--profile-steps", type=int, default=3)
    p.add_argument("--out-dir", default=str(HERE / "results"))
    p.add_argument("--part-out", default=None)
    return p


def main():
    args = build_parser().parse_args()
    if args.worker:
        if not args.part_out:
            raise SystemExit("--part-out required for --worker")
        run_worker(args)
        return
    if args.summarize_only:
        summarize(args)
        return
    if args.orchestrate:
        orchestrate(args)
        return
    build_parser().print_help()


if __name__ == "__main__":
    main()
