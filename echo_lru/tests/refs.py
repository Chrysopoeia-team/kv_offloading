#!/usr/bin/env python3
"""Pure-host references and input fixtures for the unified KV-offload test.

This module never touches an NPU.  It provides:

* the two QLI input families of the offload matrix:
  - ``glm``          : GLM-5.2 C8 fixture
  - ``constructed``  : the deterministic synthetic ranker
* plan-case/state builders for the Echo ABI and for the new
  ``1120_ordered_bitmap_512`` ABI;
* CPU references (vendored under ``ref/``):
  - ``plan_reference``           (admission order = token ascending);
  - ``plan_unordered_reference`` (admission order = flat first occurrence);
  - a direct reference for the ordered-bitmap joint kernel.

No candidate source is modified.  This module only reads vendored files; its
fixture templates are cached under ``/dev/shm/union_kv_fixtures/``.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

import config

# ---------------------------------------------------------------------------
# Constants of the offload matrix
# ---------------------------------------------------------------------------
TOPK = 2048
PAGE = 128
DIM = 128
HEADS = 32
FP8_MAX = 448.0
SEED = 17
I32_MAX = np.iinfo(np.int32).max
BLOCK_SIZE = 64            # block table size used by the plan operator
WIDTH_OF_MTP = {0: 1, 1: 2, 3: 4}
CAP_OF_MTP = {0: 4096, 1: 4096, 3: 8192}


def load_module(name: str, path: Path):
    """Load a standalone .py file under a private module name (read-only)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Two CPU references with different admission-order contracts.
PLAN_COMPARE = load_module("plan_reference", config.PLAN_REFERENCE)
PLAN_UNORDERED_COMPARE = load_module("plan_unordered_reference",
                                     config.PLAN_UNORDERED_REFERENCE)


def reference_for(method: str):
    """Return (make_echo_initial, reference_echo) for the method's contract."""
    if method == "plan_unordered":
        return (PLAN_UNORDERED_COMPARE.make_echo_initial,
                PLAN_UNORDERED_COMPARE.reference_echo)
    return PLAN_COMPARE.make_echo_initial, PLAN_COMPARE.reference_echo


# ---------------------------------------------------------------------------
# GLM-5.2 C8 fixture.
# ---------------------------------------------------------------------------
def cpu_quantize(values: torch.Tensor):
    flat = values.float().reshape(-1, values.shape[-1])
    scale = flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / FP8_MAX
    quantized = (flat / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quantized.reshape(values.shape), scale.reshape(values.shape[:-1]).float()


def page_bytes() -> int:
    return DIM * PAGE + 4 * PAGE


def make_tables(batch: int, seq_len: int, device, seed: int = SEED):
    pages = math.ceil(seq_len / PAGE)
    blocks = batch * pages
    gen = torch.Generator(device="cpu").manual_seed(seed)
    physical = torch.randperm(blocks, generator=gen).to(torch.int32)
    qli_table = physical.reshape(batch, pages).to(device)
    host_table = torch.arange(blocks, dtype=torch.int32, device=device).reshape(batch, pages)
    return qli_table, host_table, blocks, pages


def build_key_template_cpu(batch_max: int, seq_len: int, seed: int = SEED):
    pages = math.ceil(seq_len / PAGE)
    blocks = batch_max * pages
    pb = page_bytes()
    raw = torch.zeros((blocks, pb), dtype=torch.uint8)
    key = raw.view(torch.float8_e4m3fn).as_strided(
        (blocks, PAGE, 1, DIM), (pb, DIM, DIM, 1))
    scales = raw.view(torch.float32).as_strided(
        (blocks, PAGE, 1), (pb // 4, 1, 1), storage_offset=(PAGE * DIM) // 4)

    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    chunk = min(256, blocks)
    for start in range(0, blocks, chunk):
        count = min(chunk, blocks - start)
        source = torch.randn((count, PAGE, 1, DIM), generator=gen, dtype=torch.float32)
        quantized, scale = cpu_quantize(source)
        key[start:start + count].copy_(quantized)
        scales[start:start + count].copy_(scale)
    return raw, pages


def key_template_views(raw, device):
    pb = page_bytes()
    blocks = raw.shape[0]
    raw_dev = raw.to(device)
    key_dev = raw_dev.view(torch.float8_e4m3fn).as_strided(
        (blocks, PAGE, 1, DIM), (pb, DIM, DIM, 1))
    scales_dev = raw_dev.view(torch.float32).as_strided(
        (blocks, PAGE, 1), (pb // 4, 1, 1), storage_offset=(PAGE * DIM) // 4)
    return key_dev, scales_dev


FIXTURE_CACHE = Path("/dev/shm/union_kv_fixtures")


def _cache_load(name: str):
    path = FIXTURE_CACHE / name
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def _cache_save(name: str, obj):
    FIXTURE_CACHE.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_CACHE / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def build_key_template(batch_max: int, seq_len: int, device, seed: int = SEED):
    name = f"glm_{seq_len}_{batch_max}_{seed}.pt"
    cached = _cache_load(name)
    if cached is None:
        raw, pages = build_key_template_cpu(batch_max, seq_len, seed)
        _cache_save(name, raw)
    else:
        raw = cached
        pages = math.ceil(seq_len / PAGE)
    pb = page_bytes()
    raw_dev = raw.to(device)
    blocks = batch_max * pages
    key_dev = raw_dev.view(torch.float8_e4m3fn).as_strided(
        (blocks, PAGE, 1, DIM), (pb, DIM, DIM, 1))
    scales_dev = raw_dev.view(torch.float32).as_strided(
        (blocks, PAGE, 1), (pb // 4, 1, 1), storage_offset=(PAGE * DIM) // 4)
    return raw_dev, key_dev, scales_dev, pages


def build_query_side(batch: int, width: int, seq_len: int, device, seed: int = SEED):
    rows = batch * width
    gen = torch.Generator(device="cpu").manual_seed(seed + 2)
    base = torch.randn((batch, 1, HEADS, DIM), generator=gen)
    lane = torch.randn((batch, width, HEADS, DIM), generator=gen)
    previous = (base + 0.05 * lane).to(torch.bfloat16).reshape(rows, HEADS, DIM)
    delta = torch.randn((batch, 1, HEADS, DIM), generator=gen)
    current = (base + 0.05 * lane + 0.15 * delta).to(torch.bfloat16).reshape(rows, HEADS, DIM)
    query, qscale = cpu_quantize(current)
    old_query, old_scale = cpu_quantize(previous)

    weight_gen = torch.Generator(device="cpu").manual_seed(seed + 3)
    w_base = torch.randn((batch, 1, HEADS), generator=weight_gen)
    w_lane = torch.randn((batch, width, HEADS), generator=weight_gen)
    weights = (w_base + 0.05 * w_lane).reshape(rows, HEADS)
    weights = (weights / math.sqrt(HEADS * DIM)).to(torch.bfloat16)

    q_ends = torch.arange(1, batch + 1, dtype=torch.int32) * width
    key_lengths = torch.full((batch,), seq_len, dtype=torch.int32)
    visible = torch.arange(1, width + 1, dtype=torch.int32).repeat(batch) + seq_len - width
    stable = (visible - width).clamp_min(0)
    token_to_req = torch.arange(batch, dtype=torch.int32).repeat_interleave(width)
    req_ids = torch.arange(batch, dtype=torch.int64)

    return {
        "query": query.to(device),
        "previous_query": old_query.to(device),
        "query_dequant_scale": qscale.to(device),
        "previous_query_dequant_scale": old_scale.to(device),
        "weights": weights.to(device),
        "actual_seq_lengths_query": q_ends.to(device),
        "actual_seq_lengths_key": key_lengths.to(device),
        "visible_seq_lens": visible.to(device),
        "stable_prefix_lens": stable.to(device),
        "token_to_req": token_to_req.to(device),
        "req_ids": req_ids.to(device),
        "rows": rows,
    }


class GlmFixture:
    """One (seq, device) fixture."""

    def __init__(self, seq_len: int, template_batch: int, device, seed: int = SEED):
        self.seq_len = seq_len
        self.template_batch = template_batch
        self.device = device
        self.seed = seed
        self.raw, self.key, self.scales, self.pages = build_key_template(
            template_batch, seq_len, device, seed)
        self.key_mb = self.raw.numel() / 2**20

    def inputs(self, batch: int, width: int):
        qli_table, host_table, blocks, pages = make_tables(
            batch, self.seq_len, self.device, self.seed)
        assert pages == self.pages
        assert blocks <= self.template_batch * self.pages
        qs = build_query_side(batch, width, self.seq_len, self.device, self.seed)
        qli = {
            "query": qs["query"],
            "key": self.key[:blocks],
            "weights": qs["weights"],
            "query_dequant_scale": qs["query_dequant_scale"],
            "key_dequant_scale": self.scales[:blocks],
            "actual_seq_lengths_query": qs["actual_seq_lengths_query"],
            "actual_seq_lengths_key": qs["actual_seq_lengths_key"],
            "block_table": qli_table,
            "query_quant_mode": 0,
            "key_quant_mode": 0,
            "layout_query": "TND",
            "layout_key": "PA_BSND",
            "sparse_count": TOPK,
            "sparse_mode": 3,
            "pre_tokens": 2**63 - 1,
            "next_tokens": 2**63 - 1,
        }
        meta = {
            "req_ids": qs["req_ids"],
            "stable_prefix_lens": qs["stable_prefix_lens"],
            "visible_seq_lens": qs["visible_seq_lens"],
            "token_to_req": qs["token_to_req"],
            "host_block_table": host_table,
            "host_num_blocks": blocks,
            "pages": pages,
        }
        return qli, qli, meta, qs

    def release(self):
        del self.key, self.scales, self.raw
        self.key = self.scales = self.raw = None


class ConstructedFixture:
    """Deterministic synthetic ranker.

    Request-major identity page tables let one max-batch K template be sliced.
    """

    def __init__(self, seq_len: int, template_batch: int, device, seed: int = SEED,
                 group_size: int = 64):
        self.seq_len = seq_len
        self.template_batch = template_batch
        self.device = device
        self.group_size = group_size
        self.seed = seed
        self.pages = math.ceil(seq_len / PAGE)
        total = template_batch * self.pages
        cache_name = (f"constructed_{seq_len}_{template_batch}_{seed}_"
                      f"{group_size}.pt")
        cached = _cache_load(cache_name)
        if cached is None:
            key_cpu = torch.zeros((total, PAGE, 1, DIM), dtype=torch.float8_e4m3fn)
            scale_cpu = torch.ones((total, PAGE, 1), dtype=torch.float32)
            mult = torch.arange(1, 2 * DIM, 2, dtype=torch.int64)
            chunk_pages = min(256, self.pages)
            for r in range(template_batch):
                base = r * self.pages
                for start in range(0, self.pages, chunk_pages):
                    count = min(chunk_pages, self.pages - start)
                    tok0 = start * PAGE
                    t = torch.arange(tok0, tok0 + count * PAGE, dtype=torch.int64)
                    vals = ((t[:, None] * mult[None, :] + r * 37) % 251 - 125).to(torch.float32)
                    key_cpu[base + start:base + start + count] = \
                        vals.reshape(count, PAGE, 1, DIM).to(torch.float8_e4m3fn)
                    scale_cpu[base + start:base + start + count] = \
                        (1.0 + t.to(torch.float32) / float(seq_len)).reshape(count, PAGE, 1)
            _cache_save(cache_name, (key_cpu, scale_cpu))
        else:
            key_cpu, scale_cpu = cached
        self.key = key_cpu.to(device)
        self.scale = scale_cpu.to(device)
        self.key_mb = (self.key.numel() + self.scale.numel() * 4) / 2**20

    def inputs(self, batch: int, width: int):
        rows = batch * width
        npages = self.pages
        blocks = batch * npages
        table = torch.arange(blocks, dtype=torch.int32,
                             device=self.device).reshape(batch, npages)
        key = self.key[:blocks]
        key_scale = self.scale[:blocks]

        query = torch.zeros((rows, self.group_size, DIM), dtype=torch.float8_e4m3fn)
        for row in range(rows):
            lane = row % width
            query[row, :, (lane * 37) % DIM] = 1
        query = query.to(self.device)
        weights = torch.ones((rows, self.group_size), dtype=torch.bfloat16,
                             device=self.device)
        qscale = torch.ones((rows, self.group_size), dtype=torch.float32,
                            device=self.device)
        q_ends = (torch.arange(1, batch + 1, dtype=torch.int32) * width).to(self.device)
        key_lengths = torch.full((batch,), self.seq_len, dtype=torch.int32,
                                 device=self.device)
        visible = torch.full((rows,), self.seq_len, dtype=torch.int32,
                             device=self.device)
        stable = torch.full((rows,), self.seq_len - width, dtype=torch.int32,
                            device=self.device)
        token_to_req = torch.arange(batch, dtype=torch.int32,
                                    device=self.device).repeat_interleave(width)
        req_ids = torch.arange(batch, dtype=torch.int64, device=self.device)

        qli = {
            "query": query,
            "key": key,
            "weights": weights,
            "query_dequant_scale": qscale,
            "key_dequant_scale": key_scale,
            "actual_seq_lengths_query": q_ends,
            "actual_seq_lengths_key": key_lengths,
            "block_table": table,
            "query_quant_mode": 0,
            "key_quant_mode": 0,
            "layout_query": "TND",
            "layout_key": "PA_BSND",
            "sparse_count": TOPK,
            "sparse_mode": 3,
            "pre_tokens": 2**63 - 1,
            "next_tokens": 2**63 - 1,
        }
        meta = {
            "req_ids": req_ids,
            "stable_prefix_lens": stable,
            "visible_seq_lens": visible,
            "token_to_req": token_to_req,
            "host_block_table": table,
            "host_num_blocks": blocks,
            "pages": npages,
        }
        qs = {"visible_seq_lens": visible}
        return qli, qli, meta, qs

    def release(self):
        del self.key, self.scale
        self.key = self.scale = None


def make_fixture(data: str, seq_len: int, template_batch: int, device):
    if data == "glm":
        return GlmFixture(seq_len, template_batch, device)
    if data == "constructed":
        return ConstructedFixture(seq_len, template_batch, device)
    raise ValueError(data)


# ---------------------------------------------------------------------------
# QLI checks
# ---------------------------------------------------------------------------
def check_common(out: torch.Tensor, rows: int, visible: torch.Tensor):
    if tuple(out.shape) != (rows, 1, TOPK) or out.dtype != torch.int32:
        raise AssertionError(f"unexpected TopK shape/dtype: {tuple(out.shape)} {out.dtype}")
    idx = out[:, 0]
    vis = visible.to(idx.device).reshape(-1)
    if not torch.all((idx >= 0) & (idx < vis[:, None])):
        raise AssertionError("TopK contains invalid / causally masked token IDs")
    flat = idx.reshape(-1, TOPK)
    s = flat.sort(dim=-1).values
    if torch.any(s[:, 1:] == s[:, :-1]):
        raise AssertionError("TopK contains duplicate token IDs")
    return idx


def check_ordered(out: torch.Tensor) -> np.ndarray:
    """Ordered QLI contract: no -1 tail, strictly increasing per row."""
    arr = out[:, 0].cpu().numpy()
    if (arr < 0).any():
        raise AssertionError("ordered output has a -1 tail")
    if np.any(np.diff(arr, axis=-1) <= 0):
        raise AssertionError("ordered output is not strictly increasing")
    return arr


def check_set_equals(a: np.ndarray, b: np.ndarray, label: str):
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape mismatch {a.shape} vs {b.shape}")
    for row in range(a.shape[0]):
        if not np.array_equal(np.sort(a[row]), np.sort(b[row])):
            diff = np.setdiff1d(np.union1d(a[row], b[row]),
                                np.intersect1d(a[row], b[row]))
            raise AssertionError(f"{label}: row {row} selected sets differ ({diff.size} ids)")


# ---------------------------------------------------------------------------
# Echo plan case / state
# ---------------------------------------------------------------------------
def echo_workspace_elements(requests: int, tokens: int, topk: int,
                            capacity: int, blocks: int) -> int:
    """Workspace upper bound, in int32 elements."""
    npad = 1 << capacity.bit_length()
    mask_row = 3 * ((capacity + 31) // 32)
    base = requests * ((capacity + 1) + 4) + tokens * (3 * topk)
    base = (base + 7) & ~7
    base += requests * (mask_row + 8)
    base += blocks * 2
    base = (base + 7) & ~7
    base += requests * (2 * capacity) + requests
    elements = ((base + 1) & ~1) + blocks * (4 * npad + 64 * 256 + 64 * 256 + 256 + 8192)
    return int(elements)


def hit_rate_of(topk_plan: np.ndarray, resident_rows: np.ndarray,
                positions: int) -> float:
    """Fraction of query positions whose token is resident before the call."""
    topk_plan = np.asarray(topk_plan)
    resident_rows = np.asarray(resident_rows)
    requests = resident_rows.shape[0]
    hits = 0
    total = 0
    for r in range(requests):
        pool = set(int(t) for t in resident_rows[r] if t >= 0)
        row = topk_plan[r * positions:(r + 1) * positions]
        hits += sum(1 for t in row.reshape(-1) if int(t) in pool)
        total += row.size
    return float(hits) / float(total) if total else 0.0


def pick_blocks(requests: int, cores: int) -> int:
    budget = min(requests, max(1, cores))
    per_core = (requests + budget - 1) // budget
    return (requests + per_core - 1) // per_core


def _choose_resident_tokens(union: np.ndarray, fraction: float, capacity: int,
                            full_cache: bool, domain_lo: int, domain_hi: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Build a packed resident array from a query union.

    ``domain_lo`` is inclusive, ``domain_hi`` exclusive.  For Echo the domain is
    ``[1, max_token)`` (token 0 reserved); for the ordered-bitmap op it is
    ``[0, max_token)``.
    """
    union = np.unique(union)
    union = union[(union >= domain_lo) & (union < domain_hi)]
    n_res = int(round(fraction * union.size))
    n_res = min(n_res, capacity)
    perm = rng.permutation(union)
    out = np.full(capacity, -1, np.int32)
    out[:n_res] = perm[:n_res]
    if full_cache and n_res < capacity:
        fill = capacity - n_res
        domain = np.setdiff1d(np.arange(domain_lo, domain_hi, dtype=np.int32),
                              union, assume_unique=False)
        if domain.size < fill:
            raise ValueError("token domain too small to fill cache")
        out[n_res:] = rng.choice(domain, fill, replace=False)
    return out


def build_echo_case(topk_logical: np.ndarray, meta, positions: int,
                    capacity: int, max_token: int, scenario: str,
                    seed: int = SEED):
    """Build the case + initial state for an Echo-ABI method.

    ``topk_logical`` is the QLI output in the *QLI* token domain (0-based).
    Echo tokens are shifted +1 (token 0 reserved).
    Returns ``(case, initial, stats)``.
    """
    topk_logical = np.asarray(topk_logical, np.int32)
    tokens, topk = topk_logical.shape
    requests = tokens // positions
    plan_tokens = topk_logical + 1
    req_ids = np.asarray(meta["req_ids"].detach().cpu(), np.int64)
    cold = scenario == "cold_reset"
    rng = np.random.default_rng(seed)

    resident = np.full((requests, capacity), -1, np.int32)
    union_sizes = []
    if not cold:
        for r in range(requests):
            u = np.unique(plan_tokens[r * positions:(r + 1) * positions])
            union_sizes.append(int(u.size))
            resident[r] = _choose_resident_tokens(
                u, 0.90, capacity, True, 1, max_token, rng)
    reset_mask = (np.ones(requests, np.uint8) if cold
                  else np.zeros(requests, np.uint8))
    case = {
        "scenario": scenario,
        "rows": requests,
        "topk": topk,
        "capacity": capacity,
        "max_token": max_token,
        "hits": int(round(float(np.mean(union_sizes)))) if union_sizes else 0,
        "mtp": positions - 1,
        "positions": positions,
        "num_tokens": tokens,
        "req_ids": req_ids,
        "last_req_ids": (np.full(tokens, -1, np.int64) if cold
                         else np.repeat(req_ids, positions)),
        "topk_indices": plan_tokens,
        "stable_prefix_lens": np.full(requests, max_token, np.int32),
        "visible_seq_lens": np.full(tokens, max_token, np.int32),
        "token_to_req": np.repeat(np.arange(requests, dtype=np.int32), positions),
        "slot_to_token": resident,
        "lru_slots": np.tile(np.arange(capacity, dtype=np.int32), (requests, 1)),
        "reset_mask": reset_mask,
    }
    make_initial, _ = reference_for("plan")    # all variants agree on the shape
    initial = make_initial(case)
    initial["fresh"] = np.zeros(requests, np.uint8)   # decode-time reset flag off
    stats = {
        "union_size_mean": float(np.mean(union_sizes)) if union_sizes else 0.0,
        "union_size_max": int(np.max(union_sizes)) if union_sizes else 0,
        "resident_per_request": int(np.count_nonzero(resident[0] >= 0)),
    }
    return case, initial, stats


def allocate_echo_state(torch_mod, initial, case, device, blocks):
    def up(a):
        return torch_mod.from_numpy(np.ascontiguousarray(a)).to(device)

    state = {k: up(initial[k]) for k in PLAN_COMPARE.ECHO_STATE}
    state["fresh"] = up(initial["fresh"])
    requests, topk, capacity = case["rows"], case["topk"], case["capacity"]
    tokens = case["num_tokens"]
    positions = case["positions"]
    outputs = {
        "current_slots": torch_mod.empty((tokens, topk), dtype=torch_mod.int32,
                                         device=device),
        "miss_count": torch_mod.empty((requests,), dtype=torch_mod.int32,
                                      device=device),
        "miss_tokens": torch_mod.empty((requests, positions * topk),
                                       dtype=torch_mod.int32, device=device),
        "miss_slots": torch_mod.empty((requests, positions * topk),
                                      dtype=torch_mod.int32, device=device),
    }
    workspace = torch_mod.empty(
        echo_workspace_elements(requests, tokens, topk, capacity, blocks),
        dtype=torch_mod.int32, device=device)
    spl = torch_mod.full((requests,), case["max_token"], dtype=torch_mod.int32,
                         device=device)
    reset = up(case["reset_mask"])
    return state, outputs, workspace, spl, reset


def echo_plan_call(plan_op, pos, spl, reset, state, outputs, workspace, blocks):
    plan_op(pos, spl, reset,
            state["htd"], state["dth"], state["lru"], state["free_slots"],
            state["avail"], state["fifo"], state["fresh"],
            outputs["current_slots"], outputs["miss_count"],
            outputs["miss_tokens"], outputs["miss_slots"],
            workspace, blocks, False)


def restore_echo_state(state, initial, device):
    for key, arr in initial.items():
        state[key].copy_(torch.from_numpy(np.ascontiguousarray(arr)).to(device))


def check_echo_golden(plan_op, pos, spl, reset, state, outputs, workspace,
                      blocks, initial, case, device, method: str):
    """One call from the pristine state vs the method's CPU reference."""
    make_initial, ref = reference_for(method)
    expected = ref(case, make_initial(case))
    plan_op(pos, spl, reset,
            state["htd"], state["dth"], state["lru"], state["free_slots"],
            state["avail"], state["fifo"], state["fresh"],
            outputs["current_slots"], outputs["miss_count"],
            outputs["miss_tokens"], outputs["miss_slots"],
            workspace, blocks, False)
    torch.npu.synchronize()
    bad = []
    for key in PLAN_COMPARE.ECHO_STATE:
        got = state[key].cpu().numpy()
        if not np.array_equal(got, expected[key]):
            bad.append(f"state/{key}")
    for key in PLAN_COMPARE.ECHO_OUTPUT:
        got = outputs[key].cpu().numpy()
        if not np.array_equal(got, expected[key]):
            bad.append(f"out/{key}")
    restore_echo_state(state, initial, device)
    torch.npu.synchronize()
    return {"ok": not bad, "mismatch": bad}


# ---------------------------------------------------------------------------
# Ordered-bitmap (1120_ordered_bitmap_512) case / state / reference
# ---------------------------------------------------------------------------
def build_newop_case(topk_logical: np.ndarray, meta, positions: int,
                     capacity: int, max_token: int, scenario: str,
                     seed: int = SEED, block_size: int = PAGE):
    """Build the joint-group state for the ordered-bitmap op (no token shift)."""
    topk_logical = np.asarray(topk_logical, np.int32)
    tokens, topk = topk_logical.shape
    requests = tokens // positions
    cold = scenario == "cold_reset"
    rng = np.random.default_rng(seed)
    resident = np.full((requests, capacity), -1, np.int32)
    union_sizes = []
    if not cold:
        for r in range(requests):
            u = np.unique(topk_logical[r * positions:(r + 1) * positions])
            union_sizes.append(int(u.size))
            resident[r] = _choose_resident_tokens(
                u, 0.90, capacity, True, 0, max_token, rng)

    req_ids = np.asarray(meta["req_ids"].detach().cpu(), np.int64).reshape(-1)
    token_to_req = np.repeat(np.arange(requests, dtype=np.int32), positions)
    last_req_ids = (np.full(tokens, -1, np.int64) if cold
                    else np.repeat(req_ids, positions))
    slot_to_token = np.repeat(resident, positions, axis=0).astype(np.int32)
    lru_slots = np.tile(np.arange(capacity, dtype=np.int32), (tokens, 1))
    row_modes = np.zeros(tokens, np.int32)
    if not cold:
        for r in range(requests):
            for p in range(positions):
                row = r * positions + p
                if positions == 1:
                    row_modes[row] = 0
                else:
                    row_modes[row] = 1 if p == 0 else 2
    state = {
        "req_ids": req_ids,
        "topk_indices": topk_logical.copy(),
        "stable_prefix_lens": np.full(tokens, max_token, np.int32),
        "visible_seq_lens": np.full(tokens, max_token, np.int32),
        "token_to_req": token_to_req,
        "active_rows": np.array([tokens], np.int32),
        "last_req_ids": last_req_ids,
        "slot_to_token": slot_to_token,
        "lru_slots": lru_slots,
        "row_modes": row_modes,
        "resident_rows": np.repeat(np.arange(requests, dtype=np.int32) * positions,
                                   positions),
        "resident_block_table": np.zeros((tokens, capacity // block_size), np.int32),
        "compact_workspace": np.full(tokens * topk // 2, 0, np.int32),
        "max_token": max_token,
        "block_size": block_size,
    }
    state["compact_workspace"][:tokens] = I32_MAX   # device error sentinel
    stats = {
        "union_size_mean": float(np.mean(union_sizes)) if union_sizes else 0.0,
        "union_size_max": int(np.max(union_sizes)) if union_sizes else 0,
        "resident_per_request": int(np.count_nonzero(resident[0] >= 0)),
    }
    return state, stats


def newop_reference(state):
    """Direct reference for one joint-group call.

    Implements the kernel's documented cold/hot semantics: globally sorted
    unique query tokens, bitmap hit classification over the leader pool, LRU
    eviction of non-hit slots, admission in ascending token order, and the
    ``survivors ++ admitted ++ hits`` LRU rebuild.
    """
    positions = int(state["token_to_req"].size // state["req_ids"].size)
    requests = int(state["req_ids"].size)
    topk = int(state["topk_indices"].shape[1])
    capacity = int(state["slot_to_token"].shape[1])
    queries = state["topk_indices"]
    pool0 = state["slot_to_token"][0::positions].copy()
    lru0 = state["lru_slots"][0::positions].copy()
    cold = bool(np.all(state["last_req_ids"] < 0))
    cur = np.full((requests * positions, topk), -1, np.int32)
    miss_count = np.zeros(requests * positions, np.int32)
    miss_tokens = np.full((requests * positions, topk), -1, np.int32)
    miss_slots = np.full((requests * positions, topk), -1, np.int32)
    pool_out = pool0.copy()
    lru_out = lru0.copy()
    for r in range(requests):
        rows = queries[r * positions:(r + 1) * positions]
        merged = np.unique(rows.reshape(-1))
        if cold:
            pool = np.full(capacity, -1, np.int32)
            u = merged.size
            pool[:u] = merged
            cur[r * positions:(r + 1) * positions] = np.searchsorted(
                merged, rows).astype(np.int32)
            flat = r * positions * topk
            miss_tokens.reshape(-1)[flat:flat + u] = merged
            miss_slots.reshape(-1)[flat:flat + u] = np.arange(u, dtype=np.int32)
            unused = capacity - u
            lru_new = np.concatenate([np.arange(u, capacity, dtype=np.int32),
                                      np.arange(0, u, dtype=np.int32)])
            for p in range(positions):
                left = u - p * topk
                miss_count[r * positions + p] = min(max(left, 0), topk)
            pool_out[r] = pool
            lru_out[r] = lru_new
            continue

        pool = pool0[r].copy()
        lru = lru0[r].copy()
        token_to_slot = {int(t): s for s, t in enumerate(pool) if t >= 0}
        qset = set(int(t) for t in merged)
        hit_slots = [int(s) for s in lru if pool[s] >= 0 and int(pool[s]) in qset]
        hit_set = set(hit_slots)
        cand = [int(s) for s in lru if int(s) not in hit_set]
        missing = [int(t) for t in merged if int(t) not in token_to_slot]
        if len(missing) > len(cand):
            raise AssertionError("ordered-bitmap reference: insufficient slots")
        assigned = cand[:len(missing)]
        for slot, token in zip(assigned, missing):
            pool[slot] = token
        token_to_slot = {int(t): s for s, t in enumerate(pool) if t >= 0}
        for p in range(positions):
            row = rows[p]
            cur[r * positions + p] = np.array([token_to_slot[int(t)] for t in row],
                                              np.int32)
        miss_count[r * positions:(r + 1) * positions] = [
            min(max(len(missing) - p * topk, 0), topk) for p in range(positions)]
        flat = r * positions * topk
        miss_tokens.reshape(-1)[flat:flat + len(missing)] = missing
        miss_slots.reshape(-1)[flat:flat + len(missing)] = assigned
        lru_new = cand[len(missing):] + assigned + hit_slots
        lru_out[r] = np.asarray(lru_new, np.int32)
        pool_out[r] = pool
    return {
        "slot_to_token": pool_out,
        "lru_slots": lru_out,
        "current_slots": cur,
        "miss_count": miss_count,
        "miss_tokens": miss_tokens,
        "miss_slots": miss_slots,
    }


def allocate_newop_state(torch_mod, np_state, device):
    """Upload the numpy state; returns (tensors, outputs, workspace)."""
    def up(a, dtype=None):
        t = torch_mod.from_numpy(np.ascontiguousarray(a))
        if dtype is not None:
            t = t.to(dtype)
        return t.to(device)

    state = {
        "req_ids": up(np_state["req_ids"]),
        "topk_indices": up(np_state["topk_indices"]),
        "stable_prefix_lens": up(np_state["stable_prefix_lens"]),
        "visible_seq_lens": up(np_state["visible_seq_lens"]),
        "token_to_req": up(np_state["token_to_req"]),
        "active_rows": up(np_state["active_rows"]),
        "last_req_ids": up(np_state["last_req_ids"]),
        "slot_to_token": up(np_state["slot_to_token"]),
        "lru_slots": up(np_state["lru_slots"]),
        "row_modes": up(np_state["row_modes"]),
        "resident_rows": up(np_state["resident_rows"]),
        "resident_block_table": up(np_state["resident_block_table"]),
        "compact_workspace": up(np_state["compact_workspace"]),
    }
    tokens, topk = np_state["topk_indices"].shape
    requests = np_state["req_ids"].size
    positions = tokens // requests
    outputs = {
        "current_slots": torch_mod.full((tokens, topk), -1, dtype=torch_mod.int32,
                                        device=device),
        "miss_count": torch_mod.zeros((tokens,), dtype=torch_mod.int32,
                                      device=device),
        "miss_tokens": torch_mod.full((tokens, topk), -1, dtype=torch_mod.int32,
                                      device=device),
        "miss_slots": torch_mod.full((tokens, topk), -1, dtype=torch_mod.int32,
                                     device=device),
    }
    return state, outputs, int(np_state["max_token"]), int(np_state["block_size"])


def newop_plan_call(plan_op, state, outputs, max_token, block_size, topk=None):
    if topk is None:
        topk = state["topk_indices"]
    plan_op(state["req_ids"], topk,
            state["stable_prefix_lens"], state["visible_seq_lens"],
            state["token_to_req"], state["active_rows"],
            state["last_req_ids"], state["slot_to_token"], state["lru_slots"],
            outputs["current_slots"], outputs["miss_count"],
            outputs["miss_tokens"], outputs["miss_slots"],
            state["compact_workspace"], state["row_modes"],
            state["resident_rows"], state["resident_block_table"],
            max_token, block_size)


def restore_newop_state(state, np_state, device):
    for key in ("last_req_ids", "slot_to_token", "lru_slots", "row_modes",
                "resident_rows", "resident_block_table", "compact_workspace"):
        state[key].copy_(torch.from_numpy(np.ascontiguousarray(np_state[key])).to(device))


def check_newop_golden(plan_op, state, outputs, max_token, block_size, np_state):
    positions = int(np_state["token_to_req"].size // np_state["req_ids"].size)
    requests = int(np_state["req_ids"].size)
    topk = int(np_state["topk_indices"].shape[1])
    expected = newop_reference(np_state)
    newop_plan_call(plan_op, state, outputs, max_token, block_size)
    torch.npu.synchronize()
    bad = []
    got_slot = state["slot_to_token"][0::positions].cpu().numpy()
    got_lru = state["lru_slots"][0::positions].cpu().numpy()
    got_cur = outputs["current_slots"].cpu().numpy()
    got_mc = outputs["miss_count"].cpu().numpy()
    got_mt = outputs["miss_tokens"].cpu().numpy()
    got_ms = outputs["miss_slots"].cpu().numpy()
    if not np.array_equal(got_slot, expected["slot_to_token"]):
        bad.append("slot_to_token")
    if not np.array_equal(got_lru, expected["lru_slots"]):
        bad.append("lru_slots")
    if not np.array_equal(got_cur, expected["current_slots"]):
        bad.append("current_slots")
    if not np.array_equal(got_mc, expected["miss_count"]):
        bad.append("miss_count")
    if not np.array_equal(got_mt, expected["miss_tokens"]):
        bad.append("miss_tokens")
    if not np.array_equal(got_ms, expected["miss_slots"]):
        bad.append("miss_slots")
    sentinel = state["compact_workspace"][:requests * positions].cpu().numpy()
    if not np.all(sentinel == I32_MAX):
        bad.append("workspace_error_sentinel")
    restore_newop_state(state, np_state, state["req_ids"].device)
    torch.npu.synchronize()
    return {"ok": not bad, "mismatch": bad}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def event_samples(fn, restore, warmup: int, samples: int):
    """Per-sample Event timing with state restore outside the Event window."""
    torch_npu = sys.modules["torch_npu"]
    for _ in range(warmup):
        if restore is not None:
            restore()
        fn()
    torch_npu.npu.synchronize()
    out = []
    for _ in range(samples):
        if restore is not None:
            restore()
        torch_npu.npu.synchronize()
        start = torch_npu.npu.Event(enable_timing=True)
        stop = torch_npu.npu.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        torch_npu.npu.synchronize()
        out.append(start.elapsed_time(stop) * 1000.0)   # us
    return out


def stats_us(samples):
    import numpy as np
    a = np.asarray(samples, np.float64)
    return {
        "p50_us": float(np.percentile(a, 50)),
        "p95_us": float(np.percentile(a, 95)),
        "mean_us": float(a.mean()),
        "min_us": float(a.min()),
        "max_us": float(a.max()),
        "n": int(a.size),
        "samples": [float(x) for x in a],
    }
