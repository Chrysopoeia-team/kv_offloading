#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU reference of the Echo LRU plan: input construction, reference algorithms, device-state handling and checks. refs.py loads this file as a module and uses its pure-host functions."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

I32_MAX = np.iinfo(np.int32).max
BLOCK_SIZE = 64
SCENARIOS = {
    "cold_reset": (0.0, False),
    "warm_75pct": (0.75, False),
    "warm_all_hit": (1.0, False),
    "full_cache_75pct": (0.75, True),


    "full_cache_90pct": (0.90, True),
    "full_cache_all_hit": (1.0, True),
}
SIMT_STATE = ("last_req_ids", "slot_to_token", "lru_slots")
SIMT_OUTPUT = ("current_slots", "miss_count", "miss_tokens", "miss_slots")
ECHO_STATE = ("htd", "dth", "lru", "free_slots", "avail", "fifo")


ECHO_OUTPUT = ("current_slots", "miss_count", "miss_tokens", "miss_slots")


def sort_topk_rows_inplace(query):
    """Sort every TopK row ascending in place (the sorted-input contract)."""
    query.sort(axis=1)
    return query


def make_case(rows, topk, capacity, max_token, scenario, seed, mtp=0, overlap=0.40,
              sorted_topk=False, union_target=None):
    """Build a single-step benchmark case, keeping the input semantics and the RNG order."""
    if union_target is not None and not mtp:
        raise ValueError("union_target 只支持 mtp>0（mtp=0 无跨位置并集语义）")
    if mtp:
        return make_case_mtp(rows, topk, capacity, max_token, scenario, seed, mtp,
                             overlap, sorted_topk, union_target)
    fraction, full_cache = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    query = np.stack([rng.choice(max_token - 1, topk, replace=False).astype(np.int32) + 1
                      for _ in range(rows)])
    hits = int(topk * fraction)
    resident = np.full((rows, capacity), -1, np.int32)
    resident[:, :hits] = query[:, :hits]

    if full_cache:

        if max_token - 1 - topk < capacity - hits:
            raise ValueError("token 值域不足：无法填满缓存并保留指定 miss 数")
        for row in range(rows):
            others = np.setdiff1d(np.arange(1, max_token, dtype=np.int32), query[row])
            resident[row, hits:] = rng.choice(others, capacity - hits, replace=False)

    req_ids = np.arange(1000, 1000 + rows, dtype=np.int64)

    last_req_ids = req_ids.copy() if hits or full_cache else np.full(rows, -1, np.int64)
    num_blocks = (max_token + BLOCK_SIZE - 1) // BLOCK_SIZE


    if sorted_topk:
        sort_topk_rows_inplace(query)
    return {
        "scenario": scenario, "rows": rows, "topk": topk, "capacity": capacity,
        "max_token": max_token, "hits": hits, "host_num_blocks": num_blocks,
        "req_ids": req_ids, "last_req_ids": last_req_ids,
        "topk_indices": query,
        "stable_prefix_lens": np.full(rows, max_token, np.int32),
        "visible_seq_lens": np.full(rows, max_token, np.int32),
        "token_to_req": np.arange(rows, dtype=np.int32),
        "block_table": np.tile(np.arange(num_blocks, dtype=np.int32), (rows, 1)),
        "slot_to_token": resident,
        "lru_slots": np.tile(np.arange(capacity, dtype=np.int32), (rows, 1)),
        "reset_mask": (req_ids != last_req_ids).astype(np.uint8),
    }


def _make_case_mtp_union(rows, topk, capacity, max_token, scenario, seed, mtp,
                         overlap, sorted_topk, union_target):
    """Build a position stream whose per-request union has a target size."""
    fraction, full_cache = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    P = mtp + 1
    tokens = rows * P
    hits = int(topk * fraction)
    miss = topk - hits
    U = int(union_target)
    total_slots = P * topk
    if not (topk <= U <= total_slots):
        raise ValueError(f"union_target 必须在 [topk={topk}, P*topk={total_slots}] 内: {U}")
    if miss == 0:
        H_u, M_u = U, 0
    elif hits == 0:
        H_u, M_u = 0, U
    else:

        M_u = min(P * miss, U - hits)
        H_u = U - M_u
    if H_u < hits or M_u < miss:
        raise ValueError(
            f"union_target={U} 无法保持每位置命中率: H_u={H_u}<hits={hits} 或 "
            f"M_u={M_u}<miss={miss}")
    if H_u > P * hits or M_u > P * miss:
        raise ValueError(f"union_target={U} 的 H_u={H_u}/M_u={M_u} 超出 P·hits/P·miss")
    if H_u > capacity:
        raise ValueError(
            f"union_target={U} 需要 hit 并集 H_u={H_u} > capacity={capacity} —— "
            f"该容量下无法让 P 个位置同时保持 {fraction:.0%} 命中率；"
            f"请提高 capacity（最坏 U=P*topk 需 capacity >= P*hits = {P*hits}）")

    domain = np.arange(1, max_token, dtype=np.int32)
    query = np.empty((tokens, topk), np.int32)
    resident = np.full((rows, capacity), -1, np.int32)
    req_ids = np.arange(1000, 1000 + rows, dtype=np.int64)
    for r in range(rows):

        u = rng.choice(max_token - 1, H_u + M_u, replace=False).astype(np.int32) + 1
        hit_tokens = u[:H_u]
        miss_tokens = u[H_u:]

        hit_slots = (hit_tokens[np.arange(P * hits) % H_u]
                     if H_u else np.empty(0, np.int32))
        miss_slots = (miss_tokens[np.arange(P * miss) % M_u]
                      if M_u else np.empty(0, np.int32))
        for p in range(P):
            hs = hit_slots[p * hits:(p + 1) * hits]
            ms = miss_slots[p * miss:(p + 1) * miss]
            query[r * P + p] = np.concatenate((hs, ms))

        resident[r, :H_u] = hit_tokens
        if full_cache:
            fill = capacity - H_u
            if fill:
                others = np.setdiff1d(domain, u)
                if others.size < fill:
                    raise ValueError("token 值域不足：无法填满缓存并避开 query")
                resident[r, H_u:] = rng.choice(others, fill, replace=False)

        got = int(np.unique(query[r * P:(r + 1) * P]).size)
        if got != U:
            raise AssertionError(f"union 构造错误: req={r} 期望 {U}, 实得 {got}")

    warm = bool(hits or full_cache)
    last_req_ids = (np.repeat(req_ids, P) if warm
                    else np.full(tokens, -1, np.int64))
    reset_mask = np.zeros(rows, np.uint8) if warm else np.ones(rows, np.uint8)
    num_blocks = (max_token + BLOCK_SIZE - 1) // BLOCK_SIZE
    if sorted_topk:
        sort_topk_rows_inplace(query)
    return {
        "scenario": scenario, "rows": rows, "topk": topk, "capacity": capacity,
        "max_token": max_token, "hits": hits, "host_num_blocks": num_blocks,
        "mtp": mtp, "positions": P, "overlap": overlap, "num_tokens": tokens,
        "union_target": U, "generation": "union",
        "req_ids": req_ids, "last_req_ids": last_req_ids,
        "topk_indices": query,
        "stable_prefix_lens": np.full(rows, max_token, np.int32),
        "stable_prefix_lens_simt": np.full(tokens, max_token, np.int32),
        "visible_seq_lens": np.full(tokens, max_token, np.int32),
        "token_to_req": np.repeat(np.arange(rows, dtype=np.int32), P),
        "block_table": np.tile(np.arange(num_blocks, dtype=np.int32), (rows, 1)),
        "slot_to_token": resident,
        "lru_slots": np.tile(np.arange(capacity, dtype=np.int32), (rows, 1)),
        "reset_mask": reset_mask,
    }


def make_case_mtp(rows, topk, capacity, max_token, scenario, seed, mtp, overlap,
                  sorted_topk=False, union_target=None):
    """Build a benchmark case with P positions per request; the rows of a request are adjacent."""
    if union_target is not None:
        return _make_case_mtp_union(rows, topk, capacity, max_token, scenario,
                                    seed, mtp, overlap, sorted_topk, union_target)
    fraction, full_cache = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    P = mtp + 1
    tokens = rows * P
    hits = int(topk * fraction)
    n_copy_in = int(overlap * hits)
    n_copy_miss = int(overlap * (topk - hits))

    domain = np.arange(1, max_token, dtype=np.int32)
    query = np.empty((tokens, topk), np.int32)
    resident = np.full((rows, capacity), -1, np.int32)
    req_ids = np.arange(1000, 1000 + rows, dtype=np.int64)
    for r in range(rows):
        q0 = rng.choice(max_token - 1, topk, replace=False).astype(np.int32) + 1
        in_seg, out_seg = q0[:hits], q0[hits:]
        resident[r, :hits] = in_seg
        if full_cache:


            if max_token - 1 - topk < capacity - hits:
                raise ValueError("token 值域不足：无法填满缓存并保留指定 miss 数")
            others = np.setdiff1d(domain, q0)
            resident[r, hits:] = rng.choice(others, capacity - hits, replace=False)
        query[r * P] = q0

        pool_arr = resident[r][resident[r] >= 0]
        out_all = np.setdiff1d(domain, pool_arr)
        for p in range(1, P):
            copy_in = in_seg[:n_copy_in]
            copy_miss = out_seg[:n_copy_miss]

            rem_in = in_seg[n_copy_in:]
            if rem_in.size:
                rem_in = rng.permutation(rem_in)


            rem_miss = np.empty(0, np.int32)
            n_rem_miss = (topk - hits) - n_copy_miss
            if n_rem_miss > 0:
                rem_src = np.setdiff1d(out_all, copy_miss)
                rem_miss = rng.choice(rem_src, n_rem_miss, replace=False)


            query[r * P + p] = np.concatenate((copy_in, rem_in, copy_miss, rem_miss))


    warm = bool(hits or full_cache)

    last_req_ids = (np.repeat(req_ids, P) if warm
                    else np.full(tokens, -1, np.int64))
    reset_mask = (np.zeros(rows, np.uint8) if warm else np.ones(rows, np.uint8))
    num_blocks = (max_token + BLOCK_SIZE - 1) // BLOCK_SIZE


    if sorted_topk:
        sort_topk_rows_inplace(query)
    return {
        "scenario": scenario, "rows": rows, "topk": topk, "capacity": capacity,
        "max_token": max_token, "hits": hits, "host_num_blocks": num_blocks,
        "mtp": mtp, "positions": P, "overlap": overlap, "num_tokens": tokens,
        "req_ids": req_ids, "last_req_ids": last_req_ids,
        "topk_indices": query,


        "stable_prefix_lens": np.full(rows, max_token, np.int32),
        "stable_prefix_lens_simt": np.full(tokens, max_token, np.int32),
        "visible_seq_lens": np.full(tokens, max_token, np.int32),
        "token_to_req": np.repeat(np.arange(rows, dtype=np.int32), P),
        "block_table": np.tile(np.arange(num_blocks, dtype=np.int32), (rows, 1)),
        "slot_to_token": resident,
        "lru_slots": np.tile(np.arange(capacity, dtype=np.int32), (rows, 1)),
        "reset_mask": reset_mask,
    }


def simt_view_of_case(case):
    """Token-row view of a position case for the SIMT-style reference."""
    t2r = case["token_to_req"]
    return {**case,
            "rows": case["num_tokens"],
            "req_ids": case["req_ids"][t2r],
            "reset_mask": case["reset_mask"][t2r],
            "slot_to_token": case["slot_to_token"][t2r],
            "lru_slots": case["lru_slots"][t2r]}


def make_echo_initial(case):
    """Convert the initial history into Echo state. Echo slots start at 1; token 0 and slot 0 are reserved."""
    rows, topk, capacity, max_token = (case[k] for k in ("rows", "topk", "capacity", "max_token"))
    lru_pad = 8
    initial = {
        "htd": np.full((rows, max_token), I32_MAX, np.int32),
        "dth": np.full((rows, capacity + 1), I32_MAX, np.int32),
        "lru": np.zeros((rows, capacity + lru_pad), np.int32),
        "free_slots": np.zeros((rows, capacity + topk), np.int32),
        "avail": np.zeros(rows, np.int32),


        "fifo": np.where(case["reset_mask"] != 0, 1, capacity + 1).astype(np.int32),
    }
    initial["htd"][:, 0] = 0
    for row in range(rows):
        resident = case["slot_to_token"][row]
        occupied = np.flatnonzero(resident >= 0)
        initial["dth"][row, occupied + 1] = resident[occupied]
        initial["htd"][row, resident[occupied]] = occupied + 1
        order = [int(s) + 1 for s in case["lru_slots"][row] if resident[s] >= 0]
        initial["lru"][row, :len(order)] = order
        empty_slots = np.flatnonzero(resident < 0) + 1
        initial["free_slots"][row, :len(empty_slots)] = empty_slots[::-1]
        initial["avail"][row] = len(empty_slots)


    initial["fresh"] = case["reset_mask"].astype(np.uint8)
    return initial


def reference_simt(case):
    """Reference of the SIMT baseline: classify hits and evictable slots by the existing LRU order, then assign the misses in query order."""
    if case.get("mtp", 0):
        case = simt_view_of_case(case)
    rows, topk, capacity = (case[k] for k in ("rows", "topk", "capacity"))
    out = {key: case[key].copy() for key in SIMT_STATE}
    out.update({key: np.full((rows, topk), -1, np.int32)
                for key in ("current_slots", "miss_tokens", "miss_slots")})
    out["miss_count"] = np.zeros(rows, np.int32)
    for row in range(rows):
        if case["reset_mask"][row]:
            out["slot_to_token"][row].fill(-1)
            out["lru_slots"][row] = np.arange(capacity, dtype=np.int32)
            out["last_req_ids"][row] = case["req_ids"][row]

        resident, lru = out["slot_to_token"][row], out["lru_slots"][row].copy()
        queries = case["topk_indices"][row]
        query_set = set(queries.tolist())
        hit_slots = [int(slot) for slot in lru if int(resident[slot]) in query_set]
        evictable = [int(slot) for slot in lru if int(resident[slot]) not in query_set]
        token_to_slot = {int(resident[slot]): slot for slot in hit_slots}
        misses = []
        for position, token in enumerate(queries):
            if int(token) in token_to_slot:
                out["current_slots"][row, position] = token_to_slot[int(token)]
            else:
                misses.append(position)


        if len(misses) > len(evictable):
            raise AssertionError("共同输入不满足可分配条件")
        for index, position in enumerate(misses):
            slot, token = evictable[index], queries[position]
            resident[slot] = token
            out["current_slots"][row, position] = slot
            out["miss_tokens"][row, index] = token
            out["miss_slots"][row, index] = slot
        out["miss_count"][row] = len(misses)

        out["lru_slots"][row] = evictable[len(misses):] + evictable[:len(misses)] + hit_slots
    return out


def reference_echo(case, initial):
    """Echo reference for P = 1."""
    if case.get("mtp", 0):
        return reference_echo_mtp(case, initial)
    rows, topk, capacity = (case[k] for k in ("rows", "topk", "capacity"))
    out = {key: initial[key].copy() for key in ECHO_STATE}


    out["current_slots"] = np.zeros((rows, topk), np.int32)
    out["miss_count"] = np.zeros(rows, np.int32)
    out["miss_tokens"] = np.full((rows, topk), -1, np.int32)
    out["miss_slots"] = np.full((rows, topk), -1, np.int32)
    for row in range(rows):
        htd, dth, lru, free = (out[key][row] for key in ("htd", "dth", "lru", "free_slots"))
        avail, fifo = int(out["avail"][row]), int(out["fifo"][row])
        if case["reset_mask"][row]:
            htd.fill(I32_MAX)
            htd[0] = 0
            dth.fill(I32_MAX)
            lru.fill(0)
            free.fill(0)
            free[:capacity] = np.arange(capacity, 0, -1, dtype=np.int32)
            avail, fifo = capacity, 1

        occ = capacity - avail
        order = [int(s) for s in lru[:occ]]
        tokens = case["topk_indices"][row]
        before = htd[tokens].copy()
        misses = before == I32_MAX
        num_misses = int(misses.sum())
        qset = {int(t) for t in tokens}

        hit_slots = [s for s in order if dth[s] != I32_MAX and int(dth[s]) in qset]
        cand = [s for s in order if dth[s] == I32_MAX or int(dth[s]) not in qset]
        fs = max(num_misses - avail, 0)
        victims = cand[:fs]
        survivors = cand[fs:]
        for slot in victims:
            htd[dth[slot]] = I32_MAX
            dth[slot] = I32_MAX
        free[avail:avail + fs] = victims
        avail += fs


        allocated = free[avail - num_misses:avail].copy()
        avail -= num_misses
        for token, slot in zip(tokens[misses], allocated):
            htd[token], dth[slot] = slot, token
        out["current_slots"][row] = htd[tokens]


        out["miss_tokens"][row, :num_misses] = tokens[misses]
        out["miss_slots"][row, :num_misses] = allocated
        out["miss_count"][row] = num_misses


        new_order = survivors + hit_slots + [int(s) for s in allocated]
        lru[:len(new_order)] = new_order
        lru[len(new_order):] = 0
        fifo += 2
        out["avail"][row], out["fifo"][row] = avail, fifo
    return out


def reference_echo_mtp(case, initial):
    """Echo reference for P > 1: one shared pool per request, the eviction window follows the raw miss count, and the unique misses are admitted in token-ascending order. Returns the final state and outputs."""
    requests, topk, capacity = (case[k] for k in ("rows", "topk", "capacity"))
    P = case["positions"]
    tokens = case["num_tokens"]
    out = {key: initial[key].copy() for key in ECHO_STATE}


    out["current_slots"] = np.zeros((tokens, topk), np.int32)
    out["miss_count"] = np.zeros(requests, np.int32)
    out["miss_tokens"] = np.full((requests, P * topk), -1, np.int32)
    out["miss_slots"] = np.full((requests, P * topk), -1, np.int32)
    for req in range(requests):
        htd, dth, lru, free = (out[key][req] for key in ("htd", "dth", "lru", "free_slots"))
        avail, fifo = int(out["avail"][req]), int(out["fifo"][req])
        if case["reset_mask"][req]:
            htd.fill(I32_MAX)
            htd[0] = 0
            dth.fill(I32_MAX)
            lru.fill(0)
            free.fill(0)
            free[:capacity] = np.arange(capacity, 0, -1, dtype=np.int32)
            avail, fifo = capacity, 1

        queries = case["topk_indices"][req * P:(req + 1) * P]

        before = htd[queries]
        misses = before == I32_MAX
        raw_nm = int(misses.sum())
        qset = {int(t) for t in queries.reshape(-1)}

        occ = capacity - avail
        order = [int(s) for s in lru[:occ]]
        hit_slots = [s for s in order if dth[s] != I32_MAX and int(dth[s]) in qset]
        cand = [s for s in order if dth[s] == I32_MAX or int(dth[s]) not in qset]
        fs = max(raw_nm - avail, 0)
        victims = cand[:fs]
        survivors = cand[fs:]
        for slot in victims:
            htd[dth[slot]] = I32_MAX
            dth[slot] = I32_MAX
        nv = len(victims)
        free[avail:avail + nv] = victims
        avail_ev = avail + nv


        unique_miss = np.unique(queries[misses]) if raw_nm else np.array([], np.int32)
        dedup = int(unique_miss.size)
        admissions = min(dedup, avail_ev)
        allocated = free[avail_ev - admissions:avail_ev].copy()
        admit_toks = unique_miss[:admissions]
        if admissions:
            aslots = allocated[:admissions]
            htd[admit_toks] = aslots
            dth[aslots] = admit_toks
        out_mt, out_ms = out["miss_tokens"][req], out["miss_slots"][req]
        out_mt[:admissions] = admit_toks
        out_ms[:admissions] = allocated[:admissions]


        cur = before.copy()
        cur[misses] = htd[queries[misses]]
        out["current_slots"][req * P:(req + 1) * P] = cur
        out["miss_count"][req] = admissions


        new_order = survivors + hit_slots + [int(s) for s in allocated]
        lru[:len(new_order)] = new_order
        lru[len(new_order):] = 0
        fifo += 2
        out["avail"][req] = avail_ev - admissions
        out["fifo"][req] = fifo
    return out


def check_common_semantics(case, simt, echo):
    """Cross-check the hit rate, the final query mapping and the set of evicted tokens."""
    if case.get("mtp", 0):
        return check_common_semantics_mtp(case, simt, echo)
    expected_misses = np.full(case["rows"], case["topk"] - case["hits"])
    np.testing.assert_array_equal(simt["miss_count"], expected_misses)


    np.testing.assert_array_equal(echo["miss_count"], expected_misses)
    for row, queries in enumerate(case["topk_indices"]):
        np.testing.assert_array_equal(simt["slot_to_token"][row, simt["current_slots"][row]], queries)
        np.testing.assert_array_equal(echo["dth"][row, echo["current_slots"][row]], queries)
        np.testing.assert_array_equal(echo["htd"][row, queries], echo["current_slots"][row])
        historical = case["slot_to_token"][row]
        historical = historical[historical >= 0]
        queries_set = set(queries.tolist())

        free_before = case["capacity"] if case["reset_mask"][row] else int((case["slot_to_token"][row] < 0).sum())
        count = max(int(expected_misses[row]) - free_before, 0)
        oldest = [int(case["slot_to_token"][row, slot]) for slot in case["lru_slots"][row]
                  if case["slot_to_token"][row, slot] >= 0 and int(case["slot_to_token"][row, slot]) not in queries_set]
        expected_evicted = np.sort(oldest[:count])
        for after in (simt["slot_to_token"][row], echo["dth"][row, 1:]):
            np.testing.assert_array_equal(np.setdiff1d(historical, after), expected_evicted)


def check_common_semantics_mtp(case, simt, echo):
    """Cross-check of the common semantics for cases with positions."""
    requests, topk, capacity = (case[k] for k in ("rows", "topk", "capacity"))
    P, tokens = case["positions"], case["num_tokens"]
    t2r = case["token_to_req"]
    expected_misses = np.full(tokens, topk - case["hits"])
    np.testing.assert_array_equal(simt["miss_count"], expected_misses)


    simt_pool = case["slot_to_token"][t2r]
    simt_lru = case["lru_slots"][t2r]
    simt_reset = case["reset_mask"][t2r].astype(bool)

    for tr in range(tokens):
        queries = case["topk_indices"][tr]
        np.testing.assert_array_equal(simt["slot_to_token"][tr, simt["current_slots"][tr]], queries)
        historical = simt_pool[tr]
        historical = historical[historical >= 0]
        queries_set = set(queries.tolist())
        free_before = capacity if simt_reset[tr] else int((simt_pool[tr] < 0).sum())
        count = max(int(expected_misses[tr]) - free_before, 0)
        oldest = [int(simt_pool[tr, slot]) for slot in simt_lru[tr]
                  if simt_pool[tr, slot] >= 0 and int(simt_pool[tr, slot]) not in queries_set]
        expected_evicted = np.sort(oldest[:count])
        np.testing.assert_array_equal(np.setdiff1d(historical, simt["slot_to_token"][tr]), expected_evicted)

    for req in range(requests):
        queries = case["topk_indices"][req * P:(req + 1) * P]
        qflat = queries.reshape(-1)
        cur = echo["current_slots"][req * P:(req + 1) * P]
        np.testing.assert_array_equal(echo["htd"][req][qflat], cur.reshape(-1))
        valid = cur != I32_MAX
        if valid.any():
            np.testing.assert_array_equal(echo["dth"][req][cur[valid]], queries[valid])
        historical = case["slot_to_token"][req]
        historical = historical[historical >= 0]
        queries_set = set(qflat.tolist())

        free_before = capacity if case["reset_mask"][req] else int((case["slot_to_token"][req] < 0).sum())
        raw_miss = P * (topk - case["hits"])
        fs = max(raw_miss - free_before, 0)
        oldest = [int(case["slot_to_token"][req, slot]) for slot in case["lru_slots"][req]
                  if case["slot_to_token"][req, slot] >= 0 and int(case["slot_to_token"][req, slot]) not in queries_set]
        expected_evicted = np.sort(oldest[:fs])
        np.testing.assert_array_equal(np.setdiff1d(historical, echo["dth"][req, 1:]), expected_evicted)


        occupied = {0}
        if not case["reset_mask"][req]:
            pool_row = case["slot_to_token"][req]
            occupied.update(int(t) for t in pool_row[pool_row >= 0])
        dedup = sum(1 for t in set(qflat.tolist()) if t not in occupied)
        nv = min(fs, len(oldest))
        np.testing.assert_array_equal(echo["miss_count"][req],
                                      min(dedup, free_before + nv),
                                      err_msg=f"mtp-common/miss_count/req{req}")


def check_mtp_semantics(case, echo):
    """Assertions on the compact outputs and the state of cases with positions."""
    requests, topk, capacity = (case[k] for k in ("rows", "topk", "capacity"))
    P = case["positions"]
    flat_dim = P * topk

    def as_np(x):
        return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


    cur = as_np(echo["current_slots"])
    mcnt = as_np(echo["miss_count"])
    mtok = as_np(echo["miss_tokens"])
    mslot = as_np(echo["miss_slots"])
    htd, dth, lru, avail = (as_np(echo[key]) for key in ("htd", "dth", "lru", "avail"))
    q = case["topk_indices"]
    for req in range(requests):
        sl = slice(req * P, (req + 1) * P)
        qf = q[sl].reshape(-1)
        cf = cur[sl].reshape(-1)


        utok, inv = np.unique(qf, return_inverse=True)
        gmax = np.full(utok.size, np.int64(-1), np.int64)
        gmin = np.full(utok.size, np.int64(I32_MAX) + 1, np.int64)
        np.maximum.at(gmax, inv, cf.astype(np.int64))
        np.minimum.at(gmin, inv, cf.astype(np.int64))
        np.testing.assert_array_equal(gmax, gmin, err_msg=f"mtp/同token同槽位/req{req}")

        cnt = int(mcnt[req])
        assert 0 <= cnt <= flat_dim, f"mtp/compact计数域/req{req}: count={cnt}"
        mtf, msf = mtok[req], mslot[req]
        toks, slots = mtf[:cnt], msf[:cnt]

        np.testing.assert_array_equal(mtf[cnt:], -1, err_msg=f"mtp/compact尾token/req{req}")
        np.testing.assert_array_equal(msf[cnt:], -1, err_msg=f"mtp/compact尾slot/req{req}")

        miss_face = cf == I32_MAX
        if miss_face.any():
            over = np.unique(qf[miss_face])
            np.testing.assert_array_equal(htd[req][over], I32_MAX,
                                          err_msg=f"mtp/溢出面htd/req{req}")

        if cnt:
            assert int(toks.min()) >= 1, \
                f"mtp/compact token值域/req{req}: token 0 恒 hit 不入 compact"
            assert np.unique(toks).size == cnt, f"mtp/compact去重/req{req}: token 须互异"
            assert int(slots.min()) >= 1 and int(slots.max()) <= capacity, \
                f"mtp/槽位值域/req{req}: 槽号须在 [1,B]（1 基，0 保留）"
            np.testing.assert_array_equal(htd[req][toks], slots, err_msg=f"mtp/htd互证/req{req}")
            np.testing.assert_array_equal(dth[req][slots], toks, err_msg=f"mtp/dth互证/req{req}")

            order = np.argsort(toks)
            st, ss = toks[order], slots[order]
            pos = np.searchsorted(st, qf)
            posc = np.minimum(pos, cnt - 1)
            in_c = (pos < cnt) & (st[posc] == qf)
            np.testing.assert_array_equal(np.where(in_c, ss[posc], cf), cf,
                                          err_msg=f"mtp/patch互证/req{req}")


        occupied = {0}
        if not case["reset_mask"][req]:
            pool_row = case["slot_to_token"][req]
            occupied.update(int(t) for t in pool_row[pool_row >= 0])
        alive = set(int(t) for t in dth[req] if t != I32_MAX)
        admitted = sorted(t for t in set(int(t) for t in qf)
                          if t not in occupied and t in alive)
        assert cnt == len(admitted), \
            f"mtp/admissions面/req{req}: count={cnt} ≠ 独立推导 {len(admitted)}"
        if cnt:
            np.testing.assert_array_equal(np.sort(toks), np.asarray(admitted, np.int64),
                                          err_msg=f"mtp/准入集/req{req}")

        drow, hrow, lrow = dth[req], htd[req], lru[req]
        occ = capacity - int(avail[req])
        occ_slots = np.flatnonzero(drow[1:] != I32_MAX) + 1
        assert occ_slots.size == occ, f"mtp/占用数/req{req}: dth 占用 {occ_slots.size} ≠ B−avail {occ}"
        toks = drow[occ_slots]
        np.testing.assert_array_equal(hrow[toks], occ_slots, err_msg=f"mtp/htd↔dth/req{req}")
        h_alive = hrow.copy()
        h_alive[0] = I32_MAX
        live = np.flatnonzero(h_alive != I32_MAX)
        np.testing.assert_array_equal(drow[hrow[live]], live, err_msg=f"mtp/dth↔htd/req{req}")
        np.testing.assert_array_equal(np.sort(lrow[:occ]), np.sort(occ_slots),
                                      err_msg=f"mtp/lru占用集/req{req}")
        np.testing.assert_array_equal(lrow[occ:], 0, err_msg=f"mtp/lru尾零/req{req}")
        assert int(hrow[0]) == 0, f"mtp/htd保留项/req{req}: htd[req,0] 须恒 0"


def prepare_npu(torch, case, echo_initial, device, blocks_override=None):
    def upload(array):
        return torch.from_numpy(np.ascontiguousarray(array)).to(device)


    simt_src = simt_view_of_case(case) if case.get("mtp", 0) else case
    inputs = {key: upload(case[key]) for key in (
        "req_ids", "topk_indices", "stable_prefix_lens", "visible_seq_lens",
        "token_to_req", "block_table", "reset_mask")}


    if case.get("mtp", 0):
        inputs["stable_prefix_lens_simt"] = upload(case["stable_prefix_lens_simt"])
    inputs["active_rows"] = torch.tensor([simt_src["rows"]], dtype=torch.int32, device=device)


    inputs["init_mask"] = upload(case["reset_mask"])
    inputs["reset_mask_echo"] = torch.zeros_like(inputs["reset_mask"])
    initial = {
        "simt": {key: upload(simt_src[key]) for key in SIMT_STATE},
        "echo": {key: upload(echo_initial[key]) for key in ECHO_STATE},
    }


    initial["echo"]["fresh"] = upload(echo_initial["fresh"])
    states = {name: {key: value.clone() for key, value in snapshot.items()}
              for name, snapshot in initial.items()}
    rows, topk, capacity = (case[k] for k in ("rows", "topk", "capacity"))
    tokens = case.get("num_tokens", rows)
    simt_rows = simt_src["rows"]
    for key in SIMT_OUTPUT:
        shape = (simt_rows,) if key == "miss_count" else (simt_rows, topk)
        states["simt"][key] = torch.empty(shape, dtype=torch.int32, device=device)
    for key in ECHO_OUTPUT:


        if key == "current_slots":
            states["echo"][key] = torch.empty((tokens, topk), dtype=torch.int32, device=device)
        elif key == "miss_count":
            states["echo"][key] = torch.empty((rows,), dtype=torch.int32, device=device)
        else:
            flat_dim = (tokens // rows) * topk
            states["echo"][key] = torch.empty((rows, flat_dim), dtype=torch.int32, device=device)


    hash_capacity = max(32, 1 << (2 * topk - 1).bit_length())
    row_elements = 3 * hash_capacity + 2 * capacity + topk
    elements = 1 if row_elements * 4 <= 112 * 1024 else simt_rows * row_elements
    if elements > I32_MAX:
        raise ValueError("SIMT workspace 超过 int32 索引范围")
    states["simt"]["workspace"] = torch.empty(elements, dtype=torch.int32, device=device)


    cores = torch.ops.sparse_kv_plan_op.echo_lru_core_count()
    if blocks_override is None:
        budget = min(rows, cores)
        rows_per_core = (rows + budget - 1) // budget
        blocks = (rows + rows_per_core - 1) // rows_per_core
    else:

        if not (1 <= blocks_override <= rows):
            raise ValueError(f"blocks_override 必须在 1..rows（当前 rows={rows}）内: {blocks_override}")
        rows_per_core = (rows + blocks_override - 1) // blocks_override
        blocks = (rows + rows_per_core - 1) // rows_per_core
    npad = 1 << capacity.bit_length()


    mask_row = 3 * ((capacity + 31) // 32)


    base = rows * ((capacity + 1) + 4) + tokens * (3 * topk)
    base = (base + 7) & ~7
    base += rows * (mask_row + 8)
    base += blocks * 2


    base = (base + 7) & ~7
    base += rows * (2 * capacity) + rows
    elements = ((base + 1) & ~1) + blocks * (4 * npad + 64 * 256 + 64 * 256 + 256 + 8192)
    if elements > I32_MAX:
        raise ValueError("Echo workspace 超过 int32 索引范围")


    states["echo"]["workspace"] = torch.empty(elements, dtype=torch.int32, device=device)
    states["echo"]["blocks"] = blocks


    if bool(case["reset_mask"].any()):
        for key in ECHO_STATE:
            states["echo"][key].fill_(-777)
        states["echo"]["fresh"].fill_(0)
        torch.ops.sparse_kv_plan_op.echo_lru_init(
            inputs["init_mask"], states["echo"]["htd"], states["echo"]["dth"],
            states["echo"]["lru"], states["echo"]["free_slots"], states["echo"]["avail"],
            states["echo"]["fifo"], states["echo"]["fresh"], states["echo"]["blocks"])
        torch.npu.synchronize()
        for key in ECHO_STATE:
            np.testing.assert_array_equal(states["echo"][key].cpu().numpy(), echo_initial[key],
                                          err_msg=f"init-golden/echo/{key}")
        np.testing.assert_array_equal(states["echo"]["fresh"].cpu().numpy(), echo_initial["fresh"],
                                      err_msg="init-golden/echo/fresh")
    torch.npu.synchronize()
    return inputs, initial, states


def restore_state(torch, name, initial, state):
    """Restore the same persistent state before each call; the workspace is scratch."""


    torch.npu.synchronize()
    for key, source in initial[name].items():
        state[key].copy_(source)
    torch.npu.synchronize()


def invoke(torch, name, case, inputs, state):
    """The only call inside the timing window; the argument order follows the operator schema."""
    if name == "simt":
        spl = inputs.get("stable_prefix_lens_simt", inputs["stable_prefix_lens"])
        torch.ops.sparse_kv_plan_op.sparse_kv_plan(
            inputs["req_ids"], inputs["topk_indices"], spl,
            inputs["visible_seq_lens"], inputs["token_to_req"], inputs["block_table"],
            inputs["active_rows"], state["last_req_ids"], state["slot_to_token"],
            state["lru_slots"], state["current_slots"], state["miss_count"],
            state["miss_tokens"], state["miss_slots"], state["workspace"],
            case["topk"], case["capacity"], case["max_token"], BLOCK_SIZE, case["host_num_blocks"])
    else:
        torch.ops.sparse_kv_plan_op.echo_lru(
            inputs["topk_indices"], inputs["stable_prefix_lens"], inputs["reset_mask_echo"],
            state["htd"], state["dth"], state["lru"], state["free_slots"],
            state["avail"], state["fifo"], state["fresh"], state["current_slots"],
            state["miss_count"], state["miss_tokens"], state["miss_slots"],
            state["workspace"], state["blocks"], False)


def cold_init(torch, inputs, state):
    """Call the init operator so that the reset happens outside the decode timing window."""


    torch.npu.synchronize()
    torch.ops.sparse_kv_plan_op.echo_lru_init(
        inputs["init_mask"], state["htd"], state["dth"], state["lru"],
        state["free_slots"], state["avail"], state["fifo"], state["fresh"],
        state["blocks"])


def prepare_iteration(torch, name, case, inputs, initial, state):
    """Restore the pre-decode state outside the timing window."""
    if name == "echo" and bool(case["reset_mask"].any()):
        cold_init(torch, inputs, state)
    else:
        restore_state(torch, name, initial, state)


def check_npu_result(name, state, expected, phase):
    """Field-by-field equality check, read after synchronisation and outside the timing window."""


    import torch
    torch.npu.synchronize()
    actual = {key: state[key].cpu().numpy() for key in expected[name]}
    for key, reference in expected[name].items():
        np.testing.assert_array_equal(actual[key], reference, err_msg=f"{phase}/{name}/{key}")
    if name == "echo":


        np.testing.assert_array_equal(state["fresh"].cpu().numpy(), 0,
                                      err_msg=f"{phase}/echo/fresh")
    return actual


def measure_one(torch, name, case, inputs, initial, state, events):
    prepare_iteration(torch, name, case, inputs, initial, state)
    start, end = events
    before = time.perf_counter_ns()
    start.record()
    invoke(torch, name, case, inputs, state)
    end.record()
    end.synchronize()
    host_us = (time.perf_counter_ns() - before) / 1000.0
    event_us = start.elapsed_time(end) * 1000.0
    if not np.isfinite(event_us) or event_us <= 0:
        raise RuntimeError(f"{name}: 无效的 Event 耗时 {event_us}")
    return event_us, host_us


def summary(samples):
    return {
        "count": len(samples), "p50_us": float(np.percentile(samples, 50)),
        "p95_us": float(np.percentile(samples, 95)),
        "min_us": float(np.min(samples)), "max_us": float(np.max(samples)),
    }


def array_hash(arrays):
    """Fingerprint of the inputs; the data itself is not written to the result JSON."""
    digest = hashlib.sha256()
    for key, array in sorted(arrays.items()):
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--max-token", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=int, default=1, help="逻辑 NPU 卡号")
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--blocks", type=int, default=None,
                        help="覆盖 Echo blockNum（默认 min(rows,cores)）；1..rows，尾核非空由公式自动保证")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--audit-repeat", type=int, default=3, help="计时前重复正确性审计次数")


    parser.add_argument("--mtp", type=int, default=0, help="MTP 深度 N（positions=N+1；0=现行 mtp=0 形态）")
    parser.add_argument("--overlap", type=float, default=0.40,
                        help="mtp>0 时位置 p>0 复制位置 0 条目的比例（默认 0.40；域 [0,1]）")


    parser.add_argument("--union-target", type=int, default=None,
                        help="mtp>0：每 request 的 P 位置 query 并集精确 = 该值"
                             "（[topk, P*topk]；默认 None = 旧 overlap 模型）")


    parser.add_argument("--sorted-topk", action="store_true",
                        help="topk_indices 每 token 行按 token 值升序行内重排（模拟 indexer "
                             "有序契约；mtp>0 时 P 路各自有序；默认 off = 现行无序输入）")
    parser.add_argument("--verify-each-sample", action="store_true", help="短跑审计：在每次预热/测量后全量校验")
    parser.add_argument("--cpu-check", action="store_true", help="只检查共同输入和 CPU 参考，不加载共享库、不计时")
    parser.add_argument("--library", type=Path,
                        default=Path(__file__).resolve().parent / "build/libsimt_vs_echo_ops.so",
                        help="本目录构建的共享库；默认 build/libsimt_vs_echo_ops.so")
    parser.add_argument("--json", type=Path, default=Path("results/simt_vs_echo.json"))
    args = parser.parse_args()
    if not (args.rows > 0 and 0 < args.topk <= args.capacity <= 8192 and args.max_token > args.topk):
        parser.error("需要 rows>0、0<topk<=capacity<=8192、max_token>topk")
    if args.mtp < 0:
        parser.error("需要 mtp>=0")
    if not 0.0 <= args.overlap <= 1.0:
        parser.error("需要 overlap 在 [0,1]（规格 §7 风险3 的旋钮域）")
    if args.union_target is not None:
        if args.mtp <= 0:
            parser.error("--union-target 只支持 --mtp>0（mtp=0 无跨位置并集语义）")
        union_hi = args.topk * (args.mtp + 1)
        if not (args.topk <= args.union_target <= union_hi):
            parser.error(f"--union-target 必须在 [topk={args.topk}, "
                         f"P*topk={union_hi}] 内: {args.union_target}")


    num_tokens = args.rows * (args.mtp + 1)
    if max(args.rows * args.max_token, args.rows * (args.capacity + args.topk),
           args.rows * (args.capacity + 8), num_tokens * args.topk, args.max_token) > I32_MAX:
        parser.error("形状超过本基准支持的 int32 索引范围")
    if args.warmup < 0 or args.repeat <= 0 or args.audit_repeat <= 0:
        parser.error("warmup>=0、repeat>0、audit-repeat>0")
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]


    if args.sorted_topk:
        print("topk 输入形态: 有序（--sorted-topk：每 token 行按 token 值升序 = "
              "indexer 有序契约模拟，mtp>0 时 P 路各自有序）", flush=True)


    if not args.cpu_check:
        try:
            import torch
            import torch_npu
        except ImportError as exc:
            parser.error(f"需要匹配的 torch/torch_npu 环境：{exc}")
        if not torch.npu.is_available():
            parser.error("没有可用 NPU；可用 --cpu-check 检查参考结果")
        torch.npu.set_device(args.device)
        device = torch.device(f"npu:{args.device}")
        library_path = args.library.resolve()
        if not library_path.is_file():
            parser.error(f"共享库不存在：{library_path}；请先在本目录运行 bash build.sh")
        try:


            torch.ops.load_library(str(library_path))
        except (OSError, RuntimeError) as exc:
            parser.error(f"无法加载 {library_path}：{exc}")
        for name in ("sparse_kv_plan", "echo_lru", "echo_lru_init", "echo_lru_core_count"):
            if not hasattr(torch.ops.sparse_kv_plan_op, name):
                parser.error(f"共享库缺少 {name}，请重新构建本目录")
        manifest_path = Path(__file__).resolve().parent / "source_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        metadata = {
            "device_index": args.device, "device": torch.npu.get_device_name(args.device),
            "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "library_path": str(library_path),
            "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
            "source_snapshot_commit": manifest.get("source_commit", "unknown"),

            "echo_revision": manifest.get("echo_revision", "main-original"),
            "echo_kernel_sha256": hashlib.sha256(
                (Path(__file__).resolve().parent / "csrc/echo/echo_lru_kernel.h").read_bytes()).hexdigest(),
            "timing": "Event: public torch.ops stream interval including tiling H2D and host submission gaps; restore/validation excluded. Host: record+invoke+record+synchronize.",


            "cold_protocol": "v2.6: init op (echo_lru_init) outside timed window; decode timed with reset_mask=0; SIMT unchanged (in-kernel reset included)",


            "shape": {k: getattr(args, k) for k in ("rows", "topk", "capacity", "max_token",
                                                    "mtp", "overlap", "sorted_topk",
                                                    "union_target")},
            "blocks_override": args.blocks,
            "seed": args.seed, "warmup": args.warmup, "repeat": args.repeat,
            "audit_repeat": args.audit_repeat, "verify_each_sample": args.verify_each_sample,
        }
        print("实际加载共享库:", library_path, flush=True)
        results = {"metadata": metadata, "scenarios": {}}

    for scenario in scenarios:


        case_name = (scenario if args.union_target is None
                     else f"{scenario}_u{args.union_target}")


        case = make_case(args.rows, args.topk, args.capacity, args.max_token, scenario,
                         args.seed, mtp=args.mtp, overlap=args.overlap,
                         sorted_topk=args.sorted_topk,
                         union_target=args.union_target)
        echo_initial = make_echo_initial(case)
        expected = {"simt": reference_simt(case), "echo": reference_echo(case, echo_initial)}
        check_common_semantics(case, expected["simt"], expected["echo"])
        if case.get("mtp", 0):
            check_mtp_semantics(case, expected["echo"])
        miss_count = args.topk - case["hits"]
        available = int(echo_initial["avail"][0])
        if case.get("mtp", 0):


            raw_miss = case["positions"] * miss_count
            print(f"{case_name}: 每位置 miss={miss_count}, 原始 miss/请求={raw_miss}, "
                  f"初始空闲={available}, 需淘汰(原始口径)={max(raw_miss - available, 0)}, "
                  f"requests={case['rows']}, token 行={case['num_tokens']}", flush=True)
        else:
            free_size = max(miss_count - available, 0)
            print(f"{case_name}: 每行 miss={miss_count}, 初始空闲={available}, 需淘汰={free_size}", flush=True)
        if args.cpu_check:
            print("  CPU 参考与共同语义检查通过（未运行 NPU）", flush=True)
            continue

        inputs, initial, states = prepare_npu(torch, case, echo_initial, device, args.blocks)


        for iteration in range(args.audit_repeat):
            for name in ("simt", "echo"):
                state = states[name]
                prepare_iteration(torch, name, case, inputs, initial, state)
                output_keys = SIMT_OUTPUT if name == "simt" else ECHO_OUTPUT
                for key in output_keys:
                    state[key].fill_(-777)


                state["workspace"].fill_(I32_MAX)
                invoke(torch, name, case, inputs, state)
                torch.npu.synchronize()
                check_npu_result(name, state, expected, f"audit-{iteration}")
                if name == "echo" and case.get("mtp", 0):
                    check_mtp_semantics(case, state)


        events = {name: (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
                  for name in ("simt", "echo")}
        for iteration in range(args.warmup):
            order = ("simt", "echo") if iteration % 2 == 0 else ("echo", "simt")
            for name in order:
                measure_one(torch, name, case, inputs, initial, states[name], events[name])
                if args.verify_each_sample:
                    check_npu_result(name, states[name], expected, f"warmup-{iteration}")
                    if name == "echo" and case.get("mtp", 0):
                        check_mtp_semantics(case, states[name])

        if args.warmup:
            for name in ("simt", "echo"):
                check_npu_result(name, states[name], expected, "after-warmup")
                if name == "echo" and case.get("mtp", 0):
                    check_mtp_semantics(case, states[name])


        samples = {name: {"event_us": [], "host_us": []} for name in ("simt", "echo")}
        orders = []
        for iteration in range(args.repeat):
            order = ("simt", "echo") if iteration % 2 == 0 else ("echo", "simt")
            orders.append(list(order))
            for name in order:
                event_us, host_us = measure_one(torch, name, case, inputs, initial, states[name], events[name])
                samples[name]["event_us"].append(event_us)
                samples[name]["host_us"].append(host_us)
                if args.verify_each_sample:
                    check_npu_result(name, states[name], expected, f"sample-{iteration}")
                    if name == "echo" and case.get("mtp", 0):
                        check_mtp_semantics(case, states[name])


        actual = {name: check_npu_result(name, states[name], expected, "after-measurement")
                  for name in ("simt", "echo")}
        check_common_semantics(case, actual["simt"], actual["echo"])
        if case.get("mtp", 0):
            check_mtp_semantics(case, actual["echo"])
        stats = {name: {"event": summary(samples[name]["event_us"]),
                        "host": summary(samples[name]["host_us"]), "raw_samples": samples[name]}
                 for name in ("simt", "echo")}
        ratio = stats["echo"]["event"]["p50_us"] / stats["simt"]["event"]["p50_us"]
        stats.update({"echo_over_simt_event_p50": ratio, "execution_orders": orders,
                      "correctness": "passed", "expected_misses_per_row": miss_count,
                      "echo_blocks": states["echo"]["blocks"],
                      "case_sha256": array_hash({k: v for k, v in case.items() if isinstance(v, np.ndarray)})})
        results["scenarios"][case_name] = stats
        print(f"  Event P50: SIMT={stats['simt']['event']['p50_us']:.2f} μs, "
              f"Echo={stats['echo']['event']['p50_us']:.2f} μs, Echo/SIMT={ratio:.2f}×", flush=True)

    if not args.cpu_check:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("结果与原始样本:", args.json.resolve())
        if args.verify_each_sample:
            print("本次为逐样本正确性审计；频繁读回改变缓存条件，正式性能对比请关闭该选项。")


if __name__ == "__main__":
    main()
