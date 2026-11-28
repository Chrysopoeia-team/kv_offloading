# SPDX-License-Identifier: Apache-2.0
"""Exercise the packaged ACLNN entry on Ascend950."""

import argparse
import ctypes as C

import torch
import torch_npu  # noqa: F401


class Plan:
    def __init__(self, opapi):
        acl = C.CDLL("libascendcl.so")
        nnopbase = C.CDLL("libnnopbase.so")
        self.create = nnopbase.aclCreateTensor
        self.create.argtypes = [
            C.POINTER(C.c_int64),
            C.c_uint64,
            C.c_int,
            C.POINTER(C.c_int64),
            C.c_int64,
            C.c_int,
            C.POINTER(C.c_int64),
            C.c_uint64,
            C.c_void_p,
        ]
        self.create.restype = C.c_void_p
        self.destroy = nnopbase.aclDestroyTensor
        self.destroy.argtypes = [C.c_void_p]
        acl.aclGetRecentErrMsg.restype = C.c_char_p
        self.error = acl.aclGetRecentErrMsg
        lib = C.CDLL(opapi)
        self.query = lib.aclnnSparseKvPlanGetWorkspaceSize
        self.query.argtypes = (
            [C.c_void_p] * 6
            + [C.c_int64] * 3
            + [C.c_void_p] * 11
            + [C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)]
        )
        self.execute = lib.aclnnSparseKvPlan
        self.execute.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]

    def __call__(self, inputs, state, capacity, context):
        torch.npu.synchronize()
        descriptors = []
        try:
            for t in (*inputs, *state):
                shape = (C.c_int64 * t.ndim)(*t.shape)
                stride = (C.c_int64 * t.ndim)(*t.stride())
                # Values from aclDataType and aclFormat in acl_base_rt.h.
                desc = self.create(
                    shape,
                    t.ndim,
                    9 if t.dtype == torch.int64 else 3,
                    stride,
                    0,
                    2,
                    shape,
                    t.ndim,
                    t.data_ptr(),
                )
                if not desc:
                    raise RuntimeError(f"aclCreateTensor: {self.error()!r}")
                descriptors.append(desc)
            size = C.c_uint64()
            executor = C.c_void_p()
            rc = self.query(
                *descriptors[:6],
                capacity,
                context,
                64,
                *descriptors[6:],
                C.byref(size),
                C.byref(executor),
            )
            if rc:
                raise RuntimeError(
                    f"aclnnSparseKvPlanGetWorkspaceSize: {rc}, {self.error()!r}"
                )
            workspace = torch.empty(
                size.value, dtype=torch.uint8, device=inputs[0].device
            )
            stream = torch.npu.current_stream().npu_stream
            rc = self.execute(workspace.data_ptr(), size, executor, stream)
            if rc:
                raise RuntimeError(f"aclnnSparseKvPlan: {rc}, {self.error()!r}")
            torch.npu.synchronize()
        finally:
            for desc in descriptors:
                self.destroy(desc)


def new_state(rows, capacity, device):
    def full(shape, value, dtype=torch.int32):
        return torch.full(shape, value, dtype=dtype, device=device)

    return [
        full((rows,), -1, torch.int64),
        full((rows, capacity), -1),
        torch.arange(capacity, dtype=torch.int32, device=device).repeat(rows, 1),
        full((rows, 2048), -1),
        full((rows,), 0),
        full((rows, 2048), -1),
        full((rows, 2048), -1),
        full((rows * 1024,), torch.iinfo(torch.int32).max),
        full((rows,), 0),
        full((rows,), -1),
        full((rows, capacity // 64), -1),
    ]


def check(plan, device, width, context):
    capacity = 8192 if width == 4 else 4096
    ranges = torch.arange(2048, dtype=torch.int32)
    topk = torch.stack([ranges + i * 512 for i in range(width)])
    # Exercise the highest token and a shortened source row.
    topk[:, 1024:] += context - width * 2048
    topk[0, -1] = context - 1
    if width > 1:
        topk[-1, -3:] = -1
    union = torch.unique(topk[topk >= 0], sorted=True)
    count = union.numel()
    expected_slots = torch.where(topk >= 0, torch.searchsorted(union, topk), -1).int()
    expected_lru = torch.cat((torch.arange(count, capacity), torch.arange(count))).int()

    def full(shape, value, dtype=torch.int32):
        return torch.full(shape, value, dtype=dtype, device=device)

    req = full((1,), 100, torch.int64)
    active = full((1,), width)
    inputs = [
        req,
        topk.to(device),
        full((width,), context),
        full((width,), context),
        full((width,), 0),
        active,
    ]
    state = new_state(width, capacity, device)
    (
        last,
        cache,
        lru,
        slots,
        misses,
        tokens,
        miss_slots,
        status,
        modes,
        resident,
        table,
    ) = state
    expected_cache = torch.cat(
        (union, torch.full((capacity - count,), -1, dtype=torch.int32))
    )
    expected_pages = torch.arange(capacity // 64, dtype=torch.int32).repeat(width, 1)
    expected_modes = [1] + [2] * (width - 1) if width > 1 else [0]

    for step, req_id in enumerate((100, 100, 101)):
        req.fill_(req_id)
        snapshots = [value.cpu().clone() for value in inputs]
        plan(inputs, state, capacity, context)
        for value, snapshot in zip(inputs, snapshots):
            torch.testing.assert_close(value.cpu(), snapshot, rtol=0, atol=0)
        assert status[:width].cpu().min().item() >= 0
        assert last[0].item() == req_id
        assert resident.cpu().tolist() == [0] * width
        assert modes.cpu().tolist() == expected_modes
        torch.testing.assert_close(table.cpu(), expected_pages)
        torch.testing.assert_close(slots.cpu(), expected_slots)
        torch.testing.assert_close(cache[0].cpu(), expected_cache)
        torch.testing.assert_close(lru[0].cpu(), expected_lru)
        written = count if step != 1 else 0
        assert misses.cpu().tolist() == [
            min(2048, max(0, written - i * 2048)) for i in range(width)
        ]
        torch.testing.assert_close(tokens.flatten()[:written].cpu(), union[:written])
        torch.testing.assert_close(
            miss_slots.flatten()[:written].cpu(),
            torch.arange(written, dtype=torch.int32),
        )
        assert (tokens.flatten()[written:] == -1).all().item()
        assert (miss_slots.flatten()[written:] == -1).all().item()

    if width > 1:
        snapshot = cache.clone()
        active.fill_(width - 1)
        plan(inputs, state, capacity, context)
        assert status[0].item() == -1
        assert torch.equal(cache, snapshot)


def check_reclamation(plan, device, width, context, high_tokens=False):
    capacity = 8192 if width == 4 else 4096
    shortages = (0, 1, 7)
    requests = len(shortages)
    rows = requests * width
    shared = torch.arange(64, dtype=torch.int32)
    query = torch.stack(
        [
            torch.cat(
                (
                    shared,
                    torch.arange(64 + i * 1984, 64 + (i + 1) * 1984, dtype=torch.int32),
                )
            )
            for i in range(width)
        ]
    )
    token_offset = context - int(query.max()) - 1 if high_tokens else 0
    query += token_offset
    topk = query.repeat(requests, 1)
    selected = set(topk[0:width].flatten().tolist())
    state = new_state(rows, capacity, device)
    (
        last,
        cache,
        lru,
        slots,
        misses,
        tokens,
        miss_slots,
        status,
        modes,
        resident,
        table,
    ) = state
    request_ids = torch.arange(100, 100 + requests, dtype=torch.int64, device=device)
    inputs = [
        request_ids,
        topk.to(device),
        torch.full((rows,), context, dtype=torch.int32, device=device),
        torch.full((rows,), context, dtype=torch.int32, device=device),
        torch.arange(requests, dtype=torch.int32, device=device).repeat_interleave(
            width
        ),
        torch.tensor([rows], dtype=torch.int32, device=device),
    ]
    generator = torch.Generator().manual_seed(2026 + width)
    expected_cache = torch.full((rows, capacity), -1, dtype=torch.int32)
    expected_lru = torch.arange(capacity, dtype=torch.int32).repeat(rows, 1)
    expected_slots = torch.empty_like(topk)
    expected_misses = torch.zeros(rows, dtype=torch.int32)
    expected_tokens = torch.full_like(topk, -1)
    expected_miss_slots = torch.full_like(topk, -1)

    for group, shortage in enumerate(shortages):
        first = group * width
        group_rows = slice(first, first + width)
        resident_count = capacity - len(selected) + 2 + shortage
        # Spread two duplicated tokens through the LRU order.
        contents = torch.full((capacity,), -1, dtype=torch.int32)
        contents[:resident_count] = token_offset + (
            torch.arange(resident_count, dtype=torch.int32) % 2
        )
        order = torch.randperm(capacity, generator=generator).tolist()
        cache[first].copy_(contents)
        lru[first].copy_(torch.tensor(order, dtype=torch.int32))
        modes[first] = int(width > 1)
        last[first] = request_ids[group]
        resident_tokens = contents.tolist()
        hits = [slot for slot in order if resident_tokens[slot] in selected]
        evictions = [slot for slot in order if resident_tokens[slot] not in selected]
        owner_by_token = {resident_tokens[slot]: slot for slot in hits}
        missing = sorted(selected.difference(owner_by_token))
        assert len(missing) - len(evictions) == shortage
        redundant = [
            slot for slot in hits if slot != owner_by_token[resident_tokens[slot]]
        ]
        reclaimed = redundant[:shortage]
        assert len(reclaimed) == shortage
        target_slots = evictions + reclaimed
        for token, slot in zip(missing, target_slots):
            contents[slot] = token
            owner_by_token[token] = slot

        expected_cache[first] = contents
        expected_lru[first] = torch.tensor(
            target_slots + [s for s in hits if s not in reclaimed], dtype=torch.int32
        )
        expected_slots[group_rows] = torch.tensor(
            [[owner_by_token[token] for token in row] for row in query.tolist()],
            dtype=torch.int32,
        )
        expected_misses[group_rows] = torch.tensor(
            [min(2048, max(0, len(missing) - i * 2048)) for i in range(width)],
            dtype=torch.int32,
        )
        expected_tokens[group_rows].view(-1)[: len(missing)] = torch.tensor(
            missing, dtype=torch.int32
        )
        expected_miss_slots[group_rows].view(-1)[: len(missing)] = torch.tensor(
            target_slots, dtype=torch.int32
        )

    expected_resident = (
        torch.arange(requests, dtype=torch.int32).repeat_interleave(width) * width
    )
    pages = capacity // 64
    expected_table = expected_resident[:, None] * pages + torch.arange(
        pages, dtype=torch.int32
    )
    expected_modes = ([1] + [2] * (width - 1) if width > 1 else [0]) * requests
    plan(inputs, state, capacity, context)
    assert status[:rows].cpu().min().item() >= 0
    assert modes.cpu().tolist() == expected_modes
    for actual, expected in (
        (resident, expected_resident),
        (table, expected_table),
        (cache, expected_cache),
        (lru, expected_lru),
        (slots, expected_slots),
        (misses, expected_misses),
        (tokens, expected_tokens),
        (miss_slots, expected_miss_slots),
        (inputs[1], topk),
    ):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    plan(inputs, state, capacity, context)
    assert status[:rows].cpu().min().item() >= 0
    assert (misses.cpu() == 0).all().item()
    assert (tokens.cpu() == -1).all().item()
    assert (miss_slots.cpu() == -1).all().item()
    torch.testing.assert_close(cache.cpu(), expected_cache)
    torch.testing.assert_close(slots.cpu(), expected_slots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--opapi", required=True)
    args = parser.parse_args()
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    plan = Plan(args.opapi)
    for width, short_context in ((1, 102400), (2, 262144), (4, 358400)):
        # 4K and 8K caches outgrow their base bitmap workspace at different lengths.
        first_expanded = (448 if width == 4 else 352) * 1024 + 1
        for context in (
            short_context,
            first_expanded,
            512 * 1024,
            512 * 1024 + 1,
            1024 * 1024,
        ):
            check(plan, device, width, context)
            check_reclamation(
                plan, device, width, context, high_tokens=context != short_context
            )
            print(
                f"PASS MTP{width - 1}, context {context}, highest token, shortages 0/1/7 and replay"
            )


if __name__ == "__main__":
    main()
