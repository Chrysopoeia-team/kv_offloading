// echo_lru_plugin.cpp - torch operator of the Echo LRU plan.
//
// Registers sparse_kv_plan_op.echo_lru, echo_lru_init and echo_lru_core_count.
// The host side validates the tensors, derives the tiling struct, uploads it once
// per shape (cached per device) and launches the kernels on the current NPU
// stream.
//
// Reading order: Check -> EchoNpu; InitCheck -> EchoInitNpu.

#include <algorithm>
#include <cstring>
#include <limits>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "echo_lru_launch.h"

#include "echo_v2_kernel2_launch.h"

namespace {

// Thread-local cache of the device copy of the tiling struct; rebuilt only
// when the struct changes.
struct EchoTilingCache {
  at::Tensor dev;
  ascend_kernel::EchoLruTiling tiling{};
  c10::DeviceIndex device = -1;
  bool valid = false;
};

// Vector-core count of the current device (echo_lru_core_count).
int64_t EchoCoreCount() {
  int32_t device = 0;
  TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS, "echo_lru: cannot query current device");
  static thread_local int32_t cachedDevice = -1;
  static thread_local int64_t cachedCores = 0;
  if (cachedDevice != device || cachedCores <= 0) {
    int64_t cores = 0;
    auto ret = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores);
    TORCH_CHECK(ret == ACL_SUCCESS && cores > 0, "echo_lru: cannot query vector cores, ret=", ret);
    cachedDevice = device;
    cachedCores = cores;
  }
  return cachedCores;
}

// Returns the device tensor that holds the tiling struct, uploading it when
// the struct changed.
const at::Tensor& CachedTilingDev(const at::Tensor& like,
                                  const ascend_kernel::EchoLruTiling& tiling) {
  static thread_local EchoTilingCache cache;
  const auto deviceIndex = like.device().index();
  if (!cache.valid || cache.device != deviceIndex ||
      std::memcmp(&cache.tiling, &tiling, sizeof(tiling)) != 0) {
    at::Tensor cpu = at::empty({static_cast<int64_t>(sizeof(tiling))},
                               at::TensorOptions().dtype(at::kByte));
    std::memcpy(cpu.data_ptr(), &tiling, sizeof(tiling));
    cache.dev = cpu.to(like.device(), false);
    cache.tiling = tiling;
    cache.device = deviceIndex;
    cache.valid = true;
  }
  return cache.dev;
}

// Validates shapes, dtypes and the 32-bit indexing limits, then derives the
// tiling struct.
ascend_kernel::EchoLruTiling Check(const at::Tensor& pos, const at::Tensor& spl, const at::Tensor& resetMask, const at::Tensor& htd, const at::Tensor& dth, const at::Tensor& lru, const at::Tensor& freeSlots, const at::Tensor& avail, const at::Tensor& fifo, const at::Tensor& fresh, const at::Tensor& curSlots, const at::Tensor& missCount, const at::Tensor& missTokens, const at::Tensor& missSlots, const at::Tensor& workspace, int64_t blocks, bool specEnabled, bool npu) {

  TORCH_CHECK(pos.dim() == 2 && pos.size(0) > 0 && pos.size(1) > 0,
              "echo_lru: pos must have shape [num_tokens, topk]");
  const int64_t tokens = pos.size(0), topk = pos.size(1);
  TORCH_CHECK(htd.dim() == 2 && htd.size(0) > 0 && htd.size(1) > 1,
              "echo_lru: invalid htd shape");
  const int64_t requests = htd.size(0), maxToken = htd.size(1);
  TORCH_CHECK(tokens % requests == 0,
              "echo_lru: pos rows (num_tokens=", tokens,
              ") must be divisible by htd rows (requests=", requests, ")");
  const int64_t positions = tokens / requests;
  TORCH_CHECK(dth.dim() == 2 && dth.size(0) == requests, "echo_lru: invalid dth shape");
  const int64_t capacity = dth.size(1) - 1;
  TORCH_CHECK(capacity > 0 && capacity <= ascend_kernel::ARGTOPK_B_MAX,
              "echo_lru: capacity must be in [1, 8192]");
  const int64_t limit = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(tokens <= limit && requests <= limit && topk <= limit && maxToken <= limit &&

              tokens <= limit / topk &&
              requests <= limit / maxToken &&

              requests <= limit / (capacity + ascend_kernel::ECHO_LRU_ROW_PAD) &&
              requests <= limit / (capacity + std::min(topk, capacity)),
              "echo_lru: shape exceeds 32-bit kernel indexing");

  TORCH_CHECK(blocks > 0 && blocks <= requests, "echo_lru: invalid block count");
  const int64_t chunk = (requests + blocks - 1) / blocks;
  TORCH_CHECK((blocks - 1) * chunk < requests, "echo_lru: last block would be empty; use echo_partition");
  auto tensor = [&](const at::Tensor& value, at::ScalarType type,
                    at::IntArrayRef shape, const char* name) {
    TORCH_CHECK(value.scalar_type() == type && value.sizes() == shape && value.is_contiguous(),
                "echo_lru: shape/dtype/contiguity mismatch for ", name);
    TORCH_CHECK(value.device() == pos.device(), "echo_lru: tensors must share a device");
  };
  tensor(pos, at::kInt, {tokens, topk}, "pos");
  tensor(spl, at::kInt, {requests}, "spl");
  tensor(resetMask, at::kByte, {requests}, "reset_mask");
  tensor(htd, at::kInt, {requests, maxToken}, "htd");
  tensor(dth, at::kInt, {requests, capacity + 1}, "dth");

  tensor(lru, at::kInt, {requests, capacity + ascend_kernel::ECHO_LRU_ROW_PAD}, "lru");
  tensor(freeSlots, at::kInt, {requests, capacity + std::min(topk, capacity)}, "free_slots");
  tensor(avail, at::kInt, {requests}, "avail");
  tensor(fifo, at::kInt, {requests}, "fifo");
  tensor(fresh, at::kByte, {requests}, "fresh");
  tensor(curSlots, at::kInt, {tokens, topk}, "current_slots");

  tensor(missCount, at::kInt, {requests}, "miss_count");
  tensor(missTokens, at::kInt, {requests, positions * topk}, "miss_tokens");
  tensor(missSlots, at::kInt, {requests, positions * topk}, "miss_slots");
  const uint64_t required = EchoWorkspaceElements(requests, tokens, topk, capacity, blocks);
  TORCH_CHECK(required <= uint64_t(limit) && workspace.dim() == 1 &&
              workspace.numel() >= static_cast<int64_t>(required) &&
              workspace.scalar_type() == at::kInt && workspace.is_contiguous() &&
              workspace.device() == pos.device(), "echo_lru: workspace is invalid or too small");
  if (npu) {
    TORCH_CHECK(pos.device().type() == c10::DeviceType::PrivateUse1, "echo_lru: expected NPU");
  }
  ascend_kernel::EchoLruTiling tiling{};
  tiling.blockNum = blocks;
  tiling.rowsPerCore = chunk;
  tiling.tailRows = requests - chunk * (blocks - 1);
  tiling.numRequests = requests;
  tiling.numTokens = tokens;
  tiling.topkBufferB = capacity;
  tiling.freeStackSize = capacity + std::min(topk, capacity);
  tiling.maxModelLenP1 = maxToken;
  tiling.topkDim = topk;
  tiling.effTopk = std::min(topk, capacity);
  tiling.specPositions = positions;
  tiling.specEnabled = specEnabled;
  tiling.argtopkDebugRow = 0xFFFFFFFFu;
  tiling.argtopkGroupThreads = ascend_kernel::ARGTOPK_GROUP_THREADS;
  return tiling;
}

// Meta implementation (shape checks only).
void EchoMeta(const at::Tensor& pos, const at::Tensor& spl, const at::Tensor& resetMask, const at::Tensor& htd, const at::Tensor& dth, const at::Tensor& lru, const at::Tensor& freeSlots, const at::Tensor& avail, const at::Tensor& fifo, const at::Tensor& fresh, const at::Tensor& curSlots, const at::Tensor& missCount, const at::Tensor& missTokens, const at::Tensor& missSlots, const at::Tensor& workspace, int64_t blocks, bool specEnabled) {
  Check(pos, spl, resetMask, htd, dth, lru, freeSlots, avail, fifo, fresh, curSlots, missCount, missTokens, missSlots, workspace, blocks, specEnabled, false);
}

// NPU implementation: launches kernel 1 and kernel 2.
void EchoNpu(const at::Tensor& pos, const at::Tensor& spl, const at::Tensor& resetMask, const at::Tensor& htd, const at::Tensor& dth, const at::Tensor& lru, const at::Tensor& freeSlots, const at::Tensor& avail, const at::Tensor& fifo, const at::Tensor& fresh, const at::Tensor& curSlots, const at::Tensor& missCount, const at::Tensor& missTokens, const at::Tensor& missSlots, const at::Tensor& workspace, int64_t blocks, bool specEnabled) {
  const c10::OptionalDeviceGuard guard(pos.device());
  const auto tiling = Check(pos, spl, resetMask, htd, dth, lru, freeSlots, avail, fifo, fresh, curSlots, missCount, missTokens, missSlots, workspace, blocks, specEnabled, true);
  TORCH_CHECK(blocks <= EchoCoreCount(), "echo_lru: block count exceeds available vector cores");
  const at::Tensor& tilingDev = CachedTilingDev(pos, tiling);
  auto stream = c10_npu::getCurrentNPUStream().stream(false);

  const uint32_t topkDim = tiling.topkDim;
  uint32_t launchThreads = (topkDim + 3u) / 4u;
  launchThreads = (launchThreads + 31u) & ~31u;
  if (launchThreads < 64u) launchThreads = 64u;
  if (launchThreads > ascend_kernel::ECHO_LRU_THREAD_NUM) launchThreads = ascend_kernel::ECHO_LRU_THREAD_NUM;

  auto launch = [=]() -> int {
    launch_echo_lru(htd.data_ptr<int32_t>(), dth.data_ptr<int32_t>(), lru.data_ptr<int32_t>(),
                    freeSlots.data_ptr<int32_t>(), avail.data_ptr<int32_t>(), fifo.data_ptr<int32_t>(),
                    pos.data_ptr<int32_t>(), spl.data_ptr<int32_t>(), resetMask.data_ptr<uint8_t>(),
                    fresh.data_ptr<uint8_t>(),
                    curSlots.data_ptr<int32_t>(), missCount.data_ptr<int32_t>(),
                    missTokens.data_ptr<int32_t>(), missSlots.data_ptr<int32_t>(),
                    workspace.data_ptr(), tilingDev.data_ptr(), blocks, launchThreads,
                    reinterpret_cast<void*>(stream));

    launch_echo_v2_kernel2(lru.data_ptr<int32_t>(), dth.data_ptr<int32_t>(),
                           htd.data_ptr<int32_t>(), freeSlots.data_ptr<int32_t>(),
                           avail.data_ptr<int32_t>(), fifo.data_ptr<int32_t>(),
                           pos.data_ptr<int32_t>(), curSlots.data_ptr<int32_t>(),
                           missCount.data_ptr<int32_t>(), missTokens.data_ptr<int32_t>(),
                           missSlots.data_ptr<int32_t>(), workspace.data_ptr(),
                           tilingDev.data_ptr(), tiling.topkBufferB, tiling.topkDim,
                           tiling.rowsPerCore, blocks,
                           reinterpret_cast<void*>(stream));
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi("EchoLru", launch);
}

// Validates the state tensors of the init call and derives its tiling struct.
ascend_kernel::EchoLruTiling InitCheck(const at::Tensor& initMask, const at::Tensor& htd, const at::Tensor& dth, const at::Tensor& lru, const at::Tensor& freeSlots, const at::Tensor& avail, const at::Tensor& fifo, const at::Tensor& fresh, int64_t blocks, bool npu) {
  TORCH_CHECK(htd.dim() == 2 && htd.size(0) > 0 && htd.size(1) > 1,
              "echo_lru_init: invalid htd shape");
  const int64_t rows = htd.size(0), maxToken = htd.size(1);
  TORCH_CHECK(dth.dim() == 2 && dth.size(0) == rows, "echo_lru_init: invalid dth shape");
  const int64_t capacity = dth.size(1) - 1;
  TORCH_CHECK(capacity > 0 && capacity <= ascend_kernel::ARGTOPK_B_MAX,
              "echo_lru_init: capacity must be in [1, 8192]");
  const int64_t freeStack = freeSlots.size(1);
  const int64_t limit = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(rows <= limit && maxToken <= limit && freeStack <= limit &&
              rows <= limit / maxToken &&

              rows <= limit / (capacity + ascend_kernel::ECHO_LRU_ROW_PAD) &&
              rows <= limit / freeStack,
              "echo_lru_init: shape exceeds 32-bit kernel indexing");
  TORCH_CHECK(blocks > 0 && blocks <= rows, "echo_lru_init: invalid block count");
  const int64_t chunk = (rows + blocks - 1) / blocks;
  TORCH_CHECK((blocks - 1) * chunk < rows, "echo_lru_init: last block would be empty; use echo_partition");
  auto tensor = [&](const at::Tensor& value, at::ScalarType type,
                    at::IntArrayRef shape, const char* name) {
    TORCH_CHECK(value.scalar_type() == type && value.sizes() == shape && value.is_contiguous(),
                "echo_lru_init: shape/dtype/contiguity mismatch for ", name);
    TORCH_CHECK(value.device() == htd.device(), "echo_lru_init: tensors must share a device");
  };
  tensor(initMask, at::kByte, {rows}, "init_mask");
  tensor(htd, at::kInt, {rows, maxToken}, "htd");
  tensor(dth, at::kInt, {rows, capacity + 1}, "dth");

  tensor(lru, at::kInt, {rows, capacity + ascend_kernel::ECHO_LRU_ROW_PAD}, "lru");
  tensor(freeSlots, at::kInt, {rows, freeStack}, "free_slots");
  TORCH_CHECK(freeStack >= capacity, "echo_lru_init: free stack must cover capacity");
  tensor(avail, at::kInt, {rows}, "avail");
  tensor(fifo, at::kInt, {rows}, "fifo");
  tensor(fresh, at::kByte, {rows}, "fresh");
  if (npu) {
    TORCH_CHECK(htd.device().type() == c10::DeviceType::PrivateUse1, "echo_lru_init: expected NPU");
  }
  ascend_kernel::EchoLruTiling tiling{};
  tiling.blockNum = blocks;
  tiling.rowsPerCore = chunk;
  tiling.tailRows = rows - chunk * (blocks - 1);

  tiling.numRequests = rows;
  tiling.numTokens = rows;
  tiling.topkBufferB = capacity;
  tiling.freeStackSize = freeStack;
  tiling.maxModelLenP1 = maxToken;
  tiling.specPositions = 1u;

  tiling.specEnabled = 0;
  tiling.argtopkDebugRow = 0xFFFFFFFFu;
  tiling.argtopkGroupThreads = ascend_kernel::ARGTOPK_GROUP_THREADS;
  return tiling;
}

// Meta implementation of the init call (shape checks only).
void EchoInitMeta(const at::Tensor& initMask, const at::Tensor& htd, const at::Tensor& dth, const at::Tensor& lru, const at::Tensor& freeSlots, const at::Tensor& avail, const at::Tensor& fifo, const at::Tensor& fresh, int64_t blocks) {
  InitCheck(initMask, htd, dth, lru, freeSlots, avail, fifo, fresh, blocks, false);
}

// NPU implementation of the init call.
void EchoInitNpu(const at::Tensor& initMask, const at::Tensor& htd, const at::Tensor& dth, const at::Tensor& lru, const at::Tensor& freeSlots, const at::Tensor& avail, const at::Tensor& fifo, const at::Tensor& fresh, int64_t blocks) {
  const c10::OptionalDeviceGuard guard(htd.device());
  const auto tiling = InitCheck(initMask, htd, dth, lru, freeSlots, avail, fifo, fresh, blocks, true);
  TORCH_CHECK(blocks <= EchoCoreCount(), "echo_lru_init: block count exceeds available vector cores");
  const at::Tensor& tilingDev = CachedTilingDev(htd, tiling);
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  const uint32_t launchThreads = 512u;
  auto launch = [=]() -> int {
    launch_echo_lru_init(htd.data_ptr<int32_t>(), dth.data_ptr<int32_t>(),
                         lru.data_ptr<int32_t>(), freeSlots.data_ptr<int32_t>(),
                         avail.data_ptr<int32_t>(), fifo.data_ptr<int32_t>(),
                         fresh.data_ptr<uint8_t>(), initMask.data_ptr<uint8_t>(),
                         tilingDev.data_ptr(), blocks, launchThreads,
                         reinterpret_cast<void*>(stream));
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi("EchoLruInit", launch);
}
}

TORCH_LIBRARY_FRAGMENT(sparse_kv_plan_op, m) {
  m.def("echo_lru_core_count() -> int", EchoCoreCount);
  m.def("echo_lru(Tensor pos, Tensor spl, Tensor reset_mask, Tensor(a!) htd, "
        "Tensor(b!) dth, Tensor(c!) lru, Tensor(d!) free_slots, Tensor(e!) avail, "
        "Tensor(f!) fifo, Tensor(g!) fresh, Tensor(h!) current_slots, "
        "Tensor(i!) miss_count, Tensor(j!) miss_tokens, Tensor(k!) miss_slots, "
        "Tensor(l!) workspace, int blocks, bool spec_enabled) -> ()");
}
TORCH_LIBRARY_IMPL(sparse_kv_plan_op, Meta, m) { m.impl("echo_lru", EchoMeta); }
TORCH_LIBRARY_IMPL(sparse_kv_plan_op, PrivateUse1, m) { m.impl("echo_lru", EchoNpu); }

TORCH_LIBRARY_FRAGMENT(sparse_kv_plan_op, m) {
  m.def("echo_lru_init(Tensor init_mask, Tensor(a!) htd, Tensor(b!) dth, "
        "Tensor(c!) lru, Tensor(d!) free_slots, Tensor(e!) avail, Tensor(f!) fifo, "
        "Tensor(g!) fresh, int blocks) -> ()");
}
TORCH_LIBRARY_IMPL(sparse_kv_plan_op, Meta, m) { m.impl("echo_lru_init", EchoInitMeta); }
TORCH_LIBRARY_IMPL(sparse_kv_plan_op, PrivateUse1, m) { m.impl("echo_lru_init", EchoInitNpu); }
