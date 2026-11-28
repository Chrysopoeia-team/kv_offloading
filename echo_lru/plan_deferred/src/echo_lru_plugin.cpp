// echo_lru_plugin.cpp - torch operator of the Echo LRU plan.
//
// Registers sparse_kv_plan_op.echo_lru, echo_lru_init, echo_lru_core_count and
// echo_lru_join.
// The host side validates the tensors, derives the tiling struct, uploads it once
// per shape (cached per device) and launches the kernels on the current NPU
// stream.
//
// Reading order: Check -> EchoNpu; InitCheck -> EchoInitNpu.

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <string>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "echo_lru_launch.h"

#include "echo_v2_kernel2_launch.h"

namespace {

// Side stream and events used to run kernel 3 asynchronously.
struct Path2AsyncState {
  aclrtStream side = nullptr;

  c10_npu::NPUEvent* e1 = nullptr;
  c10_npu::NPUEvent* e3 = nullptr;
  bool e3_armed = false;
  bool in_capture = false;
  int device = -1;
};
Path2AsyncState g_path2;

// True when ECHO_PATH2_CAPTURE=1: the side stream stays in use while an NPU
// graph is captured (prototype, default off).
bool Path2CaptureEnabled() {
  static const bool enabled = []() {
    const char* raw = std::getenv("ECHO_PATH2_CAPTURE");
    if (raw == nullptr) {
      return false;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(value == "0" || value == "off" || value == "false");
  }();
  return enabled;
}

// True when kernel 3 runs on the side stream (ECHO_PATH2_ASYNC, default on;
// "0" keeps it on the calling stream).
bool Path2AsyncEnabled() {
  static const bool enabled = []() {
    const char* raw = std::getenv("ECHO_PATH2_ASYNC");
    if (raw == nullptr) {
      return true;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(value == "0" || value == "off" || value == "false");
  }();
  return enabled;
}

// Releases the side stream and its events.
void Path2AsyncRelease() {
  if (g_path2.side != nullptr) {

    TORCH_CHECK(aclrtSynchronizeStream(g_path2.side) == ACL_SUCCESS,
                "echo_lru: aclrtSynchronizeStream(path2 side) failed");
    TORCH_CHECK(aclrtDestroyStream(g_path2.side) == ACL_SUCCESS,
                "echo_lru: aclrtDestroyStream(path2 side) failed");
    g_path2.side = nullptr;
  }

  g_path2.e1 = nullptr;
  g_path2.e3 = nullptr;
  g_path2.e3_armed = false;
}

// Creates the side stream and its events for the current device on first use.
void Path2AsyncEnsure(int device) {
  if (!Path2AsyncEnabled()) {
    return;
  }
  if (g_path2.device == device && g_path2.side != nullptr) {
    return;
  }
  Path2AsyncRelease();
  TORCH_CHECK(aclrtCreateStream(&g_path2.side) == ACL_SUCCESS,
              "echo_lru: aclrtCreateStream(path2 side) failed");

  g_path2.e1 = new c10_npu::NPUEvent();
  g_path2.e3 = new c10_npu::NPUEvent();

  g_path2.e1->record(c10_npu::getCurrentNPUStream());
  g_path2.e3->record(c10_npu::getCurrentNPUStream());
  g_path2.device = device;
}

// True while the given stream is capturing an NPU graph.
bool StreamIsCapturing(aclrtStream stream) {
  aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI model = nullptr;
  if (aclmdlRICaptureGetInfo(stream, &status, &model) != ACL_SUCCESS) {
    return false;
  }
  return status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
}

// Entry wait: the calling stream waits for the kernel 3 launched by the
// previous call.
void Path2AsyncEntryWait(aclrtStream mainStream) {
  if (!Path2AsyncEnabled()) {
    return;
  }

  if (StreamIsCapturing(mainStream) && !Path2CaptureEnabled()) {
    return;
  }
  int device = -1;
  TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS,
              "echo_lru: cannot query current device (path2 entry wait)");
  if (g_path2.device != device || !g_path2.e3_armed) {
    return;
  }
  TORCH_CHECK(aclrtStreamWaitEvent(mainStream, g_path2.e3->event()) == ACL_SUCCESS,
              "echo_lru: aclrtStreamWaitEvent(main, E3) failed");
}

// Thread-local cache of the device copy of the tiling struct.
struct TilingCache {
  std::mutex mutex;
  std::unordered_map<std::string, at::Tensor> map;
};

TilingCache& GetTilingCache() {
  static TilingCache* cache = new TilingCache();
  return *cache;
}

// Returns the device tensor that holds the tiling struct, uploading it when
// the struct changes.
at::Tensor GetOrCreateTilingDev(const ascend_kernel::EchoLruTiling& tiling,
                                const at::Device& device) {
  std::string key = std::to_string(static_cast<int>(device.index()));
  key.push_back(':');
  key.append(reinterpret_cast<const char*>(&tiling), sizeof(tiling));
  TilingCache& cache = GetTilingCache();
  {
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto it = cache.map.find(key);
    if (it != cache.map.end()) {
      return it->second;
    }
  }

  at::Tensor tilingCpu = at::empty({static_cast<int64_t>(sizeof(tiling))},
                                   at::TensorOptions().dtype(at::kByte));
  std::memcpy(tilingCpu.data_ptr(), &tiling, sizeof(tiling));
  at::Tensor tilingDev = tilingCpu.to(device, false);
  std::lock_guard<std::mutex> lock(cache.mutex);
  auto inserted = cache.map.emplace(std::move(key), tilingDev);
  return inserted.first->second;
}

// Vector-core count of the current device (echo_lru_core_count).
int64_t EchoCoreCount() {
  int32_t device = 0;
  TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS, "echo_lru: cannot query current device");
  int64_t cores = 0;
  auto ret = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores);
  TORCH_CHECK(ret == ACL_SUCCESS && cores > 0, "echo_lru: cannot query vector cores, ret=", ret);
  return cores;
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

  TORCH_CHECK(positions == 1 || positions == 2 || positions == 4,
              "echo_lru: v3.2 k1-ws supports positions(P) in {1,2,4}; got P=",
              positions, " (fallback = main-branch v3.0 chain)");
  TORCH_CHECK(topk <= 65535, "echo_lru: v3.2 k1-ws requires topk <= 65535 (MrgSort lens domain)");
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

// NPU implementation: launches kernel 1 and kernel 2 on the calling stream
// and kernel 3 on the side stream (or on the calling stream when
// ECHO_PATH2_ASYNC=0).
void EchoNpu(const at::Tensor& pos, const at::Tensor& spl, const at::Tensor& resetMask, const at::Tensor& htd, const at::Tensor& dth, const at::Tensor& lru, const at::Tensor& freeSlots, const at::Tensor& avail, const at::Tensor& fifo, const at::Tensor& fresh, const at::Tensor& curSlots, const at::Tensor& missCount, const at::Tensor& missTokens, const at::Tensor& missSlots, const at::Tensor& workspace, int64_t blocks, bool specEnabled) {
  const c10::OptionalDeviceGuard guard(pos.device());
  const auto tiling = Check(pos, spl, resetMask, htd, dth, lru, freeSlots, avail, fifo, fresh, curSlots, missCount, missTokens, missSlots, workspace, blocks, specEnabled, true);
  TORCH_CHECK(blocks <= EchoCoreCount(), "echo_lru: block count exceeds available vector cores");

  at::Tensor tilingDev = GetOrCreateTilingDev(tiling, pos.device());
  auto stream = c10_npu::getCurrentNPUStream().stream(false);

  const bool capturing = StreamIsCapturing(stream);
  if (capturing && !g_path2.in_capture) {

    g_path2.in_capture = true;
    g_path2.e3_armed = false;
  } else if (!capturing) {
    g_path2.in_capture = false;
  }
  bool path2_async = Path2AsyncEnabled() && (!capturing || Path2CaptureEnabled());
  if (path2_async && capturing) {
    int device = -1;
    TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS,
                "echo_lru: cannot query current device (path2 capture)");
    if (g_path2.side == nullptr || g_path2.device != device) {
      path2_async = false;
    }
  }
  int device_index = -1;
  if (path2_async) {
    TORCH_CHECK(aclrtGetDevice(&device_index) == ACL_SUCCESS,
                "echo_lru: cannot query current device (path2)");
  }

  auto launch = [=]() -> int {

    if (path2_async) {
      Path2AsyncEnsure(device_index);
      Path2AsyncEntryWait(stream);
    }
    launch_echo_lru(htd.data_ptr<int32_t>(), dth.data_ptr<int32_t>(), lru.data_ptr<int32_t>(),
                    freeSlots.data_ptr<int32_t>(), avail.data_ptr<int32_t>(), fifo.data_ptr<int32_t>(),
                    pos.data_ptr<int32_t>(), spl.data_ptr<int32_t>(),
                    resetMask.data_ptr<uint8_t>(),
                    fresh.data_ptr<uint8_t>(),
                    curSlots.data_ptr<int32_t>(), missCount.data_ptr<int32_t>(),
                    missTokens.data_ptr<int32_t>(), missSlots.data_ptr<int32_t>(),
                    workspace.data_ptr(), tilingDev.data_ptr(),
                    tiling.topkBufferB, tiling.topkDim, tiling.specPositions,
                    blocks, reinterpret_cast<void*>(stream));

    launch_echo_v2_kernel2(lru.data_ptr<int32_t>(), dth.data_ptr<int32_t>(),
                           htd.data_ptr<int32_t>(), freeSlots.data_ptr<int32_t>(),
                           avail.data_ptr<int32_t>(), fifo.data_ptr<int32_t>(),
                           pos.data_ptr<int32_t>(), curSlots.data_ptr<int32_t>(),
                           missCount.data_ptr<int32_t>(), missTokens.data_ptr<int32_t>(),
                           missSlots.data_ptr<int32_t>(), workspace.data_ptr(),
                           tilingDev.data_ptr(), tiling.topkBufferB, tiling.topkDim,
                           tiling.rowsPerCore, tiling.specPositions, blocks,
                           reinterpret_cast<void*>(stream));

    if (!path2_async) {
      launch_echo_v2_kernel3(lru.data_ptr<int32_t>(), dth.data_ptr<int32_t>(),
                             freeSlots.data_ptr<int32_t>(), avail.data_ptr<int32_t>(),
                             fifo.data_ptr<int32_t>(), missCount.data_ptr<int32_t>(),
                             workspace.data_ptr(), tilingDev.data_ptr(),
                             tiling.topkBufferB, tiling.topkDim, tiling.rowsPerCore,
                             blocks, reinterpret_cast<void*>(stream));
      return 0;
    }

    aclError rc_e1 = aclrtRecordEvent(g_path2.e1->event(), stream);
    TORCH_CHECK(rc_e1 == ACL_SUCCESS, "echo_lru: record(E1) failed rc=", (int)rc_e1);
    aclError rc_w1 = aclrtStreamWaitEvent(g_path2.side, g_path2.e1->event());
    TORCH_CHECK(rc_w1 == ACL_SUCCESS, "echo_lru: aclrtStreamWaitEvent(side, E1) failed rc=", (int)rc_w1);
    launch_echo_v2_kernel3(lru.data_ptr<int32_t>(), dth.data_ptr<int32_t>(),
                           freeSlots.data_ptr<int32_t>(), avail.data_ptr<int32_t>(),
                           fifo.data_ptr<int32_t>(), missCount.data_ptr<int32_t>(),
                           workspace.data_ptr(), tilingDev.data_ptr(),
                           tiling.topkBufferB, tiling.topkDim, tiling.rowsPerCore,
                           blocks, reinterpret_cast<void*>(g_path2.side));
    aclError rc_e3 = aclrtRecordEvent(g_path2.e3->event(), g_path2.side);
    TORCH_CHECK(rc_e3 == ACL_SUCCESS, "echo_lru: record(E3) failed rc=", (int)rc_e3);
    g_path2.e3_armed = true;
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

  at::Tensor tilingDev = GetOrCreateTilingDev(tiling, htd.device());
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  auto launch = [=]() -> int {

    Path2AsyncEntryWait(stream);

    launch_echo_lru_init(htd.data_ptr<int32_t>(), dth.data_ptr<int32_t>(),
                         lru.data_ptr<int32_t>(), freeSlots.data_ptr<int32_t>(),
                         avail.data_ptr<int32_t>(), fifo.data_ptr<int32_t>(),
                         fresh.data_ptr<uint8_t>(), initMask.data_ptr<uint8_t>(),
                         tilingDev.data_ptr(), blocks, reinterpret_cast<void*>(stream));
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi("EchoLruInit", launch);
}

// echo_lru_join: makes the calling stream wait for the last deferred kernel 3.
void EchoJoinNpu() {
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  if (!Path2AsyncEnabled()) {
    return;
  }
  int device = -1;
  TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS,
              "echo_lru_join: cannot query current device");

  at_npu::native::OpCommand::RunOpApi("EchoLruJoin", [=]() -> int {
    if (g_path2.side == nullptr || !g_path2.e3_armed || g_path2.device != device) {
      return 0;
    }
    TORCH_CHECK(aclrtStreamWaitEvent(stream, g_path2.e3->event()) == ACL_SUCCESS,
                "echo_lru_join: aclrtStreamWaitEvent(main, E3) failed");
    g_path2.e3_armed = false;
    return 0;
  });
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

TORCH_LIBRARY_FRAGMENT(sparse_kv_plan_op, m) {
  m.def("echo_lru_join() -> ()");
}
TORCH_LIBRARY_IMPL(sparse_kv_plan_op, CatchAll, m) { m.impl("echo_lru_join", EchoJoinNpu); }
