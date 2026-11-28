#!/usr/bin/env bash
# Build the Echo LRU plan operator library (Ascend 950 / dav-3510).
#
# Requires an initialised CANN development environment (provides bisheng) and a
# Python with torch and torch_npu. Verified with CANN 9.2.1 on aarch64.
#
#   bash build.sh                      # -> build/libecho_lru_plan_deferred.so
#   bash build.sh /tmp/out             # another output directory
#
# Two device translation units are compiled with bisheng, then the host-side
# torch operator is compiled and linked with the system C++ compiler. The flags
# are the ones used for the measurements in RESULTS.md; keep them unchanged.
set -euo pipefail
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/src
out=${1:-"$src/../build"}
python=${PYTHON:-python3}
cann=${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:?Source the CANN environment first}}
arch=$(uname -m)
mapfile -t paths < <("$python" - <<'PY'
import importlib.util, pathlib, sysconfig, torch
for package in ('torch', 'torch_npu'):
    print(pathlib.Path(importlib.util.find_spec(package).origin).parent)
print(sysconfig.get_path('include'))
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
PY
)
inc=(-I"$cann/include")
if [[ -d "$cann/asc/include" ]]; then inc+=(-I"$cann/asc/include"); fi
kernel=(--npu-arch=dav-3510 -xasc -std=c++17 -fPIC -O3 --cce-auto-sync=off
        -Wno-deprecated-declarations -mllvm -cce-vf-remove-membar=false
        -mllvm -cce-aicore-hoist-movemask=false)
mkdir -p "$out"
# Kernel 1 (lookup and classification) and kernel 2 (merge, admission, bookkeeping).
"$cann/bin/bisheng" "${kernel[@]}" -DECHO_LRU_THREAD_NUM_OVERRIDE=1024 "${inc[@]}" -I"$src" \
    -c "$src/echo_lru_launch.cpp" -o "$out/kernel_scan.o"
"$cann/bin/bisheng" "${kernel[@]}" "${inc[@]}" -I"$src" \
    -c "$src/echo_v2_kernel2.cpp" -o "$out/kernel_merge.o"
"${CXX:-c++}" -std=c++17 -O3 -DNDEBUG -fPIC -shared -DECHO_LRU_THREAD_NUM_OVERRIDE=1024 \
    -D_GLIBCXX_USE_CXX11_ABI="${paths[3]}" \
    -I"${paths[0]}/include" -I"${paths[0]}/include/torch/csrc/api/include" -I"${paths[1]}/include" \
    -I"${paths[2]}" "${inc[@]}" -I"$src" "$src/echo_lru_plugin.cpp" "$out/kernel_scan.o" "$out/kernel_merge.o" \
    -L"${paths[0]}/lib" -L"${paths[1]}/lib" -L"$cann/lib64" -L"$cann/$arch-linux/lib64" \
    -ltorch -ltorch_cpu -ltorch_python -lc10 -ltorch_npu -lascendcl -ltiling_api -lplatform \
    "$cann/$arch-linux/lib64/libascendc_runtime.a" -o "$out/libecho_lru_plan_deferred.so"
echo "built: $out/libecho_lru_plan_deferred.so"
