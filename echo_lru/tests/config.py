"""Paths and candidate matrix of the echo_lru test harness.

This is the only file that usually needs editing on another machine. Every
path can be overridden with an environment variable.

Shape/method facts (do not silently broaden):
  * every candidate takes topk=2048;
  * capacity follows the method and the number of positions (P = MTP + 1):
      P=4 -> 8192, P=1/2 -> 4096.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # the echo_lru directory

# ---------------------------------------------------------------------------
# CPU references: the golden oracles. Do not edit them.
#   ref/plan_reference.py            admission order = token ascending
#   ref/plan_unordered_reference.py  admission order = flat first occurrence
# ---------------------------------------------------------------------------
REF_DIR = HERE / "ref"
PLAN_REFERENCE = REF_DIR / "plan_reference.py"
PLAN_UNORDERED_REFERENCE = REF_DIR / "plan_unordered_reference.py"

# ---------------------------------------------------------------------------
# Candidate libraries. All of them register the same `sparse_kv_plan_op.echo_lru`
# operator, so the harness runs one worker process per method.
# ---------------------------------------------------------------------------
PLAN_SO = os.environ.get("KV_PLAN_SO", str(ROOT / "plan" / "build" / "libecho_lru_plan.so"))
PLAN_DEFERRED_SO = os.environ.get(
    "KV_PLAN_DEFERRED_SO", str(ROOT / "plan_deferred" / "build" / "libecho_lru_plan_deferred.so"))
PLAN_UNORDERED_SO = os.environ.get(
    "KV_PLAN_UNORDERED_SO",
    str(ROOT / "plan_unordered" / "build" / "libecho_lru_plan_unordered.so"))

# ---------------------------------------------------------------------------
# Upstream QLI.
#   original : torch_npu.npu_quant_lightning_indexer, no extra package needed.
#   ordered  : the ordered QLI package (third-party, not part of this
#              repository). Set KV_ORDER_ENV to its set_env.bash and, if the
#              module name differs, KV_ORDER_MODULE.
# ---------------------------------------------------------------------------
ORDER_ENV = os.environ.get("KV_ORDER_ENV", "/path/to/qli_vendor/bin/set_env.bash")
ORDER_MODULE = os.environ.get(
    "KV_ORDER_MODULE",
    "cann_ops_transformer_qliorder.ops.attention.quant_lightning_indexer_ordered")

# ---------------------------------------------------------------------------
# Candidate matrix
#   qli: "ordered"   -> the plan consumes the ordered QLI output
#        "original"  -> the plan consumes the original (unordered) QLI output
#   abi: "echo"      -> sparse_kv_plan_op.echo_lru
#   ref: "plan"           -> token-ascending reference
#        "plan_unordered" -> flat first-occurrence reference
# ---------------------------------------------------------------------------
METHODS = {
    "plan": {"so": PLAN_SO, "qli": "ordered", "abi": "echo", "ref": "plan"},
    "plan_deferred": {"so": PLAN_DEFERRED_SO, "qli": "ordered", "abi": "echo",
                      "ref": "plan"},
    "plan_unordered": {"so": PLAN_UNORDERED_SO, "qli": "original", "abi": "echo",
                       "ref": "plan_unordered"},
}


def method_enabled(method: str) -> bool:
    """A method is runnable only if its library exists."""
    return Path(METHODS[method]["so"]).exists()


def missing_methods(methods) -> dict:
    return {m: METHODS[m]["so"] for m in methods if not method_enabled(m)}
