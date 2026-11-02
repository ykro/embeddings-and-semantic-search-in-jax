"""Experiment 04: An Exact Vector Search Engine in JAX (einsum + jax.lax.top_k).

Builds an in-memory index of 50,000 L2-normalized vectors (D=256) and answers a batch of
128 queries with one @jax.jit function: normalize the queries, score all 6.4 million
(query, document) pairs with `jnp.einsum("qd,nd->qn", ...)` and keep the best 5 per query
with `jax.lax.top_k`. Every query is a noisy copy of a known index row (a planted needle).

Two NumPy baselines:
- naive (the original one): normalizes the raw index inside every call, then loops over
  the queries in Python, scores each one with an elementwise multiply + sum and sorts all
  50,000 scores with argsort to keep 5
- fair: the same pre-normalized index as JAX and the same batched math (one matmul +
  argpartition), no JAX

The 128-query time is batch throughput; the script also measures one query on its own
(latency), which still has to scan the whole index.

Run with `--show-ir` to print what XLA actually compiled (fusions and the TopK call).
"""

from __future__ import annotations

import re
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

INDEX_SIZE = 50_000
EMBED_DIM = 256
BATCH_QUERIES = 128
TOP_K = 5


@jax.jit
def build_normalized_index(raw_vectors: jax.Array) -> jax.Array:
    """Pre-normalizes the document matrix (N, D) onto the unit hypersphere (done once)."""
    norms = jnp.linalg.norm(raw_vectors, axis=-1, keepdims=True)
    return raw_vectors / jnp.maximum(norms, 1e-8)


@jax.jit
def jit_search_top_k(
    query_batch: jax.Array,
    normalized_index: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compiled search: L2 normalize queries -> einsum scores -> lax.top_k."""
    q_norms = jnp.linalg.norm(query_batch, axis=-1, keepdims=True)
    q_unit = query_batch / jnp.maximum(q_norms, 1e-8)
    scores = jnp.einsum("qd,nd->qn", q_unit, normalized_index)
    top_scores, top_indices = jax.lax.top_k(scores, k=TOP_K)
    return top_scores, top_indices


def numpy_sequential_search_top_k(
    query_batch: np.ndarray,
    raw_index: np.ndarray,
    k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Naive baseline: normalizes the index, then one query at a time with a full argsort."""
    index_norms = np.linalg.norm(raw_index, axis=-1, keepdims=True)
    norm_index = raw_index / np.maximum(index_norms, 1e-8)
    num_q = query_batch.shape[0]
    top_scores = np.empty((num_q, k), dtype=np.float32)
    top_indices = np.empty((num_q, k), dtype=np.int32)
    for i in range(num_q):
        q = query_batch[i]
        q_unit = q / max(float(np.linalg.norm(q)), 1e-8)
        sims_full = np.sum(norm_index * q_unit, axis=-1)
        idx = np.argsort(-sims_full)[:k]
        top_scores[i] = sims_full[idx]
        top_indices[i] = idx
    return top_scores, top_indices


def numpy_batched_search_top_k(
    query_batch: np.ndarray,
    norm_index: np.ndarray,
    k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Fair baseline: the same batched math in NumPy (one matmul + argpartition)."""
    q_unit = query_batch / np.maximum(np.linalg.norm(query_batch, axis=-1, keepdims=True), 1e-8)
    scores = q_unit @ norm_index.T
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part_scores, order, axis=1), np.take_along_axis(part, order, axis=1)


def _median_ms(fn: Callable[[], Any], n_iters: int) -> float:
    """Median wall time of `fn()` in ms (robust to noisy neighbours on a shared machine)."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def compiled_program_summary(query_batch: jax.Array, norm_index: jax.Array) -> dict[str, Any]:
    """Inspects the optimized HLO that XLA compiled for `jit_search_top_k`."""
    hlo = jit_search_top_k.lower(query_batch, norm_index).compile().as_text()
    fusions = len(re.findall(r"^\s*(?:ROOT )?%\S+ = \S+ fusion\(", hlo, re.MULTILINE))
    topk_lines = [ln.strip() for ln in hlo.splitlines() if 'custom_call_target="TopK"' in ln]
    return {"hlo_fusions": fusions, "hlo_topk_custom_call": bool(topk_lines), "hlo_text": hlo}


def run_benchmark(
    index_size: int = INDEX_SIZE,
    dim: int = EMBED_DIM,
    num_queries: int = BATCH_QUERIES,
    num_iters: int = 20,
    loop_iters: int = 3,
) -> dict[str, Any]:
    rng = np.random.default_rng(123)
    raw_index_np = rng.standard_normal((index_size, dim), dtype=np.float32)

    # Planted needles: every query is a noisy copy of a known index row
    planted_ids = np.linspace(100, index_size - 100, num_queries, dtype=np.int32)
    noise = rng.normal(0.0, 0.15, size=(num_queries, dim)).astype(np.float32)
    queries_np = raw_index_np[planted_ids] + noise

    # JAX and the fair baseline share an index normalized once, before any timing;
    # the naive baseline receives the raw index and normalizes it inside every call
    norm_index_jax = build_normalized_index(jnp.asarray(raw_index_np)).block_until_ready()
    norm_index_np = np.asarray(norm_index_jax)
    queries_jax = jnp.asarray(queries_np)

    # Warmup: the first call traces and compiles; it is not timed
    jax_scores, jax_indices = jit_search_top_k(queries_jax, norm_index_jax)
    jax_scores.block_until_ready()
    seq_scores, seq_indices = numpy_sequential_search_top_k(queries_np, raw_index_np)
    one_query = queries_jax[:1]
    jit_search_top_k(one_query, norm_index_jax)[0].block_until_ready()
    _, bat_indices = numpy_batched_search_top_k(queries_np, norm_index_np)

    numpy_seq_ms = _median_ms(
        lambda: numpy_sequential_search_top_k(queries_np, raw_index_np), loop_iters
    )
    numpy_batched_ms = _median_ms(
        lambda: numpy_batched_search_top_k(queries_np, norm_index_np), num_iters
    )
    jax_batch_ms = _median_ms(
        lambda: jit_search_top_k(queries_jax, norm_index_jax)[0].block_until_ready(), num_iters
    )

    single_query_ms = _median_ms(
        lambda: jit_search_top_k(one_query, norm_index_jax)[0].block_until_ready(), num_iters
    )

    jax_idx = np.asarray(jax_indices)
    summary = compiled_program_summary(queries_jax, norm_index_jax)
    return {
        "backend": jax.default_backend(),
        "index_shape": (index_size, dim),
        "index_mib": (index_size * dim * 4) / (1024 * 1024),
        "queries_shape": (num_queries, dim),
        "top_k": TOP_K,
        "scored_pairs": num_queries * index_size,
        "timing": f"median ms per call after a warmup, n={num_iters} (loop n={loop_iters})",
        "numpy_seq_ms": numpy_seq_ms,
        "numpy_batched_ms": numpy_batched_ms,
        "jax_batch_ms": jax_batch_ms,
        "per_query_us": (jax_batch_ms * 1000.0) / num_queries,
        "single_query_ms": single_query_ms,
        "speedup": numpy_seq_ms / jax_batch_ms,
        "speedup_vs_numpy_batched": numpy_batched_ms / jax_batch_ms,
        "exact_match_rate": float(np.mean(jax_idx == seq_indices)),
        "exact_match_rate_batched": float(np.mean(jax_idx == bat_indices)),
        "needle_recall_at_1": float(np.mean(jax_idx[:, 0] == planted_ids)),
        "max_score_diff": float(np.max(np.abs(np.asarray(jax_scores) - seq_scores))),
        "hlo_fusions": summary["hlo_fusions"],
        "hlo_topk_custom_call": summary["hlo_topk_custom_call"],
        "_hlo_text": summary["hlo_text"],
    }


def main() -> None:
    m = run_benchmark()
    print("=" * 72)
    print("Experiment 04: @jax.jit Vector Search Engine (einsum + lax.top_k)")
    print("=" * 72)
    print(f"Backend Device             : {m['backend']}")
    print(f"Vector Index Shape         : {m['index_shape']} float32 ({m['index_mib']:.1f} MiB)")
    print(f"Query Batch Shape          : {m['queries_shape']} (Top-K={m['top_k']})")
    print(f"Scores per Batch           : {m['scored_pairs']:,} (query, doc) pairs")
    print(f"Planted Needle Recall@1    : {m['needle_recall_at_1'] * 100:.1f}%")
    print(
        f"Same Top-5 IDs as NumPy    : {m['exact_match_rate'] * 100:.1f}% "
        f"(max score diff {m['max_score_diff']:.2e})"
    )
    print(
        f"Compiled XLA program       : {m['hlo_fusions']} fusions + TopK call "
        f"({m['hlo_topk_custom_call']})"
    )
    if "--show-ir" in sys.argv:
        print("-" * 72)
        print("Optimized HLO entry computation (jit_search_top_k):")
        entry = m["_hlo_text"].split("ENTRY", 1)[1]
        for line in entry.splitlines()[1:]:
            line = line.strip()
            if not line or line == "}":
                break
            op = re.match(r"(?:ROOT )?(%\S+) = (\(.*?\)|\S+) ([\w-]+)\(", line)
            if op:
                shape = re.sub(r"\{[\d,]*\}", "", op.group(2))
                target = re.search(r'custom_call_target="(\w+)"', line)
                kind = f"custom-call {target.group(1)}" if target else op.group(3)
                print(f"  {op.group(1)} = {kind} -> {shape}")
    print("-" * 72)
    print("Median ms per call after a warmup (loop n=3, others n=20):")
    print(f"Python loop + argsort      : {m['numpy_seq_ms']:.4f} ms (naive)")
    print(f"NumPy batched (matmul)     : {m['numpy_batched_ms']:.4f} ms (no JAX)")
    print(f"JAX @jit einsum + top_k    : {m['jax_batch_ms']:.4f} ms")
    print(f"Batch / 128 (throughput)   : {m['per_query_us']:.1f} us per query")
    print(f"1 Query Alone (latency)    : {m['single_query_ms']:.4f} ms (@jit)")
    print(f"@jit vs Python loop        : {m['speedup']:.1f}x")
    print(f"@jit vs NumPy batched      : {m['speedup_vs_numpy_batched']:.2f}x")


if __name__ == "__main__":
    main()
