"""Experiment 04: A Vector Search Engine in 40 Lines of JAX (Exact Top-K with jax.lax.top_k).

Builds an in-memory vector index of 50,000 documents (D=256) and resolves a
batch of 128 simultaneous queries using a single fused @jax.jit kernel combining
L2 normalization, `jnp.einsum("qd,nd->qn", queries, index)`, and `jax.lax.top_k`.
"""

from __future__ import annotations

import time
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
    """Pre-normalizes the document matrix (N, D) onto the unit hypersphere."""
    norms = jnp.linalg.norm(raw_vectors, axis=-1, keepdims=True)
    return raw_vectors / jnp.maximum(norms, 1e-8)


@jax.jit
def jit_search_top_k(
    query_batch: jax.Array,
    normalized_index: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Fused OpenXLA vector search engine: L2 normalize queries -> einsum -> lax.top_k."""
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
    """Baseline sequential Python per-query search over unnormalized NumPy arrays."""
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
        top_indices[i] = idx.astype(np.int32)

    return top_scores, top_indices


def run_benchmark(
    index_size: int = INDEX_SIZE,
    dim: int = EMBED_DIM,
    num_queries: int = BATCH_QUERIES,
    num_iters: int = 20,
) -> dict[str, Any]:
    rng = np.random.default_rng(123)
    raw_index_np = rng.standard_normal((index_size, dim), dtype=np.float32)

    # Plant needles in the haystack: first 8 queries are perturbed versions of known index rows
    planted_ids = np.linspace(100, index_size - 100, num_queries, dtype=np.int32)
    queries_np = raw_index_np[planted_ids] + rng.normal(0.0, 0.15, size=(num_queries, dim)).astype(np.float32)

    raw_index_jax = jnp.asarray(raw_index_np)
    queries_jax = jnp.asarray(queries_np)

    # Pre-normalize index on device
    norm_index_jax = build_normalized_index(raw_index_jax)
    norm_index_jax.block_until_ready()

    # Warmup @jax.jit search kernel
    jax_scores, jax_indices = jit_search_top_k(queries_jax, norm_index_jax)
    jax_scores.block_until_ready()

    # Baseline timing (sequential per-query search in Python/NumPy)
    t0 = time.perf_counter()
    np_scores, np_indices = numpy_sequential_search_top_k(queries_np, raw_index_np, k=TOP_K)
    numpy_ms = (time.perf_counter() - t0) * 1000.0

    # JAX @jit batched search timing
    t1 = time.perf_counter()
    for _ in range(num_iters):
        s_out, _ = jit_search_top_k(queries_jax, norm_index_jax)
        s_out.block_until_ready()
    jax_batch_ms = ((time.perf_counter() - t1) / num_iters) * 1000.0

    per_query_us = (jax_batch_ms * 1000.0) / num_queries
    speedup = numpy_ms / jax_batch_ms

    # Verify exact Top-K ID agreement and planted needle recovery
    exact_match_rate = float(np.mean(np.asarray(jax_indices) == np_indices))
    needle_recall_at_1 = float(np.mean(np.asarray(jax_indices[:, 0]) == planted_ids))
    max_score_diff = float(np.max(np.abs(np.asarray(jax_scores) - np_scores)))

    return {
        "backend": jax.default_backend(),
        "index_shape": (index_size, dim),
        "index_mib": (index_size * dim * 4) / (1024 * 1024),
        "queries_shape": (num_queries, dim),
        "top_k": TOP_K,
        "numpy_seq_ms": numpy_ms,
        "jax_batch_ms": jax_batch_ms,
        "per_query_us": per_query_us,
        "speedup": speedup,
        "exact_match_rate": exact_match_rate,
        "needle_recall_at_1": needle_recall_at_1,
        "max_score_diff": max_score_diff,
    }


def main() -> None:
    metrics = run_benchmark()
    print("=" * 72)
    print("Experiment 04: @jax.jit Vector Search Engine (einsum + lax.top_k)")
    print("=" * 72)
    print(f"Backend Device             : {metrics['backend']}")
    print(f"Vector Index Shape         : {metrics['index_shape']} float32 ({metrics['index_mib']:.1f} MiB)")
    print(f"Query Batch Shape          : {metrics['queries_shape']} (Top-K={metrics['top_k']})")
    print(f"Planted Needle Recall@1    : {metrics['needle_recall_at_1'] * 100:.1f}%")
    print(f"Exact Top-K Agreement      : {metrics['exact_match_rate'] * 100:.1f}% (max diff={metrics['max_score_diff']:.2e})")
    print("-" * 72)
    print(f"Sequential Python/NumPy    : {metrics['numpy_seq_ms']:.4f} ms (128 queries)")
    print(f"Fused JAX @jit Search      : {metrics['jax_batch_ms']:.4f} ms (128 queries)")
    print(f"Latency per Query (@jit)   : {metrics['per_query_us']:.2f} us/query")
    print(f"Speedup vs Sequential      : {metrics['speedup']:.2f}x")


if __name__ == "__main__":
    main()
