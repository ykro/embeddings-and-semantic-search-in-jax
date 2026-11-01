"""Experiment 02: Geometry of Meaning — L2 Normalization and Cosine Similarity in @jax.jit.

Shows why Euclidean distance breaks when document length scales vector magnitude,
and how projecting vectors onto the unit hypersphere (`v / norm(v)`) reduces
all-pairs cosine similarity to a single batched `jnp.einsum("qd,nd->qn", Q, K)`
compiled by OpenXLA.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def l2_normalize(vectors: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Projects vectors onto the unit hypersphere along the last axis."""
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / jnp.maximum(norms, eps)


@jax.jit
def batched_cosine_similarity(queries: jax.Array, documents: jax.Array) -> jax.Array:
    """Computes exact all-pairs cosine similarity via L2 norm + fused einsum."""
    q_norm = l2_normalize(queries)
    d_norm = l2_normalize(documents)
    return jnp.einsum("qd,nd->qn", q_norm, d_norm)


def numpy_loop_cosine_similarity(queries: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """Baseline sequential Python/NumPy loop computing cosine similarity pair-by-pair."""
    num_q = queries.shape[0]
    num_d = documents.shape[0]
    out = np.empty((num_q, num_d), dtype=np.float32)
    doc_norms = np.linalg.norm(documents, axis=-1)
    for i in range(num_q):
        q = queries[i]
        q_norm = np.linalg.norm(q)
        for j in range(num_d):
            out[i, j] = np.dot(q, documents[j]) / (q_norm * doc_norms[j] + 1e-8)
    return out


def demonstrate_length_invariance() -> dict[str, Any]:
    """Demonstrates how Euclidean distance fails under document length scaling while cosine succeeds."""
    # Query: short ticket about database timeout
    query = jnp.array([0.82, 0.54, 0.12, -0.10], dtype=jnp.float32)

    # Doc A: long detailed runbook on the exact same topic (5x magnitude due to word count)
    doc_same_topic_long = query * 5.0 + jnp.array([0.02, -0.01, 0.01, 0.0], dtype=jnp.float32)

    # Doc B: short unrelated ticket about CSS button alignment (similar magnitude to query)
    doc_unrelated_short = jnp.array([0.10, -0.15, 0.85, 0.48], dtype=jnp.float32)

    docs = jnp.stack([doc_same_topic_long, doc_unrelated_short], axis=0)

    # Euclidean distances (smaller = closer)
    euclid_dists = jnp.linalg.norm(docs - query[None, :], axis=-1)
    euclid_winner = int(jnp.argmin(euclid_dists))  # Wrongly picks index 1 (unrelated short doc)

    # Cosine similarities (larger = closer)
    cos_sims = batched_cosine_similarity(query[None, :], docs)[0]
    cosine_winner = int(jnp.argmax(cos_sims))  # Correctly picks index 0 (same topic long doc)

    return {
        "euclid_same_topic_long": float(euclid_dists[0]),
        "euclid_unrelated_short": float(euclid_dists[1]),
        "euclid_picks_correct_doc": euclid_winner == 0,
        "cosine_same_topic_long": float(cos_sims[0]),
        "cosine_unrelated_short": float(cos_sims[1]),
        "cosine_picks_correct_doc": cosine_winner == 0,
    }


def run_benchmark(
    num_queries: int = 64,
    num_docs: int = 2048,
    dim: int = 256,
    num_iters: int = 50,
) -> dict[str, Any]:
    geom_check = demonstrate_length_invariance()

    rng = np.random.default_rng(42)
    q_np = rng.standard_normal((num_queries, dim), dtype=np.float32)
    d_np = rng.standard_normal((num_docs, dim), dtype=np.float32)

    q_jax = jnp.asarray(q_np)
    d_jax = jnp.asarray(d_np)

    # Warmup JAX @jit
    jax_out = batched_cosine_similarity(q_jax, d_jax)
    jax_out.block_until_ready()

    # Check unit norms on hypersphere
    norms = jnp.linalg.norm(l2_normalize(d_jax), axis=-1)
    max_norm_error = float(jnp.max(jnp.abs(norms - 1.0)))

    # Measure Python/NumPy loop baseline (1 run since loop over 64x2048 = 131,072 pairs is slow)
    t0 = time.perf_counter()
    np_out = numpy_loop_cosine_similarity(q_np, d_np)
    numpy_loop_ms = (time.perf_counter() - t0) * 1000.0

    # Measure JAX @jit batched einsum
    t1 = time.perf_counter()
    for _ in range(num_iters):
        res = batched_cosine_similarity(q_jax, d_jax)
        res.block_until_ready()
    jax_jit_ms = ((time.perf_counter() - t1) / num_iters) * 1000.0

    max_diff = float(np.max(np.abs(np_out - np.asarray(jax_out))))
    speedup = numpy_loop_ms / jax_jit_ms

    return {
        "backend": jax.default_backend(),
        "queries_shape": (num_queries, dim),
        "docs_shape": (num_docs, dim),
        "max_norm_error": max_norm_error,
        "numpy_loop_ms": numpy_loop_ms,
        "jax_jit_ms": jax_jit_ms,
        "speedup": speedup,
        "max_diff": max_diff,
        **geom_check,
    }


def main() -> None:
    metrics = run_benchmark()
    print("=" * 72)
    print("Experiment 02: Vector Space Geometry, L2 Norm & Cosine Similarity")
    print("=" * 72)
    print(f"Backend Device             : {metrics['backend']}")
    print(f"Query x Document Shapes    : {metrics['queries_shape']} x {metrics['docs_shape']}")
    print(f"Unit Hypersphere Norm Err  : {metrics['max_norm_error']:.2e}")
    print("-" * 72)
    print("Length-Scale Trap (Long Paraphrase vs Short Unrelated Ticket):")
    print(
        f"  Euclidean Distance       : Long Same-Topic={metrics['euclid_same_topic_long']:.3f} "
        f"| Short Unrelated={metrics['euclid_unrelated_short']:.3f} "
        f"(Correct={metrics['euclid_picks_correct_doc']})"
    )
    print(
        f"  Cosine Similarity        : Long Same-Topic={metrics['cosine_same_topic_long']:.4f} "
        f"| Short Unrelated={metrics['cosine_unrelated_short']:.4f} "
        f"(Correct={metrics['cosine_picks_correct_doc']})"
    )
    print("-" * 72)
    print(f"Python/NumPy Pairwise Loop : {metrics['numpy_loop_ms']:.4f} ms")
    print(f"JAX @jit L2 + einsum       : {metrics['jax_jit_ms']:.4f} ms")
    print(f"Speedup vs Pairwise Loop   : {metrics['speedup']:.2f}x")
    print(f"Max |NumPy - JAX @jit|     : {metrics['max_diff']:.2e}")


if __name__ == "__main__":
    main()
