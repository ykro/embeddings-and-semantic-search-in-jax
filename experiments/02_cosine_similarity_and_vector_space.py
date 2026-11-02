"""Experiment 02: Geometry of Meaning — L2 Normalization and Cosine Similarity in @jax.jit.

Shows, with two hand-made 4-D vectors, why Euclidean distance breaks when document
length scales vector magnitude, and how projecting vectors onto the unit hypersphere
(`v / norm(v)`) reduces all-pairs cosine similarity to one batched
`jnp.einsum("qd,nd->qn", Q, K)` compiled by OpenXLA.

Timings: the pair-by-pair Python loop is the naive baseline; the same math vectorized in
NumPy (normalize + one matmul) is the fair one. Both are reported.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
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
    """Exact all-pairs cosine similarity: L2 normalize both sides + one einsum."""
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


def numpy_vectorized_cosine_similarity(queries: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """Fair baseline: the same math vectorized in NumPy (L2 normalize + one matmul)."""
    q_unit = queries / np.maximum(np.linalg.norm(queries, axis=-1, keepdims=True), 1e-8)
    d_unit = documents / np.maximum(np.linalg.norm(documents, axis=-1, keepdims=True), 1e-8)
    return q_unit @ d_unit.T


def _median_ms(fn: Callable[[], Any], n_iters: int) -> float:
    """Median wall time of `fn()` in ms (robust to noisy neighbours on a shared machine)."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def demonstrate_length_invariance() -> dict[str, Any]:
    """Euclidean distance fails when a document is just longer; cosine similarity does not."""
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
    loop_iters: int = 3,
) -> dict[str, Any]:
    geom_check = demonstrate_length_invariance()

    rng = np.random.default_rng(42)
    q_np = rng.standard_normal((num_queries, dim), dtype=np.float32)
    d_np = rng.standard_normal((num_docs, dim), dtype=np.float32)

    q_jax = jnp.asarray(q_np)
    d_jax = jnp.asarray(d_np)

    # Warmup: the first call traces and compiles; it is not timed
    jax_out = batched_cosine_similarity(q_jax, d_jax).block_until_ready()

    # Check unit norms on hypersphere
    norms = jnp.linalg.norm(l2_normalize(d_jax), axis=-1)
    max_norm_error = float(jnp.max(jnp.abs(norms - 1.0)))

    # Parity of both baselines against the compiled einsum
    np_out = numpy_loop_cosine_similarity(q_np, d_np)
    np_vec_out = numpy_vectorized_cosine_similarity(q_np, d_np)
    max_diff = float(np.max(np.abs(np_out - np.asarray(jax_out))))
    max_diff_vectorized = float(np.max(np.abs(np_vec_out - np.asarray(jax_out))))

    # Timing: median per call (the loop does 131,072 Python iterations, so fewer repeats)
    numpy_loop_ms = _median_ms(lambda: numpy_loop_cosine_similarity(q_np, d_np), loop_iters)
    numpy_vec_ms = _median_ms(lambda: numpy_vectorized_cosine_similarity(q_np, d_np), num_iters)
    jax_jit_ms = _median_ms(
        lambda: batched_cosine_similarity(q_jax, d_jax).block_until_ready(), num_iters
    )

    return {
        "backend": jax.default_backend(),
        "queries_shape": (num_queries, dim),
        "docs_shape": (num_docs, dim),
        "max_norm_error": max_norm_error,
        "timing": f"median ms per call after a warmup, n={num_iters} (loop n={loop_iters})",
        "numpy_loop_ms": numpy_loop_ms,
        "numpy_vectorized_ms": numpy_vec_ms,
        "jax_jit_ms": jax_jit_ms,
        "speedup": numpy_loop_ms / jax_jit_ms,
        "speedup_vs_numpy_vectorized": numpy_vec_ms / jax_jit_ms,
        "max_diff": max_diff,
        "max_diff_vectorized": max_diff_vectorized,
        **geom_check,
    }


def main() -> None:
    m = run_benchmark()
    print("=" * 72)
    print("Experiment 02: Vector Space Geometry, L2 Norm & Cosine Similarity")
    print("=" * 72)
    print(f"Backend Device             : {m['backend']}")
    print(f"Query x Document Shapes    : {m['queries_shape']} x {m['docs_shape']}")
    print(f"Unit Hypersphere Norm Err  : {m['max_norm_error']:.2e}")
    print("-" * 72)
    print("Length trap (toy 4-D vectors): long same-topic doc vs short unrelated")
    print(
        f"Euclidean (lower=closer)   : same={m['euclid_same_topic_long']:.3f} "
        f"| unrelated={m['euclid_unrelated_short']:.3f} -> correct={m['euclid_picks_correct_doc']}"
    )
    print(
        f"Cosine (higher=closer)     : same={m['cosine_same_topic_long']:.4f} "
        f"| unrelated={m['cosine_unrelated_short']:.4f} -> correct={m['cosine_picks_correct_doc']}"
    )
    print("-" * 72)
    print("Median ms per call after a warmup (loop n=3, others n=50):")
    print(f"Python Pairwise Loop       : {m['numpy_loop_ms']:.4f} ms (131,072 pairs)")
    print(f"NumPy Vectorized (matmul)  : {m['numpy_vectorized_ms']:.4f} ms (no JAX)")
    print(f"JAX @jit L2 + einsum       : {m['jax_jit_ms']:.4f} ms")
    print(f"@jit vs Pairwise Loop      : {m['speedup']:.1f}x")
    print(f"@jit vs NumPy Vectorized   : {m['speedup_vs_numpy_vectorized']:.2f}x")
    print(f"Max |Loop - JAX @jit|      : {m['max_diff']:.2e}")


if __name__ == "__main__":
    main()
