"""Experiment 01: From Discrete Token IDs to Dense Embedding Lookup Tables.

Compares two ways of turning token IDs into vectors inside @jax.jit: materializing a
one-hot tensor and multiplying it by `W_embed`, or indexing `W_embed[token_ids]`
directly. The script inspects the real Jaxpr of both functions (the indexing becomes a
`gather`; the one-hot becomes `eq` against 32,000 columns plus a `dot_general`), checks
that both give the same result, that the gradient only reaches the rows of the token IDs
that were used, and times both, plus plain NumPy fancy indexing as a no-JAX baseline.

Run with `--show-ir` to print the compact Jaxpr of both functions.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

VOCAB_SIZE = 32_000
EMBED_DIM = 128
BATCH_SIZE = 32
SEQ_LEN = 64
PAD_ID = 0


def one_hot_matmul_embed(W_embed: jax.Array, token_ids: jax.Array, mask: jax.Array) -> jax.Array:
    """Naive embedding lookup: materialize a one-hot tensor, then matmul."""
    one_hot = jax.nn.one_hot(token_ids, num_classes=W_embed.shape[0], dtype=W_embed.dtype)
    token_vectors = jnp.einsum("bsv,vd->bsd", one_hot, W_embed)
    masked_sum = jnp.sum(token_vectors * mask[..., None], axis=1)
    lengths = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    return masked_sum / lengths


def lookup_table_embed(W_embed: jax.Array, token_ids: jax.Array, mask: jax.Array) -> jax.Array:
    """Idiomatic JAX embedding lookup: direct indexing (XLA gather) + masked mean pool."""
    token_vectors = W_embed[token_ids]
    masked_sum = jnp.sum(token_vectors * mask[..., None], axis=1)
    lengths = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    return masked_sum / lengths


def numpy_lookup_embed(W_embed: np.ndarray, token_ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """No-JAX baseline: the same lookup with NumPy fancy indexing."""
    token_vectors = W_embed[token_ids]
    masked_sum = np.sum(token_vectors * mask[..., None], axis=1)
    lengths = np.maximum(np.sum(mask, axis=1, keepdims=True), 1.0)
    return masked_sum / lengths


jit_one_hot_embed = jax.jit(one_hot_matmul_embed)
jit_lookup_embed = jax.jit(lookup_table_embed)


def _loss_lookup(W_embed: jax.Array, token_ids: jax.Array, mask: jax.Array) -> jax.Array:
    pooled = lookup_table_embed(W_embed, token_ids, mask)
    return jnp.sum(pooled**2)


grad_lookup_fn = jax.jit(jax.grad(_loss_lookup, argnums=0))


def _median_ms(fn: Callable[[], Any], n_iters: int) -> float:
    """Median wall time of `fn()` in ms (robust to noisy neighbours on a shared machine)."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def compact_jaxpr(fn: Callable[..., Any], *args: Any) -> list[str]:
    """The real Jaxpr of `fn`, one equation per line, without the long parameter dicts."""
    closed = jax.make_jaxpr(fn)(*args)
    lines: list[str] = []

    def walk(jaxpr: Any, indent: str) -> None:
        for eqn in jaxpr.eqns:
            outs = ", ".join(v.aval.str_short() for v in eqn.outvars)
            ins = ", ".join(v.aval.str_short() for v in eqn.invars)
            name = eqn.primitive.name
            if name in ("jit", "pjit") and "jaxpr" in eqn.params:
                lines.append(f"{indent}{outs} = jit[{eqn.params.get('name', '')}]({ins})")
                walk(eqn.params["jaxpr"].jaxpr, indent + "  ")
            else:
                lines.append(f"{indent}{outs} = {name}({ins})")

    walk(closed.jaxpr, "")
    return lines


def make_inputs() -> tuple[jax.Array, jax.Array, jax.Array]:
    key = jax.random.key(42)
    k_w, k_tok, k_len = jax.random.split(key, 3)
    W_embed = jax.random.normal(k_w, (VOCAB_SIZE, EMBED_DIM), dtype=jnp.float32) * 0.02
    raw_tokens = jax.random.randint(k_tok, (BATCH_SIZE, SEQ_LEN), minval=1, maxval=VOCAB_SIZE)
    valid_lens = jax.random.randint(k_len, (BATCH_SIZE, 1), minval=16, maxval=SEQ_LEN + 1)
    mask = (jnp.arange(SEQ_LEN)[None, :] < valid_lens).astype(jnp.float32)
    token_ids = jnp.where(mask > 0, raw_tokens, PAD_ID)
    return W_embed, token_ids, mask


def run_benchmark(num_iters: int = 30) -> dict[str, Any]:
    W_embed, token_ids, mask = make_inputs()
    W_np, ids_np, mask_np = np.asarray(W_embed), np.asarray(token_ids), np.asarray(mask)

    # Warmup (compiles both jitted functions) and numerical parity
    out_onehot = jit_one_hot_embed(W_embed, token_ids, mask).block_until_ready()
    out_lookup = jit_lookup_embed(W_embed, token_ids, mask).block_until_ready()
    out_numpy = numpy_lookup_embed(W_np, ids_np, mask_np)
    max_diff = float(jnp.max(jnp.abs(out_onehot - out_lookup)))
    max_diff_numpy = float(np.max(np.abs(out_numpy - np.asarray(out_lookup))))

    # The gradient w.r.t. W_embed is a dense (V, D) array whose only nonzero rows are
    # the token IDs that appear under the mask; the PAD row stays at zero.
    grads = grad_lookup_fn(W_embed, token_ids, mask).block_until_ready()
    nonzero_rows = int(jnp.sum(jnp.any(grads != 0.0, axis=1)))
    real_ids = np.unique(ids_np[mask_np > 0])
    pad_grad_norm = float(jnp.linalg.norm(grads[PAD_ID]))

    # Real Jaxpr of both functions (what @jax.jit hands to OpenXLA)
    lookup_ir = compact_jaxpr(lookup_table_embed, W_embed, token_ids, mask)
    onehot_ir = compact_jaxpr(one_hot_matmul_embed, W_embed, token_ids, mask)

    # Sizes computed from the shapes (bytes = elements x 4 for float32)
    onehot_bytes = BATCH_SIZE * SEQ_LEN * VOCAB_SIZE * 4
    gathered_bytes = BATCH_SIZE * SEQ_LEN * EMBED_DIM * 4

    # Timing: median per call after the warmup above
    onehot_ms = _median_ms(
        lambda: jit_one_hot_embed(W_embed, token_ids, mask).block_until_ready(), num_iters
    )
    lookup_ms = _median_ms(
        lambda: jit_lookup_embed(W_embed, token_ids, mask).block_until_ready(), num_iters
    )
    numpy_ms = _median_ms(lambda: numpy_lookup_embed(W_np, ids_np, mask_np), num_iters)

    return {
        "backend": jax.default_backend(),
        "vocab_size": VOCAB_SIZE,
        "embed_dim": EMBED_DIM,
        "batch_shape": (BATCH_SIZE, SEQ_LEN),
        "real_tokens": int(np.sum(mask_np)),
        "onehot_buffer_mib": onehot_bytes / (1024 * 1024),
        "gathered_buffer_mib": gathered_bytes / (1024 * 1024),
        "memory_saving_ratio": onehot_bytes / gathered_bytes,
        "lookup_has_gather": any("= gather(" in line for line in lookup_ir),
        "onehot_has_dot_general": any("= dot_general(" in line for line in onehot_ir),
        "lookup_jaxpr": lookup_ir,
        "onehot_jaxpr": onehot_ir,
        "nonzero_grad_rows": nonzero_rows,
        "unique_real_ids": int(real_ids.size),
        "pad_grad_norm": pad_grad_norm,
        "timing": f"median ms per call after a warmup, n={num_iters}",
        "onehot_ms": onehot_ms,
        "lookup_ms": lookup_ms,
        "numpy_lookup_ms": numpy_ms,
        "speedup": onehot_ms / lookup_ms,
        "speedup_vs_numpy_gather": numpy_ms / lookup_ms,
        "max_diff": max_diff,
        "max_diff_numpy": max_diff_numpy,
    }


def main() -> None:
    m = run_benchmark()
    print("=" * 72)
    print("Experiment 01: Discrete Token IDs to Dense Embedding Lookup Table")
    print("=" * 72)
    print(f"Backend Device             : {m['backend']}")
    print(f"Vocabulary & Dimension     : V={m['vocab_size']:,} | D={m['embed_dim']}")
    print(f"Token IDs Shape            : {m['batch_shape']} int32, {m['real_tokens']:,} real")
    print(f"One-Hot Tensor (computed)  : {m['onehot_buffer_mib']:.2f} MiB f32[32,64,32000]")
    print(
        f"Gathered Rows (computed)   : {m['gathered_buffer_mib']:.2f} MiB "
        f"({m['memory_saving_ratio']:.0f}x smaller)"
    )
    print(f"Lookup Jaxpr has gather    : {m['lookup_has_gather']}")
    print(f"One-Hot Jaxpr dot_general  : {m['onehot_has_dot_general']}")
    print(
        f"Rows with Nonzero Grad     : {m['nonzero_grad_rows']:,} of {m['vocab_size']:,} "
        f"(unique real IDs: {m['unique_real_ids']:,})"
    )
    print(f"PAD Row Grad Norm (ID=0)   : {m['pad_grad_norm']:.6f}")
    if "--show-ir" in sys.argv:
        print("-" * 72)
        print("Jaxpr of lookup_table_embed (W_embed[token_ids]):")
        for line in m["lookup_jaxpr"]:
            print("  " + line)
        print("Jaxpr of one_hot_matmul_embed:")
        for line in m["onehot_jaxpr"]:
            print("  " + line)
    print("-" * 72)
    print("Median ms per call after a warmup (n=30):")
    print(f"One-Hot @ W_embed (@jit)   : {m['onehot_ms']:.4f} ms")
    print(f"W_embed[token_ids] (@jit)  : {m['lookup_ms']:.4f} ms")
    print(f"NumPy W_embed[token_ids]   : {m['numpy_lookup_ms']:.4f} ms (no JAX)")
    print(f"Gather vs One-Hot (@jit)   : {m['speedup']:.1f}x")
    print(f"Gather (@jit) vs NumPy     : {m['speedup_vs_numpy_gather']:.2f}x")
    print(f"Max |One-Hot - Gather|     : {m['max_diff']:.2e}")


if __name__ == "__main__":
    main()
