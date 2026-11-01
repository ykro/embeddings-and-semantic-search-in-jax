"""Experiment 01: From Discrete Token IDs to Dense Embedding Lookup Tables.

Demonstrates why one-hot vectors over a 32,000-token vocabulary waste memory and
cannot represent similarity, and how JAX's array indexing `W_embed[token_ids]`
lowers to an XLA Gather primitive that is mathematically identical to
`one_hot(token_ids) @ W_embed`, differentiable via `jax.grad`, and orders of
magnitude faster.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp

VOCAB_SIZE = 32_000
EMBED_DIM = 128
BATCH_SIZE = 32
SEQ_LEN = 64
PAD_ID = 0


def one_hot_matmul_embed(
    W_embed: jax.Array,
    token_ids: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Naive embedding lookup via explicit one-hot materialization + matmul."""
    one_hot = jax.nn.one_hot(token_ids, num_classes=W_embed.shape[0], dtype=W_embed.dtype)
    token_vectors = jnp.einsum("bsv,vd->bsd", one_hot, W_embed)
    masked_sum = jnp.sum(token_vectors * mask[..., None], axis=1)
    lengths = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    return masked_sum / lengths


def lookup_table_embed(
    W_embed: jax.Array,
    token_ids: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Idiomatic JAX embedding lookup via direct indexing (XLA Gather) + masked pool."""
    token_vectors = W_embed[token_ids]
    masked_sum = jnp.sum(token_vectors * mask[..., None], axis=1)
    lengths = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    return masked_sum / lengths


jit_one_hot_embed = jax.jit(one_hot_matmul_embed)
jit_lookup_embed = jax.jit(lookup_table_embed)


def _loss_lookup(W_embed: jax.Array, token_ids: jax.Array, mask: jax.Array) -> jax.Array:
    pooled = lookup_table_embed(W_embed, token_ids, mask)
    return jnp.sum(pooled**2)


grad_lookup_fn = jax.jit(jax.grad(_loss_lookup, argnums=0))


def run_benchmark(
    num_iters: int = 30,
) -> dict[str, Any]:
    key = jax.random.key(42)
    k_w, k_tok, k_len = jax.random.split(key, 3)

    W_embed = jax.random.normal(k_w, (VOCAB_SIZE, EMBED_DIM), dtype=jnp.float32) * 0.02
    raw_tokens = jax.random.randint(k_tok, (BATCH_SIZE, SEQ_LEN), minval=1, maxval=VOCAB_SIZE)
    valid_lens = jax.random.randint(k_len, (BATCH_SIZE, 1), minval=16, maxval=SEQ_LEN + 1)
    positions = jnp.arange(SEQ_LEN)[None, :]
    mask = (positions < valid_lens).astype(jnp.float32)
    token_ids = jnp.where(mask > 0, raw_tokens, PAD_ID)

    # Warmup and numerical parity check
    out_onehot = jit_one_hot_embed(W_embed, token_ids, mask)
    out_onehot.block_until_ready()
    out_lookup = jit_lookup_embed(W_embed, token_ids, mask)
    out_lookup.block_until_ready()

    max_diff = float(jnp.max(jnp.abs(out_onehot - out_lookup)))

    # Verify sparse gradient flow only touches active token IDs
    grads = grad_lookup_fn(W_embed, token_ids, mask)
    grads.block_until_ready()
    pad_grad_norm = float(jnp.linalg.norm(grads[PAD_ID]))
    active_token = int(token_ids[0, 0])
    active_grad_norm = float(jnp.linalg.norm(grads[active_token]))

    # Memory footprint comparison (in MiB)
    onehot_bytes = BATCH_SIZE * SEQ_LEN * VOCAB_SIZE * 4
    dense_out_bytes = BATCH_SIZE * SEQ_LEN * EMBED_DIM * 4
    memory_ratio = onehot_bytes / dense_out_bytes

    # Timing One-Hot Matmul (@jit)
    t0 = time.perf_counter()
    for _ in range(num_iters):
        res_oh = jit_one_hot_embed(W_embed, token_ids, mask)
        res_oh.block_until_ready()
    onehot_ms = ((time.perf_counter() - t0) / num_iters) * 1000.0

    # Timing Direct Lookup Table (@jit)
    t1 = time.perf_counter()
    for _ in range(num_iters):
        res_lu = jit_lookup_embed(W_embed, token_ids, mask)
        res_lu.block_until_ready()
    lookup_ms = ((time.perf_counter() - t1) / num_iters) * 1000.0

    speedup = onehot_ms / lookup_ms

    return {
        "backend": jax.default_backend(),
        "vocab_size": VOCAB_SIZE,
        "embed_dim": EMBED_DIM,
        "batch_shape": (BATCH_SIZE, SEQ_LEN),
        "onehot_buffer_mib": onehot_bytes / (1024 * 1024),
        "dense_buffer_mib": dense_out_bytes / (1024 * 1024),
        "memory_saving_ratio": memory_ratio,
        "onehot_ms": onehot_ms,
        "lookup_ms": lookup_ms,
        "speedup": speedup,
        "max_diff": max_diff,
        "pad_grad_norm": pad_grad_norm,
        "active_grad_norm": active_grad_norm,
    }


def main() -> None:
    metrics = run_benchmark()
    print("=" * 72)
    print("Experiment 01: Discrete Token IDs to Dense Embedding Lookup Table")
    print("=" * 72)
    print(f"Backend Device             : {metrics['backend']}")
    print(f"Vocabulary & Dimension     : V={metrics['vocab_size']:,} | D={metrics['embed_dim']}")
    print(f"Batch Token Shape          : {metrics['batch_shape']} int32")
    print(f"One-Hot Tensor Footprint   : {metrics['onehot_buffer_mib']:.2f} MiB")
    print(f"Dense Gather Footprint     : {metrics['dense_buffer_mib']:.2f} MiB ({metrics['memory_saving_ratio']:.0f}x smaller)")
    print(f"PAD Token Grad Norm (ID=0) : {metrics['pad_grad_norm']:.6f} (masked out)")
    print(f"Active Token Grad Norm     : {metrics['active_grad_norm']:.6f} (differentiable)")
    print("-" * 72)
    print(f"One-Hot @ W_embed (@jit)   : {metrics['onehot_ms']:.4f} ms")
    print(f"W_embed[token_ids] (@jit)  : {metrics['lookup_ms']:.4f} ms")
    print(f"Lookup Speedup vs One-Hot  : {metrics['speedup']:.2f}x")
    print(f"Max |One-Hot - Gather|     : {metrics['max_diff']:.2e}")


if __name__ == "__main__":
    main()
