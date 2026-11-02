"""Experiment 05: Ticket & Intent Classification on Top of Frozen Embeddings.

Compares two practical ways to build an intent classifier on top of a frozen
contrastive embedding encoder (from Experiment 03) without touching its weights:
1. Centroid classifier, often called "zero-shot" because nothing is trained: the 32
   labeled training embeddings are averaged into 4 class prototypes and a new ticket goes
   to the closest one by cosine. Strictly speaking it uses labeled examples (few-shot
   prototypes), it just trains 0 parameters.
2. Linear probe: a 64 x 4 linear layer + bias (260 parameters) trained with cross-entropy.

Both are evaluated on 12 unseen tickets (3 per category), so each ticket is 8.3 points of
accuracy. The tickets reuse vocabulary from the training domains; it is a small toy test.
"""

from __future__ import annotations

import importlib
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

exp03 = importlib.import_module("experiments.03_training_contrastive_embeddings")

# Unseen evaluation support tickets in Spanish (3 per category = 12 unseen test tickets)
UNSEEN_TEST_TICKETS: list[tuple[str, int]] = [
    ("necesito factura fiscal con IVA desglosado del cobro mensual en tarjeta", 0),
    ("me hicieron un cargo duplicado en la renovacion anual de la cuenta", 0),
    ("aplicar nota de credito y reembolso en mi recibo de suscripcion", 0),
    ("el certificado SSL de nuestro balanceador devuelve timeout en HTTPS", 1),
    ("falla la resolucion DNS con error NXDOMAIN y perdida de paquetes TCP", 1),
    ("el tunel VPN IPsec se corta por problemas de MTU en el firewall de red", 1),
    ("error 403 PermissionDenied en la service account al acceder al bucket", 2),
    ("configurar federacion OIDC en IAM para evitar credenciales JSON y dar acceso RBAC", 2),
    ("otorgar rol y permisos de deploy bajo principio de menor privilegio en auditoria", 2),
    ("bloqueo deadlock en transacciones SQL y tabla lenta sin indice B-tree", 3),
    ("error TooManyConnections en el pool de Postgres y sesiones colgadas en DB", 3),
    ("crecimiento de logs WAL en disco porque autovacuum no limpia la consulta SQL", 3),
]


def _median_ms(fn: Callable[[], Any], n_iters: int) -> float:
    """Median wall time of `fn()` in ms (robust to noisy neighbours on a shared machine)."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


@jax.jit
def compute_class_centroids(
    embeddings: jax.Array, labels: jax.Array, num_classes: int = 4
) -> jax.Array:
    """Computes L2-normalized class prototype centroids from frozen embeddings."""
    one_hot = jax.nn.one_hot(labels, num_classes=num_classes, dtype=embeddings.dtype)
    class_sums = jnp.einsum("nc,nd->cd", one_hot, embeddings)
    norms = jnp.linalg.norm(class_sums, axis=-1, keepdims=True)
    return class_sums / jnp.maximum(norms, 1e-8)


@jax.jit
def zero_shot_predict(embeddings: jax.Array, centroids: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Classifies embeddings by nearest class centroid in cosine space."""
    sims = jnp.einsum("nd,cd->nc", embeddings, centroids)
    preds = jnp.argmax(sims, axis=-1)
    return preds, sims


def probe_loss(
    probe_params: dict[str, jax.Array],
    embeddings: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    logits = embeddings @ probe_params["W"] + probe_params["b"]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    targets = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    ce = -jnp.mean(jnp.sum(targets * log_probs, axis=-1))
    reg = 1e-3 * jnp.sum(probe_params["W"] ** 2)
    return ce + reg


@jax.jit
def probe_train_step(
    probe_params: dict[str, jax.Array],
    embeddings: jax.Array,
    labels: jax.Array,
    lr: float = 0.5,
) -> tuple[dict[str, jax.Array], jax.Array]:
    loss, grads = jax.value_and_grad(probe_loss)(probe_params, embeddings, labels)
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, probe_params, grads)
    return new_params, loss


@jax.jit
def linear_probe_predict(
    probe_params: dict[str, jax.Array],
    embeddings: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    logits = embeddings @ probe_params["W"] + probe_params["b"]
    probs = jax.nn.softmax(logits, axis=-1)
    preds = jnp.argmax(probs, axis=-1)
    return preds, probs


def run_experiment() -> dict[str, Any]:
    # 1. Train (or obtain) the frozen contrastive encoder from Experiment 03
    trained = exp03.train_contrastive_model(num_steps=180)
    encoder_params = trained["params"]

    q_raw, d_raw, labels = exp03.build_dataset()
    train_raw = jnp.concatenate([q_raw, d_raw], axis=0)
    train_labels = jnp.concatenate([labels, labels], axis=0)

    # Encode training bank and unseen test tickets with frozen encoder
    train_emb = exp03.encode_texts(encoder_params, train_raw)
    test_raw = jnp.asarray(
        np.stack([exp03.text_to_raw_features(t) for t, _ in UNSEEN_TEST_TICKETS], axis=0)
    )
    test_labels = jnp.asarray([c for _, c in UNSEEN_TEST_TICKETS], dtype=jnp.int32)
    test_emb = exp03.encode_texts(encoder_params, test_raw)

    # Compare against untrained random encoder zero-shot accuracy
    untrained_params = exp03.init_projector_params(jax.random.key(99))
    untrained_train_emb = exp03.encode_texts(untrained_params, train_raw)
    untrained_test_emb = exp03.encode_texts(untrained_params, test_raw)
    untrained_centroids = compute_class_centroids(untrained_train_emb, train_labels)
    untrained_preds, _ = zero_shot_predict(untrained_test_emb, untrained_centroids)

    # 2. Centroid classifier ("zero-shot": 0 trained parameters) on frozen embeddings
    centroids = compute_class_centroids(train_emb, train_labels)
    zs_preds, zs_sims = zero_shot_predict(test_emb, centroids)

    def centroid_call() -> jax.Array:
        return zero_shot_predict(test_emb, compute_class_centroids(train_emb, train_labels))[0]

    zs_ms = _median_ms(lambda: centroid_call().block_until_ready(), 50)
    zs_correct = int(jnp.sum(zs_preds == test_labels))

    # 3. Linear probe on frozen embeddings (warmup compiles probe_train_step; not timed)
    embed_dim = train_emb.shape[-1]
    num_classes = 4
    probe_params = {
        "W": jnp.zeros((embed_dim, num_classes), dtype=jnp.float32),
        "b": jnp.zeros((num_classes,), dtype=jnp.float32),
    }
    jax.block_until_ready(probe_train_step(probe_params, train_emb, train_labels))

    t1 = time.perf_counter()
    for _ in range(80):
        probe_params, p_loss = probe_train_step(probe_params, train_emb, train_labels)
    p_loss.block_until_ready()
    probe_train_ms = (time.perf_counter() - t1) * 1000.0

    lp_preds, lp_probs = linear_probe_predict(probe_params, test_emb)
    lp_preds.block_until_ready()
    lp_correct = int(jnp.sum(lp_preds == test_labels))
    lp_confidence = float(jnp.mean(jnp.max(lp_probs, axis=-1)))
    num_test = int(test_emb.shape[0])
    untrained_correct = int(jnp.sum(untrained_preds == test_labels))

    return {
        "backend": jax.default_backend(),
        "num_train_samples": int(train_emb.shape[0]),
        "num_test_tickets": num_test,
        "embed_dim": int(embed_dim),
        "untrained_correct": untrained_correct,
        "untrained_zero_shot_acc": untrained_correct / num_test,
        "zero_shot_correct": zs_correct,
        "zero_shot_acc": zs_correct / num_test,
        "zero_shot_trained_params": 0,
        "zero_shot_ms": zs_ms,
        "linear_probe_correct": lp_correct,
        "linear_probe_acc": lp_correct / num_test,
        "linear_probe_params": int(embed_dim * num_classes + num_classes),
        "linear_probe_confidence": lp_confidence,
        "linear_probe_train_ms": probe_train_ms,
        "linear_probe_loss": float(p_loss),
        "sample_predictions": [
            {
                "ticket": UNSEEN_TEST_TICKETS[i][0],
                "expected": exp03.CATEGORY_NAMES[int(test_labels[i])],
                "zero_shot": exp03.CATEGORY_NAMES[int(zs_preds[i])],
                "linear_probe": exp03.CATEGORY_NAMES[int(lp_preds[i])],
                "cosine_score": float(jnp.max(zs_sims[i])),
            }
            for i in (0, 3, 6, 9)
        ],
    }


def main() -> None:
    res = run_experiment()
    n = res["num_test_tickets"]
    print("=" * 72)
    print("Experiment 05: Centroids (zero-shot) vs Linear Probe on Frozen Embeddings")
    print("=" * 72)
    print(f"{'Backend Device':<29}: {res['backend']}")
    print(
        f"{'Frozen Embeddings':<29}: D={res['embed_dim']} | "
        f"{res['num_train_samples']} labeled train, {n} unseen test"
    )
    print(
        f"{'Centroids, Untrained Encoder':<29}: {res['untrained_correct']}/{n} "
        f"({res['untrained_zero_shot_acc'] * 100:.1f}%, chance 25%)"
    )
    print(
        f"{'Centroids, 0 Trained Params':<29}: {res['zero_shot_correct']}/{n} "
        f"({res['zero_shot_acc'] * 100:.1f}%) in {res['zero_shot_ms']:.3f} ms"
    )
    probe = f"Linear Probe, {res['linear_probe_params']} Params"
    print(
        f"{probe:<29}: {res['linear_probe_correct']}/{n} "
        f"({res['linear_probe_acc'] * 100:.1f}%), 80 steps in {res['linear_probe_train_ms']:.1f} ms"
    )
    print(f"{'Probe Mean Confidence':<29}: {res['linear_probe_confidence'] * 100:.1f}%")
    print("-" * 72)
    print("Sample Unseen Ticket Predictions:")
    for item in res["sample_predictions"]:
        print(
            f"  [{item['expected']:<13}] -> Centroid={item['zero_shot']:<13} "
            f"| Probe={item['linear_probe']}"
        )


if __name__ == "__main__":
    main()
