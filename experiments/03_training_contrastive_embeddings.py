"""Experiment 03: Contrastive Representation Learning with InfoNCE and jax.value_and_grad.

Trains a small projection head (192 -> 96 -> 64) on 16 Spanish technical support
(query, document) pairs from 4 categories (Facturacion, Redes/DNS, Permisos IAM, Base de
Datos). The loss is symmetric InfoNCE with temperature 0.08, plus a supervised category
term and a small L2 penalty. Recall@1 is measured on the same 16 training pairs (it shows
that the space was rearranged; generalization to unseen tickets is tested in Experiment 05).
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# 16 curated (query, positive_doc, category_id) pairs in Spanish technical support
# Categories: 0=Facturacion, 1=Redes/DNS, 2=Permisos IAM, 3=Base de Datos
SUPPORT_PAIRS: list[tuple[str, str, int]] = [
    # 0: Facturacion e Impuestos
    (
        "cobro duplicado en mi tarjeta de credito este mes",
        "reembolso por cargo doble en factura mensual y nota de credito",
        0,
    ),
    (
        "como descargar la factura electronica con RFC e IVA",
        "portal de facturacion fiscal para emitir comprobante con impuestos",
        0,
    ),
    (
        "actualizar metodo de pago rechazado en renovacion anual",
        "cambiar tarjeta vencida para evitar suspension de suscripcion y cobro",
        0,
    ),
    (
        "descuento aplicado no aparece en el estado de cuenta",
        "ajuste de tarifa promocional y saldo a favor en recibo de pago",
        0,
    ),
    # 1: Caidas de Red y DNS
    (
        "el dominio principal no resuelve y devuelve error NXDOMAIN",
        "configuracion de registros DNS tipo A y propagacion de nameservers",
        1,
    ),
    (
        "perdida de paquetes y latencia alta hacia el balanceador",
        "diagnostico de ruteo BGP conectividad TCP y timeout en firewall de red",
        1,
    ),
    (
        "certificado TLS expirado bloquea conexion HTTPS en el navegador",
        "renovacion automatica de SSL en load balancer y validacion de dominio DNS",
        1,
    ),
    (
        "tunel VPN corporativo se desconecta cada cinco minutos",
        "estabilidad de enlace IPsec MTU de red y keepalive en gateway remoto",
        1,
    ),
    # 2: Permisos IAM y Service Accounts
    (
        "error 403 PermissionDenied al leer bucket desde el worker",
        "asignar rol roles/storage.objectViewer a la service account en IAM",
        2,
    ),
    (
        "como rotar credenciales sin usar llaves JSON estaticas",
        "configurar Workload Identity Federation y tokens OIDC temporales en IAM",
        2,
    ),
    (
        "usuario nuevo no puede desplegar contenedores en produccion",
        "politica de control de acceso RBAC y permisos de deploy para desarrolladores",
        2,
    ),
    (
        "auditoria de accesos administrativos y revocacion de sesion",
        "registro de actividad IAM principio de menor privilegio y bloqueo de cuenta",
        2,
    ),
    # 3: Latencia y Locks en Base de Datos
    (
        "consulta SQL tarda treinta segundos por falta de indice",
        "optimizacion de query plan creacion de indice compuesto B-tree en tabla",
        3,
    ),
    (
        "deadlock frecuente al actualizar inventario en transacciones concurrentes",
        "aislamiento transaccional bloqueo de filas row lock y orden de commits SQL",
        3,
    ),
    (
        "conexiones agotadas TooManyConnections en el pool de Postgres",
        "dimensionamiento de connection pooling PgBouncer y cierre de sesiones idle",
        3,
    ),
    (
        "disco lleno por crecimiento de archivos WAL y falta de vacuum",
        "mantenimiento de autovacuum retencion de logs transaccionales y espacio en DB",
        3,
    ),
]

CATEGORY_NAMES: tuple[str, ...] = (
    "Facturacion",
    "Redes/DNS",
    "Permisos IAM",
    "Base de Datos",
)

# Domain synonym clusters that share co-occurrence context in technical corpora
# fmt: off
DOMAIN_CLUSTERS: tuple[tuple[str, ...], ...] = (
    (
        "cobro", "cargo", "doble", "duplicado", "tarjeta", "credito", "factura",
        "facturacion", "reembolso", "fiscal", "iva", "rfc", "pago", "suscripcion",
        "descuento", "tarifa", "recibo", "cuenta", "mensual", "anual", "comprobante",
    ),
    (
        "dominio", "dns", "nxdomain", "registros", "nameservers", "paquetes",
        "latencia", "balanceador", "load", "balancer", "bgp", "tcp", "timeout",
        "firewall", "red", "tls", "ssl", "https", "certificado", "vpn", "ipsec", "mtu",
    ),
    (
        "403", "permissiondenied", "bucket", "worker", "rol", "storage", "iam",
        "service", "account", "credenciales", "json", "workload", "identity",
        "oidc", "rbac", "permisos", "deploy", "auditoria", "privilegio", "acceso",
    ),
    (
        "consulta", "sql", "indice", "query", "btree", "tabla", "deadlock",
        "transacciones", "lock", "commits", "conexiones", "toomanyconnections",
        "pool", "postgres", "pgbouncer", "wal", "vacuum", "autovacuum", "db",
    ),
)
# fmt: on

RAW_DIM = 192
EMBED_DIM = 64


def text_to_raw_features(text: str, dim: int = RAW_DIM) -> np.ndarray:
    """Converts Spanish text into deterministic character n-gram + co-occurrence features.

    Toy featurizer: dims [16..192) hold hashed words and character 3-grams; dims [0..16)
    hold a hand-made domain signal (a word from DOMAIN_CLUSTERS adds weight to its
    category's 4 channels). Queries and documents use different surface words, so the
    hashed features dominate before training (Recall@1 = 2/16); the projector learns to
    amplify the shared domain channels and suppress the word-hash noise.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = [t.strip(".,;:?!") for t in text.lower().split() if t.strip(".,;:?!")]

    for tok in tokens:
        # High-variance lexical token + character 3-gram hashes in dims [16..dim)
        for gram in [tok] + [tok[i : i + 3] for i in range(max(1, len(tok) - 2))]:
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
            idx = 16 + (h % (dim - 16))
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[idx] += sign * 2.2

        # Latent domain co-occurrence channels in dims [0..16)
        for c_idx, cluster in enumerate(DOMAIN_CLUSTERS):
            if tok in cluster:
                slot = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:4], 16) % 4
                vec[c_idx * 4 + slot] += 0.65
                vec[c_idx * 4 + ((slot + 1) % 4)] += 0.45

    return vec


def build_dataset() -> tuple[jax.Array, jax.Array, jax.Array]:
    q_feats = np.stack([text_to_raw_features(q) for q, _, _ in SUPPORT_PAIRS], axis=0)
    d_feats = np.stack([text_to_raw_features(d) for _, d, _ in SUPPORT_PAIRS], axis=0)
    labels = np.array([cat for _, _, cat in SUPPORT_PAIRS], dtype=np.int32)
    return jnp.asarray(q_feats), jnp.asarray(d_feats), jnp.asarray(labels)


def init_projector_params(
    key: jax.Array, in_dim: int = RAW_DIM, out_dim: int = EMBED_DIM
) -> dict[str, jax.Array]:
    k1, k2 = jax.random.split(key)
    hidden_dim = 96
    return {
        "W1": jax.random.normal(k1, (in_dim, hidden_dim), dtype=jnp.float32)
        * (1.0 / np.sqrt(in_dim)),
        "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "W2": jax.random.normal(k2, (hidden_dim, out_dim), dtype=jnp.float32)
        * (1.0 / np.sqrt(hidden_dim)),
        "b2": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


@jax.jit
def encode_texts(params: dict[str, jax.Array], raw_x: jax.Array) -> jax.Array:
    """Projects raw features into L2-normalized semantic embeddings on the unit sphere."""
    h = jax.nn.gelu(raw_x @ params["W1"] + params["b1"])
    z = h @ params["W2"] + params["b2"]
    norms = jnp.linalg.norm(z, axis=-1, keepdims=True)
    return z / jnp.maximum(norms, 1e-8)


def infonce_contrastive_loss(
    params: dict[str, jax.Array],
    q_raw: jax.Array,
    d_raw: jax.Array,
    labels: jax.Array,
    temperature: float = 0.08,
) -> jax.Array:
    """Training loss: 0.65 x symmetric InfoNCE + 0.35 x category term + L2 penalty."""
    q_emb = encode_texts(params, q_raw)
    d_emb = encode_texts(params, d_raw)

    # Cosine similarity matrix scaled by temperature tau
    logits = jnp.einsum("qd,nd->qn", q_emb, d_emb) / temperature
    batch_size = q_raw.shape[0]
    targets = jnp.arange(batch_size)

    # Cross-entropy along queries->docs and docs->queries
    log_probs_q2d = jax.nn.log_softmax(logits, axis=1)
    log_probs_d2q = jax.nn.log_softmax(logits, axis=0)
    pair_loss = -0.5 * (
        jnp.mean(log_probs_q2d[targets, targets]) + jnp.mean(log_probs_d2q[targets, targets])
    )

    # Category-level contrastive alignment so tickets of the same intent cluster together
    same_cat = (labels[:, None] == labels[None, :]).astype(jnp.float32)
    cat_weights = same_cat / jnp.sum(same_cat, axis=1, keepdims=True)
    cat_loss = -jnp.mean(jnp.sum(cat_weights * log_probs_q2d, axis=1))

    # Small L2 penalty on the projector weights
    l2_reg = 1e-4 * (jnp.sum(params["W1"] ** 2) + jnp.sum(params["W2"] ** 2))
    return 0.65 * pair_loss + 0.35 * cat_loss + l2_reg


@jax.jit
def train_step(
    params: dict[str, jax.Array],
    q_raw: jax.Array,
    d_raw: jax.Array,
    labels: jax.Array,
    lr: float = 0.08,
) -> tuple[dict[str, jax.Array], jax.Array]:
    loss, grads = jax.value_and_grad(infonce_contrastive_loss)(params, q_raw, d_raw, labels)
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss


@jax.jit
def evaluate_retrieval(
    params: dict[str, jax.Array],
    q_raw: jax.Array,
    d_raw: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    q_emb = encode_texts(params, q_raw)
    d_emb = encode_texts(params, d_raw)
    sim_matrix = jnp.einsum("qd,nd->qn", q_emb, d_emb)
    preds = jnp.argmax(sim_matrix, axis=1)
    recall_at_1 = jnp.mean((preds == jnp.arange(q_raw.shape[0])).astype(jnp.float32))

    eye = jnp.eye(q_raw.shape[0], dtype=jnp.bool_)
    pos_sim = jnp.mean(jnp.diag(sim_matrix))
    neg_sim = jnp.mean(jnp.where(~eye, sim_matrix, 0.0)) * (q_raw.shape[0] / (q_raw.shape[0] - 1))
    return recall_at_1, pos_sim, neg_sim


def train_contrastive_model(num_steps: int = 180) -> dict[str, Any]:
    q_raw, d_raw, labels = build_dataset()
    params = init_projector_params(jax.random.key(7))

    init_r1, init_pos, init_neg = evaluate_retrieval(params, q_raw, d_raw)
    init_loss = float(infonce_contrastive_loss(params, q_raw, d_raw, labels))

    # Warmup: the first call of train_step traces and compiles it. Its result is
    # discarded (train_step is pure), so the timed loop below starts from the same params.
    t_c = time.perf_counter()
    jax.block_until_ready(train_step(params, q_raw, d_raw, labels))
    compile_ms = (time.perf_counter() - t_c) * 1000.0

    t0 = time.perf_counter()
    for _ in range(num_steps):
        params, loss_arr = train_step(params, q_raw, d_raw, labels)
    loss_arr.block_until_ready()
    train_ms = (time.perf_counter() - t0) * 1000.0
    loss_val = float(loss_arr)

    final_r1, final_pos, final_neg = evaluate_retrieval(params, q_raw, d_raw)

    return {
        "backend": jax.default_backend(),
        "num_pairs": int(q_raw.shape[0]),
        "embed_dim": EMBED_DIM,
        "num_steps": num_steps,
        "first_call_compile_ms": compile_ms,
        "train_ms": train_ms,
        "ms_per_step": train_ms / num_steps,
        "init_loss": init_loss,
        "final_loss": loss_val,
        "recall_measured_on": "the 16 training pairs (chance = 1/16 = 6.25%)",
        "init_recall_at_1": float(init_r1),
        "final_recall_at_1": float(final_r1),
        "init_pos_sim": float(init_pos),
        "init_neg_sim": float(init_neg),
        "final_pos_sim": float(final_pos),
        "final_neg_sim": float(final_neg),
        "margin_gain": float((final_pos - final_neg) - (init_pos - init_neg)),
        "params": params,
    }


def main() -> None:
    res = train_contrastive_model()
    n = res["num_pairs"]
    print("=" * 72)
    print("Experiment 03: Contrastive InfoNCE Training with jax.value_and_grad")
    print("=" * 72)
    print(f"Backend Device             : {res['backend']}")
    print(f"Support Corpus Pairs       : {n} pairs across 4 domains (D={res['embed_dim']})")
    print(f"First Call (trace+compile) : {res['first_call_compile_ms']:.2f} ms")
    print(
        f"Training Steps (@jax.jit)  : {res['num_steps']} steps in {res['train_ms']:.2f} ms "
        f"({res['ms_per_step']:.3f} ms/step)"
    )
    print("-" * 72)
    print(
        f"Training Loss (Before -> After)     : {res['init_loss']:.4f} -> {res['final_loss']:.4f}"
    )
    print(
        f"{'Recall@1 on ' + str(n) + ' train pairs':<36}: "
        f"{res['init_recall_at_1'] * 100:.1f}% -> {res['final_recall_at_1'] * 100:.1f}% "
        f"(chance {100 / n:.2f}%)"
    )
    print(
        f"Positive Pair Cosine (Before->After): "
        f"{res['init_pos_sim']:+.4f} -> {res['final_pos_sim']:+.4f}"
    )
    print(
        f"Negative Pair Cosine (Before->After): "
        f"{res['init_neg_sim']:+.4f} -> {res['final_neg_sim']:+.4f}"
    )
    print(f"Semantic Separation Margin Gain     : +{res['margin_gain']:.4f}")


if __name__ == "__main__":
    main()
