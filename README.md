# De Números a Significado: Embeddings, Geometría Vectorial y Búsqueda Semántica en JAX

- **Por qué es relevante:** Todo sistema moderno de búsqueda semántica, RAG, recomendación y clasificación de intenciones transforma IDs discretos en vectores densos sobre una hiperesfera unitaria. Construir ese pipeline desde cero en JAX (`W_embed[token_ids]`, normalización L2, entrenamiento contrastivo `InfoNCE` con `jax.value_and_grad`, búsqueda exacta con `jnp.einsum` + `jax.lax.top_k` y clasificadores `Zero-Shot` / `Linear Probe`) desmitifica cómo las palabras y documentos adquieren significado geométrico.
- **Para quién es:** Ingenieros de software y desarrolladores que ya comprenden cómo aprende una red pequeña (Proyecto B2) y quieren entender cómo se entrenan, indexan y consultan `embeddings` de alta dimensión sin depender de cajas negras.
- **Quién debería verlo:** Equipos que diseñan motores de búsqueda vectorial, ruteadores de tickets de soporte, catálogos de productos o sistemas de recuperación de contexto que requieren latencias de microsegundos por query en CPU, GPU o Cloud TPU.
- **Serie completa:** Tercer repositorio de la Serie B dentro de la guía práctica [JAX & OpenXLA Field Guide](https://github.com/ykro/jax-openxla-field-guide).

---

## Arquitectura y Modelo Mental

### 1. De IDs Discretos a la Hiperesfera Unitaria (`Lookup Table`, Máscara y Normalización L2)

Representar un vocabulario de `32,000` tokens con vectores `one-hot` desperdicia `250x` más memoria RAM y deja a todas las palabras ortogonales entre sí. En JAX, indexar directamente la matriz `W_embed[token_ids]` compila a una primitiva `Gather` de OpenXLA matemáticamente idéntica al producto `one_hot @ W_embed`, diferenciable respecto a los tokens activos y combinable con una máscara de padding y normalización L2 (`v / jnp.linalg.norm`).

![De IDs discretos a W_embed[token_ids], pooling con máscara y normalización L2](diagrams/01-discrete-ids-to-embedding-space.svg)

### 2. Cómo Aprende el Espacio Vectorial (`Contrastive Learning` e `InfoNCE`)

Un proyector inicializado al azar mezcla tickets de distintos dominios porque las palabras superficiales no coinciden exactamente. Al optimizar la función de pérdida contrastiva `InfoNCE` escalada por temperatura ($\tau = 0.08$) con `@jax.jit(jax.value_and_grad(infonce_loss))`, los gradientes acercan los pares positivos sobre la esfera unitaria y empujan los pares negativos, elevando el `Recall@1` de `12.5%` a `100.0%`.

![Geometría contrastiva InfoNCE antes y después del entrenamiento en JAX](diagrams/02-contrastive-infonce-geometry.svg)

### 3. Motor de Búsqueda Vectorial Compilado (`jnp.einsum` + `jax.lax.top_k`)

Cuando los vectores del índice `(50000, 256)` y el `batch` de queries `(128, 256)` viven en la esfera unitaria, la `cosine similarity` de los `6.4 millones` de pares se reduce a una sola contracción matricial `jnp.einsum("qd,nd->qn", q_unit, index)` fusionada con `jax.lax.top_k(scores, k=5)` dentro de un único binario XLA.

![Motor de búsqueda vectorial compilado con @jax.jit, einsum y lax.top_k](diagrams/03-jit-einsum-vector-search.svg)

---

## Resultados Reales de Verificación (`verify_embeddings.py`)

Benchmarks deterministas ejecutados con Python `3.12.11`, JAX `0.12.0.1` y NumPy `2.5.4`:

| Experimento | Archivo | Baseline (One-Hot / Loop / Sin entrenar) | JAX + OpenXLA | Speedup / Verificación |
| :--- | :--- | :--- | :--- | :--- |
| **01. Gather Lookup vs One-Hot** | `experiments/01_tokens_to_embedding_table.py` | `35.8146 ms` (`250.00 MiB` One-Hot) | `0.1211 ms` (`1.00 MiB` Gather) | **295.75x** (`250x` menos RAM, `max_diff = 0.0`) |
| **02. `@jit` L2 + Cosine `einsum`** | `experiments/02_cosine_similarity_and_vector_space.py` | `86.9407 ms` (Python/NumPy pairwise `loop`) | `0.4352 ms` (`@jax.jit` L2 + `einsum`) | **199.77x** (`max_diff = 2.24e-07`, invariante a longitud) |
| **03. `InfoNCE` Contrastive Training** | `experiments/03_training_contrastive_embeddings.py` | `12.5%` `Recall@1` (`loss = 3.7216`) | `100.0%` `Recall@1` (`loss = 0.8806`) | **+0.658** margen semántico (`180` steps en `184.6 ms`) |
| **04. `@jit` Top-K Search (`50k x 256`)** | `experiments/04_jit_vector_search_engine.py` | `1191.9367 ms` (Búsqueda secuencial NumPy) | `15.7486 ms` (`@jit` `einsum` + `lax.top_k`) | **75.69x** (`123.0 us/query`, `100.0%` acuerdo Top-5) |
| **05. Zero-Shot & Linear Probe** | `experiments/05_zero_shot_and_linear_probe_classifier.py` | `58.3%` Acc (Encoder sin entrenar) | `100.0%` Acc (`Zero-Shot` y `Linear Probe`) | **100%** en tickets nuevos (`0` params en `Zero-Shot`) |

Salida íntegra de terminal generada por `verify_embeddings.py`:

```text
==============================================================================
THE OUTSIDE JUDGE — Embeddings & Semantic Search in JAX Verification Report
==============================================================================
Runtime : Python 3.12.11 | JAX 0.12.0.1 | NumPy 2.5.4 | Backend: cpu
------------------------------------------------------------------------------
Experiment                          | Baseline       | JAX/OpenXLA    | Speedup / Check
------------------------------------------------------------------------------
01. Gather Lookup vs One-Hot        |    35.8146 ms |     0.1211 ms | 295.75x (250x less RAM)
02. @jit L2 + Cosine einsum         |    86.9407 ms |     0.4352 ms | 199.77x (PASS)
03. InfoNCE Contrastive Training    |      12.5% R@1 |     100.0% R@1 | +0.658 margin (184.6 ms)
04. @jit Top-K Search (50k x 256)   |  1191.9367 ms |    15.7486 ms | 75.69x (123.0 us/q)
05. Zero-Shot & Linear Probe        |      58.3% Acc |     100.0% Acc | ZS=100% / Probe=100%
------------------------------------------------------------------------------
Saved JSON report to: benchmarks/latest_results.json
ALL DETERMINISTIC CHECKS PASSED.
```

---

## Estructura del Repositorio

```text
.
├── benchmarks/
│   ├── latest_results.json                             # Métricas verificadas por verify_embeddings.py
│   └── terminal_output.txt                             # Salida íntegra del Outside Judge
├── diagrams/
│   ├── 01-discrete-ids-to-embedding-space.{html,svg,png}
│   ├── 02-contrastive-infonce-geometry.{html,svg,png}
│   └── 03-jit-einsum-vector-search.{html,svg,png}
├── experiments/
│   ├── 01_tokens_to_embedding_table.py                 # Lookup table W_embed[ids] vs one-hot matmul y máscara
│   ├── 02_cosine_similarity_and_vector_space.py        # Geometría L2, invarianza a longitud y einsum en @jax.jit
│   ├── 03_training_contrastive_embeddings.py           # Entrenamiento contrastivo InfoNCE con jax.value_and_grad
│   ├── 04_jit_vector_search_engine.py                  # Motor de búsqueda Top-K sobre 50,000 vectores con lax.top_k
│   └── 05_zero_shot_and_linear_probe_classifier.py     # Clasificación Zero-Shot Centroids vs Linear Probe
├── gcp/
│   └── run_on_demand_embeddings.sh                     # Runner on-demand con auto-teardown en Cloud TPU v5e y L4 GPU
├── tests/
│   └── test_embeddings.py                              # Suite pytest determinista (5 tests)
├── Makefile
├── pyproject.toml
└── verify_embeddings.py                                # El Outside Judge (aserciones de paridad, Recall@1 y speedup)
```

---

## Onboarding Paso a Paso

### 1. Instalación desde cero (macOS / Linux)

```bash
git clone https://github.com/ykro/embeddings-and-semantic-search-in-jax.git
cd embeddings-and-semantic-search-in-jax
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Recorrido guiado por los 5 experimentos (Orden recomendado)

Cada script se ejecuta en menos de 2 segundos y construye un bloque fundamental del sistema de `embeddings` y búsqueda semántica:

```bash
# Paso 1 — De IDs discretos a W_embed[token_ids], máscara de padding y ahorro de 250x en RAM frente a one-hot
python experiments/01_tokens_to_embedding_table.py

# Paso 2 — Geometría del significado: por qué falla la distancia euclidiana y cómo normalizar en la esfera L2
python experiments/02_cosine_similarity_and_vector_space.py

# Paso 3 — Cómo aprende el espacio vectorial: pérdida contrastiva InfoNCE en español (Recall@1 de 12.5% a 100%)
python experiments/03_training_contrastive_embeddings.py

# Paso 4 — Motor de búsqueda vectorial en JAX: 128 queries contra 50,000 vectores con einsum + lax.top_k
python experiments/04_jit_vector_search_engine.py

# Paso 5 — Clasificación de intenciones sin reentrenar el modelo base: Zero-Shot Centroids vs Linear Probe
python experiments/05_zero_shot_and_linear_probe_classifier.py
```

### 3. Verificación completa (Linter + Tests + Outside Judge)

```bash
# Ejecutar únicamente el Outside Judge y actualizar benchmarks/latest_results.json
python verify_embeddings.py

# Ejecutar ruff, pytest (5 tests) y verify_embeddings.py en un solo comando
make all
```

### 4. Ejecución on-demand con auto-teardown en Google Cloud (Cloud TPU v5e o NVIDIA L4 GPU)

El script `gcp/run_on_demand_embeddings.sh` aprovisiona instancias on-demand con auto-teardown en el proyecto `ai-experiments-487722` y garantiza la eliminación inmediata de recursos mediante `trap cleanup EXIT INT TERM`.

```bash
# Ejecutar benchmark en una instancia on-demand de Cloud TPU v5e-1 (us-central1-a)
make gcp-tpu

# Ejecutar benchmark en un Vertex AI Custom Job on-demand con GPU NVIDIA L4
make gcp-gpu
```
