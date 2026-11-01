"""Deterministic pytest suite for embeddings-and-semantic-search-in-jax."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp

exp01 = importlib.import_module("experiments.01_tokens_to_embedding_table")
exp02 = importlib.import_module("experiments.02_cosine_similarity_and_vector_space")
exp03 = importlib.import_module("experiments.03_training_contrastive_embeddings")
exp04 = importlib.import_module("experiments.04_jit_vector_search_engine")
exp05 = importlib.import_module("experiments.05_zero_shot_and_linear_probe_classifier")


def test_01_lookup_table_matches_one_hot_and_masks_pad_gradients() -> None:
    metrics = exp01.run_benchmark(num_iters=5)
    assert metrics["max_diff"] < 1e-5
    assert metrics["pad_grad_norm"] == 0.0
    assert metrics["active_grad_norm"] > 0.0
    assert metrics["memory_saving_ratio"] == 250.0


def test_02_l2_normalization_and_cosine_length_invariance() -> None:
    geom = exp02.demonstrate_length_invariance()
    assert not geom["euclid_picks_correct_doc"]
    assert geom["cosine_picks_correct_doc"]

    key = jax.random.key(0)
    vecs = jax.random.normal(key, (16, 64)) * 10.0
    normed = exp02.l2_normalize(vecs)
    norms = jnp.linalg.norm(normed, axis=-1)
    assert float(jnp.max(jnp.abs(norms - 1.0))) < 1e-5


def test_03_contrastive_infonce_reaches_100_percent_recall_at_1() -> None:
    res = exp03.train_contrastive_model(num_steps=180)
    assert res["final_loss"] < res["init_loss"] * 0.25
    assert res["final_recall_at_1"] == 1.0
    assert res["final_pos_sim"] > res["final_neg_sim"] + 0.5


def test_04_jit_vector_search_top_k_exact_parity() -> None:
    res = exp04.run_benchmark(index_size=5000, dim=128, num_queries=32, num_iters=5)
    assert res["exact_match_rate"] == 1.0
    assert res["needle_recall_at_1"] == 1.0
    assert res["max_score_diff"] < 1e-5


def test_05_zero_shot_and_linear_probe_intent_classification() -> None:
    res = exp05.run_experiment()
    assert res["zero_shot_acc"] == 1.0
    assert res["linear_probe_acc"] == 1.0
    assert res["zero_shot_acc"] > res["untrained_zero_shot_acc"]
