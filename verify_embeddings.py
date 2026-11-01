"""The Outside Judge — Verification & Benchmark Harness for embeddings-and-semantic-search-in-jax.

Executes all 5 experiments, enforces deterministic mathematical assertions
(numerical equivalence, L2 hypersphere normalization, Recall@1 = 100%, exact Top-K
search parity, and classifier accuracy), and writes both `benchmarks/latest_results.json`
and `benchmarks/terminal_output.txt`.
"""

from __future__ import annotations

import importlib
import json
import platform
from pathlib import Path
from typing import Any

import jax
import numpy as np

exp01 = importlib.import_module("experiments.01_tokens_to_embedding_table")
exp02 = importlib.import_module("experiments.02_cosine_similarity_and_vector_space")
exp03 = importlib.import_module("experiments.03_training_contrastive_embeddings")
exp04 = importlib.import_module("experiments.04_jit_vector_search_engine")
exp05 = importlib.import_module("experiments.05_zero_shot_and_linear_probe_classifier")


def run_all_verifications() -> dict[str, Any]:
    r01 = exp01.run_benchmark()
    assert r01["max_diff"] < 1e-5, f"Exp 01 parity failed: {r01['max_diff']}"
    assert r01["pad_grad_norm"] == 0.0, "Exp 01 PAD token should receive zero gradient"
    assert r01["active_grad_norm"] > 0.0, "Exp 01 active token should receive non-zero gradient"
    assert r01["speedup"] > 1.5, f"Exp 01 expected Gather speedup over One-Hot: {r01['speedup']}"

    r02 = exp02.run_benchmark()
    assert r02["max_norm_error"] < 1e-5, f"Exp 02 L2 norm error too high: {r02['max_norm_error']}"
    assert not r02["euclid_picks_correct_doc"], "Exp 02 Euclidean should fail on length-scaled doc"
    assert r02["cosine_picks_correct_doc"], "Exp 02 Cosine must succeed on length-scaled doc"
    assert r02["max_diff"] < 1e-5, f"Exp 02 parity failed: {r02['max_diff']}"
    assert r02["speedup"] > 5.0, f"Exp 02 expected @jit einsum speedup: {r02['speedup']}"

    r03 = exp03.train_contrastive_model()
    assert r03["final_loss"] < r03["init_loss"] * 0.25, "Exp 03 InfoNCE loss did not converge"
    assert r03["final_recall_at_1"] == 1.0, f"Exp 03 expected 100% Recall@1, got {r03['final_recall_at_1']}"
    assert r03["margin_gain"] > 0.5, f"Exp 03 margin gain too small: {r03['margin_gain']}"
    r03_clean = {k: v for k, v in r03.items() if k != "params"}

    r04 = exp04.run_benchmark()
    assert r04["exact_match_rate"] == 1.0, f"Exp 04 Top-K parity failed: {r04['exact_match_rate']}"
    assert r04["needle_recall_at_1"] == 1.0, f"Exp 04 needle recall failed: {r04['needle_recall_at_1']}"
    assert r04["max_score_diff"] < 1e-5, f"Exp 04 score diff too high: {r04['max_score_diff']}"
    assert r04["speedup"] > 5.0, f"Exp 04 expected @jit Top-K speedup: {r04['speedup']}"

    r05 = exp05.run_experiment()
    assert r05["zero_shot_acc"] >= 0.90, f"Exp 05 zero-shot acc too low: {r05['zero_shot_acc']}"
    assert r05["linear_probe_acc"] >= 0.90, f"Exp 05 linear probe acc too low: {r05['linear_probe_acc']}"
    assert r05["zero_shot_acc"] > r05["untrained_zero_shot_acc"], "Exp 05 trained encoder must beat random"

    return {
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
        },
        "experiments": {
            "01_tokens_to_embedding_table": r01,
            "02_cosine_similarity_and_vector_space": r02,
            "03_training_contrastive_embeddings": r03_clean,
            "04_jit_vector_search_engine": r04,
            "05_zero_shot_and_linear_probe_classifier": r05,
        },
    }


def format_report(summary: dict[str, Any]) -> str:
    rt = summary["runtime"]
    ex = summary["experiments"]
    e1 = ex["01_tokens_to_embedding_table"]
    e2 = ex["02_cosine_similarity_and_vector_space"]
    e3 = ex["03_training_contrastive_embeddings"]
    e4 = ex["04_jit_vector_search_engine"]
    e5 = ex["05_zero_shot_and_linear_probe_classifier"]

    lines = [
        "=" * 78,
        "THE OUTSIDE JUDGE — Embeddings & Semantic Search in JAX Verification Report",
        "=" * 78,
        f"Runtime : Python {rt['python']} | JAX {rt['jax']} | NumPy {rt['numpy']} | Backend: {rt['backend']}",
        "-" * 78,
        f"{'Experiment':<35} | {'Baseline':<14} | {'JAX/OpenXLA':<14} | {'Speedup / Check'}",
        "-" * 78,
        (
            f"{'01. Gather Lookup vs One-Hot':<35} | "
            f"{e1['onehot_ms']:>10.4f} ms | "
            f"{e1['lookup_ms']:>10.4f} ms | "
            f"{e1['speedup']:.2f}x ({e1['memory_saving_ratio']:.0f}x less RAM)"
        ),
        (
            f"{'02. @jit L2 + Cosine einsum':<35} | "
            f"{e2['numpy_loop_ms']:>10.4f} ms | "
            f"{e2['jax_jit_ms']:>10.4f} ms | "
            f"{e2['speedup']:.2f}x (PASS)"
        ),
        (
            f"{'03. InfoNCE Contrastive Training':<35} | "
            f"{e3['init_recall_at_1'] * 100:>9.1f}% R@1 | "
            f"{e3['final_recall_at_1'] * 100:>9.1f}% R@1 | "
            f"+{e3['margin_gain']:.3f} margin ({e3['train_ms']:.1f} ms)"
        ),
        (
            f"{'04. @jit Top-K Search (50k x 256)':<35} | "
            f"{e4['numpy_seq_ms']:>10.4f} ms | "
            f"{e4['jax_batch_ms']:>10.4f} ms | "
            f"{e4['speedup']:.2f}x ({e4['per_query_us']:.1f} us/q)"
        ),
        (
            f"{'05. Zero-Shot & Linear Probe':<35} | "
            f"{e5['untrained_zero_shot_acc'] * 100:>9.1f}% Acc | "
            f"{e5['linear_probe_acc'] * 100:>9.1f}% Acc | "
            f"ZS={e5['zero_shot_acc'] * 100:.0f}% / Probe={e5['linear_probe_acc'] * 100:.0f}%"
        ),
        "-" * 78,
        "Saved JSON report to: benchmarks/latest_results.json",
        "ALL DETERMINISTIC CHECKS PASSED.",
    ]
    return "\n".join(lines)


def main() -> None:
    summary = run_all_verifications()
    report_text = format_report(summary)
    print(report_text)

    bench_dir = Path(__file__).resolve().parent / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    (bench_dir / "latest_results.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (bench_dir / "terminal_output.txt").write_text(
        report_text + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
