"""The Outside Judge — Verification & Benchmark Harness for embeddings-and-semantic-search-in-jax.

Executes all 5 experiments, enforces deterministic assertions (numerical parity against
NumPy, gather in the Jaxpr, gradient only on used rows, L2 hypersphere normalization,
Recall@1 on the training pairs, exact Top-K parity and classifier accuracy) and writes
`benchmarks/latest_results.json`, `benchmarks/terminal_output.txt` and the results block
of `README.md`, so the README numbers are always the ones in the JSON.

The checks are deterministic. The timings are medians per call after a warmup and vary
run to run; every speedup row also reports a fairer baseline (see "Cómo leer estas
cifras" in the README).
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import jax
import numpy as np

exp01 = importlib.import_module("experiments.01_tokens_to_embedding_table")
exp02 = importlib.import_module("experiments.02_cosine_similarity_and_vector_space")
exp03 = importlib.import_module("experiments.03_training_contrastive_embeddings")
exp04 = importlib.import_module("experiments.04_jit_vector_search_engine")
exp05 = importlib.import_module("experiments.05_zero_shot_and_linear_probe_classifier")

ROOT = Path(__file__).resolve().parent
README_START = "<!-- judge:start -->"
README_END = "<!-- judge:end -->"


def _cpu_model() -> str:
    """Human-readable CPU model (platform.processor() only says 'arm' on macOS)."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _load_average() -> list[float]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return []


def _public(d: dict[str, Any]) -> dict[str, Any]:
    """Drops non-serializable or bulky internals (trained params, raw HLO text)."""
    return {k: v for k, v in d.items() if k != "params" and not k.startswith("_")}


def run_all_verifications() -> dict[str, Any]:
    load_before = _load_average()

    r01 = exp01.run_benchmark()
    assert r01["max_diff"] < 1e-5, f"Exp 01 parity failed: {r01['max_diff']}"
    assert r01["max_diff_numpy"] < 1e-5, f"Exp 01 NumPy parity failed: {r01['max_diff_numpy']}"
    assert r01["lookup_has_gather"], "Exp 01 W_embed[token_ids] should lower to gather"
    assert r01["onehot_has_dot_general"], "Exp 01 one-hot path should contain dot_general"
    assert r01["pad_grad_norm"] == 0.0, "Exp 01 PAD row should receive zero gradient"
    assert r01["nonzero_grad_rows"] == r01["unique_real_ids"], (
        "Exp 01 gradient should reach exactly the rows of the real token IDs"
    )
    assert r01["speedup"] > 1.5, f"Exp 01 expected gather faster than one-hot: {r01['speedup']}"

    r02 = exp02.run_benchmark()
    assert r02["max_norm_error"] < 1e-5, f"Exp 02 L2 norm error too high: {r02['max_norm_error']}"
    assert not r02["euclid_picks_correct_doc"], "Exp 02 Euclidean should fail on length-scaled doc"
    assert r02["cosine_picks_correct_doc"], "Exp 02 Cosine must succeed on length-scaled doc"
    assert r02["max_diff"] < 1e-5, f"Exp 02 parity failed: {r02['max_diff']}"
    assert r02["max_diff_vectorized"] < 1e-5, (
        f"Exp 02 vectorized NumPy parity failed: {r02['max_diff_vectorized']}"
    )
    assert r02["speedup"] > 5.0, f"Exp 02 expected @jit faster than the pair loop: {r02['speedup']}"

    r03 = exp03.train_contrastive_model()
    assert r03["final_loss"] < r03["init_loss"] * 0.25, "Exp 03 loss did not converge"
    assert r03["final_recall_at_1"] == 1.0, (
        f"Exp 03 expected 100% Recall@1: {r03['final_recall_at_1']}"
    )
    assert r03["margin_gain"] > 0.5, f"Exp 03 margin gain too small: {r03['margin_gain']}"

    r04 = exp04.run_benchmark()
    assert r04["exact_match_rate"] == 1.0, f"Exp 04 Top-K parity failed: {r04['exact_match_rate']}"
    assert r04["exact_match_rate_batched"] == 1.0, "Exp 04 batched NumPy Top-K parity failed"
    assert r04["needle_recall_at_1"] == 1.0, (
        f"Exp 04 needle recall failed: {r04['needle_recall_at_1']}"
    )
    assert r04["max_score_diff"] < 1e-5, f"Exp 04 score diff too high: {r04['max_score_diff']}"
    assert r04["speedup"] > 5.0, (
        f"Exp 04 expected @jit faster than the query loop: {r04['speedup']}"
    )

    r05 = exp05.run_experiment()
    assert r05["zero_shot_acc"] >= 0.90, f"Exp 05 centroid acc too low: {r05['zero_shot_acc']}"
    assert r05["linear_probe_acc"] >= 0.90, (
        f"Exp 05 linear probe acc too low: {r05['linear_probe_acc']}"
    )
    assert r05["zero_shot_acc"] > r05["untrained_zero_shot_acc"], "Exp 05 trained encoder must win"

    return {
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "load_average_before": load_before,
            "load_average_after": _load_average(),
            "timing": "median ms per call after one warmup call; varies run to run",
        },
        "experiments": {
            "01_tokens_to_embedding_table": _public(r01),
            "02_cosine_similarity_and_vector_space": _public(r02),
            "03_training_contrastive_embeddings": _public(r03),
            "04_jit_vector_search_engine": _public(r04),
            "05_zero_shot_and_linear_probe_classifier": _public(r05),
        },
    }


def fmt_ms(v: float) -> str:
    if v >= 100:
        return f"{v:.1f} ms"
    if v >= 1:
        return f"{v:.2f} ms"
    return f"{v:.3f} ms"


def fmt_x(v: float) -> str:
    return f"{v:.1f}x" if v >= 10 else f"{v:.2f}x"


def table_rows(summary: dict[str, Any]) -> list[tuple[str, str, str, str, str]]:
    ex = summary["experiments"]
    e1 = ex["01_tokens_to_embedding_table"]
    e2 = ex["02_cosine_similarity_and_vector_space"]
    e3 = ex["03_training_contrastive_embeddings"]
    e4 = ex["04_jit_vector_search_engine"]
    e5 = ex["05_zero_shot_and_linear_probe_classifier"]
    n5 = e5["num_test_tickets"]
    return [
        (
            "01. Gather vs one-hot (jit)",
            fmt_ms(e1["onehot_ms"]),
            fmt_ms(e1["lookup_ms"]),
            fmt_x(e1["speedup"]),
            f"{fmt_x(e1['speedup_vs_numpy_gather'])} vs NumPy W[ids]",
        ),
        (
            "02. jit cosine vs pair loop",
            fmt_ms(e2["numpy_loop_ms"]),
            fmt_ms(e2["jax_jit_ms"]),
            fmt_x(e2["speedup"]),
            f"{fmt_x(e2['speedup_vs_numpy_vectorized'])} vs NumPy matmul",
        ),
        (
            "03. InfoNCE, 16 train pairs",
            f"{e3['init_recall_at_1'] * 100:.1f}% R@1",
            f"{e3['final_recall_at_1'] * 100:.1f}% R@1",
            f"+{e3['margin_gain']:.3f} margin",
            f"chance {100 / e3['num_pairs']:.2f}% R@1",
        ),
        (
            "04. jit top-k vs query loop",
            fmt_ms(e4["numpy_seq_ms"]),
            fmt_ms(e4["jax_batch_ms"]),
            fmt_x(e4["speedup"]),
            f"{fmt_x(e4['speedup_vs_numpy_batched'])} vs NumPy batched",
        ),
        (
            "05. Centroids & linear probe",
            f"{e5['untrained_correct']}/{n5} acc",
            f"{e5['zero_shot_correct']}/{n5} acc",
            f"probe {e5['linear_probe_correct']}/{n5}",
            f"untrained enc, {n5} unseen",
        ),
    ]


def format_report(summary: dict[str, Any]) -> str:
    rt = summary["runtime"]
    load = rt["load_average_before"], rt["load_average_after"]
    width = 98
    lines = [
        "=" * width,
        "THE OUTSIDE JUDGE — Embeddings & Semantic Search in JAX Verification Report",
        "=" * width,
        f"Runtime : Python {rt['python']} | JAX {rt['jax']} | NumPy {rt['numpy']} | Backend: {rt['backend']}",
        (
            f"CPU     : {rt['cpu_model']} ({rt['cpu_count']} cores) | 1-min load average "
            f"{load[0][0] if load[0] else 'n/a'} before, {load[1][0] if load[1] else 'n/a'} after"
        ),
        "-" * width,
        f"{'Experiment':<28} | {'Baseline':<10} | {'JAX':<10} | {'Speedup/Check':<13} | Fair baseline / note",
        "-" * width,
    ]
    for name, base, jx, speed, fair in table_rows(summary):
        lines.append(f"{name:<28} | {base:<10} | {jx:<10} | {speed:<13} | {fair}")
    lines += [
        "-" * width,
        "Times: median per call after a warmup; they vary run to run and with machine load.",
        "Saved JSON report to: benchmarks/latest_results.json",
        "ALL DETERMINISTIC CHECKS PASSED.",
    ]
    return "\n".join(lines)


def readme_block(summary: dict[str, Any], report_text: str) -> str:
    rt = summary["runtime"]
    load0 = rt["load_average_before"][0] if rt["load_average_before"] else "n/a"
    out = [
        README_START,
        "",
        (
            f"Corrida del juez en {rt['cpu_model']} ({rt['cpu_count']} núcleos), "
            f"Python {rt['python']}, JAX {rt['jax']}, NumPy {rt['numpy']}, backend "
            f"`{rt['backend']}`. Carga del sistema (load average de 1 min): {load0} al empezar. "
            "Tiempos = mediana por llamada después de una llamada de calentamiento."
        ),
        "",
        "| Experimento | Baseline | JAX + OpenXLA | Speedup / Verificación | Baseline justo / nota |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for name, base, jx, speed, fair in table_rows(summary):
        out.append(f"| **{name}** | `{base}` | `{jx}` | **{speed}** | {fair} |")
    out += [
        "",
        "Salida íntegra de terminal generada por `verify_embeddings.py`:",
        "",
        "```text",
        report_text,
        "```",
        "",
        README_END,
    ]
    return "\n".join(out)


def update_readme(summary: dict[str, Any], report_text: str) -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    readme.write_text(head + readme_block(summary, report_text) + tail, encoding="utf-8")


def main() -> None:
    summary = run_all_verifications()
    report_text = format_report(summary)
    print(report_text)

    bench_dir = ROOT / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "latest_results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (bench_dir / "terminal_output.txt").write_text(report_text + "\n", encoding="utf-8")
    update_readme(summary, report_text)


if __name__ == "__main__":
    main()
