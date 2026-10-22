#!/usr/bin/env python3
"""
eval_pi.py: Benchmark Evaluation Engine for Edge Multimodal Clinical OCR
Computes Character Error Rate (CER) and Word Error Rate (WER) using Levenshtein distance.
"""

import os
import sys
import json
import glob
from pathlib import Path

def levenshtein_distance(ref: list, hyp: list) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(ref)][len(hyp)]

def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER)."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)

def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)

def run_evaluation(predictions_dir: str, truth_dir: str, output_report: str):
    pred_path = Path(predictions_dir)
    truth_path = Path(truth_dir)

    results = []
    total_cer = 0.0
    total_wer = 0.0
    evaluated_count = 0

    print(f"Loading predictions from: {predictions_dir}")
    print(f"Loading ground truth from: {truth_dir}")

    for pred_file in sorted(pred_path.glob("*.txt")):
        stem = pred_file.stem.replace("_PREDICTION", "")
        truth_file = truth_path / f"{stem}.txt"

        if not truth_file.exists():
            continue

        with open(pred_file, "r", encoding="utf-8", errors="ignore") as f:
            hyp = f.read().strip()
        with open(truth_file, "r", encoding="utf-8", errors="ignore") as f:
            ref = f.read().strip()

        cer = compute_cer(ref, hyp)
        wer = compute_wer(ref, hyp)

        total_cer += cer
        total_wer += wer
        evaluated_count += 1

        results.append({
            "sample_id": stem,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "ref_length_chars": len(ref),
            "hyp_length_chars": len(hyp)
        })

    if evaluated_count == 0:
        print("Warning: No matching prediction/ground truth pairs found.")
        return

    avg_cer = total_cer / evaluated_count
    avg_wer = total_wer / evaluated_count

    summary = {
        "model": "Gemma 3 4B Instruct (Q4_K_M)",
        "hardware": "Raspberry Pi 4B (4GB LPDDR4)",
        "evaluated_samples": evaluated_count,
        "average_cer": round(avg_cer, 4),
        "average_wer": round(avg_wer, 4),
        "sample_breakdown": results
    }

    os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print("           BENCHMARK EVALUATION SUMMARY                ")
    print("=======================================================")
    print(f"Evaluated Samples: {evaluated_count}")
    print(f"Average CER:       {avg_cer:.4f}")
    print(f"Average WER:       {avg_wer:.4f}")
    print(f"Detailed Report:   {output_report}")
    print("=======================================================\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <predictions_dir> <ground_truth_dir> [output_report_json]")
        sys.exit(1)

    preds = sys.argv[1]
    truth = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "benchmark_report.json"
    run_evaluation(preds, truth, out)
