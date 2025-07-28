"""Evaluate translation outputs using BLEU and chrF++ metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import evaluate


def _load_lines(path: str) -> list[str]:
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()]


def compute_metrics(pred_file: str, ref_file: str) -> Tuple[float, float]:
    """Return corpus BLEU and chrF++ scores for the given files."""

    preds = _load_lines(pred_file)
    refs = _load_lines(ref_file)

    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    bleu = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])["score"]
    chrf = chrf_metric.compute(predictions=preds, references=refs)["score"]

    return bleu, chrf


def print_samples(pred_file: str, ref_file: str, num: int = 3) -> None:
    preds = _load_lines(pred_file)[:num]
    refs = _load_lines(ref_file)[:num]
    for i, (p, r) in enumerate(zip(preds, refs), 1):
        print(f"[{i}]\n  REF: {r}\n  HYP: {p}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred', help='Predictions file')
    parser.add_argument('ref', help='Reference file')
    parser.add_argument('--samples', type=int, default=3, help='Number of example translations to display')
    args = parser.parse_args()
    bleu, chrf_score = compute_metrics(args.pred, args.ref)
    print(f"BLEU: {bleu:.2f}")
    print(f"chrF++: {chrf_score:.2f}\n")

    print_samples(args.pred, args.ref, args.samples)


if __name__ == '__main__':
    main()
