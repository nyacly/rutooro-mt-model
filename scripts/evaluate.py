"""Evaluate translation model with sacreBLEU and chrF++"""
import argparse
from pathlib import Path

from datasets import load_metric
from sacrebleu.metrics import CHRF


def compute_metrics(pred_file: str, ref_file: str):
    preds = [line.strip() for line in Path(pred_file).read_text(encoding='utf-8').splitlines()]
    refs = [line.strip() for line in Path(ref_file).read_text(encoding='utf-8').splitlines()]
    bleu = load_metric('bleu')
    bleu_result = bleu.compute(predictions=preds, references=[[r] for r in refs])
    chrf = CHRF()
    chrf_result = chrf.corpus_score(preds, [refs])
    return bleu_result['bleu'], chrf_result.score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred', help='Predictions file')
    parser.add_argument('ref', help='Reference file')
    args = parser.parse_args()
    bleu, chrf_score = compute_metrics(args.pred, args.ref)
    print(f"BLEU: {bleu:.2f}")
    print(f"chrF++: {chrf_score:.2f}")


if __name__ == '__main__':
    main()
