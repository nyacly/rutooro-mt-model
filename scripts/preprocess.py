
"""Clean and split the dataset for training.

The input is expected to be a list of dictionaries containing a ``translation``
field.  Text is normalised by lowercasing and removing characters outside the
basic Latin alphabet and punctuation. Duplicate or empty pairs are discarded and
the final result is written to ``train.json``, ``dev.json`` and ``test.json``
inside the given output directory.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Tuple


_CLEAN_RE = re.compile(r"[^a-z0-9\s.,!?\-']+")


def _clean(text: str) -> str:
    text = text.lower()
    text = _CLEAN_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(input_path: str) -> List[Tuple[str, str]]:
    """Read and clean raw data returning a list of sentence pairs."""

    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pairs = []
    seen = set()
    for row in raw:
        if "translation" not in row:
            continue
        en = row["translation"].get("en") or row["translation"].get("english")
        tt = (
            row["translation"].get("ttj")
            or row["translation"].get("rutooro")
            or row["translation"].get("tt")
        )
        if not en or not tt:
            continue
        en_c = _clean(en)
        tt_c = _clean(tt)
        if not en_c or not tt_c:
            continue
        key = (en_c, tt_c)
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    return pairs


def split_and_save(pairs: List[Tuple[str, str]], output_dir: str) -> None:
    """Split data and write ``train.json``, ``dev.json`` and ``test.json``."""

    random.shuffle(pairs)
    n = len(pairs)
    train_end = int(n * 0.8)
    dev_end = int(n * 0.9)

    splits = {
        "train": pairs[:train_end],
        "dev": pairs[train_end:dev_end],
        "test": pairs[dev_end:],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        data = [
            {"translation": {"en": en, "ttj": tt}}
            for en, tt in subset
        ]
        (out_dir / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Raw JSON file produced by download script")
    parser.add_argument(
        "output_dir",
        default="data/clean",
        help="Directory to store train/dev/test JSON files",
    )
    args = parser.parse_args()

    pairs = preprocess(args.input)
    split_and_save(pairs, args.output_dir)


if __name__ == "__main__":
    main()
