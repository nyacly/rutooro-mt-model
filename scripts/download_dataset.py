
"""Download English↔Rutooro datasets and convert to a unified JSON format.

Two sources are supported:
 - The small dataset hosted on HuggingFace under
   ``michsethowusu/english-tooro_sentence-pairs_mt560``.
 - The MaNy-Eng collection available from
   https://rtg.isi.edu/many-eng/data-v1.html.

The resulting file is a list of dictionaries following the structure used by
``datasets.load_dataset(..., 'json')`` where each entry is of the form::

    {"translation": {"en": "english text", "ttj": "rutooro text"}}

The script only performs downloading/formatting so that downstream
preprocessing can operate on a single consistent file.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import requests
from datasets import load_dataset

HF_DATASET = "michsethowusu/english-tooro_sentence-pairs_mt560"


def load_from_hf() -> List[dict]:
    """Load dataset from the HuggingFace Hub."""
    ds = load_dataset(HF_DATASET, split="train")
    pairs = []
    for row in ds:
        en = row.get("english") or row.get("source")
        tt = row.get("rutooro") or row.get("target")
        if en and tt:
            pairs.append({"translation": {"en": en, "ttj": tt}})
    return pairs


def load_from_many_eng() -> List[dict]:
    """Attempt to download the Rutooro–English data from MaNy-Eng.

    The dataset webpage lists several tarballs for each language pair.  We use a
    small helper that tries to fetch ``ttj-eng-v1.tar.gz`` (the naming used on
    release ``data-v1.html``).  If the download fails an informative error is
    raised so users know they may have to retrieve the data manually.
    """

    url = (
        "https://rtg.isi.edu/many-eng/releases/many-eng-v1.0/ttj-eng-v1.tsv"
    )
    logging.info("Downloading MaNy-Eng dataset from %s", url)
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to download dataset (status code {resp.status_code})."
        )

    pairs = []
    for line in resp.text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            en, tt = parts[0], parts[1]
            pairs.append({"translation": {"en": en, "ttj": tt}})
    return pairs


def main(output_path: str = "data/english_rutooro.json", source: str = "hf"):
    if source == "hf":
        pairs = load_from_hf()
    elif source == "many-eng":
        pairs = load_from_many_eng()
    else:
        raise ValueError("source must be 'hf' or 'many-eng'")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/english_rutooro.json",
        help="Where to write the unified JSON file",
    )
    parser.add_argument(
        "--source",
        choices=["hf", "many-eng"],
        default="hf",
        help="Dataset source to download",
    )
    args = parser.parse_args()
    main(args.output, args.source)
