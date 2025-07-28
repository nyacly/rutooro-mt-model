"""Download English-Rutooro dataset from HuggingFace and save as JSON."""
from datasets import load_dataset
import json
from pathlib import Path

DATASET_NAME = "michsethowusu/english-tooro_sentence-pairs_mt560"


def main(output_path: str = "data/english_rutooro.json"):
    ds = load_dataset(DATASET_NAME, split="train")
    pairs = []
    for row in ds:
        en = row.get("english") or row.get("source")
        tt = row.get("rutooro") or row.get("target")
        if en and tt:
            pairs.append({"translation": {"en": en, "ttj": tt}})
    Path(output_path).write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
